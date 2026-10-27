
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "outputs" / "map_matched_gps_segments.csv"
MODEL_FILE = ROOT / "models" / "gps_reliability_model.joblib"
OUTPUT_DIR = ROOT / "outputs"

TARGET = "is_outlier"
FEATURES = [
    "route_distance_m",
    "time_gap_s",
    "step_distance_m",
    "implied_speed_kmh",
    "speed_change_kmh",
    "heading_change_deg",
]


def add_features(df):
    df = df.sort_values(["trip_id", "ping_seq"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    g = df.groupby("trip_id", sort=False)

    df["time_gap_s"] = g["timestamp"].diff().dt.total_seconds()

    lat1 = np.radians(g["lat"].shift())
    lon1 = np.radians(g["lon"].shift())
    lat2 = np.radians(df["lat"])
    lon2 = np.radians(df["lon"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    df["step_distance_m"] = (
        2 * 6371000 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    )

    df["implied_speed_kmh"] = (
        df["step_distance_m"]
        / df["time_gap_s"].where(df["time_gap_s"] > 0)
        * 3.6
    )

    df["speed_change_kmh"] = (
        df.groupby("trip_id", sort=False)["implied_speed_kmh"]
        .diff().abs()
    )

    y = np.sin(dlon) * np.cos(lat2)
    x = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    )
    df["heading_deg"] = (np.degrees(np.arctan2(y, x)) + 360) % 360

    previous_heading = (
        df.groupby("trip_id", sort=False)["heading_deg"].shift()
    )
    diff = (df["heading_deg"] - previous_heading).abs()
    df["heading_change_deg"] = np.minimum(diff, 360 - diff)

    df["route_distance_m"] = df["match_distance_m"]
    return df


def build_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])


def scores(y_true, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, predicted),
        "precision_outlier": precision_score(
            y_true, predicted, zero_division=0
        ),
        "recall_outlier": recall_score(
            y_true, predicted, zero_division=0
        ),
        "f1_outlier": f1_score(
            y_true, predicted, zero_division=0
        ),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = add_features(pd.read_csv(DATA_FILE))
    df = df.dropna(subset=[TARGET, "trip_id"]).copy()
    df[TARGET] = df[TARGET].astype(int)

    X = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    y = df[TARGET]
    groups = df["trip_id"]

    # Recreate the original held-out test trips.
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=42
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    train_groups = groups.iloc[train_idx]

    # Generate out-of-fold probabilities for training trips only.
    cv = StratifiedGroupKFold(
        n_splits=4, shuffle=True, random_state=123
    )
    oof_probabilities = np.full(len(X_train), np.nan)

    for fold, (fit_idx, valid_idx) in enumerate(
        cv.split(X_train, y_train, groups=train_groups), start=1
    ):
        y_fit = y_train.iloc[fit_idx]
        if y_fit.nunique() < 2:
            raise ValueError(
                f"Fold {fold} training data contains only one class."
            )

        fold_model = build_model()
        fold_model.fit(X_train.iloc[fit_idx], y_fit)

        classes = fold_model.named_steps["classifier"].classes_
        positive_column = int(np.where(classes == 1)[0][0])

        oof_probabilities[valid_idx] = fold_model.predict_proba(
            X_train.iloc[valid_idx]
        )[:, positive_column]

    valid = np.isfinite(oof_probabilities)
    if not valid.all():
        raise ValueError("Some training rows lack out-of-fold predictions.")

    # Choose the threshold maximizing out-of-fold F1 for class 1.
    thresholds = np.arange(0.05, 0.951, 0.01)
    threshold_results = [
        scores(y_train.to_numpy(), oof_probabilities, t)
        for t in thresholds
    ]
    threshold_df = pd.DataFrame(threshold_results)

    best_f1 = threshold_df["f1_outlier"].max()
    tied = threshold_df[
        np.isclose(threshold_df["f1_outlier"], best_f1)
    ]
    # If tied, choose the higher threshold (fewer positive predictions).
    best_threshold = float(tied["threshold"].max())

    threshold_df.to_csv(
        OUTPUT_DIR / "gps_threshold_cv_results.csv", index=False
    )

    # Fit once on all training trips and evaluate on the untouched test trips.
    final_model = build_model()
    final_model.fit(X_train, y_train)

    classes = final_model.named_steps["classifier"].classes_
    positive_column = int(np.where(classes == 1)[0][0])
    test_probabilities = final_model.predict_proba(X_test)[:, positive_column]

    default_predictions = (test_probabilities >= 0.50).astype(int)
    tuned_predictions = (test_probabilities >= best_threshold).astype(int)

    comparison = pd.DataFrame([
        scores(y_test, test_probabilities, 0.50),
        scores(y_test, test_probabilities, best_threshold),
    ])
    comparison.insert(0, "method", ["Default 0.50", "Tuned threshold"])
    comparison.to_csv(
        OUTPUT_DIR / "gps_threshold_test_comparison.csv", index=False
    )

    print("Held-out trips:")
    print(sorted(groups.iloc[test_idx].unique()))
    print(f"\nSelected threshold from training CV: {best_threshold:.2f}")
    print(f"Best training CV outlier F1: {best_f1:.4f}")

    print("\nTest comparison:")
    print(comparison.round(4).to_string(index=False))

    print("\nDefault threshold classification report:")
    print(classification_report(
        y_test, default_predictions,
        labels=[0, 1],
        target_names=["Normal GPS", "Synthetic outlier"],
        zero_division=0
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, default_predictions, labels=[0, 1]))

    print("\nTuned threshold classification report:")
    print(classification_report(
        y_test, tuned_predictions,
        labels=[0, 1],
        target_names=["Normal GPS", "Synthetic outlier"],
        zero_division=0
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, tuned_predictions, labels=[0, 1]))

    if y_test.nunique() == 2:
        print(
            "\nTest ROC-AUC (threshold independent):",
            round(roc_auc_score(y_test, test_probabilities), 4)
        )

    test_output = df.iloc[test_idx][
        ["trip_id", "ping_seq", TARGET]
    ].copy()
    test_output["outlier_probability"] = test_probabilities
    test_output["prediction_default_050"] = default_predictions
    test_output["prediction_tuned"] = tuned_predictions
    test_output.to_csv(
        OUTPUT_DIR / "gps_reliability_threshold_test_predictions.csv",
        index=False
    )

    # Save a deployable model with the selected threshold.
    joblib.dump({
        "model": final_model,
        "feature_columns": FEATURES,
        "threshold": best_threshold,
        "target": TARGET,
    }, ROOT / "models" / "gps_reliability_model_tuned.joblib")

    print("\nSaved:")
    print(OUTPUT_DIR / "gps_threshold_cv_results.csv")
    print(OUTPUT_DIR / "gps_threshold_test_comparison.csv")
    print(OUTPUT_DIR / "gps_reliability_threshold_test_predictions.csv")
    print(ROOT / "models" / "gps_reliability_model_tuned.joblib")


if __name__ == "__main__":
    main()
