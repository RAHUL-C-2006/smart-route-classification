
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "outputs" / "map_matched_gps_segments.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

TARGET = "is_outlier"

def haversine_m(lat1, lon1, lat2, lon2):
    """Distance between two latitude/longitude points in metres."""
    lat1, lon1, lat2, lon2 = map(
        np.radians, [lat1, lon1, lat2, lon2]
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    return 2 * 6371000 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def add_features(df):
    df = df.sort_values(["trip_id", "ping_seq"]).copy()

    grouped = df.groupby("trip_id", sort=False)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["time_gap_s"] = grouped["timestamp"].diff().dt.total_seconds()

    df["step_distance_m"] = haversine_m(
        grouped["lat"].shift(),
        grouped["lon"].shift(),
        df["lat"],
        df["lon"]
    )

    # Speed between consecutive GPS observations
    df["implied_speed_kmh"] = (
        df["step_distance_m"] / df["time_gap_s"].replace(0, np.nan) * 3.6
    )

    df["speed_change_kmh"] = (
        df.groupby("trip_id", sort=False)["implied_speed_kmh"].diff().abs()
    )

    # Heading/bearing between consecutive observed GPS points
    prev_lat = np.radians(grouped["lat"].shift())
    prev_lon = np.radians(grouped["lon"].shift())
    curr_lat = np.radians(df["lat"])
    curr_lon = np.radians(df["lon"])

    dlon = curr_lon - prev_lon
    y = np.sin(dlon) * np.cos(curr_lat)
    x = (
        np.cos(prev_lat) * np.sin(curr_lat)
        - np.sin(prev_lat) * np.cos(curr_lat) * np.cos(dlon)
    )

    df["heading_deg"] = (np.degrees(np.arctan2(y, x)) + 360) % 360

    previous_heading = df.groupby("trip_id", sort=False)["heading_deg"].shift()
    heading_diff = (df["heading_deg"] - previous_heading).abs()
    df["heading_change_deg"] = np.minimum(heading_diff, 360 - heading_diff)

    # Observable map-matching feature
    df["route_distance_m"] = df["match_distance_m"]

    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)

    if TARGET not in df.columns:
        raise ValueError(
            f"Missing '{TARGET}' in {DATA_FILE}. "
            "Check that the map-matched output preserves the synthetic label."
        )

    df = add_features(df)
    df = df.dropna(subset=[TARGET, "trip_id"]).copy()
    df[TARGET] = df[TARGET].astype(int)

    feature_cols = [
        "route_distance_m",
        "time_gap_s",
        "step_distance_m",
        "implied_speed_kmh",
        "speed_change_kmh",
        "heading_change_deg"
    ]

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = df[TARGET]
    groups = df["trip_id"]

    # Hold out entire trips so rows from the same trip
    # cannot appear in both training and testing.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42
    )

    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nHeld-out trips:")
    print(sorted(groups.iloc[test_idx].unique()))

    print("\nTrain/test observations:", len(train_idx), "/", len(test_idx))
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Normal GPS", "Synthetic outlier"],
        zero_division=0
    ))

    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    if y_test.nunique() == 2:
        print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
    else:
        print("ROC-AUC unavailable: test set contains only one class.")

    # Save predictions for later inspection
    evaluation = df.iloc[test_idx][
        ["trip_id", "ping_seq", TARGET]
    ].copy()
    evaluation["predicted_outlier"] = predictions
    evaluation["outlier_probability"] = probabilities
    evaluation.to_csv(
        OUTPUT_DIR / "gps_reliability_test_predictions.csv",
        index=False
    )

    # Save the trained model and feature list
    joblib.dump(
        {"model": model, "feature_columns": feature_cols},
        MODEL_DIR / "gps_reliability_model.joblib"
    )

    # Permutation importance on held-out trips
    importance = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        scoring="f1",
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std
    }).sort_values("importance_mean", ascending=False)

    importance_df.to_csv(
        OUTPUT_DIR / "gps_reliability_feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=importance_df,
        x="importance_mean",
        y="feature"
    )
    plt.title("GPS Reliability Model: Permutation Importance")
    plt.xlabel("Decrease in F1 when feature is shuffled")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "gps_reliability_feature_importance.png",
        dpi=200
    )
    plt.close()

    print("\nSaved model:", MODEL_DIR / "gps_reliability_model.joblib")
    print("Saved test predictions.")
    print("Saved feature importance and plot.")


if __name__ == "__main__":
    main()
