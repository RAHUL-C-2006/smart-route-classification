from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(r"C:\smart route classification")
MODEL_PATH = ROOT / "models" / "gps_reliability_model_tuned.joblib"
DATA_PATH = ROOT / "outputs" / "map_matched_gps_segments.csv"

HELD_OUT_TRIPS = [
    "synth_0005", "synth_0007", "synth_0013", "synth_0016", "synth_0017",
    "synth_0020", "synth_0027", "synth_0028", "synth_0038", "synth_0040",
]


def haversine_m(lat1, lon1, lat2, lon2):
    """Standard haversine distance in metres (assumed; verified below)."""
    r = 6371000.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * r * np.arcsin(np.sqrt(a))


def add_features(df):
    """Copied from src/gps_reliability_model.py."""
    df = df.sort_values(["trip_id", "ping_seq"]).copy()

    grouped = df.groupby("trip_id", sort=False)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["time_gap_s"] = grouped["timestamp"].diff().dt.total_seconds()

    df["step_distance_m"] = haversine_m(
        grouped["lat"].shift(),
        grouped["lon"].shift(),
        df["lat"],
        df["lon"],
    )

    df["implied_speed_kmh"] = (
        df["step_distance_m"] / df["time_gap_s"].replace(0, np.nan) * 3.6
    )

    df["speed_change_kmh"] = (
        df.groupby("trip_id", sort=False)["implied_speed_kmh"].diff().abs()
    )

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

    df["route_distance_m"] = df["match_distance_m"]

    return df


def load_model_bundle():
    return joblib.load(MODEL_PATH)


def add_reliability_scores(df, bundle):
    """Adds reliability_outlier_prob and reliability_flag (model threshold)."""
    cols = bundle["feature_columns"]
    df = df.copy()
    df["reliability_outlier_prob"] = bundle["model"].predict_proba(df[cols])[:, 1]
    df["reliability_flag"] = df["reliability_outlier_prob"] >= bundle["threshold"]
    return df


if __name__ == "__main__":
    # Verification: reproduce the tuned held-out result from the handover.
    bundle = load_model_bundle()
    raw = pd.read_csv(DATA_PATH)
    feats = add_features(raw)
    scored = add_reliability_scores(feats, bundle)

    test = scored[scored["trip_id"].isin(HELD_OUT_TRIPS)]
    actual = test["is_outlier"].astype(bool)   # evaluation only
    pred = test["reliability_flag"]

    tn = int(((~actual) & (~pred)).sum())
    fp = int(((~actual) & pred).sum())
    fn = int((actual & (~pred)).sum())
    tp = int((actual & pred).sum())

    print("Held-out observations:", len(test), "(expected 4364)")
    print("Held-out outliers    :", int(actual.sum()), "(expected 168)")
    print("Threshold used       :", bundle["threshold"])
    print("Confusion matrix [[TN, FP], [FN, TP]]:", [[tn, fp], [fn, tp]])
    print("Expected             : [[4170, 26], [99, 69]]")
    ok = (tn, fp, fn, tp) == (4170, 26, 99, 69)
    print()
    print("FEATURE PIPELINE VERIFIED" if ok else "MISMATCH - do not proceed; share this output")