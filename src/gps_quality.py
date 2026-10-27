
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
GPS_PATH = (
    PROJECT_DIR
    / "data"
    / "synthetic"
    / "synthetic_raw_gps_traces.csv"
)
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

gps = pd.read_csv(GPS_PATH)

# Haversine distance between two latitude/longitude coordinates.
# Returns distance in metres.
def haversine_m(lat1, lon1, lat2, lon2):
    earth_radius_m = 6_371_000

    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    a = np.clip(a, 0, 1)
    return 2 * earth_radius_m * np.arcsin(np.sqrt(a))


gps["gps_error_m"] = haversine_m(
    gps["true_lat"],
    gps["true_lon"],
    gps["lat"],
    gps["lon"],
)

# Summarise GPS error for every trip
trip_summary = (
    gps.groupby("trip_id")
    .agg(
        observations=("gps_error_m", "size"),
        mean_error_m=("gps_error_m", "mean"),
        median_error_m=("gps_error_m", "median"),
        p95_error_m=("gps_error_m", lambda s: s.quantile(0.95)),
        max_error_m=("gps_error_m", "max"),
        labelled_outliers=("is_outlier", "sum"),
    )
    .reset_index()
)

trip_summary.to_csv(
    OUTPUT_DIR / "gps_quality_by_trip.csv",
    index=False,
)

print("OVERALL GPS ERROR (metres)")
print(gps["gps_error_m"].describe().round(2))

print("\nOUTLIER LABEL VS ERROR")
gps["is_outlier"] = (
    gps["is_outlier"].astype(str).str.lower().eq("true")
)
print(
    gps.groupby("is_outlier")["gps_error_m"]
    .agg(["count", "mean", "median", "max"])
    .round(2)
)

print("\nTRIPS WITH HIGHEST 95TH-PERCENTILE ERROR")
print(
    trip_summary.sort_values("p95_error_m", ascending=False)
    .head(10)
    .round(2)
    .to_string(index=False)
)

print("\nSaved:", OUTPUT_DIR / "gps_quality_by_trip.csv")
