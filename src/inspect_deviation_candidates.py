import pandas as pd

from deviation_features import DATA_PATH, ROOT, haversine_m

d = pd.read_csv(ROOT / "outputs" / "deviation_candidates.csv")
raw = pd.read_csv(DATA_PATH)[
    ["trip_id", "ping_seq", "true_lat", "true_lon", "matched_segment_index"]
]
d = (
    d.merge(raw, on=["trip_id", "ping_seq"], how="left")
    .sort_values(["trip_id", "ping_seq"])
    .reset_index(drop=True)
)

d["timestamp"] = pd.to_datetime(d["timestamp"])
d["time_gap_s"] = d.groupby("trip_id", sort=False)["timestamp"].diff().dt.total_seconds()
# Offline evaluation only (uses true positions)
d["gps_error_m"] = haversine_m(d["true_lat"], d["true_lon"], d["lat"], d["lon"])
d["true_to_matched_m"] = haversine_m(
    d["true_lat"], d["true_lon"], d["matched_lat"], d["matched_lon"]
)

show = [
    "ping_seq", "time_gap_s", "route_distance_m", "gps_error_m",
    "true_to_matched_m", "reliability_outlier_prob", "reliability_flag",
    "matched_segment_index", "eval_is_outlier", "candidate_label",
]

cands = d[d["candidate_label"] == "potential_route_deviation"]
print("Candidates per trip:")
print(cands.groupby("trip_id").size().to_string())
print()

for (trip, run), g in cands.groupby(["trip_id", "run_id"]):
    lo = max(g.index.min() - 2, 0)
    hi = min(g.index.max() + 2, len(d) - 1)
    w = d.loc[lo:hi]
    w = w[w["trip_id"] == trip]
    print("=" * 70)
    print(f"Trip {trip}, run {run}: {len(g)} candidate pings (with 2 pings of context)")
    print("=" * 70)
    print(w[show].round(2).to_string(index=False))
    print()