import numpy as np
import pandas as pd

from deviation_features import ROOT, haversine_m

CAND = ROOT / "outputs" / "deviation_candidates_exact.csv"
META = ROOT / "data" / "synthetic" / "trip_metadata.csv"
OUT_TRIP = ROOT / "outputs" / "trip_summary.csv"
OUT_ROUTE = ROOT / "outputs" / "route_summary.csv"

d = pd.read_csv(CAND)
d["timestamp"] = pd.to_datetime(d["timestamp"])
d = d.sort_values(["trip_id", "ping_seq"])

# Time and ping counts use every ping of the trip
per_trip = d.groupby("trip_id").agg(
    pings=("ping_seq", "size"),
    noise_pings=("candidate_label", lambda s: int((s == "likely_gps_noise").sum())),
    start=("timestamp", "min"),
    end=("timestamp", "max"),
)
per_trip["duration_min"] = (per_trip["end"] - per_trip["start"]).dt.total_seconds() / 60

# Distance uses only pings that were NOT flagged as likely GPS noise,
# measured between consecutive matched (on-route) positions
clean = d[d["candidate_label"] != "likely_gps_noise"].copy()
g = clean.groupby("trip_id", sort=False)
clean["step_m"] = haversine_m(
    g["matched_lat"].shift(), g["matched_lon"].shift(),
    clean["matched_lat"], clean["matched_lon"],
)
per_trip["observed_distance_m"] = clean.groupby("trip_id")["step_m"].sum()

meta = pd.read_csv(META)[
    ["trip_id", "route_short_name", "distance_m", "avg_speed_kmph"]
].rename(columns={"distance_m": "planned_distance_m",
                  "avg_speed_kmph": "planned_speed_kmph"})

t = per_trip.reset_index().merge(meta, on="trip_id", how="left")
t["observed_speed_kmph"] = t["observed_distance_m"] / 1000 / (t["duration_min"] / 60)
t["distance_diff_pct"] = (t["observed_distance_m"] / t["planned_distance_m"] - 1) * 100
t["speed_ratio"] = t["observed_speed_kmph"] / t["planned_speed_kmph"]

cols = ["trip_id", "route_short_name", "pings", "noise_pings", "duration_min",
        "planned_distance_m", "observed_distance_m", "distance_diff_pct",
        "planned_speed_kmph", "observed_speed_kmph", "speed_ratio"]
t = t[cols]
t.to_csv(OUT_TRIP, index=False)

print("PER-TRIP SUMMARY (first 15 trips)")
print(t.head(15).round(2).to_string(index=False))
print()
print("Across all", len(t), "trips:")
print(t[["duration_min", "planned_distance_m", "observed_distance_m",
         "distance_diff_pct", "planned_speed_kmph", "observed_speed_kmph",
         "speed_ratio"]].describe().loc[["mean", "min", "50%", "max"]].round(2).to_string())

r = (t.groupby("route_short_name")
       .agg(trips=("trip_id", "size"),
            mean_distance_km=("observed_distance_m", lambda s: s.mean() / 1000),
            mean_duration_min=("duration_min", "mean"),
            mean_observed_speed=("observed_speed_kmph", "mean"),
            mean_planned_speed=("planned_speed_kmph", "mean"))
       .sort_values("mean_observed_speed", ascending=False)
       .reset_index())
r.to_csv(OUT_ROUTE, index=False)
print()
print("ROUTES RANKED BY OBSERVED AVERAGE SPEED")
print(r.round(2).to_string(index=False))
print()
print("Saved:", OUT_TRIP)
print("Saved:", OUT_ROUTE)