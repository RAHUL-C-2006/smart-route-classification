import pandas as pd

from deviation_features import ROOT, haversine_m

V2 = ROOT / "data" / "synthetic_v2"
CAND = ROOT / "outputs" / "v2" / "v2_deviation_candidates.csv"
OUT_TRIP = ROOT / "outputs" / "v2" / "v2_trip_summary.csv"
OUT_ROUTE = ROOT / "outputs" / "v2" / "v2_route_summary.csv"

c = pd.read_csv(CAND)
c["timestamp"] = pd.to_datetime(c["timestamp"])
c = c.sort_values(["trip_id", "ping_seq"])

per = c.groupby("trip_id").agg(
    pings=("ping_seq", "size"),
    noise_pings=("candidate_label", lambda s: int((s == "likely_gps_noise").sum())),
    start=("timestamp", "min"),
    end=("timestamp", "max"),
)
per["duration_min"] = (per["end"] - per["start"]).dt.total_seconds() / 60

# Distance: only pings not flagged as likely GPS noise, measured between matched route points
along = pd.read_csv(ROOT / "outputs" / "v2" / "v2_map_matched_exact.csv",
                    usecols=["trip_id", "ping_seq", "matched_along_m"])
clean = c[c["candidate_label"] != "likely_gps_noise"].merge(along, on=["trip_id", "ping_seq"], how="left")
per["observed_distance_m"] = clean.groupby("trip_id")["matched_along_m"].agg(lambda s: s.max() - s.min())

meta = pd.read_csv(V2 / "trip_metadata_v2.csv")[
    ["trip_id", "route_short_name", "distance_m", "avg_speed_kmph",
     "scheduled_duration_min", "has_slow_zone", "n_stops", "sim_duration_min"]
].rename(columns={"distance_m": "planned_distance_m",
                  "avg_speed_kmph": "timetable_speed_kmph",
                  "sim_duration_min": "true_duration_min"})

t = per.reset_index().merge(meta, on="trip_id", how="left")
t["observed_speed_kmph"] = t["observed_distance_m"] / 1000 / (t["duration_min"] / 60)
t["delay_min"] = t["duration_min"] - t["scheduled_duration_min"]
t["delay_pct"] = 100 * t["delay_min"] / t["scheduled_duration_min"]
t["mid_hour"] = (t["start"].dt.hour + t["start"].dt.minute / 60 + t["duration_min"] / 120)
t["period"] = pd.cut(t["mid_hour"], [0, 8, 10.5, 17, 20, 24],
                     labels=["before 8", "morning peak 8-10:30", "midday 10:30-17",
                             "evening peak 17-20", "after 20"], right=False)
t["distance_err_pct"] = 100 * (t["observed_distance_m"] / t["planned_distance_m"] - 1)
t["duration_err_pct"] = 100 * (t["duration_min"] / t["true_duration_min"] - 1)
t.to_csv(OUT_TRIP, index=False)

cols = ["trip_id", "route_short_name", "duration_min", "scheduled_duration_min",
        "delay_pct", "observed_speed_kmph", "timetable_speed_kmph", "period"]
print("FIRST 10 TRIPS")
print(t[cols].head(10).round(2).to_string(index=False))

print()
print("CHECK AGAINST THE GENERATOR'S KNOWN TRUTH")
print(f"  Observed distance vs route length: mean {t['distance_err_pct'].mean():+.2f}% | "
      f"largest {t['distance_err_pct'].abs().max():.2f}%")
print(f"  Observed duration vs true duration: mean {t['duration_err_pct'].mean():+.2f}% | "
      f"largest {t['duration_err_pct'].abs().max():.2f}%")

print()
print("DELAY BY TIME OF DAY (middle of the journey)")
print(t.groupby("period", observed=True)
       .agg(trips=("trip_id", "size"), mean_delay_pct=("delay_pct", "mean"),
            observed_speed=("observed_speed_kmph", "mean"),
            timetable_speed=("timetable_speed_kmph", "mean"))
       .round(2).to_string())

print()
print("ROUTES WITH A PLANTED SLOW ZONE vs WITHOUT")
print(t.groupby("has_slow_zone")
       .agg(trips=("trip_id", "size"), mean_delay_pct=("delay_pct", "mean"),
            observed_speed=("observed_speed_kmph", "mean"))
       .round(2).to_string())

r = (t.groupby("route_short_name")
       .agg(trips=("trip_id", "size"), has_slow_zone=("has_slow_zone", "first"),
            n_stops=("n_stops", "first"), mean_delay_pct=("delay_pct", "mean"),
            observed_speed=("observed_speed_kmph", "mean"),
            timetable_speed=("timetable_speed_kmph", "mean"))
       .sort_values("mean_delay_pct", ascending=False).reset_index())
r.to_csv(OUT_ROUTE, index=False)
print()
print("ROUTES BY AVERAGE DELAY (most delayed first)")
print(r.round(2).to_string(index=False))
print()
print("Saved:", OUT_TRIP)
print("Saved:", OUT_ROUTE)