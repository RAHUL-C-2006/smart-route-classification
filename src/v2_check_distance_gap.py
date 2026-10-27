import pandas as pd

from deviation_features import ROOT

OUT = ROOT / "outputs" / "v2"

t = pd.read_csv(OUT / "v2_trip_summary.csv")[
    ["trip_id", "route_short_name", "planned_distance_m", "observed_distance_m"]]
m = pd.read_csv(OUT / "v2_map_matched_exact.csv", usecols=["trip_id", "ping_seq", "true_dist_m"])
c = pd.read_csv(OUT / "v2_deviation_candidates.csv", usecols=["trip_id", "ping_seq", "candidate_label"])
m = m.merge(c, on=["trip_id", "ping_seq"], how="left")
clean = m[m["candidate_label"] != "likely_gps_noise"]

ends = clean.groupby("trip_id")["true_dist_m"].agg(first_m="min", last_m="max").reset_index()
d = t.merge(ends, on="trip_id")
route = d["planned_distance_m"]
d["start_gap_m"] = d["first_m"]
d["end_gap_m"] = route - d["last_m"]
d["covered_m"] = d["last_m"] - d["first_m"]       # true distance between first and last used ping
d["ends_missing_pct"] = 100 * (d["start_gap_m"] + d["end_gap_m"]) / route
d["jitter_pct"] = 100 * (d["observed_distance_m"] - d["covered_m"]) / route
d["total_err_pct"] = 100 * (d["observed_distance_m"] / route - 1)

print("WHERE THE DISTANCE SHORTFALL COMES FROM (percent of route length, mean over 120 trips)")
print(f"  Route length not covered by first/last ping (ends missing): -{d['ends_missing_pct'].mean():.2f}%")
print(f"  Extra distance from GPS wobble between pings (jitter):      {d['jitter_pct'].mean():+.2f}%")
print(f"  Net (should match the earlier -1.68%):                      {d['total_err_pct'].mean():+.2f}%")
print()

d["length_group"] = pd.cut(route / 1000, [0, 5, 15, 30, 100],
                           labels=["under 5 km", "5-15 km", "15-30 km", "over 30 km"])
print("BY ROUTE LENGTH")
print(d.groupby("length_group", observed=True)
       .agg(trips=("trip_id", "size"), ends_missing_pct=("ends_missing_pct", "mean"),
            jitter_pct=("jitter_pct", "mean"), net_err_pct=("total_err_pct", "mean"))
       .round(2).to_string())
print()
print("5 TRIPS WITH THE LARGEST SHORTFALL")
cols = ["trip_id", "route_short_name", "planned_distance_m", "start_gap_m", "end_gap_m",
        "jitter_pct", "total_err_pct"]
print(d.sort_values("total_err_pct")[cols].head(5).round(1).to_string(index=False))