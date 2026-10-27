import pandas as pd

from deviation_features import ROOT, haversine_m

CAND = ROOT / "outputs" / "deviation_candidates_exact.csv"
TRIPS = ROOT / "outputs" / "trip_summary.csv"
OUT = ROOT / "outputs" / "trip_dwell_summary.csv"

STILL_KMH = 3.0   # rough rule: slower than this between pings = standing still

d = pd.read_csv(CAND)
d["timestamp"] = pd.to_datetime(d["timestamp"])
c = d[d["candidate_label"] != "likely_gps_noise"].sort_values(["trip_id", "ping_seq"]).copy()

g = c.groupby("trip_id", sort=False)
c["dt_s"] = g["timestamp"].diff().dt.total_seconds()
c["step_m"] = haversine_m(g["matched_lat"].shift(), g["matched_lon"].shift(),
                          c["matched_lat"], c["matched_lon"])
c = c.dropna(subset=["dt_s", "step_m"])
c = c[c["dt_s"] > 0]
c["speed_kmh"] = c["step_m"] / c["dt_s"] * 3.6
c["still"] = c["speed_kmh"] < STILL_KMH

prev = c.groupby("trip_id", sort=False)["still"].shift(fill_value=False)
c["stop_start"] = c["still"] & ~prev

def summarise(x):
    total = x["dt_s"].sum()
    still_t = x.loc[x["still"], "dt_s"].sum()
    move_t = total - still_t
    dist = x["step_m"].sum()
    return pd.Series({
        "standing_share_pct": 100 * still_t / total,
        "standing_stops": int(x["stop_start"].sum()),
        "moving_min": move_t / 60,
        "moving_speed_kmph": (dist / 1000) / (move_t / 3600) if move_t > 0 else float("nan"),
    })

s = c.groupby("trip_id").apply(summarise).reset_index()
t = pd.read_csv(TRIPS)[["trip_id", "route_short_name", "planned_speed_kmph",
                        "observed_speed_kmph"]]
s = t.merge(s, on="trip_id")
s["moving_vs_planned"] = s["moving_speed_kmph"] / s["planned_speed_kmph"]
s.to_csv(OUT, index=False)

print("FIRST 12 TRIPS")
print(s.head(12).round(2).to_string(index=False))
print()
print("ACROSS ALL", len(s), "TRIPS")
print(s[["standing_share_pct", "standing_stops", "planned_speed_kmph",
         "observed_speed_kmph", "moving_speed_kmph", "moving_vs_planned"]]
      .describe().loc[["mean", "min", "50%", "max"]].round(2).to_string())
print()
print("Saved:", OUT)