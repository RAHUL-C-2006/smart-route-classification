import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from deviation_features import ROOT, haversine_m

CAND = ROOT / "outputs" / "deviation_candidates_exact.csv"
DWELL = ROOT / "outputs" / "trip_dwell_summary.csv"
OUT_PNG = ROOT / "outputs" / "mobility_overview.png"
OUT_HOUR = ROOT / "outputs" / "hourly_pattern.csv"
STILL_KMH = 3.0   # same rough rule as the previous step

d = pd.read_csv(CAND)
d["timestamp"] = pd.to_datetime(d["timestamp"])
c = d[d["candidate_label"] != "likely_gps_noise"].sort_values(["trip_id", "ping_seq"]).copy()
g = c.groupby("trip_id", sort=False)
c["dt_s"] = g["timestamp"].diff().dt.total_seconds()
c["step_m"] = haversine_m(g["matched_lat"].shift(), g["matched_lon"].shift(),
                          c["matched_lat"], c["matched_lon"])
c = c.dropna(subset=["dt_s", "step_m"])
c = c[c["dt_s"] > 0].copy()
c["still"] = (c["step_m"] / c["dt_s"] * 3.6) < STILL_KMH
c["hour"] = c["timestamp"].dt.hour
c["still_s"] = c["dt_s"].where(c["still"], 0.0)
c["move_s"] = c["dt_s"].where(~c["still"], 0.0)
c["move_m"] = c["step_m"].where(~c["still"], 0.0)

h = c.groupby("hour").agg(pings=("dt_s", "size"), total_s=("dt_s", "sum"),
                          still_s=("still_s", "sum"), move_s=("move_s", "sum"),
                          move_m=("move_m", "sum"))
h["standing_share_pct"] = 100 * h["still_s"] / h["total_s"]
h["moving_speed_kmph"] = (h["move_m"] / 1000) / (h["move_s"] / 3600)
h = h[["pings", "standing_share_pct", "moving_speed_kmph"]].reset_index()
h.to_csv(OUT_HOUR, index=False)

s = pd.read_csv(DWELL)
r = (s.groupby("route_short_name")
       .agg(moving=("moving_speed_kmph", "mean"), standing=("standing_share_pct", "mean"))
       .sort_values("moving"))

fig, ax = plt.subplots(2, 2, figsize=(15, 10))

ax[0, 0].barh(r.index, r["moving"], color="#2E75B6")
ax[0, 0].set_title("Moving speed by route (km/h, average of 2 trips)")
ax[0, 0].tick_params(axis="y", labelsize=8)

ax[0, 1].scatter(s["planned_speed_kmph"], s["moving_speed_kmph"], color="#2E75B6")
lim = max(s["planned_speed_kmph"].max(), s["moving_speed_kmph"].max()) + 1
ax[0, 1].plot([0, lim], [0, lim], "--", color="gray", label="perfect agreement")
ax[0, 1].set_xlabel("Planned speed (km/h)")
ax[0, 1].set_ylabel("Moving speed (km/h)")
ax[0, 1].set_title("Planned vs moving speed, one dot per trip")
ax[0, 1].legend()

ax[1, 0].plot(h["hour"], h["moving_speed_kmph"], marker="o", color="#2E75B6")
ax[1, 0].set_title("Moving speed by hour of day (km/h)")
ax[1, 0].set_xlabel("Hour of day")

ax[1, 1].plot(h["hour"], h["standing_share_pct"], marker="o", color="#C55A11")
ax[1, 1].set_title("Share of time standing still, by hour (%)")
ax[1, 1].set_xlabel("Hour of day")

fig.suptitle("Phase 3 first look: synthetic data, standing = under 3 km/h (rough estimate)")
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=130)

print("HOURLY PATTERN (pings = number of readings in that hour)")
print(h.round(2).to_string(index=False))
print()
print("ROUTES: slowest 3 and fastest 3 by moving speed")
print(pd.concat([r.head(3), r.tail(3)]).round(2).to_string())
print()
print("Saved:", OUT_PNG)
print("Saved:", OUT_HOUR)