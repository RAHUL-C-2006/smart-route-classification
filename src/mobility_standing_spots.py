import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from deviation_features import ROOT, haversine_m

CAND = ROOT / "outputs" / "deviation_candidates_exact.csv"
DWELL = ROOT / "outputs" / "trip_dwell_summary.csv"
STOPS = ROOT / "data" / "gtfs" / "stops.txt"
OUT_PTS = ROOT / "outputs" / "standing_away_from_stops.csv"
OUT_CELLS = ROOT / "outputs" / "standing_spot_cells.csv"
OUT_MAP = ROOT / "outputs" / "standing_away_from_stops_map.png"

STILL_KMH = 3.0      # same rough rule as before
NEAR_STOP_M = 40.0   # my assumption: standing within 40 m of a stop is normal waiting
CELL_DEG = 0.002     # about 200 m grid squares

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

routes = pd.read_csv(DWELL)[["trip_id", "route_short_name"]]
c = c.merge(routes, on="trip_id", how="left")

# Distance from each reading to the nearest bus stop
stops = pd.read_csv(STOPS, usecols=["stop_lat", "stop_lon"]).dropna()
LAT0, LON0, R = stops["stop_lat"].mean(), stops["stop_lon"].mean(), 6371000.0

def to_xy(lat, lon):
    return np.column_stack([
        np.radians(np.asarray(lon, float) - LON0) * np.cos(np.radians(LAT0)) * R,
        np.radians(np.asarray(lat, float) - LAT0) * R,
    ])

tree = cKDTree(to_xy(stops["stop_lat"].values, stops["stop_lon"].values))
c["nearest_stop_m"] = tree.query(to_xy(c["matched_lat"].values, c["matched_lon"].values))[0]

still = c[c["still"]].copy()
still["near_stop"] = still["nearest_stop_m"] <= NEAR_STOP_M
total_still_min = still["dt_s"].sum() / 60
away = still[~still["near_stop"]].copy()

print("STANDING TIME AND BUS STOPS")
print(f"Total standing time: {total_still_min:,.0f} min")
print(f"  within {NEAR_STOP_M:.0f} m of a stop (normal waiting): "
      f"{still.loc[still['near_stop'], 'dt_s'].sum() / 60:,.0f} min")
print(f"  away from any stop: {away['dt_s'].sum() / 60:,.0f} min "
      f"({100 * away['dt_s'].sum() / 60 / total_still_min:.1f}% of standing time)")
print()

per_trip = (away.groupby("trip_id")["dt_s"].sum() / 60).rename("away_standing_min")
print("Away-from-stop standing per trip (minutes): mean "
      f"{per_trip.mean():.1f}, max {per_trip.max():.1f}, trips with any: {len(per_trip)} of {c['trip_id'].nunique()}")
print()

# Grid squares where standing away from stops happens
away["cell"] = (np.floor(away["matched_lat"] / CELL_DEG).astype(int).astype(str) + "_" +
                np.floor(away["matched_lon"] / CELL_DEG).astype(int).astype(str))
cells = (away.groupby("cell")
             .agg(standing_min=("dt_s", lambda s: s.sum() / 60),
                  trips=("trip_id", "nunique"),
                  routes=("route_short_name", "nunique"),
                  lat=("matched_lat", "mean"), lon=("matched_lon", "mean"))
             .sort_values("standing_min", ascending=False).reset_index())
cells.to_csv(OUT_CELLS, index=False)
away[["trip_id", "route_short_name", "ping_seq", "timestamp", "matched_lat",
      "matched_lon", "dt_s", "nearest_stop_m"]].to_csv(OUT_PTS, index=False)

print(f"Grid squares with away-from-stop standing: {len(cells)}")
print(f"  hit by 2 or more trips: {int((cells['trips'] >= 2).sum())}")
print()
print("TOP 10 SQUARES BY STANDING MINUTES")
print(cells.head(10).round(4).to_string(index=False))

# Map
fig, ax = plt.subplots(figsize=(9, 9))
ax.scatter(c["matched_lon"], c["matched_lat"], s=1, color="lightgray", label="all bus positions")
ax.scatter(away["matched_lon"], away["matched_lat"], s=8, color="#C00000",
           label=f"standing, more than {NEAR_STOP_M:.0f} m from a stop")
ax.set_aspect(1 / np.cos(np.radians(LAT0)))
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Standing away from bus stops (synthetic data, rough estimate)")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(OUT_MAP, dpi=130)

print()
print("Saved:", OUT_PTS)
print("Saved:", OUT_CELLS)
print("Saved:", OUT_MAP)