import numpy as np
import pandas as pd

from deviation_features import ROOT, haversine_m

V2 = ROOT / "data" / "synthetic_v2"
OUT = ROOT / "outputs" / "v2"

SEC_M = 200.0            # section length
STOP_BUFFER_M = 30.0     # sections this close to a bus stop are excluded (waiting is normal)
HOT_RATIO = 1.8          # hotspot: average section time at least 1.8x the trip's typical section
SLOW_TRIP_RATIO = 1.5    # a trip counts as "slow here" if its ratio is at least this
MIN_SHARE = 0.6          # share of trips that must be slow there
MIN_TRIPS = 4            # minimum trips covering a section

m = pd.read_csv(
    OUT / "v2_map_matched_exact.csv", dtype={"shape_id": str},
    usecols=["trip_id", "ping_seq", "timestamp", "shape_id", "route_short_name",
             "matched_lat", "matched_lon", "matched_segment_index", "true_dist_m"])
cand = pd.read_csv(OUT / "v2_deviation_candidates.csv",
                   usecols=["trip_id", "ping_seq", "candidate_label"])
m = m.merge(cand, on=["trip_id", "ping_seq"], how="left")
m = m[m["candidate_label"] != "likely_gps_noise"].copy()
m["timestamp"] = pd.to_datetime(m["timestamp"])

# Distance along the route for every matched ping
sh = pd.read_csv(ROOT / "data" / "gtfs" / "shapes.txt",
                 usecols=["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"],
                 dtype={"shape_id": str})
sh = sh[sh["shape_id"].isin(m["shape_id"].unique())]
m["along_m"] = np.nan
for sid, s in sh.groupby("shape_id"):
    s = s.sort_values("shape_pt_sequence")
    lat, lon = s["shape_pt_lat"].to_numpy(), s["shape_pt_lon"].to_numpy()
    cum = np.concatenate([[0.0], np.cumsum(haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:]))])
    mask = m["shape_id"] == sid
    seg = m.loc[mask, "matched_segment_index"].to_numpy()
    m.loc[mask, "along_m"] = cum[seg] + haversine_m(
        lat[seg], lon[seg],
        m.loc[mask, "matched_lat"].to_numpy(), m.loc[mask, "matched_lon"].to_numpy())

e = (m["along_m"] - m["true_dist_m"]).abs()    # true_dist_m used for this check only
print(f"Check: distance along route vs the generator's truth: median error {e.median():.1f} m, "
      f"95th percentile {e.quantile(0.95):.1f} m")

# Time taken to cross each section, per trip
m = m.sort_values(["trip_id", "ping_seq"])
rows = []
for trip, g in m.groupby("trip_id", sort=False):
    sid, rname = g["shape_id"].iloc[0], g["route_short_name"].iloc[0]
    along = np.maximum.accumulate(g["along_m"].to_numpy()) + np.arange(len(g)) * 1e-6
    ts = (g["timestamp"] - g["timestamp"].iloc[0]).dt.total_seconds().to_numpy()
    first, last = int(np.ceil(along[0] / SEC_M)), int(np.floor(along[-1] / SEC_M))
    if last - first < 3:
        continue
    tb = np.interp(np.arange(first, last + 1) * SEC_M, along, ts)
    sect_t = np.diff(tb)
    med = np.median(sect_t)
    for i, t_ in zip(range(first, last), sect_t):
        rows.append((sid, rname, trip, i, t_, t_ / med))
secs = pd.DataFrame(rows, columns=["shape_id", "route_short_name", "trip_id", "section",
                                   "section_time_s", "ratio"])

agg = (secs.groupby(["shape_id", "route_short_name", "section"])
           .agg(trips=("trip_id", "nunique"), mean_ratio=("ratio", "mean"),
                slow_share=("ratio", lambda r: (r >= SLOW_TRIP_RATIO).mean()))
           .reset_index())
agg["start_m"] = agg["section"] * SEC_M
agg["end_m"] = agg["start_m"] + SEC_M

# Sections next to a bus stop are expected to be slower
stops = pd.read_csv(V2 / "planted_stops.csv", dtype={"shape_id": str})
agg["near_stop"] = False
for sid, s in stops.groupby("shape_id"):
    a = np.sort(s["along_m"].to_numpy())
    mk = agg["shape_id"] == sid
    lo = np.searchsorted(a, agg.loc[mk, "start_m"].to_numpy() - STOP_BUFFER_M)
    hi = np.searchsorted(a, agg.loc[mk, "end_m"].to_numpy() + STOP_BUFFER_M)
    agg.loc[mk, "near_stop"] = (hi - lo) > 0

agg["hotspot"] = ((agg["trips"] >= MIN_TRIPS) & (agg["mean_ratio"] >= HOT_RATIO)
                  & (agg["slow_share"] >= MIN_SHARE) & (~agg["near_stop"]))
agg.to_csv(OUT / "v2_section_stats.csv", index=False)
hot = agg[agg["hotspot"]].sort_values("mean_ratio", ascending=False)
hot.to_csv(OUT / "v2_hotspots.csv", index=False)

print()
print(f"Sections analysed: {len(agg)} | near a bus stop (excluded): {int(agg['near_stop'].sum())} "
      f"| hotspots found: {len(hot)}")

# Compare with the planted slow zones (evaluation only)
zones = pd.read_csv(V2 / "planted_slow_zones.csv", dtype={"shape_id": str})
print()
print("PLANTED SLOW ZONES: FOUND OR MISSED?")
res, inside = [], pd.Series(False, index=hot.index)
for z in zones.itertuples():
    ov = agg[(agg["shape_id"] == z.shape_id) & (agg["start_m"] < z.end_m) & (agg["end_m"] > z.start_m)]
    ok = ov[~ov["near_stop"]]
    found = bool(ov["hotspot"].any())
    res.append({"zone": z.zone_id, "route": z.route_short_name, "sections_overlapping": len(ov),
                "excluded_near_stop": int(ov["near_stop"].sum()),
                "best_mean_ratio": round(ok["mean_ratio"].max(), 2) if len(ok) else np.nan,
                "found": "YES" if found else "no"})
    near = (hot["shape_id"] == z.shape_id) & (hot["start_m"] < z.end_m + SEC_M) & (hot["end_m"] > z.start_m - SEC_M)
    inside = inside | near
print(pd.DataFrame(res).to_string(index=False))

false_hot = hot[~inside]
print()
print(f"Hotspots NOT near any planted zone (false alarms): {len(false_hot)} of {len(hot)}")
if len(false_hot):
    print(false_hot[["route_short_name", "start_m", "end_m", "trips", "mean_ratio", "slow_share"]]
          .head(10).round(2).to_string(index=False))
print()
print("Saved:", OUT / "v2_section_stats.csv")
print("Saved:", OUT / "v2_hotspots.csv")