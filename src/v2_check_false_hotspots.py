import numpy as np
import pandas as pd

from deviation_features import ROOT, haversine_m

OUT = ROOT / "outputs" / "v2"
WINDOW_M = 200.0
# The two hotspots that were not near any planted zone (from the previous output)
TARGETS = [("500-EB", 7200.0, 7400.0), ("MKT-5H SNBS-JBS-HRH", 15600.0, 15800.0)]

m = pd.read_csv(
    OUT / "v2_map_matched_exact.csv", dtype={"shape_id": str},
    usecols=["trip_id", "ping_seq", "timestamp", "shape_id", "route_short_name",
             "matched_lat", "matched_lon", "matched_segment_index", "true_dist_m"])
cand = pd.read_csv(OUT / "v2_deviation_candidates.csv",
                   usecols=["trip_id", "ping_seq", "candidate_label"])
m = m.merge(cand, on=["trip_id", "ping_seq"], how="left")
m["timestamp"] = pd.to_datetime(m["timestamp"])
m = m.sort_values(["trip_id", "ping_seq"]).copy()
m["noise"] = m["candidate_label"] == "likely_gps_noise"

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

for route, s0, e0 in TARGETS:
    print("=" * 78)
    print(f"Route {route}: section {s0:.0f}-{e0:.0f} m (a normal 200 m section takes roughly 40-60 s)")
    print("=" * 78)
    rows = []
    for trip, g in m[m["route_short_name"] == route].groupby("trip_id", sort=False):
        t0 = g["timestamp"].iloc[0]
        ts_all = (g["timestamp"] - t0).dt.total_seconds().to_numpy()
        true_d = np.maximum.accumulate(g["true_dist_m"].to_numpy()) + np.arange(len(g)) * 1e-6
        if true_d[0] > s0 or true_d[-1] < e0:
            continue
        true_s = np.interp(e0, true_d, ts_all) - np.interp(s0, true_d, ts_all)

        c = g[~g["noise"]]
        ts_c = (c["timestamp"] - t0).dt.total_seconds().to_numpy()
        al_raw = c["along_m"].to_numpy()
        al = np.maximum.accumulate(al_raw) + np.arange(len(c)) * 1e-6
        obs_s = np.interp(e0, al, ts_c) - np.interp(s0, al, ts_c)

        win_c = ((c["true_dist_m"] >= s0 - WINDOW_M) & (c["true_dist_m"] <= e0 + WINDOW_M)).to_numpy()
        win_a = ((g["true_dist_m"] >= s0 - WINDOW_M) & (g["true_dist_m"] <= e0 + WINDOW_M)).to_numpy()
        err = np.abs(al_raw[win_c] - c["true_dist_m"].to_numpy()[win_c])
        step = np.diff(al_raw, prepend=al_raw[0])[win_c]
        rows.append({
            "trip_id": trip,
            "start_hour": round(t0.hour + t0.minute / 60, 1),
            "measured_s": obs_s,
            "true_s": true_s,
            "pings_near": int(win_c.sum()),
            "noise_removed_near": int((g["noise"].to_numpy() & win_a).sum()),
            "max_along_error_m": err.max() if len(err) else np.nan,
            "biggest_backward_step_m": float(-step.min()) if len(step) and step.min() < 0 else 0.0,
        })
    print(pd.DataFrame(rows).round(1).to_string(index=False))
    print()