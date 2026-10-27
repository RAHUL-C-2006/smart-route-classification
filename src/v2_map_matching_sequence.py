import shutil

import numpy as np
import pandas as pd

from deviation_features import ROOT, haversine_m

R = 6371000.0
V2 = ROOT / "data" / "synthetic_v2"
OUT_DIR = ROOT / "outputs" / "v2"
OUT_PATH = OUT_DIR / "v2_map_matched_exact.csv"
BACKUP = OUT_DIR / "v2_map_matched_nearest_only.csv"

# Matcher settings (first-pass choices, not tuned)
K = 6                  # candidate positions kept per ping (one per distinct stretch of road)
CAND_MARGIN_M = 60.0   # a candidate may be this much farther than the nearest one
PASS_SEP_M = 100.0     # candidates must differ by this much along the route
SIGMA_M = 10.0         # typical GPS error
EMISSION_CAP = 8.0     # limits the penalty for a very noisy ping
JUMP_PENALTY = 10.0    # penalty for an implausible jump along the route
MAX_SPEED_MS = 25.0    # fastest believable bus speed (90 km/h)
FWD_SLACK_M = 60.0
BACK_TOL_M = 40.0      # small backward movement is allowed (GPS jitter)

if OUT_PATH.exists() and not BACKUP.exists():
    shutil.copy(OUT_PATH, BACKUP)
    print("Backed up the previous matched file to:", BACKUP)

gps = pd.read_csv(V2 / "synthetic_raw_gps_traces_v2.csv")
meta = pd.read_csv(V2 / "trip_metadata_v2.csv", dtype={"shape_id": str})
gps = gps.merge(meta[["trip_id", "shape_id", "route_id", "route_short_name"]],
                on="trip_id", how="left")
gps["timestamp"] = pd.to_datetime(gps["timestamp"])
gps = gps.sort_values(["trip_id", "ping_seq"]).reset_index(drop=True)

sh = pd.read_csv(ROOT / "data" / "gtfs" / "shapes.txt",
                 usecols=["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"],
                 dtype={"shape_id": str})
sh = sh[sh["shape_id"].isin(gps["shape_id"].unique())]

geo = {}
for sid, s in sh.groupby("shape_id"):
    s = s.sort_values("shape_pt_sequence")
    lat, lon = s["shape_pt_lat"].to_numpy(), s["shape_pt_lon"].to_numpy()
    lat0, lon0 = lat.mean(), lon.mean()
    cos0 = np.cos(np.radians(lat0))
    S = np.column_stack([np.radians(lon - lon0) * cos0 * R, np.radians(lat - lat0) * R])
    A, B = S[:-1], S[1:]
    AB = B - A
    cum = np.concatenate([[0.0], np.cumsum(haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:]))])
    geo[sid] = dict(lat0=lat0, lon0=lon0, cos0=cos0, A=A, AB=AB,
                    AB2=np.maximum((AB ** 2).sum(axis=1), 1e-9),
                    cum=cum, seglen=np.diff(cum))


def emission(d):
    return min(d * d / (2 * SIGMA_M ** 2), EMISSION_CAP)


def best_path(cands, ts):
    """Viterbi: cheapest sequence of candidates, penalising implausible jumps along the route."""
    n = len(cands)
    cost = [[emission(c[2]) for c in cands[0]]]
    back = [[0] * len(cands[0])]
    for i in range(1, n):
        max_fwd = MAX_SPEED_MS * max(ts[i] - ts[i - 1], 0.0) + FWD_SLACK_M
        row, brow = [], []
        for c in cands[i]:
            best, bj = None, 0
            for j, pc in enumerate(cands[i - 1]):
                delta = c[1] - pc[1]
                v = cost[i - 1][j] + (0.0 if -BACK_TOL_M <= delta <= max_fwd else JUMP_PENALTY)
                if best is None or v < best:
                    best, bj = v, j
            row.append(best + emission(c[2]))
            brow.append(bj)
        cost.append(row)
        back.append(brow)
    k = int(np.argmin(cost[-1]))
    path = [k]
    for i in range(n - 1, 0, -1):
        k = back[i][k]
        path.append(k)
    return path[::-1]


n_all = len(gps)
seg_new = np.zeros(n_all, dtype=int); seg_old = np.zeros(n_all, dtype=int)
along_new = np.zeros(n_all); along_old = np.zeros(n_all)
lat_new = np.zeros(n_all); lon_new = np.zeros(n_all)
multi = 0

for trip, idx in gps.groupby("trip_id", sort=False).groups.items():
    idx = np.asarray(idx)
    G = geo[gps.at[idx[0], "shape_id"]]
    la, lo = gps.loc[idx, "lat"].to_numpy(), gps.loc[idx, "lon"].to_numpy()
    P_all = np.column_stack([np.radians(lo - G["lon0"]) * G["cos0"] * R,
                             np.radians(la - G["lat0"]) * R])
    ts = (gps.loc[idx, "timestamp"] - gps.at[idx[0], "timestamp"]).dt.total_seconds().to_numpy()

    cands = []
    for a in range(0, len(idx), 100):
        P = P_all[a:a + 100]
        t = np.clip(((P[:, None, :] - G["A"][None]) * G["AB"][None]).sum(axis=2) / G["AB2"][None], 0, 1)
        proj = G["A"][None] + t[:, :, None] * G["AB"][None]
        d = np.linalg.norm(P[:, None, :] - proj, axis=2)
        along = G["cum"][:-1][None] + t * G["seglen"][None]
        for r in range(len(P)):
            d_row = d[r]
            near = np.where(d_row <= d_row.min() + CAND_MARGIN_M)[0]
            near = near[np.argsort(d_row[near])]
            chosen = []
            for s_ in near:
                a_ = along[r, s_]
                if all(abs(a_ - c[1]) > PASS_SEP_M for c in chosen):
                    chosen.append((int(s_), float(a_), float(d_row[s_]),
                                   float(proj[r, s_, 0]), float(proj[r, s_, 1])))
                    if len(chosen) == K:
                        break
            cands.append(chosen)

    path = best_path(cands, ts)
    for pos, k in enumerate(path):
        c = cands[pos][k]
        g_i = idx[pos]
        seg_new[g_i], along_new[g_i] = c[0], c[1]
        lat_new[g_i] = G["lat0"] + np.degrees(c[4] / R)
        lon_new[g_i] = G["lon0"] + np.degrees(c[3] / (R * G["cos0"]))
        seg_old[g_i], along_old[g_i] = cands[pos][0][0], cands[pos][0][1]   # nearest-segment choice
        multi += len(cands[pos]) > 1

gps["matched_segment_index"] = seg_new
gps["matched_lat"], gps["matched_lon"] = lat_new, lon_new
gps["matched_along_m"] = along_new
gps["match_distance_m"] = haversine_m(gps["lat"], gps["lon"], gps["matched_lat"], gps["matched_lon"])

e_old = np.abs(along_old - gps["true_dist_m"])     # true_dist_m used for evaluation only
e_new = np.abs(along_new - gps["true_dist_m"])
print()
print("DISTANCE ALONG THE ROUTE vs THE GENERATOR'S TRUTH")
print(f"{'':28s}{'nearest-only':>14s}{'sequence-aware':>16s}")
print(f"{'median error (m)':28s}{e_old.median():>14.1f}{e_new.median():>16.1f}")
print(f"{'95th percentile (m)':28s}{e_old.quantile(0.95):>14.1f}{e_new.quantile(0.95):>16.1f}")
print(f"{'largest error (m)':28s}{e_old.max():>14.0f}{e_new.max():>16.0f}")
print(f"{'pings wrong by over 100 m':28s}{int((e_old > 100).sum()):>14d}{int((e_new > 100).sum()):>16d}")
print()
print(f"Pings with more than one candidate road position: {multi} of {n_all}")
print(f"Pings where the chosen segment changed: {int((seg_new != seg_old).sum())}")
bad = (pd.DataFrame({"route": gps["route_short_name"], "old": e_old > 100, "new": e_new > 100})
         .groupby("route").sum().astype(int))
bad = bad[(bad["old"] > 0) | (bad["new"] > 0)].sort_values("old", ascending=False)
print()
print("ROUTES WITH PINGS WRONG BY OVER 100 m (before -> after)")
print(bad.to_string() if len(bad) else "none")

md = gps["match_distance_m"]
err = haversine_m(gps["true_lat"], gps["true_lon"], gps["matched_lat"], gps["matched_lon"])
print()
print(f"Distance from ping to matched point: mean {md.mean():.2f} m | median {md.median():.2f} m | max {md.max():.2f} m")
print(f"Distance from TRUE position to matched point (non-outlier pings): "
      f"mean {err[~gps['is_outlier']].mean():.2f} m | 95th percentile {err[~gps['is_outlier']].quantile(0.95):.2f} m")

gps.to_csv(OUT_PATH, index=False)
print()
print("Saved:", OUT_PATH)