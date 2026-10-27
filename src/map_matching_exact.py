import numpy as np
import pandas as pd

from deviation_features import DATA_PATH, ROOT, haversine_m

R = 6371000.0
OUT_PATH = ROOT / "outputs" / "map_matched_gps_segments_exact.csv"
CHUNK = 400

old = pd.read_csv(DATA_PATH, dtype={"shape_id": str})
shape_ids = old["shape_id"].unique()

sh = pd.read_csv(
    ROOT / "data" / "gtfs" / "shapes.txt",
    usecols=["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"],
    dtype={"shape_id": str},
)
sh = sh[sh["shape_id"].isin(shape_ids)]

new = old.copy()

for shape_id, idx in old.groupby("shape_id").groups.items():
    s = sh[sh["shape_id"] == shape_id].sort_values("shape_pt_sequence")
    lat0 = s["shape_pt_lat"].mean()
    lon0 = s["shape_pt_lon"].mean()
    cos0 = np.cos(np.radians(lat0))

    def to_xy(lat, lon):
        return np.column_stack([
            np.radians(np.asarray(lon, float) - lon0) * cos0 * R,
            np.radians(np.asarray(lat, float) - lat0) * R,
        ])

    S = to_xy(s["shape_pt_lat"].values, s["shape_pt_lon"].values)
    A, B = S[:-1], S[1:]
    AB = B - A
    AB2 = np.maximum((AB ** 2).sum(axis=1), 1e-9)

    idx = np.array(list(idx))
    P_all = to_xy(old.loc[idx, "lat"].values, old.loc[idx, "lon"].values)

    seg_out = np.empty(len(idx), dtype=int)
    proj_out = np.empty((len(idx), 2))
    for a in range(0, len(idx), CHUNK):
        P = P_all[a:a + CHUNK]
        diff = P[:, None, :] - A[None, :, :]
        t = np.clip((diff * AB[None]).sum(axis=2) / AB2[None], 0, 1)
        proj = A[None] + t[:, :, None] * AB[None]
        d = np.linalg.norm(P[:, None, :] - proj, axis=2)
        best = d.argmin(axis=1)
        rows = np.arange(len(P))
        seg_out[a:a + CHUNK] = best
        proj_out[a:a + CHUNK] = proj[rows, best]

    new.loc[idx, "matched_segment_index"] = seg_out
    new.loc[idx, "matched_lat"] = lat0 + np.degrees(proj_out[:, 1] / R)
    new.loc[idx, "matched_lon"] = lon0 + np.degrees(proj_out[:, 0] / (R * cos0))

new["match_distance_m"] = haversine_m(
    new["lat"], new["lon"], new["matched_lat"], new["matched_lon"]
)

# ---------- Compare old vs new ----------
print("Observations:", len(new), "| trips:", new["trip_id"].nunique())
print()
for name, col in [("Old (midpoint search)", old["match_distance_m"]),
                  ("New (exact search)", new["match_distance_m"])]:
    print(f"{name}: mean {col.mean():.2f} m | median {col.median():.2f} m | "
          f"p75 {col.quantile(0.75):.2f} m | max {col.max():.2f} m | "
          f">30 m: {int((col > 30).sum())}")

delta = old["match_distance_m"] - new["match_distance_m"]
print()
print("Pings whose distance fell by more than 1 m:", int((delta > 1).sum()))
print("Pings whose distance rose by more than 1 m:", int((delta < -1).sum()))
print("Pings whose matched segment changed:",
      int((old["matched_segment_index"] != new["matched_segment_index"]).sum()))
changed = new.assign(old_dist=old["match_distance_m"], delta=delta)
print("Trips with any distance change > 1 m:")
print(changed[changed["delta"].abs() > 1].groupby("trip_id").size().to_string())

print()
print("Trip synth_0017, pings 109-118 (old vs new):")
w = changed[(changed["trip_id"] == "synth_0017")
            & changed["ping_seq"].between(109, 118)]
print(w[["ping_seq", "old_dist", "match_distance_m", "matched_segment_index"]]
      .round(2).to_string(index=False))

new.to_csv(OUT_PATH, index=False)
print()
print("Saved:", OUT_PATH)