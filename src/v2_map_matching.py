import numpy as np
import pandas as pd

from deviation_features import ROOT, haversine_m

R = 6371000.0
CHUNK = 400
V2 = ROOT / "data" / "synthetic_v2"
OUT_DIR = ROOT / "outputs" / "v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "v2_map_matched_exact.csv"

gps = pd.read_csv(V2 / "synthetic_raw_gps_traces_v2.csv")
meta = pd.read_csv(V2 / "trip_metadata_v2.csv", dtype={"shape_id": str})
gps = gps.merge(meta[["trip_id", "shape_id", "route_id", "route_short_name"]],
                on="trip_id", how="left")
assert gps["shape_id"].notna().all(), "some trips have no shape_id"

sh = pd.read_csv(
    ROOT / "data" / "gtfs" / "shapes.txt",
    usecols=["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"],
    dtype={"shape_id": str},
)
sh = sh[sh["shape_id"].isin(gps["shape_id"].unique())]

gps["matched_lat"] = np.nan
gps["matched_lon"] = np.nan
gps["matched_segment_index"] = -1

for shape_id, idx in gps.groupby("shape_id").groups.items():
    s = sh[sh["shape_id"] == shape_id].sort_values("shape_pt_sequence")
    lat0, lon0 = s["shape_pt_lat"].mean(), s["shape_pt_lon"].mean()
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
    P_all = to_xy(gps.loc[idx, "lat"].values, gps.loc[idx, "lon"].values)

    seg_out = np.empty(len(idx), dtype=int)
    proj_out = np.empty((len(idx), 2))
    for a in range(0, len(idx), CHUNK):
        P = P_all[a:a + CHUNK]
        t = np.clip(((P[:, None, :] - A[None]) * AB[None]).sum(axis=2) / AB2[None], 0, 1)
        proj = A[None] + t[:, :, None] * AB[None]
        d = np.linalg.norm(P[:, None, :] - proj, axis=2)
        best = d.argmin(axis=1)
        rows = np.arange(len(P))
        seg_out[a:a + CHUNK] = best
        proj_out[a:a + CHUNK] = proj[rows, best]

    gps.loc[idx, "matched_segment_index"] = seg_out
    gps.loc[idx, "matched_lat"] = lat0 + np.degrees(proj_out[:, 1] / R)
    gps.loc[idx, "matched_lon"] = lon0 + np.degrees(proj_out[:, 0] / (R * cos0))

gps["match_distance_m"] = haversine_m(gps["lat"], gps["lon"],
                                      gps["matched_lat"], gps["matched_lon"])

print("Observations:", len(gps), "| trips:", gps["trip_id"].nunique(),
      "| unmatched:", int(gps["matched_lat"].isna().sum()))
md = gps["match_distance_m"]
print(f"Distance from GPS ping to its matched route point: mean {md.mean():.2f} m | "
      f"median {md.median():.2f} m | p75 {md.quantile(0.75):.2f} m | max {md.max():.2f} m | "
      f">30 m: {int((md > 30).sum())}")

# Accuracy against the generator's known true positions (evaluation only)
err = haversine_m(gps["true_lat"], gps["true_lon"], gps["matched_lat"], gps["matched_lon"])
print()
print("Matching accuracy: distance from the TRUE bus position to the matched point")
for name, mask in [("all pings", slice(None)), ("non-outlier pings", ~gps["is_outlier"])]:
    e = err[mask]
    print(f"  {name}: mean {e.mean():.2f} m | median {e.median():.2f} m | "
          f"95th percentile {e.quantile(0.95):.2f} m | max {e.max():.2f} m")

gps.to_csv(OUT_PATH, index=False)
print()
print("Saved:", OUT_PATH)