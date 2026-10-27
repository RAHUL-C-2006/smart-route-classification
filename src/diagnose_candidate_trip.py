import numpy as np
import pandas as pd

from deviation_features import DATA_PATH, ROOT, haversine_m

TRIP = "synth_0017"
FIRST, LAST = 108, 120
R = 6371000.0

m = pd.read_csv(DATA_PATH)
m = m[m["trip_id"] == TRIP].sort_values("ping_seq").reset_index(drop=True)
shape_id = m["shape_id"].iloc[0]
print("Trip:", TRIP, "| shape_id:", shape_id)

sh = pd.read_csv(
    ROOT / "data" / "gtfs" / "shapes.txt",
    usecols=["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"],
    dtype={"shape_id": str},
)
sh = sh[sh["shape_id"] == shape_id].sort_values("shape_pt_sequence").reset_index(drop=True)
print("Shape points:", len(sh), "| segments:", len(sh) - 1)

stops = pd.read_csv(
    ROOT / "data" / "gtfs" / "stops.txt",
    usecols=["stop_id", "stop_name", "stop_lat", "stop_lon"],
)

lat0 = sh["shape_pt_lat"].mean()
lon0 = sh["shape_pt_lon"].mean()


def to_xy(lat, lon):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    x = np.radians(lon - lon0) * np.cos(np.radians(lat0)) * R
    y = np.radians(lat - lat0) * R
    return np.column_stack([x, y])


S = to_xy(sh["shape_pt_lat"], sh["shape_pt_lon"])
A, B = S[:-1], S[1:]
AB = B - A
AB2 = np.maximum((AB ** 2).sum(axis=1), 1e-9)


def exact_dist_to_shape(lat, lon):
    P = to_xy([lat], [lon])[0]
    t = np.clip(((P - A) * AB).sum(axis=1) / AB2, 0, 1)
    proj = A + t[:, None] * AB
    d = np.linalg.norm(P - proj, axis=1)
    i = int(d.argmin())
    return float(d[i]), i


# Segment lengths around the suspicious segment
seg_len = haversine_m(
    sh["shape_pt_lat"].values[:-1], sh["shape_pt_lon"].values[:-1],
    sh["shape_pt_lat"].values[1:], sh["shape_pt_lon"].values[1:],
)
print("\nSegment lengths (m) around the matched segment:")
for i in range(79, 85):
    if i < len(seg_len):
        print(f"  segment {i}: {seg_len[i]:.1f}")
print(f"Shape-wide segment length: median {np.median(seg_len):.1f}, max {seg_len.max():.1f}")

w = m[(m["ping_seq"] >= FIRST) & (m["ping_seq"] <= LAST)].copy().reset_index(drop=True)

rows = []
for i, r in w.iterrows():
    obs_d, obs_seg = exact_dist_to_shape(r["lat"], r["lon"])
    true_d, true_seg = exact_dist_to_shape(r["true_lat"], r["true_lon"])
    sd = haversine_m(stops["stop_lat"].values, stops["stop_lon"].values,
                     r["true_lat"], r["true_lon"])
    j = int(np.argmin(sd))
    prev = w.loc[i - 1] if i > 0 else None
    step = (haversine_m(prev["true_lat"], prev["true_lon"], r["true_lat"], r["true_lon"])
            if prev is not None else np.nan)
    rows.append({
        "ping_seq": r["ping_seq"],
        "true_step_m": step,
        "matcher_dist_m": r["match_distance_m"],
        "exact_obs_dist_m": obs_d,
        "matcher_seg": r["matched_segment_index"],
        "exact_obs_seg": obs_seg,
        "exact_true_dist_m": true_d,
        "nearest_stop_to_true_m": sd[j],
        "nearest_stop": str(stops["stop_name"].iloc[j])[:28],
    })

print("\nPing-by-ping check (true_* columns are offline evaluation only):")
print(pd.DataFrame(rows).round(2).to_string(index=False))