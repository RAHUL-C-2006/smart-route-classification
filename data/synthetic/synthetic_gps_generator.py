"""
Synthetic GPS trace generator for the Smart Route Classification project.

Takes clean GTFS shape geometries (the "intended" path a route follows)
and simulates a vehicle physically driving that path, then samples noisy,
irregularly-timed GPS pings the way a real AVL/phone GPS receiver would.

This gives you:
  - synthetic "raw" GPS traces to feed into your map-matching step
  - a ground-truth path for each trace, so you can measure matching accuracy
    (since you already know exactly which shape the noise came from)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

EARTH_RADIUS_M = 6371000.0


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters between two lat/lon points (vectorized)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def _add_cum_dist(shape):
    shape = shape.sort_values("shape_pt_sequence").reset_index(drop=True)
    lat = shape["shape_pt_lat"].to_numpy()
    lon = shape["shape_pt_lon"].to_numpy()
    seg_dist = np.zeros(len(shape))
    seg_dist[1:] = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    shape["cum_dist_m"] = np.cumsum(seg_dist)
    return shape


def load_shape(shapes_path, shape_id):
    """Load one shape's points, sorted, with cumulative distance along the path."""
    shapes = pd.read_csv(shapes_path)
    shape = shapes[shapes["shape_id"] == shape_id]
    if shape.empty:
        raise ValueError(f"shape_id {shape_id!r} not found")
    return _add_cum_dist(shape)


def load_all_shapes(shapes_path, shape_ids=None):
    """
    Load every shape from shapes.txt once and return {shape_id: shape_df}.
    Pass `shape_ids` to only keep a subset (much faster for batch generation
    when you only need a sample of routes, not all ~5,600 of them).
    """
    shapes = pd.read_csv(shapes_path)
    if shape_ids is not None:
        shapes = shapes[shapes["shape_id"].isin(shape_ids)]
    return {sid: _add_cum_dist(df) for sid, df in shapes.groupby("shape_id")}


def batch_generate(shapes_dict, trip_specs, **default_kwargs):
    """
    Generate many trips at once.
    trip_specs: list of dicts, each at least {"shape_id": ..., "trip_id": ..., "start_time": ...}.
    Any other generate_trip_trace kwarg can be overridden per trip; otherwise
    default_kwargs (e.g. avg_speed_kmph) applies to all of them.
    Returns one concatenated DataFrame of noisy pings across all trips.
    """
    frames = []
    for spec in trip_specs:
        shape = shapes_dict[spec["shape_id"]]
        kwargs = {**default_kwargs, **{k: v for k, v in spec.items() if k not in ("shape_id",)}}
        frames.append(generate_trip_trace(shape, **kwargs))
    return pd.concat(frames, ignore_index=True)


def interpolate_position(shape, target_dist_m):
    """Linear-interpolate a lat/lon at an arbitrary distance along the shape."""
    cum = shape["cum_dist_m"].to_numpy()
    target_dist_m = np.clip(target_dist_m, 0, cum[-1])
    idx = np.searchsorted(cum, target_dist_m, side="right")
    idx = np.clip(idx, 1, len(cum) - 1)

    d0, d1 = cum[idx - 1], cum[idx]
    lat0, lat1 = shape["shape_pt_lat"].iloc[idx - 1], shape["shape_pt_lat"].iloc[idx]
    lon0, lon1 = shape["shape_pt_lon"].iloc[idx - 1], shape["shape_pt_lon"].iloc[idx]

    frac = 0.0 if d1 == d0 else (target_dist_m - d0) / (d1 - d0)
    return lat0 + frac * (lat1 - lat0), lon0 + frac * (lon1 - lon0)


def simulate_movement(total_dist_m, avg_speed_kmph=18.0, speed_var=0.25,
                       stop_prob_per_min=2.0, stop_duration_s=(10, 45), dt_s=2.0, seed=None):
    """
    Fine-grained (dt-second resolution) simulation of a vehicle covering
    total_dist_m, with randomly varying speed and occasional stops
    (traffic lights, congestion, bus stops).

    Returns arrays (t_seconds, distance_m) describing true position over time.
    """
    rng = np.random.default_rng(seed)
    avg_speed_ms = avg_speed_kmph * 1000 / 3600

    t, dist = 0.0, 0.0
    ts, ds = [0.0], [0.0]
    stop_prob_per_step = stop_prob_per_min * (dt_s / 60.0)

    while dist < total_dist_m:
        if rng.random() < stop_prob_per_step:
            t += rng.uniform(*stop_duration_s)
        else:
            speed = max(0.0, avg_speed_ms * (1 + rng.normal(0, speed_var)))
            dist += speed * dt_s
            t += dt_s
        ts.append(t)
        ds.append(min(dist, total_dist_m))

    return np.array(ts), np.array(ds)


def generate_trip_trace(shape, trip_id, start_time, avg_speed_kmph=18.0,
                         ping_interval_s=(10, 30), gps_noise_m=8.0,
                         outlier_prob=0.04, outlier_noise_m=45.0,
                         dropout_prob=0.06, seed=None):
    """
    Simulate one vehicle trip along `shape` and sample noisy GPS pings.

    Returns a DataFrame with both the ground-truth position and the
    noisy observed position for every ping, so matching accuracy can
    later be scored directly.
    """
    rng = np.random.default_rng(seed)
    total_dist_m = shape["cum_dist_m"].iloc[-1]

    t_fine, d_fine = simulate_movement(total_dist_m, avg_speed_kmph=avg_speed_kmph, seed=seed)

    # schedule raw ping times independently of the fine simulation grid
    ping_times = [0.0]
    while ping_times[-1] < t_fine[-1]:
        ping_times.append(ping_times[-1] + rng.uniform(*ping_interval_s))
    ping_times = np.array(ping_times[:-1])

    records = []
    for seq, pt in enumerate(ping_times):
        if rng.random() < dropout_prob:
            continue  # simulate a dropped/missed GPS fix

        true_dist = np.interp(pt, t_fine, d_fine)
        true_lat, true_lon = interpolate_position(shape, true_dist)

        noise_std_m = outlier_noise_m if rng.random() < outlier_prob else gps_noise_m
        # convert an isotropic meter-scale noise draw into lat/lon degree offsets
        bearing = rng.uniform(0, 2 * np.pi)
        offset_m = rng.normal(0, noise_std_m)
        dlat = (offset_m * np.cos(bearing)) / 111320.0
        dlon = (offset_m * np.sin(bearing)) / (111320.0 * np.cos(np.radians(true_lat)))

        records.append({
            "trip_id": trip_id,
            "ping_seq": seq,
            "timestamp": start_time + timedelta(seconds=float(pt)),
            "true_lat": true_lat,
            "true_lon": true_lon,
            "lat": true_lat + dlat,
            "lon": true_lon + dlon,
            "is_outlier": noise_std_m == outlier_noise_m,
        })

    return pd.DataFrame.from_records(records)


def build_sample_dataset(gtfs_dir, n_routes=20, trips_per_route=2, seed=7):
    """
    Pick a diverse sample of real BMTC routes/shapes and generate a full
    synthetic raw-GPS dataset from them, with trip/route metadata attached.
    """
    rng = np.random.default_rng(seed)

    trips = pd.read_csv(f"{gtfs_dir}/trips.txt")
    routes = pd.read_csv(f"{gtfs_dir}/routes.txt")
    trip_shapes = trips[["shape_id", "route_id", "trip_headsign"]].drop_duplicates("shape_id")
    trip_shapes = trip_shapes.merge(routes[["route_id", "route_short_name", "route_long_name"]],
                                     on="route_id", how="left")

    sample_shape_ids = rng.choice(trip_shapes["shape_id"].unique(), size=n_routes, replace=False)
    shapes_dict = load_all_shapes(f"{gtfs_dir}/shapes.txt", shape_ids=sample_shape_ids)

    trip_specs, meta_rows = [], []
    base_day = datetime(2026, 9, 11, 6, 0, 0)
    trip_counter = 0

    for shape_id in sample_shape_ids:
        if shape_id not in shapes_dict:
            continue  # shape had no rows in shapes.txt (shouldn't normally happen)
        info = trip_shapes[trip_shapes["shape_id"] == shape_id].iloc[0]
        for _ in range(trips_per_route):
            trip_counter += 1
            trip_id = f"synth_{trip_counter:04d}"
            start_time = base_day + timedelta(minutes=int(rng.uniform(0, 720)))  # spread across 6am-6pm
            avg_speed = float(np.clip(rng.normal(18, 4), 8, 30))  # km/h, urban bus range
            trip_specs.append({"shape_id": shape_id, "trip_id": trip_id,
                                "start_time": start_time, "avg_speed_kmph": avg_speed,
                                "seed": int(rng.integers(0, 1_000_000))})
            meta_rows.append({"trip_id": trip_id, "shape_id": shape_id,
                               "route_id": info["route_id"], "route_short_name": info["route_short_name"],
                               "route_long_name": info["route_long_name"],
                               "distance_m": round(shapes_dict[shape_id]["cum_dist_m"].iloc[-1], 1),
                               "avg_speed_kmph": round(avg_speed, 1), "start_time": start_time})

    gps_df = batch_generate(shapes_dict, trip_specs)
    meta_df = pd.DataFrame(meta_rows)
    return gps_df, meta_df


if __name__ == "__main__":
    gps_df, meta_df = build_sample_dataset("/home/claude/gtfs_bmtc", n_routes=20, trips_per_route=2)
    print(f"{len(meta_df)} synthetic trips across {meta_df['shape_id'].nunique()} real BMTC routes")
    print(f"{len(gps_df)} total noisy GPS pings generated")
    print(f"Dropout rate check: pings/trip avg = {len(gps_df)/len(meta_df):.1f}")
    gps_df.to_csv("/mnt/user-data/outputs/synthetic_raw_gps_traces.csv", index=False)
    meta_df.to_csv("/mnt/user-data/outputs/trip_metadata.csv", index=False)
    print("Saved synthetic_raw_gps_traces.csv and trip_metadata.csv")
