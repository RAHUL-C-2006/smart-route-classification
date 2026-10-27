from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic_gps_generator import _add_cum_dist  # reused from the original generator

ROOT = Path(r"C:\smart route classification")
GTFS = ROOT / "data" / "gtfs"
V1_META = ROOT / "data" / "synthetic" / "trip_metadata.csv"
OUT = ROOT / "data" / "synthetic_v2"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 11
TRIPS_PER_ROUTE = 6
BASE_DAY = datetime(2026, 9, 11)
DAY_START_H, DAY_END_H = 6.0, 19.0          # trip start times are spread over this window

# ---- Planted patterns (my assumptions; all are saved to files) ----
# name, start hour, end hour, speed factor, dwell-time factor
PEAKS = [("morning_peak", 8.0, 10.5, 0.70, 1.5),
         ("evening_peak", 17.0, 20.0, 0.65, 1.5)]
SIGNAL_STOPS_PER_MIN = 0.4                   # random traffic-light style stops
SIGNAL_STOP_S = (10, 45)
DWELL_PROB = 0.85                            # chance of waiting at a bus stop
DWELL_S = (10, 40)
SCHED_DWELL_S = 25.0                         # timetable allowance per stop
N_BOTTLENECK_ROUTES = 8
ZONE_LEN_M = 250.0
ZONE_SPEED_FACTOR = 0.35
ZONE_SIGNAL_MULT = 3.0
STOP_MAX_OFFSET_M = 50.0                     # GTFS stop must be this close to the route shape
STOP_MIN_GAP_M = 60.0

# ---- GPS behaviour (same as the original generator) ----
PING_INTERVAL_S = (10, 30)
GPS_NOISE_M = 8.0
OUTLIER_PROB = 0.04
OUTLIER_NOISE_M = 45.0
DROPOUT_PROB = 0.06
R = 6371000.0


def peak_factors(h):
    for _, lo, hi, speed_f, dwell_f in PEAKS:
        if lo <= h < hi:
            return speed_f, dwell_f
    return 1.0, 1.0


def stop_positions(shape, stop_rows):
    """Distance along the route shape for each GTFS stop (exact projection)."""
    lat0, lon0 = shape["shape_pt_lat"].mean(), shape["shape_pt_lon"].mean()
    cos0 = np.cos(np.radians(lat0))

    def to_xy(lat, lon):
        return np.column_stack([np.radians(np.asarray(lon, float) - lon0) * cos0 * R,
                                np.radians(np.asarray(lat, float) - lat0) * R])

    S = to_xy(shape["shape_pt_lat"].values, shape["shape_pt_lon"].values)
    A, B = S[:-1], S[1:]
    AB = B - A
    AB2 = np.maximum((AB ** 2).sum(axis=1), 1e-9)
    cum = shape["cum_dist_m"].to_numpy()
    seglen = np.diff(cum)

    P = to_xy(stop_rows["stop_lat"].values, stop_rows["stop_lon"].values)
    t = np.clip(((P[:, None, :] - A[None]) * AB[None]).sum(axis=2) / AB2[None], 0, 1)
    proj = A[None] + t[:, :, None] * AB[None]
    d = np.linalg.norm(P[:, None, :] - proj, axis=2)
    best = d.argmin(axis=1)
    rows = np.arange(len(P))

    out = stop_rows[["stop_id", "stop_name", "stop_lat", "stop_lon"]].copy().reset_index(drop=True)
    out["offset_m"] = d[rows, best]
    out["along_m"] = cum[best] + t[rows, best] * seglen[best]
    total = cum[-1]
    out = out[(out["offset_m"] <= STOP_MAX_OFFSET_M)
              & (out["along_m"] > 20) & (out["along_m"] < total - 20)]
    out = out.sort_values("along_m")

    keep, last = [], -1e9
    for r in out.itertuples():
        if r.along_m - last >= STOP_MIN_GAP_M:
            keep.append(r.Index)
            last = r.along_m
    return out.loc[keep]


def simulate_movement_v2(total, free_ms, start_h, stop_d, zone, rng, dt=2.0):
    """True distance along the route over time, with the planted patterns."""
    t, dist, k = 0.0, 0.0, 0
    ts, ds = [0.0], [0.0]
    p_sig = SIGNAL_STOPS_PER_MIN * dt / 60.0
    while dist < total:
        h = start_h + t / 3600.0
        speed_f, dwell_f = peak_factors(h)
        in_zone = zone is not None and zone[0] <= dist <= zone[1]

        if rng.random() < p_sig * (ZONE_SIGNAL_MULT if in_zone else 1.0):
            t += rng.uniform(*SIGNAL_STOP_S)
            ts.append(t)
            ds.append(dist)
            continue

        speed = max(0.0, free_ms * speed_f * (ZONE_SPEED_FACTOR if in_zone else 1.0)
                    * (1 + rng.normal(0, 0.2)))
        new = dist + speed * dt

        if k < len(stop_d) and stop_d[k] <= new:      # reached a bus stop
            d_stop = stop_d[k]
            t += max(0.0, d_stop - dist) / max(speed, 0.5)
            ts.append(t)
            ds.append(d_stop)
            if rng.random() < DWELL_PROB:
                t += rng.uniform(*DWELL_S) * dwell_f
                ts.append(t)
                ds.append(d_stop)
            dist = d_stop
            k += 1
            continue

        dist = new
        t += dt
        ts.append(t)
        ds.append(min(dist, total))
    return np.array(ts), np.array(ds)


def sample_pings(trip_id, start_dt, shape, ts, ds, rng):
    cum = shape["cum_dist_m"].to_numpy()
    latarr = shape["shape_pt_lat"].to_numpy()
    lonarr = shape["shape_pt_lon"].to_numpy()

    times = [0.0]
    while times[-1] < ts[-1]:
        times.append(times[-1] + rng.uniform(*PING_INTERVAL_S))
    pt = np.array(times[:-1])
    seq = np.arange(len(pt))
    keep = rng.random(len(pt)) >= DROPOUT_PROB
    pt, seq = pt[keep], seq[keep]

    true_dist = np.interp(pt, ts, ds)
    tlat = np.interp(true_dist, cum, latarr)
    tlon = np.interp(true_dist, cum, lonarr)

    is_out = rng.random(len(pt)) < OUTLIER_PROB
    std = np.where(is_out, OUTLIER_NOISE_M, GPS_NOISE_M)
    bearing = rng.uniform(0, 2 * np.pi, len(pt))
    off = rng.normal(0, 1, len(pt)) * std
    dlat = off * np.cos(bearing) / 111320.0
    dlon = off * np.sin(bearing) / (111320.0 * np.cos(np.radians(tlat)))

    return pd.DataFrame({
        "trip_id": trip_id,
        "ping_seq": seq,
        "timestamp": pd.Timestamp(start_dt) + pd.to_timedelta(pt, unit="s"),
        "true_lat": tlat, "true_lon": tlon,
        "lat": tlat + dlat, "lon": tlon + dlon,
        "is_outlier": is_out,
        "true_dist_m": true_dist,      # ground truth, for validation only
    })


def main():
    rng = np.random.default_rng(SEED)

    meta1 = pd.read_csv(V1_META, dtype={"shape_id": str})
    routes = (meta1[["shape_id", "route_id", "route_short_name", "route_long_name"]]
              .drop_duplicates("shape_id").reset_index(drop=True))
    shape_ids = routes["shape_id"].tolist()
    print("Routes (shapes) to use:", len(shape_ids))

    print("Reading shapes...")
    sh = pd.read_csv(GTFS / "shapes.txt",
                     usecols=["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"],
                     dtype={"shape_id": str})
    sh = sh[sh["shape_id"].isin(shape_ids)]
    shapes = {sid: _add_cum_dist(df) for sid, df in sh.groupby("shape_id")}

    print("Reading stops and stop times...")
    trips = pd.read_csv(GTFS / "trips.txt", usecols=["trip_id", "shape_id"], dtype=str)
    cand = trips[trips["shape_id"].isin(shape_ids)]
    st = pd.read_csv(GTFS / "stop_times.txt", usecols=["trip_id", "stop_id", "stop_sequence"],
                     dtype={"trip_id": str, "stop_id": str})
    st = st[st["trip_id"].isin(cand["trip_id"])]
    counts = st.groupby("trip_id").size().rename("n_st").reset_index()
    cand = cand.merge(counts, on="trip_id", how="left").fillna({"n_st": 0})
    rep = cand.sort_values("n_st", ascending=False).drop_duplicates("shape_id")
    stops = pd.read_csv(GTFS / "stops.txt", usecols=["stop_id", "stop_name", "stop_lat", "stop_lon"],
                        dtype={"stop_id": str})

    stop_tables = {}
    for sid in shape_ids:
        gtfs_trip = rep.loc[rep["shape_id"] == sid, "trip_id"].iloc[0]
        ids = st[st["trip_id"] == gtfs_trip].sort_values("stop_sequence")["stop_id"]
        rows = stops[stops["stop_id"].isin(ids)].drop_duplicates("stop_id")
        tab = stop_positions(shapes[sid], rows).assign(stop_source="gtfs_trip")
        if len(tab) < 3:      # GTFS lists no intermediate stops: fall back to nearby stops
            s_ = shapes[sid]
            box = stops[
                stops["stop_lat"].between(s_["shape_pt_lat"].min() - 0.0005,
                                          s_["shape_pt_lat"].max() + 0.0005)
                & stops["stop_lon"].between(s_["shape_pt_lon"].min() - 0.0005,
                                            s_["shape_pt_lon"].max() + 0.0005)
            ].drop_duplicates("stop_id")
            tab = stop_positions(s_, box)
            tab = tab[tab["offset_m"] <= 30].assign(stop_source="nearby_stops_fallback")
            print(f"Route {sid}: GTFS stop list too short, used {len(tab)} nearby stops instead")
        tab.insert(0, "shape_id", sid)
        stop_tables[sid] = tab

    # Plant slow zones on some routes
    bott = set(rng.choice(shape_ids, size=N_BOTTLENECK_ROUTES, replace=False))
    zones, zone_rows = {}, []
    for sid in shape_ids:
        if sid not in bott:
            continue
        sh_ = shapes[sid]
        total = sh_["cum_dist_m"].iloc[-1]
        start = rng.uniform(0.2, 0.8) * total
        start = min(start, total - ZONE_LEN_M - 50)
        end = start + ZONE_LEN_M
        mid = (start + end) / 2
        zones[sid] = (start, end)
        zone_rows.append({
            "zone_id": f"Z{len(zone_rows) + 1:02d}", "shape_id": sid,
            "route_short_name": routes.loc[routes["shape_id"] == sid, "route_short_name"].iloc[0],
            "start_m": round(start, 1), "end_m": round(end, 1),
            "centre_lat": float(np.interp(mid, sh_["cum_dist_m"], sh_["shape_pt_lat"])),
            "centre_lon": float(np.interp(mid, sh_["cum_dist_m"], sh_["shape_pt_lon"])),
            "speed_factor": ZONE_SPEED_FACTOR, "signal_multiplier": ZONE_SIGNAL_MULT,
        })

    # Free-flow speed per route
    free_kmph = {sid: float(np.clip(rng.normal(24, 3), 16, 32)) for sid in shape_ids}

    frames, meta_rows, n = [], [], 0
    edges = np.linspace(DAY_START_H, DAY_END_H, TRIPS_PER_ROUTE + 1)
    for _, r in routes.iterrows():
        sid = r["shape_id"]
        shape = shapes[sid]
        total = shape["cum_dist_m"].iloc[-1]
        stop_d = stop_tables[sid]["along_m"].to_numpy()
        for j in range(TRIPS_PER_ROUTE):
            n += 1
            trip_id = f"v2_{n:04d}"
            start_h = float(rng.uniform(edges[j], edges[j + 1]))
            start_dt = BASE_DAY + timedelta(hours=start_h)
            trip_free = free_kmph[sid] * float(rng.uniform(0.95, 1.05))
            free_ms = trip_free * 1000 / 3600

            ts, ds = simulate_movement_v2(total, free_ms, start_h, stop_d, zones.get(sid), rng)
            frames.append(sample_pings(trip_id, start_dt, shape, ts, ds, rng))

            sched_s = total / free_ms + len(stop_d) * DWELL_PROB * SCHED_DWELL_S
            meta_rows.append({
                "trip_id": trip_id, "shape_id": sid, "route_id": r["route_id"],
                "route_short_name": r["route_short_name"], "route_long_name": r["route_long_name"],
                "distance_m": round(total, 1),
                "avg_speed_kmph": round(total / 1000 / (sched_s / 3600), 2),   # timetable speed
                "scheduled_duration_min": round(sched_s / 60, 2),
                "free_flow_kmph": round(trip_free, 2),
                "n_stops": len(stop_d), "has_slow_zone": sid in zones,
                "start_time": start_dt, "sim_duration_min": round(ts[-1] / 60, 2),
            })

    gps = pd.concat(frames, ignore_index=True)
    meta = pd.DataFrame(meta_rows)
    meta["true_avg_speed_kmph"] = meta["distance_m"] / 1000 / (meta["sim_duration_min"] / 60)

    gps.to_csv(OUT / "synthetic_raw_gps_traces_v2.csv", index=False)
    meta.to_csv(OUT / "trip_metadata_v2.csv", index=False)
    pd.concat(stop_tables.values()).to_csv(OUT / "planted_stops.csv", index=False)
    pd.DataFrame(zone_rows).to_csv(OUT / "planted_slow_zones.csv", index=False)
    pd.DataFrame(PEAKS, columns=["name", "start_hour", "end_hour", "speed_factor",
                                 "dwell_factor"]).to_csv(OUT / "planted_peak_windows.csv", index=False)

    print()
    print("Trips:", len(meta), "| pings:", len(gps), "| outlier share:",
          round(100 * gps["is_outlier"].mean(), 2), "%")
    print("Bus stops used per route: min", int(meta["n_stops"].min()),
          "| median", int(meta["n_stops"].median()), "| max", int(meta["n_stops"].max()))
    print()
    print("Planted slow zones:")
    print(pd.DataFrame(zone_rows)[["zone_id", "route_short_name", "start_m", "end_m"]].to_string(index=False))
    print()
    meta["start_bucket"] = pd.cut(meta["start_time"].dt.hour + meta["start_time"].dt.minute / 60,
                                  [0, 8, 10.5, 17, 24],
                                  labels=["before 8", "8-10:30", "10:30-17", "after 17"], right=False)
    print("Average true speed (km/h) by trip start time, and timetable speed:")
    print(meta.groupby("start_bucket", observed=True)
              .agg(trips=("trip_id", "size"), true_speed=("true_avg_speed_kmph", "mean"),
                   timetable_speed=("avg_speed_kmph", "mean")).round(2).to_string())
    print()
    print("Saved to:", OUT)


if __name__ == "__main__":
    main()