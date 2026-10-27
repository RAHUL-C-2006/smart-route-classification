import sys

import pandas as pd

ROOT = r"C:\smart route classification"
sys.path.insert(0, ROOT + r"\data\synthetic")
import synthetic_gps_generator_v2 as g
from synthetic_gps_generator import _add_cum_dist

meta = pd.read_csv(g.OUT / "trip_metadata_v2.csv", dtype={"shape_id": str})
per = (meta.drop_duplicates("shape_id")
           [["shape_id", "route_short_name", "distance_m", "n_stops"]]
           .sort_values("n_stops").reset_index(drop=True))
print("ROUTES WITH THE FEWEST STOPS")
print(per.head(6).to_string(index=False))
print()

low = per[per["n_stops"] < 10]
if low.empty:
    print("No route has fewer than 10 stops.")
    raise SystemExit

ids = low["shape_id"].tolist()
sh = pd.read_csv(g.GTFS / "shapes.txt",
                 usecols=["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"],
                 dtype={"shape_id": str})
sh = sh[sh["shape_id"].isin(ids)]
trips = pd.read_csv(g.GTFS / "trips.txt", usecols=["trip_id", "shape_id"], dtype=str)
rep = trips[trips["shape_id"].isin(ids)].drop_duplicates("shape_id")
st = pd.read_csv(g.GTFS / "stop_times.txt", usecols=["trip_id", "stop_id", "stop_sequence"],
                 dtype={"trip_id": str, "stop_id": str})
stops = pd.read_csv(g.GTFS / "stops.txt", usecols=["stop_id", "stop_name", "stop_lat", "stop_lon"],
                    dtype={"stop_id": str})

g.STOP_MAX_OFFSET_M = 1e9        # keep every stop so we can see its distance from the route
g.STOP_MIN_GAP_M = 0.0

for _, r in low.iterrows():
    sid = r["shape_id"]
    print("=" * 60)
    print(f"Route {r['route_short_name']} | shape {sid} | length {r['distance_m']:.0f} m | stops used: {r['n_stops']}")
    gtfs_trip = rep.loc[rep["shape_id"] == sid, "trip_id"].iloc[0]
    rows_st = st[st["trip_id"] == gtfs_trip]
    print("  stop_times rows for its GTFS trip:", len(rows_st))
    srows = stops[stops["stop_id"].isin(rows_st["stop_id"])].drop_duplicates("stop_id")
    print("  of those, found in stops.txt:", len(srows))
    if srows.empty:
        continue
    shape = _add_cum_dist(sh[sh["shape_id"] == sid])
    tab = g.stop_positions(shape, srows)
    o = tab["offset_m"]
    print(f"  distance of each stop from the route line (m): min {o.min():.0f}, median {o.median():.0f}, max {o.max():.0f}")
    print("  stops within 50 m:", int((o <= 50).sum()), "| within 100 m:", int((o <= 100).sum()),
          "| within 200 m:", int((o <= 200).sum()))