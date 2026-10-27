import json

import pandas as pd

from deviation_features import ROOT

OUT = ROOT / "outputs" / "v2"
SYN2 = ROOT / "data" / "synthetic_v2"
DASH = ROOT / "dashboard"
BIN_M = 100          # route shape is stored every 100 m along the route

with open(DASH / "data.json", encoding="utf-8") as f:
    data = json.load(f)

meta = (pd.read_csv(SYN2 / "trip_metadata_v2.csv", dtype={"shape_id": str})
          .drop_duplicates("shape_id")[["shape_id", "route_short_name", "distance_m"]])
m = pd.read_csv(OUT / "v2_map_matched_exact.csv", dtype={"shape_id": str},
                usecols=["shape_id", "matched_lat", "matched_lon", "matched_along_m"])
stops = pd.read_csv(SYN2 / "planted_stops.csv", dtype={"shape_id": str})

geo = {}
for r in meta.itertuples():
    p = m[m["shape_id"] == r.shape_id].copy()
    p["bin"] = (p["matched_along_m"].clip(lower=0) // BIN_M).astype(int)
    g = (p.groupby("bin").agg(lat=("matched_lat", "mean"), lon=("matched_lon", "mean"))
          .reset_index().sort_values("bin"))
    path = [[round(float(a), 5), round(float(b), 5), int(k * BIN_M)]
            for k, a, b in zip(g["bin"], g["lat"], g["lon"])]
    s = stops[stops["shape_id"] == r.shape_id].sort_values("along_m")
    geo[r.route_short_name] = {
        "length_m": int(round(float(r.distance_m))),
        "path": path,
        "stops": [[round(float(a), 5), round(float(b), 5), int(round(float(c))),
                   n if isinstance(n, str) else "Bus stop"]
                  for a, b, c, n in zip(s["stop_lat"], s["stop_lon"], s["along_m"], s["stop_name"])],
    }

data["geo"] = geo
data.pop("network", None)

with open(DASH / "data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, allow_nan=False)
with open(DASH / "data.js", "w", encoding="utf-8") as f:
    f.write("window.DASH = " + json.dumps(data, allow_nan=False) + ";")

n_pts = sum(len(v["path"]) for v in geo.values())
n_stops = sum(len(v["stops"]) for v in geo.values())
print("Routes with geometry:", len(geo), "| route points:", n_pts, "| bus stops:", n_stops)
print("Routes with no bus stops:", [k for k, v in geo.items() if not v["stops"]] or "none")
print("Routes with a gap of more than 400 m in the shape:",
      [k for k, v in geo.items()
       if any(b[2] - a[2] > 400 for a, b in zip(v["path"], v["path"][1:]))] or "none")
print("Saved:", DASH / "data.json", "and", DASH / "data.js", f"({(DASH / 'data.js').stat().st_size / 1024:.0f} KB)")