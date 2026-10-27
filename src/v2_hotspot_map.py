import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deviation_features import ROOT

OUT = ROOT / "outputs" / "v2"
V2 = ROOT / "data" / "synthetic_v2"

hot = pd.read_csv(OUT / "v2_hotspots.csv", dtype={"shape_id": str})
zones = pd.read_csv(V2 / "planted_slow_zones.csv", dtype={"shape_id": str})
m = pd.read_csv(OUT / "v2_map_matched_exact.csv", dtype={"shape_id": str},
                usecols=["shape_id", "matched_lat", "matched_lon", "matched_along_m"])

# Merge neighbouring sections of the same route into one hotspot
hot = hot.sort_values(["shape_id", "start_m"]).reset_index(drop=True)
new_group = (hot["shape_id"] != hot["shape_id"].shift()) | (hot["start_m"] != hot["end_m"].shift())
hot["hotspot_id"] = new_group.cumsum()
rows = []
for hid, g in hot.groupby("hotspot_id"):
    sid, s0, e0 = g["shape_id"].iloc[0], g["start_m"].min(), g["end_m"].max()
    pts = m[(m["shape_id"] == sid) & (m["matched_along_m"] >= s0) & (m["matched_along_m"] <= e0)]
    rows.append({
        "hotspot_id": hid, "route": g["route_short_name"].iloc[0], "shape_id": sid,
        "start_m": s0, "end_m": e0, "sections": len(g),
        "trips": int(g["trips"].max()), "max_ratio": g["mean_ratio"].max(),
        "slow_share": g["slow_share"].mean(),
        "lat": pts["matched_lat"].mean(), "lon": pts["matched_lon"].mean(),
    })
h = pd.DataFrame(rows)

# Offline check against the planted zones (within 200 m along the same route)
def near_zone(r):
    z = zones[(zones["shape_id"] == r["shape_id"])
              & (zones["start_m"] < r["end_m"] + 200) & (zones["end_m"] > r["start_m"] - 200)]
    return z["zone_id"].iloc[0] if len(z) else ""

h["planted_zone"] = h.apply(near_zone, axis=1)
h.to_csv(OUT / "v2_hotspot_table.csv", index=False)

print("HOTSPOTS (neighbouring sections merged)")
print(h[["hotspot_id", "route", "start_m", "end_m", "trips", "max_ratio", "slow_share",
         "lat", "lon", "planted_zone"]].round(3).to_string(index=False))
print()
found = set(h["planted_zone"]) - {""}
print(f"Planted zones found: {len(found)} of {len(zones)}  | hotspots with no planted zone nearby: "
      f"{int((h['planted_zone'] == '').sum())}")

# Map
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(m["matched_lon"], m["matched_lat"], s=0.4, color="#BFBFBF", label="route network (matched positions)")
ax.scatter(zones["centre_lon"], zones["centre_lat"], s=260, facecolors="none", edgecolors="black",
           linewidths=1.4, label="planted slow zone (truth)")
ax.scatter(h["lon"], h["lat"], s=40 + 60 * h["max_ratio"], color="#C00000", alpha=0.8,
           label="hotspot found (bigger = slower)")
for r in h.itertuples():
    ax.annotate(r.route, (r.lon, r.lat), textcoords="offset points", xytext=(8, 6), fontsize=7)
ax.set_aspect(1 / np.cos(np.radians(m["matched_lat"].mean())))
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Recurring slow sections found on synthetic v2 data")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "v2_hotspot_map.png", dpi=130)

print()
print("Saved:", OUT / "v2_hotspot_table.csv")
print("Saved:", OUT / "v2_hotspot_map.png")