
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GPS_FILE = ROOT / "outputs" / "map_matched_gps_segments.csv"
SHAPES_FILE = ROOT / "data" / "gtfs" / "shapes.txt"
OUTPUT_FILE = ROOT / "outputs" / "segment_matching_comparison.png"

gps = pd.read_csv(GPS_FILE)
shapes = pd.read_csv(SHAPES_FILE)

trip_id = "synth_0001"

trip = gps[gps["trip_id"] == trip_id].sort_values("ping_seq")

if trip.empty:
    raise ValueError(f"No observations found for {trip_id}")

shape_id = trip["shape_id"].iloc[0]

route = shapes[shapes["shape_id"] == shape_id].sort_values(
    "shape_pt_sequence"
)

if route.empty:
    raise ValueError(f"No GTFS shape found for {shape_id}")

plt.figure(figsize=(10, 8))

plt.plot(
    route["shape_pt_lon"],
    route["shape_pt_lat"],
    label="Planned GTFS route",
    linewidth=2
)

plt.scatter(
    trip["lon"],
    trip["lat"],
    label="Observed GPS",
    s=8,
    alpha=0.6
)

plt.scatter(
    trip["matched_lon"],
    trip["matched_lat"],
    label="Matched positions",
    s=8,
    alpha=0.7
)

# Highlight observations with large route distances
far = trip[trip["match_distance_m"] > 30]

plt.scatter(
    far["lon"],
    far["lat"],
    label="GPS points over 30 m from route",
    marker="x",
    s=35
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"Segment-Based Map Matching: {trip_id}")
plt.legend()
plt.tight_layout()

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
plt.close()

print(f"Plot saved to: {OUTPUT_FILE}")
print(f"Trip: {trip_id}")
print(f"Observations: {len(trip)}")
print(f"Points over 30 m from route: {len(far)}")
