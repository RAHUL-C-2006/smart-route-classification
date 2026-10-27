
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
GTFS_DIR = PROJECT_DIR / "data" / "gtfs"
SYNTHETIC_DIR = PROJECT_DIR / "data" / "synthetic"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load synthetic GPS and trip metadata
gps = pd.read_csv(SYNTHETIC_DIR / "synthetic_raw_gps_traces.csv")
metadata = pd.read_csv(SYNTHETIC_DIR / "trip_metadata.csv")

# Select the first trip for our initial visual inspection
trip_id = "synth_0001"

trip_gps = gps[gps["trip_id"].astype(str) == trip_id].copy()
trip_info = metadata[metadata["trip_id"].astype(str) == trip_id]

if trip_gps.empty or trip_info.empty:
    raise ValueError(f"Could not find GPS data or metadata for {trip_id}")

shape_id = str(trip_info.iloc[0]["shape_id"]).strip()

print("Trip ID:", trip_id)
print("Route:", trip_info.iloc[0]["route_short_name"])
print("Shape ID:", shape_id)
print("GPS observations:", len(trip_gps))

# Read only the selected GTFS shape from the large shapes file
shapes = pd.read_csv(
    GTFS_DIR / "shapes.txt",
    dtype={"shape_id": str}
)

planned = shapes[shapes["shape_id"].str.strip() == shape_id].copy()

if planned.empty:
    raise ValueError(f"Shape {shape_id} was not found in shapes.txt")

planned = planned.sort_values("shape_pt_sequence")

# Ensure coordinates are numeric and order GPS points by sequence
for col in ["lat", "lon", "true_lat", "true_lon"]:
    trip_gps[col] = pd.to_numeric(trip_gps[col], errors="coerce")

trip_gps = trip_gps.sort_values("ping_seq")
trip_gps = trip_gps.dropna(subset=["lat", "lon", "true_lat", "true_lon"])

# Plot longitude on x-axis and latitude on y-axis
plt.figure(figsize=(11, 8))

plt.plot(
    planned["shape_pt_lon"],
    planned["shape_pt_lat"],
    label="Planned GTFS route",
    linewidth=2
)

plt.plot(
    trip_gps["true_lon"],
    trip_gps["true_lat"],
    label="Synthetic reference path",
    marker=".",
    markersize=3,
    linewidth=1
)

outliers = (
    trip_gps["is_outlier"].astype(str).str.lower().eq("true")
)

normal = trip_gps[~outliers]
abnormal = trip_gps[outliers]

plt.scatter(
    normal["lon"], normal["lat"],
    label="Observed GPS",
    s=12
)

if not abnormal.empty:
    plt.scatter(
        abnormal["lon"], abnormal["lat"],
        label="GPS points labelled outliers",
        marker="x",
        s=35
    )

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"Planned Route vs GPS Observations — {trip_id}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_path = OUTPUT_DIR / f"route_comparison_{trip_id}.png"
plt.savefig(output_path, dpi=200)
print("Plot saved to:", output_path)

plt.show()
