
from pathlib import Path
import pandas as pd
import numpy as np

# Locate project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
GTFS_DIR = PROJECT_DIR / "data" / "gtfs"
SYNTHETIC_DIR = PROJECT_DIR / "data" / "synthetic"

# Load the two synthetic datasets
gps = pd.read_csv(SYNTHETIC_DIR / "synthetic_raw_gps_traces.csv")
metadata = pd.read_csv(SYNTHETIC_DIR / "trip_metadata.csv")

print("=" * 55)
print("1. TRIP AND METADATA CHECKS")
print("=" * 55)

gps_trips = set(gps["trip_id"].astype(str))
metadata_trips = set(metadata["trip_id"].astype(str))

print("Unique GPS trips:", len(gps_trips))
print("Metadata trips:", len(metadata_trips))
print("GPS trips without metadata:", len(gps_trips - metadata_trips))
print("Metadata trips without GPS:", len(metadata_trips - gps_trips))

print("\n" + "=" * 55)
print("2. GPS COORDINATE CHECKS")
print("=" * 55)

for col in ["lat", "lon", "true_lat", "true_lon"]:
    gps[col] = pd.to_numeric(gps[col], errors="coerce")

print("Missing/invalid coordinate values:")
print(gps[["lat", "lon", "true_lat", "true_lon"]].isna().sum())

valid_coords = (
    gps["lat"].between(-90, 90)
    & gps["lon"].between(-180, 180)
    & gps["true_lat"].between(-90, 90)
    & gps["true_lon"].between(-180, 180)
)

print("Rows with valid coordinate ranges:", int(valid_coords.sum()))
print("Rows with invalid coordinate ranges:", int((~valid_coords).sum()))

print("\n" + "=" * 55)
print("3. GPS OUTLIER LABELS")
print("=" * 55)

print(gps["is_outlier"].value_counts(dropna=False))
print("\nOutlier percentage:")
print(round(gps["is_outlier"].astype(str).str.lower()
            .eq("true").mean() * 100, 2), "%")

print("\n" + "=" * 55)
print("4. TIMESTAMP AND SEQUENCE CHECKS")
print("=" * 55)

gps["timestamp"] = pd.to_datetime(gps["timestamp"], errors="coerce")
print("Invalid timestamps:", int(gps["timestamp"].isna().sum()))

duplicate_rows = gps.duplicated(["trip_id", "ping_seq"]).sum()
print("Duplicate trip_id + ping_seq records:", int(duplicate_rows))

gps = gps.sort_values(["trip_id", "ping_seq"])
gps["sequence_gap"] = gps.groupby("trip_id")["ping_seq"].diff()

print("Missing sequence numbers between observations:",
      int((gps["sequence_gap"] > 1).sum()))

print("\nObservations per trip:")
print(gps.groupby("trip_id").size().describe())

print("\n" + "=" * 55)
print("5. SYNTHETIC SHAPES VS GTFS SHAPES")
print("=" * 55)

synthetic_shapes = set(metadata["shape_id"].dropna().astype(str).str.strip())

# Read only the shape_id column, not all 2.49 million shape points.
gtfs_shape_ids = pd.read_csv(
    GTFS_DIR / "shapes.txt",
    usecols=["shape_id"],
    dtype={"shape_id": str}
)["shape_id"].dropna().str.strip().unique()

gtfs_shapes = set(gtfs_shape_ids)
matched_shapes = synthetic_shapes & gtfs_shapes

print("Unique synthetic metadata shapes:", len(synthetic_shapes))
print("Unique GTFS shapes:", len(gtfs_shapes))
print("Matching shape IDs:", len(matched_shapes))
print("Synthetic shape IDs not found in GTFS:",
      len(synthetic_shapes - gtfs_shapes))

if synthetic_shapes - gtfs_shapes:
    print("\nExamples of unmatched synthetic shape IDs:")
    print(sorted(synthetic_shapes - gtfs_shapes)[:10])

print("\n" + "=" * 55)
print("6. SAVE VALIDATION SUMMARY")
print("=" * 55)

summary = pd.DataFrame({
    "check": [
        "GPS observations",
        "Unique GPS trips",
        "Metadata trips",
        "Invalid timestamps",
        "Duplicate trip/sequence records",
        "Invalid coordinate rows",
        "Matching shape IDs",
        "Unmatched synthetic shape IDs",
    ],
    "result": [
        len(gps),
        len(gps_trips),
        len(metadata_trips),
        int(gps["timestamp"].isna().sum()),
        int(duplicate_rows),
        int((~valid_coords).sum()),
        len(matched_shapes),
        len(synthetic_shapes - gtfs_shapes),
    ]
})

output_dir = PROJECT_DIR / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

summary.to_csv(output_dir / "validation_summary.csv", index=False)
print(summary.to_string(index=False))
print("\nSaved: outputs/validation_summary.csv")
