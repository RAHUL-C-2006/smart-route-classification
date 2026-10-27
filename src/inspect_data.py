
from pathlib import Path
import pandas as pd

# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
GTFS_DIR = PROJECT_DIR / "data" / "gtfs"
SYNTHETIC_DIR = PROJECT_DIR / "data" / "synthetic"

print("Project folder:", PROJECT_DIR)

# 1. Check which GTFS files exist
print("\n--- GTFS FILES ---")
if GTFS_DIR.exists():
    for file in sorted(GTFS_DIR.iterdir()):
        if file.is_file():
            print(file.name)
else:
    print("GTFS folder not found:", GTFS_DIR)

# 2. Check which synthetic files exist
print("\n--- SYNTHETIC FILES ---")
if SYNTHETIC_DIR.exists():
    for file in sorted(SYNTHETIC_DIR.iterdir()):
        if file.is_file():
            print(file.name)
else:
    print("Synthetic folder not found:", SYNTHETIC_DIR)

# 3. Load CSV files safely
def inspect_csv(file_path):
    print(f"\n--- Reading {file_path.name} ---")

    if not file_path.exists():
        print("File not found:", file_path)
        return

    try:
        df = pd.read_csv(file_path)

        print("Shape (rows, columns):", df.shape)
        print("\nColumn names:")
        print(df.columns.tolist())

        print("\nFirst 5 rows:")
        print(df.head())

        print("\nMissing values per column:")
        print(df.isnull().sum())

    except Exception as error:
        print("Could not read file:", error)


# Synthetic GPS datasets
inspect_csv(SYNTHETIC_DIR / "synthetic_raw_gps_traces.csv")
inspect_csv(SYNTHETIC_DIR / "trip_metadata.csv")

# Important GTFS tables
for filename in [
    "routes.txt",
    "stops.txt",
    "trips.txt",
    "shapes.txt",
    "stop_times.txt",
]:
    inspect_csv(GTFS_DIR / filename)
