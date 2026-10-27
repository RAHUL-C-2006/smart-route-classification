import pandas as pd

ROOT = r"C:\smart route classification"
GTFS = ROOT + r"\data\gtfs"

src = open(ROOT + r"\data\synthetic\synthetic_gps_generator_v2.py",
           encoding="utf-8", errors="ignore").read()
print("Edit present in generator file:", "cand = trips[trips" in src)
print()

trips = pd.read_csv(GTFS + r"\trips.txt", usecols=["trip_id", "shape_id"], dtype=str)
t = trips[trips["shape_id"] == "BEMLA-12 DOWN"]
print("GTFS trips that use shape BEMLA-12 DOWN:", len(t))

st = pd.read_csv(GTFS + r"\stop_times.txt", usecols=["trip_id", "stop_id"],
                 dtype={"trip_id": str, "stop_id": str})
st = st[st["trip_id"].isin(t["trip_id"])]
counts = st.groupby("trip_id").size()
print("Of those, trips that have stop times:", len(counts))
print()
print("Number of stops in a trip -> how many trips have that many:")
print(counts.value_counts().sort_index().to_string())