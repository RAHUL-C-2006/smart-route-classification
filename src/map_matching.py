
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]

GPS_FILE = ROOT / "data" / "synthetic" / "synthetic_raw_gps_traces.csv"
META_FILE = ROOT / "data" / "synthetic" / "trip_metadata.csv"
SHAPES_FILE = ROOT / "data" / "gtfs" / "shapes.txt"

OUTPUT_FILE = ROOT / "outputs" / "map_matched_gps_segments.csv"


def to_local_xy(lat, lon, ref_lat):
    """Convert latitude/longitude to approximate local metre coordinates."""
    x = np.asarray(lon) * 111320.0 * np.cos(np.radians(ref_lat))
    y = np.asarray(lat) * 111320.0
    return np.column_stack((x, y))


def project_points_to_segments(points, starts, ends):
    """
    Project each point onto its nearest candidate line segment.
    Returns projected coordinates and distances in metres.
    """
    vectors = ends - starts
    lengths_squared = np.sum(vectors * vectors, axis=1)

    relative = points - starts
    denominator = np.maximum(lengths_squared, 1e-12)

    t = np.sum(relative * vectors, axis=1) / denominator
    t = np.clip(t, 0.0, 1.0)

    projected = starts + t[:, None] * vectors
    distances = np.linalg.norm(points - projected, axis=1)

    return projected, distances, t


def main():
    gps = pd.read_csv(GPS_FILE)
    metadata = pd.read_csv(META_FILE)
    shapes = pd.read_csv(SHAPES_FILE)

    gps = gps.merge(
        metadata[
            ["trip_id", "shape_id", "route_id", "route_short_name"]
        ],
        on="trip_id",
        how="left",
        validate="many_to_one"
    )

    shapes = shapes[
        ["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"]
    ].dropna()

    results = []

    for shape_id, trip_points in gps.groupby("shape_id", sort=False):
        route_shape = shapes[
            shapes["shape_id"] == shape_id
        ].sort_values("shape_pt_sequence").copy()

        trip_points = trip_points.copy()

        if len(route_shape) < 2:
            trip_points["matched_lat"] = np.nan
            trip_points["matched_lon"] = np.nan
            trip_points["match_distance_m"] = np.nan
            trip_points["matched_segment_index"] = np.nan
            results.append(trip_points)
            continue

        ref_lat = route_shape["shape_pt_lat"].mean()

        route_xy = to_local_xy(
            route_shape["shape_pt_lat"].to_numpy(),
            route_shape["shape_pt_lon"].to_numpy(),
            ref_lat
        )

        gps_xy = to_local_xy(
            trip_points["lat"].to_numpy(),
            trip_points["lon"].to_numpy(),
            ref_lat
        )

        starts = route_xy[:-1]
        ends = route_xy[1:]

        # Build a spatial index of segment midpoints to find
        # a manageable set of candidate segments for each GPS point.
        midpoints = (starts + ends) / 2.0
        tree = cKDTree(midpoints)

        candidate_count = min(8, len(midpoints))
        _, candidate_indices = tree.query(
            gps_xy, k=candidate_count
        )

        if candidate_count == 1:
            candidate_indices = candidate_indices[:, None]

        matched_xy = np.empty_like(gps_xy)
        matched_distances = np.empty(len(gps_xy))
        matched_segment_indices = np.empty(len(gps_xy), dtype=int)

        for i, point in enumerate(gps_xy):
            candidates = np.atleast_1d(candidate_indices[i])

            projected, distances, _ = project_points_to_segments(
                point[None, :],
                starts[candidates],
                ends[candidates]
            )

            best = int(np.argmin(distances))
            matched_xy[i] = projected[best]
            matched_distances[i] = distances[best]
            matched_segment_indices[i] = candidates[best]

        trip_points["matched_lon"] = (
            matched_xy[:, 0]
            / (111320.0 * np.cos(np.radians(ref_lat)))
        )
        trip_points["matched_lat"] = matched_xy[:, 1] / 111320.0
        trip_points["match_distance_m"] = matched_distances
        trip_points["matched_segment_index"] = matched_segment_indices

        results.append(trip_points)

    matched = pd.concat(results, ignore_index=True)
    matched = matched.sort_values(
        ["trip_id", "ping_seq"]
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(OUTPUT_FILE, index=False)

    print("Segment-based map matching completed.")
    print(f"GPS observations: {len(matched)}")
    print(f"Trips processed: {matched['trip_id'].nunique()}")
    print(
        "Unmatched observations:",
        int(matched["match_distance_m"].isna().sum())
    )
    print(
        "\nDistance to planned route (metres):\n",
        matched["match_distance_m"].describe().round(2)
    )
    print(f"\nSaved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
