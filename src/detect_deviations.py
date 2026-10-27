import numpy as np
import pandas as pd

from deviation_features import (
    DATA_PATH, HELD_OUT_TRIPS, ROOT,
    add_features, add_reliability_scores, load_model_bundle,
)

# ---- Fixed first-pass settings (not tuned) ----
OFF_ROUTE_M = 30.0        # distance from planned route counted as "off route"
MIN_RUN = 3               # consecutive off-route pings needed for a deviation candidate
MAX_FLAGGED_SHARE = 0.5   # if more of the run is flagged unreliable, do not call it a deviation

OUT_PINGS = ROOT / "outputs" / "deviation_candidates.csv"
OUT_SUMMARY = ROOT / "outputs" / "deviation_candidates_summary.csv"


def detect_candidates(df):
    """Uses ONLY observable columns. No labels, no true positions."""
    cols = [
        "trip_id", "ping_seq", "timestamp", "lat", "lon",
        "matched_lat", "matched_lon", "route_distance_m",
        "reliability_outlier_prob", "reliability_flag",
    ]
    d = df[cols].sort_values(["trip_id", "ping_seq"]).reset_index(drop=True)

    d["off_route"] = d["route_distance_m"] > OFF_ROUTE_M

    # Number each run of consecutive off-route (or on-route) pings within a trip
    prev = d.groupby("trip_id", sort=False)["off_route"].shift(fill_value=False)
    changed = d["off_route"] != prev
    d["run_id"] = changed.groupby(d["trip_id"], sort=False).cumsum()

    # Length and flagged-share of each off-route run
    stats = (
        d[d["off_route"]]
        .groupby(["trip_id", "run_id"])
        .agg(run_length=("off_route", "size"),
             run_flagged_share=("reliability_flag", "mean"))
        .reset_index()
    )
    d = d.merge(stats, on=["trip_id", "run_id"], how="left")
    d["run_length"] = d["run_length"].fillna(0).astype(int)
    d["run_flagged_share"] = d["run_flagged_share"].fillna(0.0)

    deviation = (
        d["off_route"]
        & (d["run_length"] >= MIN_RUN)
        & (d["run_flagged_share"] <= MAX_FLAGGED_SHARE)
    )
    noise = (
        d["reliability_flag"]
        & ~deviation
        & (~d["off_route"] | (d["run_length"] < MIN_RUN))
    )
    uncertain = ~deviation & ~noise & (d["off_route"] | d["reliability_flag"])

    d["candidate_label"] = np.select(
        [deviation, noise, uncertain],
        ["potential_route_deviation", "likely_gps_noise", "uncertain"],
        default="normal",
    )
    return d


def report(d, title):
    print("=" * 60)
    print(f"{title}: {len(d)} pings, {d['trip_id'].nunique()} trips")
    print("=" * 60)
    print("Detector label (rows) vs injected-outlier label (columns, evaluation only):")
    print(pd.crosstab(d["candidate_label"], d["eval_is_outlier"], margins=True).to_string())

    inj = d[d["eval_is_outlier"]]
    print(f"\nInjected outliers: {len(inj)}")
    print(f"  of which more than {OFF_ROUTE_M:.0f} m from the route: {int(inj['off_route'].sum())}")
    print("  Labels the detector gave them:")
    print(inj["candidate_label"].value_counts().to_string())
    n_dev = int((d["candidate_label"] == "potential_route_deviation").sum())
    print(f"\nPotential-deviation candidates: {n_dev}")
    print()


if __name__ == "__main__":
    print(f"Settings: off-route > {OFF_ROUTE_M} m, min run {MIN_RUN}, "
          f"max flagged share {MAX_FLAGGED_SHARE}")
    print()

    bundle = load_model_bundle()
    raw = pd.read_csv(DATA_PATH)
    scored = add_reliability_scores(add_features(raw), bundle)
    assert not scored.duplicated(["trip_id", "ping_seq"]).any()

    d = detect_candidates(scored)

    # Attach the synthetic label AFTER detection, for offline evaluation only
    eval_labels = scored[["trip_id", "ping_seq", "is_outlier"]].rename(
        columns={"is_outlier": "eval_is_outlier"}
    )
    d = d.merge(eval_labels, on=["trip_id", "ping_seq"], how="left")
    d["split"] = np.where(d["trip_id"].isin(HELD_OUT_TRIPS), "held_out", "training_trips")

    report(d[d["split"] == "held_out"], "HELD-OUT TRIPS (headline)")
    report(d[d["split"] == "training_trips"], "TRAINING TRIPS (in-sample, interpret cautiously)")

    d.to_csv(OUT_PINGS, index=False)
    summary = d.groupby(["split", "candidate_label"]).size().unstack(fill_value=0)
    summary.to_csv(OUT_SUMMARY)
    print("Saved:", OUT_PINGS)
    print("Saved:", OUT_SUMMARY)