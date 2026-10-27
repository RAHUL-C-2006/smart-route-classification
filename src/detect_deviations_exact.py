import numpy as np
import pandas as pd

import deviation_features as f
import detect_deviations as dd

EXACT_PATH = f.ROOT / "outputs" / "map_matched_gps_segments_exact.csv"

bundle = f.load_model_bundle()
raw = pd.read_csv(EXACT_PATH, dtype={"shape_id": str})
scored = f.add_reliability_scores(f.add_features(raw), bundle)

# Check: reliability model on held-out trips with the corrected matching
test = scored[scored["trip_id"].isin(f.HELD_OUT_TRIPS)]
actual = test["is_outlier"].astype(bool)
pred = test["reliability_flag"]
tn = int(((~actual) & (~pred)).sum()); fp = int(((~actual) & pred).sum())
fn = int((actual & (~pred)).sum());   tp = int((actual & pred).sum())
print("Reliability model on corrected matching (held-out):")
print("  [[TN, FP], [FN, TP]] =", [[tn, fp], [fn, tp]], "| before: [[4170, 26], [99, 69]]")
print()

d = dd.detect_candidates(scored)
eval_labels = scored[["trip_id", "ping_seq", "is_outlier"]].rename(
    columns={"is_outlier": "eval_is_outlier"})
d = d.merge(eval_labels, on=["trip_id", "ping_seq"], how="left")
d["split"] = np.where(d["trip_id"].isin(f.HELD_OUT_TRIPS), "held_out", "training_trips")

dd.report(d[d["split"] == "held_out"], "HELD-OUT TRIPS (corrected matching)")
dd.report(d[d["split"] == "training_trips"], "TRAINING TRIPS (in-sample)")

d.to_csv(f.ROOT / "outputs" / "deviation_candidates_exact.csv", index=False)
print("Saved: outputs\\deviation_candidates_exact.csv")