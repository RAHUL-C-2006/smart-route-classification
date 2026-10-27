import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

import deviation_features as f
import detect_deviations as dd

V2_MATCHED = f.ROOT / "outputs" / "v2" / "v2_map_matched_exact.csv"
OUT = f.ROOT / "outputs" / "v2" / "v2_deviation_candidates.csv"

bundle = f.load_model_bundle()
raw = pd.read_csv(V2_MATCHED, dtype={"shape_id": str})
scored = f.add_reliability_scores(f.add_features(raw), bundle)
assert not scored.duplicated(["trip_id", "ping_seq"]).any()

y = scored["is_outlier"].astype(bool)          # evaluation only
flag = scored["reliability_flag"]
prob = scored["reliability_outlier_prob"]
tn = int(((~y) & (~flag)).sum()); fp = int(((~y) & flag).sum())
fn = int((y & (~flag)).sum());   tp = int((y & flag).sum())

print("RELIABILITY MODEL ON 120 NEW TRIPS (never seen in training)")
print(f"Pings: {len(scored)} | injected outliers: {int(y.sum())} | threshold: {bundle['threshold']:.2f}")
print("Confusion matrix [[TN, FP], [FN, TP]]:", [[tn, fp], [fn, tp]])
print(f"Precision {precision_score(y, flag):.3f} | recall {recall_score(y, flag):.3f} | "
      f"F1 {f1_score(y, flag):.3f} | ROC-AUC {roc_auc_score(y, prob):.3f}")
print("Original held-out result for comparison: precision 0.726 | recall 0.411 | F1 0.525 | ROC-AUC 0.797")
print()

d = dd.detect_candidates(scored)
labels = scored[["trip_id", "ping_seq", "is_outlier"]].rename(columns={"is_outlier": "eval_is_outlier"})
d = d.merge(labels, on=["trip_id", "ping_seq"], how="left")
dd.report(d, "DETECTOR ON 120 NEW TRIPS")

d.to_csv(OUT, index=False)
print("Saved:", OUT)