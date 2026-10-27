from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(r"C:\smart route classification")
MODEL_PATH = ROOT / "models" / "gps_reliability_model_tuned.joblib"
DATA_PATH = ROOT / "outputs" / "map_matched_gps_segments.csv"

# ---------- 1. Saved tuned model ----------
print("=" * 60)
print("TUNED MODEL FILE")
print("=" * 60)
bundle = joblib.load(MODEL_PATH)
print("Loaded object type:", type(bundle).__name__)

feature_cols = None
if isinstance(bundle, dict):
    for key, value in bundle.items():
        if key == "model":
            print(f"  {key}: {type(value).__name__}")
        elif key in ("feature_cols", "feature_columns", "features"):
            print(f"  {key}: {list(value)}")
            feature_cols = list(value)
        else:
            print(f"  {key}: {value}")
else:
    print("Not a dict; attributes:", [a for a in dir(bundle) if not a.startswith("_")][:20])

# ---------- 2. Matched dataset ----------
print()
print("=" * 60)
print("MATCHED DATASET")
print("=" * 60)
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print()
print("Columns and dtypes:")
print(df.dtypes.to_string())
print()
print("First 3 rows:")
print(df.head(3).T.to_string())
print()
print("Missing values per column (only columns with any):")
missing = df.isna().sum()
print(missing[missing > 0].to_string() if (missing > 0).any() else "  none")

# ---------- 3. Compatibility check ----------
print()
print("=" * 60)
print("COMPATIBILITY CHECK")
print("=" * 60)
if feature_cols:
    present = [c for c in feature_cols if c in df.columns]
    absent = [c for c in feature_cols if c not in df.columns]
    print("Model features present in CSV:", present)
    print("Model features MISSING from CSV:", absent)
else:
    print("Could not read feature list from model; share the output above.")

eval_only = [c for c in ("is_outlier", "true_lat", "true_lon") if c in df.columns]
print("Evaluation-only columns present (never used in detector logic):", eval_only)