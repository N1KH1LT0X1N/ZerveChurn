import pandas as pd
import numpy as np

# ── Helper ────────────────────────────────────────────────────────────────────
def describe_dataset(df, name):
    print("\n" + "=" * 100)
    print(f"  DATASET: {name}")
    print(f"  Shape  : {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    print("=" * 100)

    # Classify columns
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    # ── Null counts ──────────────────────────────────────────────────────────
    null_counts = df.isnull().sum()
    null_pct    = (null_counts / len(df) * 100).round(2)

    # ── Numeric stats ────────────────────────────────────────────────────────
    if num_cols:
        print(f"\n{'─'*100}")
        print(f"  NUMERIC FIELDS ({len(num_cols)})")
        print(f"{'─'*100}")
        hdr = f"{'Field':<45}{'Dtype':<15}{'Nulls':>8}{'Null%':>8}{'Mean':>14}{'Median':>14}{'Std':>14}{'Min':>14}{'Max':>14}"
        print(hdr)
        print("─" * 146)
        for c in num_cols:
            col = df[c].dropna()
            if len(col) > 0:
                mean_v   = f"{col.mean():.4g}"
                med_v    = f"{col.median():.4g}"
                std_v    = f"{col.std():.4g}"
                min_v    = f"{col.min():.4g}"
                max_v    = f"{col.max():.4g}"
            else:
                mean_v = med_v = std_v = min_v = max_v = "N/A"
            dtype_str = str(df[c].dtype)
            print(f"  {c:<43}{dtype_str:<15}{null_counts[c]:>8,}{null_pct[c]:>8.2f}%{mean_v:>14}{med_v:>14}{std_v:>14}{min_v:>14}{max_v:>14}")

    # ── Categorical / object stats ───────────────────────────────────────────
    if cat_cols:
        print(f"\n{'─'*100}")
        print(f"  CATEGORICAL / OTHER FIELDS ({len(cat_cols)})")
        print(f"{'─'*100}")
        hdr2 = f"{'Field':<45}{'Dtype':<15}{'Nulls':>8}{'Null%':>8}{'Cardinality':>14}{'Top Value':<30}"
        print(hdr2)
        print("─" * 120)
        for c in cat_cols:
            vc = df[c].value_counts(dropna=True)
            card = vc.shape[0]
            top  = str(vc.index[0])[:28] if card > 0 else "N/A"
            dtype_str = str(df[c].dtype)
            print(f"  {c:<43}{dtype_str:<15}{null_counts[c]:>8,}{null_pct[c]:>8.2f}%{card:>14,}  {top:<28}")

    print(f"\n{'─'*100}")
    print(f"  DTYPE SUMMARY")
    print(f"{'─'*100}")
    for dtype_name, cnt in df.dtypes.value_counts().items():
        print(f"  {str(dtype_name):<30}: {cnt:>4} columns")
    print()


# ── Load all three datasets ───────────────────────────────────────────────────
print("Loading datasets...")
retention_df    = pd.read_parquet("user_retention.parquet")
segments_df     = pd.read_csv("user_segments.csv")
intelligence_df = pd.read_csv("user_intelligence_export.csv")

print(f"  user_retention.parquet     → {retention_df.shape[0]:,} rows × {retention_df.shape[1]} cols")
print(f"  user_segments.csv          → {segments_df.shape[0]:,} rows × {segments_df.shape[1]} cols")
print(f"  user_intelligence_export   → {intelligence_df.shape[0]:,} rows × {intelligence_df.shape[1]} cols")

# ── Describe each dataset ─────────────────────────────────────────────────────
describe_dataset(retention_df,    "user_retention.parquet")
describe_dataset(segments_df,     "user_segments.csv")
describe_dataset(intelligence_df, "user_intelligence_export.csv")

print("\n" + "=" * 100)
print("  ALL THREE DATASETS DESCRIBED SUCCESSFULLY")
print("=" * 100)
