import pandas as pd
import numpy as np

# 1. Dataset Dimensions
print("# 1. Dataset Dimensions")
print(f"- **Rows**: {user_retention.shape[0]:,}")
print(f"- **Columns**: {user_retention.shape[1]}")
print()

# 2. Schema Information
print("# 2. Schema Information")
print("\nColumn details:")
schema_info = []
for col in user_retention.columns:
    dtype = str(user_retention[col].dtype)
    schema_info.append(f"- `{col}`: {dtype}")
print("\n".join(schema_info[:20]))
print(f"\n... and {len(user_retention.columns) - 20} more columns")
print()

# 3. Sample Rows
print("# 3. Sample Data (First 5 Rows)")
print("\nShowing key columns:")
key_cols = ['distinct_id', 'person_id', 'event', 'timestamp', 'created_at', 'prop_$pathname', 'prop_$browser', 'prop_$os', 'prop_$geoip_country_code']
available_key_cols = [col for col in key_cols if col in user_retention.columns]
sample_df = user_retention[available_key_cols].head()
for idx, row in sample_df.iterrows():
    print(f"\n**Row {idx+1}:**")
    for col in available_key_cols:
        val = row[col]
        if pd.isna(val):
            val = "null"
        elif isinstance(val, pd.Timestamp):
            val = val.strftime('%Y-%m-%d %H:%M:%S')
        print(f"  - {col}: {val}")
print()

# 4. Missing Values Analysis
print("# 4. Missing Values Analysis")
missing_stats = user_retention.isnull().sum()
missing_pct = (missing_stats / len(user_retention) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing_stats,
    'Missing %': missing_pct
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
print(f"\n**Total columns with missing values**: {len(missing_df)} out of {len(user_retention.columns)}")
print(f"\n**Top 10 columns with most missing values:**")
for col, row in missing_df.head(10).iterrows():
    print(f"- `{col}`: {int(row['Missing Count']):,} ({row['Missing %']:.1f}%)")
print()

# 5. Timestamp Analysis
print("# 5. Timestamp Analysis")
timestamp_cols = [col for col in user_retention.columns if user_retention[col].dtype == 'datetime64[us]' or user_retention[col].dtype == 'datetime64[ns]']
print(f"\n**Timestamp columns found**: {len(timestamp_cols)}")
for col in timestamp_cols[:5]:
    non_null = user_retention[col].dropna()
    if len(non_null) > 0:
        print(f"\n`{col}:`")
        print(f"  - Min: {non_null.min()}")
        print(f"  - Max: {non_null.max()}")
        print(f"  - Range: {(non_null.max() - non_null.min()).days} days")
print()

# 6. Data Quality Checks
print("# 6. Data Quality Checks")
print(f"\n- **Duplicate rows**: {user_retention.duplicated().sum():,}")
print(f"- **Completely empty rows**: {user_retention.isnull().all(axis=1).sum():,}")
print(f"- **Memory usage**: {user_retention.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"- **Unique events**: {user_retention['event'].nunique() if 'event' in user_retention.columns else 'N/A'}")
print(f"- **Unique users (distinct_id)**: {user_retention['distinct_id'].nunique() if 'distinct_id' in user_retention.columns else 'N/A':,}")
print(f"- **Unique persons (person_id)**: {user_retention['person_id'].nunique() if 'person_id' in user_retention.columns else 'N/A':,}")
print()

# 7. Summary Statistics
print("# 7. Data Type Distribution")
dtype_counts = user_retention.dtypes.value_counts()
print("\n**Column types:**")
for dtype, count in dtype_counts.items():
    print(f"- {dtype}: {count} columns")
print()

print("---")
print("**✓ Exploration complete!**")