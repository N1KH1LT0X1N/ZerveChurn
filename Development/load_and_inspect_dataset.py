import pandas as pd
import numpy as np

# Load the parquet file
df = pd.read_parquet('user_retention.parquet')

# Display basic dataset information
print("=" * 80)
print("DATASET DIMENSIONS")
print("=" * 80)
print(f"Total Rows: {df.shape[0]:,}")
print(f"Total Columns: {df.shape[1]}")
print()

# Memory usage
print("=" * 80)
print("MEMORY USAGE")
print("=" * 80)
memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
print(f"Total Memory Usage: {memory_mb:.2f} MB")
print(f"Average Memory per Row: {memory_mb * 1024 / df.shape[0]:.2f} KB")
print()

# Column types summary
print("=" * 80)
print("COLUMN TYPES SUMMARY")
print("=" * 80)
dtype_summary = df.dtypes.value_counts()
for dtype, count in dtype_summary.items():
    print(f"{str(dtype):30s}: {count:3d} columns")
print()

# Display first few rows
print("=" * 80)
print("SAMPLE DATA (First 3 Rows)")
print("=" * 80)
print(df.head(3).to_string())
