import pandas as pd
import numpy as np

print("=" * 80)
print("DATA QUALITY ASSESSMENT")
print("=" * 80)

# Missing data analysis
print("\n" + "-" * 80)
print("MISSING DATA ANALYSIS")
print("-" * 80)

missing_summary = pd.DataFrame({
    'missing_count': df.isnull().sum(),
    'missing_pct': (df.isnull().sum() / len(df) * 100).round(2)
})
missing_summary = missing_summary[missing_summary['missing_count'] > 0].sort_values('missing_count', ascending=False)

print(f"\nColumns with Missing Data: {len(missing_summary)} out of {df.shape[1]} total columns")
print(f"Total Missing Values: {missing_summary['missing_count'].sum():,}")
print(f"\nTop 15 Columns with Most Missing Data:")
print(missing_summary.head(15).to_string())

# Columns with complete data
complete_cols = df.shape[1] - len(missing_summary)
print(f"\n✓ Columns with Complete Data (No Missing): {complete_cols}")

# Duplicate analysis
print("\n" + "-" * 80)
print("DUPLICATE RECORDS ANALYSIS")
print("-" * 80)

# Check for duplicate rows
duplicate_rows = df.duplicated().sum()
print(f"Duplicate Rows (all columns identical): {duplicate_rows:,}")

# Check for duplicate UUIDs (should be unique identifiers)
if 'uuid' in df.columns:
    duplicate_uuids = df['uuid'].duplicated().sum()
    unique_uuids = df['uuid'].nunique()
    print(f"Duplicate UUIDs: {duplicate_uuids:,}")
    print(f"Unique UUIDs: {unique_uuids:,} out of {len(df):,} rows")

# Check for duplicate events per user
if 'person_id' in df.columns and 'event' in df.columns:
    duplicate_events = df.duplicated(subset=['person_id', 'event', 'timestamp']).sum()
    print(f"Duplicate Events (same user, event, timestamp): {duplicate_events:,}")

# Key identifiers
print("\n" + "-" * 80)
print("KEY IDENTIFIERS")
print("-" * 80)

identifier_cols = ['person_id', 'distinct_id', 'uuid', 'prop_$user_id']
for col in identifier_cols:
    if col in df.columns:
        unique_vals = df[col].nunique()
        null_vals = df[col].isnull().sum()
        print(f"{col:20s}: {unique_vals:,} unique values, {null_vals:,} nulls")

# Event types
print("\n" + "-" * 80)
print("EVENT TYPES SUMMARY")
print("-" * 80)

if 'event' in df.columns:
    event_counts = df['event'].value_counts()
    print(f"Total Unique Event Types: {len(event_counts)}")
    print(f"\nTop 10 Most Frequent Events:")
    for event, count in event_counts.head(10).items():
        print(f"  {event:50s}: {count:,} ({count/len(df)*100:.1f}%)")
