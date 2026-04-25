import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Load dataset
df_stat = pd.read_parquet('user_retention.parquet')

print("=" * 80)
print("📊 STATISTICAL PROFILING & DATA QUALITY ASSESSMENT")
print("=" * 80)

# ============================================================================
# SECTION 1: NUMERICAL COLUMN STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("1. NUMERICAL COLUMN COMPREHENSIVE STATISTICS")
print("=" * 80)

# Select numerical columns
numerical_columns = df_stat.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nTotal numerical columns: {len(numerical_columns)}")

# Calculate comprehensive statistics for numerical columns
numerical_stats = pd.DataFrame({
    'column': numerical_columns,
    'count': [df_stat[col].count() for col in numerical_columns],
    'missing': [df_stat[col].isna().sum() for col in numerical_columns],
    'missing_pct': [df_stat[col].isna().sum() / len(df_stat) * 100 for col in numerical_columns],
    'mean': [df_stat[col].mean() if df_stat[col].count() > 0 else np.nan for col in numerical_columns],
    'std': [df_stat[col].std() if df_stat[col].count() > 0 else np.nan for col in numerical_columns],
    'min': [df_stat[col].min() if df_stat[col].count() > 0 else np.nan for col in numerical_columns],
    'q25': [df_stat[col].quantile(0.25) if df_stat[col].count() > 0 else np.nan for col in numerical_columns],
    'median': [df_stat[col].median() if df_stat[col].count() > 0 else np.nan for col in numerical_columns],
    'q75': [df_stat[col].quantile(0.75) if df_stat[col].count() > 0 else np.nan for col in numerical_columns],
    'max': [df_stat[col].max() if df_stat[col].count() > 0 else np.nan for col in numerical_columns],
    'skewness': [df_stat[col].skew() if df_stat[col].count() > 0 else np.nan for col in numerical_columns],
    'kurtosis': [df_stat[col].kurtosis() if df_stat[col].count() > 0 else np.nan for col in numerical_columns]
})

print("\nNumerical Statistics Summary:")
print(numerical_stats.to_string(index=False))

# ============================================================================
# SECTION 2: CATEGORICAL COLUMN CARDINALITY ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("2. CATEGORICAL COLUMN CARDINALITY ANALYSIS")
print("=" * 80)

# Select categorical columns (string, object, category types)
categorical_columns = df_stat.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
print(f"\nTotal categorical columns: {len(categorical_columns)}")

# Calculate cardinality statistics
categorical_stats = pd.DataFrame({
    'column': categorical_columns,
    'count': [df_stat[col].count() for col in categorical_columns],
    'missing': [df_stat[col].isna().sum() for col in categorical_columns],
    'missing_pct': [df_stat[col].isna().sum() / len(df_stat) * 100 for col in categorical_columns],
    'unique_values': [df_stat[col].nunique() for col in categorical_columns],
    'cardinality_ratio': [df_stat[col].nunique() / df_stat[col].count() * 100 if df_stat[col].count() > 0 else 0 for col in categorical_columns],
    'most_common': [df_stat[col].value_counts().index[0] if df_stat[col].count() > 0 else None for col in categorical_columns],
    'most_common_count': [df_stat[col].value_counts().iloc[0] if df_stat[col].count() > 0 else 0 for col in categorical_columns],
    'most_common_pct': [df_stat[col].value_counts().iloc[0] / df_stat[col].count() * 100 if df_stat[col].count() > 0 else 0 for col in categorical_columns]
})

# Sort by cardinality ratio
categorical_stats = categorical_stats.sort_values('cardinality_ratio', ascending=False)
print("\nCategorical Statistics Summary (sorted by cardinality ratio):")
print(categorical_stats.to_string(index=False))

# ============================================================================
# SECTION 3: MISSING VALUE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("3. COMPREHENSIVE MISSING VALUE ANALYSIS")
print("=" * 80)

# Calculate missing values for all columns
all_columns = df_stat.columns.tolist()
missing_analysis = pd.DataFrame({
    'column': all_columns,
    'missing_count': [df_stat[col].isna().sum() for col in all_columns],
    'missing_percentage': [df_stat[col].isna().sum() / len(df_stat) * 100 for col in all_columns],
    'non_missing_count': [df_stat[col].count() for col in all_columns],
    'dtype': [str(df_stat[col].dtype) for col in all_columns]
})

# Filter columns with missing values
missing_analysis_filtered = missing_analysis[missing_analysis['missing_count'] > 0].sort_values('missing_percentage', ascending=False)

print(f"\nTotal columns: {len(all_columns)}")
print(f"Columns with missing values: {len(missing_analysis_filtered)}")
print(f"Complete columns (no missing): {len(all_columns) - len(missing_analysis_filtered)}")

print("\nTop 20 columns with highest missing percentage:")
print(missing_analysis_filtered.head(20).to_string(index=False))

# ============================================================================
# SECTION 4: DATA QUALITY SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("4. DATA QUALITY ASSESSMENT SUMMARY")
print("=" * 80)

total_rows = len(df_stat)
total_cols = len(all_columns)
total_cells = total_rows * total_cols
total_missing_cells = df_stat.isna().sum().sum()
data_completeness = (1 - total_missing_cells / total_cells) * 100

print(f"\n📊 Overall Data Quality Metrics:")
print(f"  • Total rows: {total_rows:,}")
print(f"  • Total columns: {total_cols}")
print(f"  • Total cells: {total_cells:,}")
print(f"  • Total missing cells: {total_missing_cells:,}")
print(f"  • Data completeness: {data_completeness:.2f}%")
print(f"  • Numerical columns: {len(numerical_columns)}")
print(f"  • Categorical columns: {len(categorical_columns)}")
print(f"  • Columns with >50% missing: {len(missing_analysis_filtered[missing_analysis_filtered['missing_percentage'] > 50])}")
print(f"  • Columns with >90% missing: {len(missing_analysis_filtered[missing_analysis_filtered['missing_percentage'] > 90])}")

# Identify high-cardinality categorical columns (potential identifiers)
high_cardinality_cats = categorical_stats[categorical_stats['cardinality_ratio'] > 80]
print(f"  • High-cardinality categorical columns (>80% unique): {len(high_cardinality_cats)}")

# Store results for downstream use
stat_profile_numerical = numerical_stats
stat_profile_categorical = categorical_stats
stat_profile_missing = missing_analysis_filtered
data_quality_summary = {
    'total_rows': total_rows,
    'total_columns': total_cols,
    'data_completeness': data_completeness,
    'numerical_cols': len(numerical_columns),
    'categorical_cols': len(categorical_columns),
    'cols_with_missing': len(missing_analysis_filtered)
}

print("\n✅ Statistical profiling complete!")