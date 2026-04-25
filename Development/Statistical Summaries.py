import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =====================================================
# 1. STATISTICAL SUMMARIES FOR NUMERICAL COLUMNS
# =====================================================

# Get numerical columns only
numerical_cols = user_retention.select_dtypes(include=[np.number]).columns.tolist()

# Calculate comprehensive statistics
stats_summary = pd.DataFrame({
    'count': [user_retention[col].count() for col in numerical_cols],
    'mean': [user_retention[col].mean() for col in numerical_cols],
    'std': [user_retention[col].std() for col in numerical_cols],
    'min': [user_retention[col].min() for col in numerical_cols],
    'q25': [user_retention[col].quantile(0.25) for col in numerical_cols],
    'median': [user_retention[col].quantile(0.50) for col in numerical_cols],
    'q75': [user_retention[col].quantile(0.75) for col in numerical_cols],
    'max': [user_retention[col].max() for col in numerical_cols]
}, index=numerical_cols)

print("=" * 80)
print("STATISTICAL SUMMARIES - NUMERICAL COLUMNS")
print("=" * 80)
print(f"\nTotal numerical columns: {len(numerical_cols)}")
print(f"\nKey Statistics:\n")
print(stats_summary.to_string())

# =====================================================
# 2. CARDINALITY ANALYSIS FOR CATEGORICAL COLUMNS
# =====================================================

categorical_cols = user_retention.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

cardinality_stats = pd.DataFrame({
    'unique_values': [user_retention[col].nunique() for col in categorical_cols],
    'null_count': [user_retention[col].isnull().sum() for col in categorical_cols],
    'null_pct': [(user_retention[col].isnull().sum() / len(user_retention) * 100) for col in categorical_cols],
    'most_common': [user_retention[col].value_counts().index[0] if user_retention[col].value_counts().shape[0] > 0 else None for col in categorical_cols],
    'most_common_count': [user_retention[col].value_counts().iloc[0] if user_retention[col].value_counts().shape[0] > 0 else 0 for col in categorical_cols]
}, index=categorical_cols)

cardinality_stats = cardinality_stats.sort_values('unique_values', ascending=False)

print("\n" + "=" * 80)
print("CARDINALITY ANALYSIS - CATEGORICAL COLUMNS")
print("=" * 80)
print(f"\nTotal categorical columns: {len(categorical_cols)}")
print(f"\nTop 20 columns by cardinality:\n")
print(cardinality_stats.head(20).to_string())

# =====================================================
# 3. TOP 20 EVENT TYPE DISTRIBUTION
# =====================================================

event_distribution = user_retention['event'].value_counts().head(20)
event_pct = (event_distribution / len(user_retention) * 100).round(2)

print("\n" + "=" * 80)
print("TOP 20 EVENT TYPE DISTRIBUTION")
print("=" * 80)
print(f"\nTotal unique events: {user_retention['event'].nunique()}")
print(f"Total events logged: {len(user_retention):,}\n")

for idx, (event_name, count) in enumerate(event_distribution.items(), 1):
    print(f"{idx:2d}. {event_name:40s} | {count:8,} ({event_pct[event_name]:6.2f}%)")

# Create interactive visualization
event_viz = go.Figure(data=[
    go.Bar(
        x=event_distribution.values,
        y=event_distribution.index,
        orientation='h',
        marker=dict(
            color=['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF',
                   '#1F77B4', '#9467BD', '#8C564B', '#C49C94', '#E377C2',
                   '#F7B6D2', '#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B',
                   '#D0BBFF', '#1F77B4', '#9467BD', '#8C564B', '#C49C94']
        ),
        text=[f'{v:,}' for v in event_distribution.values],
        textposition='auto'
    )
])

event_viz.update_layout(
    title='Top 20 Event Types - Distribution',
    xaxis_title='Event Count',
    yaxis_title='Event Type',
    plot_bgcolor='#1D1D20',
    paper_bgcolor='#1D1D20',
    font=dict(color='#fbfbff', size=12),
    height=700,
    yaxis={'categoryorder': 'total ascending'}
)

event_viz.show()