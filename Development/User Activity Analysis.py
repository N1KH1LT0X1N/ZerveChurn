import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# =====================================================
# 4. USER ACTIVITY DISTRIBUTION
# =====================================================

# Calculate events per user
user_activity = user_retention.groupby('distinct_id').size().reset_index(name='event_count')

print("=" * 80)
print("USER ACTIVITY DISTRIBUTION")
print("=" * 80)
print(f"\nTotal unique users: {user_activity['distinct_id'].nunique():,}")
print(f"Total events: {user_activity['event_count'].sum():,}")
print(f"\nActivity Statistics:")
print(f"  - Mean events per user: {user_activity['event_count'].mean():.2f}")
print(f"  - Median events per user: {user_activity['event_count'].median():.0f}")
print(f"  - Std dev: {user_activity['event_count'].std():.2f}")
print(f"  - Min events: {user_activity['event_count'].min()}")
print(f"  - Max events: {user_activity['event_count'].max()}")
print(f"  - 25th percentile: {user_activity['event_count'].quantile(0.25):.0f}")
print(f"  - 75th percentile: {user_activity['event_count'].quantile(0.75):.0f}")
print(f"  - 95th percentile: {user_activity['event_count'].quantile(0.95):.0f}")

# Activity distribution breakdown
print(f"\nActivity Level Breakdown:")
activity_levels = [
    (1, 1, "Single event users"),
    (2, 5, "Low activity (2-5 events)"),
    (6, 10, "Medium activity (6-10 events)"),
    (11, 50, "High activity (11-50 events)"),
    (51, 100, "Very high activity (51-100 events)"),
    (101, float('inf'), "Power users (100+ events)")
]

for min_events, max_events, label in activity_levels:
    if max_events == float('inf'):
        _count = len(user_activity[user_activity['event_count'] >= min_events])
    else:
        _count = len(user_activity[(user_activity['event_count'] >= min_events) & (user_activity['event_count'] <= max_events)])
    _pct = (_count / len(user_activity) * 100)
    print(f"  - {label:30s}: {_count:6,} users ({_pct:5.2f}%)")

# Create histogram with appropriate bins
activity_hist = go.Figure()

activity_hist.add_trace(go.Histogram(
    x=user_activity['event_count'],
    nbinsx=50,
    marker=dict(color='#A1C9F4', line=dict(color='#fbfbff', width=0.5)),
    name='User Activity'
))

activity_hist.update_layout(
    title='User Activity Distribution (Events per User)',
    xaxis_title='Number of Events',
    yaxis_title='Number of Users',
    plot_bgcolor='#1D1D20',
    paper_bgcolor='#1D1D20',
    font=dict(color='#fbfbff', size=12),
    height=500,
    showlegend=False
)

activity_hist.show()

# Create log-scale visualization for better visibility
activity_log = go.Figure()

activity_log.add_trace(go.Histogram(
    x=user_activity['event_count'],
    nbinsx=50,
    marker=dict(color='#8DE5A1', line=dict(color='#fbfbff', width=0.5)),
    name='User Activity'
))

activity_log.update_layout(
    title='User Activity Distribution (Log Scale Y-Axis)',
    xaxis_title='Number of Events',
    yaxis_title='Number of Users (Log Scale)',
    yaxis_type='log',
    plot_bgcolor='#1D1D20',
    paper_bgcolor='#1D1D20',
    font=dict(color='#fbfbff', size=12),
    height=500,
    showlegend=False
)

activity_log.show()