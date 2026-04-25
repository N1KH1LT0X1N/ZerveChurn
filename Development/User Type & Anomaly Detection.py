import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =====================================================
# 6. FREE VS UPGRADED USER INDICATORS
# =====================================================

print("=" * 80)
print("USER TYPE ANALYSIS - FREE VS UPGRADED")
print("=" * 80)

# Identify potential upgrade indicators
# Look for upgrade-related events
upgrade_events = user_retention[user_retention['event'].str.contains('upgrade|subscription|payment|billing|tier|plan', case=False, na=False)]
print(f"\nUpgrade-related events found: {len(upgrade_events):,}")
if len(upgrade_events) > 0:
    print(f"Upgrade event types:")
    for event_type, _count in upgrade_events['event'].value_counts().head(10).items():
        print(f"  - {event_type}: {_count:,}")

# Analyze credit usage as indicator (prop_credits_used)
users_with_credits = user_retention[user_retention['prop_credits_used'].notna()]['distinct_id'].unique()
print(f"\nUsers with credit usage data: {len(users_with_credits):,}")

# Check for tool usage patterns (premium features)
tool_users = user_retention[user_retention['prop_tool_name'].notna()].groupby('distinct_id').size().reset_index(name='tool_usage_count')
print(f"\nUsers with tool usage: {len(tool_users):,}")
if len(tool_users) > 0:
    print(f"  - Mean tool usage: {tool_users['tool_usage_count'].mean():.2f}")
    print(f"  - Median tool usage: {tool_users['tool_usage_count'].median():.0f}")

# Identify potential paid users based on multiple indicators
paid_user_indicators = user_retention.groupby('distinct_id').agg({
    'prop_credits_used': lambda x: x.notna().sum(),
    'prop_tool_name': lambda x: x.notna().sum(),
    'event': 'count'
}).reset_index()

paid_user_indicators['potential_paid'] = (
    (paid_user_indicators['prop_credits_used'] > 0) |
    (paid_user_indicators['prop_tool_name'] > 5)  # Heavy tool usage
)

potential_paid_users = paid_user_indicators[paid_user_indicators['potential_paid']]['distinct_id'].nunique()
potential_free_users = len(paid_user_indicators) - potential_paid_users

print(f"\n**User Type Breakdown:**")
print(f"  - Potential Paid Users: {potential_paid_users:,} ({potential_paid_users/len(paid_user_indicators)*100:.1f}%)")
print(f"  - Potential Free Users: {potential_free_users:,} ({potential_free_users/len(paid_user_indicators)*100:.1f}%)")

# Activity comparison
paid_activity = paid_user_indicators[paid_user_indicators['potential_paid']]['event'].mean()
free_activity = paid_user_indicators[~paid_user_indicators['potential_paid']]['event'].mean()
print(f"\n**Activity Comparison:**")
print(f"  - Avg events per paid user: {paid_activity:.2f}")
print(f"  - Avg events per free user: {free_activity:.2f}")
print(f"  - Ratio (paid/free): {paid_activity/free_activity:.2f}x")

# =====================================================
# 7. DUPLICATE & ANOMALY DETECTION
# =====================================================

print("\n" + "=" * 80)
print("DUPLICATE & ANOMALY DETECTION")
print("=" * 80)

# Check for duplicate rows
duplicate_rows = user_retention.duplicated().sum()
print(f"\n1. DUPLICATE ROWS:")
print(f"   Total duplicate rows: {duplicate_rows:,} ({duplicate_rows/len(user_retention)*100:.2f}%)")

# Check for duplicate events (same user, event, timestamp)
duplicate_events = user_retention.duplicated(subset=['distinct_id', 'event', 'timestamp']).sum()
print(f"\n2. DUPLICATE EVENTS (same user, event, timestamp):")
print(f"   Duplicate events: {duplicate_events:,} ({duplicate_events/len(user_retention)*100:.2f}%)")

# Check for duplicate UUIDs (should be unique)
duplicate_uuids = user_retention['uuid'].duplicated().sum()
print(f"\n3. DUPLICATE UUIDs:")
print(f"   Duplicate UUIDs: {duplicate_uuids:,}")

# Detect anomalies - users with extremely high activity
user_event_counts = user_retention.groupby('distinct_id').size()
q99 = user_event_counts.quantile(0.99)
q95 = user_event_counts.quantile(0.95)
anomalous_users = user_event_counts[user_event_counts > q99]

print(f"\n4. ACTIVITY ANOMALIES:")
print(f"   99th percentile threshold: {q99:.0f} events")
print(f"   Users exceeding 99th percentile: {len(anomalous_users):,}")
print(f"   Total events from anomalous users: {anomalous_users.sum():,} ({anomalous_users.sum()/len(user_retention)*100:.1f}%)")

if len(anomalous_users) > 0:
    print(f"\n   Top 10 most active users:")
    for _idx, (_user_id, _event_count) in enumerate(anomalous_users.nlargest(10).items(), 1):
        _short_id = _user_id[:16] + "..."
        print(f"     {_idx:2d}. {_short_id:20s}: {_event_count:6,} events")

# Detect timestamp anomalies
timestamp_diffs = user_retention.sort_values('timestamp')['timestamp'].diff()
same_timestamp_count = (timestamp_diffs == pd.Timedelta(0)).sum()
print(f"\n5. TIMESTAMP ANOMALIES:")
print(f"   Events with identical consecutive timestamps: {same_timestamp_count:,}")

# Create visualization comparing user types
user_type_viz = make_subplots(
    rows=1, cols=2,
    subplot_titles=('User Type Distribution', 'Activity Comparison'),
    specs=[[{"type": "pie"}, {"type": "bar"}]]
)

# Pie chart for user distribution
user_type_viz.add_trace(
    go.Pie(
        labels=['Potential Paid Users', 'Potential Free Users'],
        values=[potential_paid_users, potential_free_users],
        marker=dict(colors=['#8DE5A1', '#FFB482']),
        textinfo='label+percent'
    ),
    row=1, col=1
)

# Bar chart for activity comparison
user_type_viz.add_trace(
    go.Bar(
        x=['Paid Users', 'Free Users'],
        y=[paid_activity, free_activity],
        marker=dict(color=['#8DE5A1', '#FFB482']),
        text=[f'{paid_activity:.1f}', f'{free_activity:.1f}'],
        textposition='auto'
    ),
    row=1, col=2
)

user_type_viz.update_xaxes(title_text="User Type", row=1, col=2)
user_type_viz.update_yaxes(title_text="Avg Events per User", row=1, col=2)

user_type_viz.update_layout(
    title_text='User Type Analysis: Distribution and Activity',
    plot_bgcolor='#1D1D20',
    paper_bgcolor='#1D1D20',
    font=dict(color='#fbfbff', size=12),
    height=500,
    showlegend=False
)

user_type_viz.show()