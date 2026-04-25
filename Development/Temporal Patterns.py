import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =====================================================
# 5. TEMPORAL PATTERNS ANALYSIS
# =====================================================

# Create temporal features from timestamp
temporal_df = user_retention[['timestamp', 'event']].copy()
temporal_df['hour'] = temporal_df['timestamp'].dt.hour
temporal_df['day_of_week'] = temporal_df['timestamp'].dt.dayofweek  # Monday=0, Sunday=6
temporal_df['week'] = temporal_df['timestamp'].dt.isocalendar().week
temporal_df['month'] = temporal_df['timestamp'].dt.month
temporal_df['date'] = temporal_df['timestamp'].dt.date

print("=" * 80)
print("TEMPORAL PATTERNS ANALYSIS")
print("=" * 80)

# Hour of day patterns
hourly_events = temporal_df.groupby('hour').size().reset_index(name='event_count')
print(f"\n1. HOURLY PATTERNS:")
print(f"   Peak hour: {hourly_events.loc[hourly_events['event_count'].idxmax(), 'hour']}:00 with {hourly_events['event_count'].max():,} events")
print(f"   Lowest hour: {hourly_events.loc[hourly_events['event_count'].idxmin(), 'hour']}:00 with {hourly_events['event_count'].min():,} events")

# Day of week patterns
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_events = temporal_df.groupby('day_of_week').size().reset_index(name='event_count')
daily_events['day_name'] = daily_events['day_of_week'].apply(lambda x: day_names[x])
print(f"\n2. DAY OF WEEK PATTERNS:")
print(f"   Peak day: {daily_events.loc[daily_events['event_count'].idxmax(), 'day_name']} with {daily_events['event_count'].max():,} events")
print(f"   Lowest day: {daily_events.loc[daily_events['event_count'].idxmin(), 'day_name']} with {daily_events['event_count'].min():,} events")
print(f"\n   Day-by-day breakdown:")
for _, row in daily_events.iterrows():
    print(f"     {row['day_name']:10s}: {row['event_count']:7,} events")

# Week patterns
weekly_events = temporal_df.groupby('week').size().reset_index(name='event_count')
print(f"\n3. WEEKLY PATTERNS:")
print(f"   Total weeks in data: {weekly_events['week'].nunique()}")
print(f"   Peak week: Week {weekly_events.loc[weekly_events['event_count'].idxmax(), 'week']} with {weekly_events['event_count'].max():,} events")
print(f"   Average events per week: {weekly_events['event_count'].mean():,.0f}")

# Monthly patterns
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_events = temporal_df.groupby('month').size().reset_index(name='event_count')
monthly_events['month_name'] = monthly_events['month'].apply(lambda x: month_names[x-1])
print(f"\n4. MONTHLY PATTERNS:")
print(f"   Months covered: {', '.join(monthly_events['month_name'].tolist())}")
for _, row in monthly_events.iterrows():
    print(f"     {row['month_name']:3s}: {row['event_count']:7,} events")

# Create comprehensive temporal visualizations
temporal_viz = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Events by Hour of Day', 'Events by Day of Week', 
                    'Events by Week', 'Events by Month'),
    vertical_spacing=0.12,
    horizontal_spacing=0.12
)

# Hour of day chart
temporal_viz.add_trace(
    go.Bar(x=hourly_events['hour'], y=hourly_events['event_count'],
           marker=dict(color='#A1C9F4'), name='Hourly'),
    row=1, col=1
)

# Day of week chart
temporal_viz.add_trace(
    go.Bar(x=daily_events['day_name'], y=daily_events['event_count'],
           marker=dict(color='#FFB482'), name='Daily'),
    row=1, col=2
)

# Week chart
temporal_viz.add_trace(
    go.Scatter(x=weekly_events['week'], y=weekly_events['event_count'],
               mode='lines+markers', marker=dict(color='#8DE5A1'), line=dict(color='#8DE5A1'),
               name='Weekly'),
    row=2, col=1
)

# Month chart
temporal_viz.add_trace(
    go.Bar(x=monthly_events['month_name'], y=monthly_events['event_count'],
           marker=dict(color='#FF9F9B'), name='Monthly'),
    row=2, col=2
)

temporal_viz.update_xaxes(title_text="Hour", row=1, col=1)
temporal_viz.update_xaxes(title_text="Day", row=1, col=2)
temporal_viz.update_xaxes(title_text="Week Number", row=2, col=1)
temporal_viz.update_xaxes(title_text="Month", row=2, col=2)

temporal_viz.update_yaxes(title_text="Event Count", row=1, col=1)
temporal_viz.update_yaxes(title_text="Event Count", row=1, col=2)
temporal_viz.update_yaxes(title_text="Event Count", row=2, col=1)
temporal_viz.update_yaxes(title_text="Event Count", row=2, col=2)

temporal_viz.update_layout(
    title_text='Temporal Patterns: Hour, Day, Week, Month',
    plot_bgcolor='#1D1D20',
    paper_bgcolor='#1D1D20',
    font=dict(color='#fbfbff', size=11),
    height=800,
    showlegend=False
)

temporal_viz.show()