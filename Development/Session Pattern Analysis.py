import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 80)
print("SESSION PATTERN ANALYSIS")
print("=" * 80)

# Sort by user and timestamp
# Defensive projection: only `distinct_id` and `timestamp` are used downstream
# in this block. Sorting + .copy() of the full 409,287×107 dataframe (with 77
# object columns) requires a contiguous ~240 MiB allocation that fails under
# memory pressure with `MemoryError: Unable to allocate 240. MiB ...`. Cuts
# the working-set down to ~6 MiB. See docs/repo_state_and_next_steps.md
# Blocker D.
df_sessions = (
    user_retention[['distinct_id', 'timestamp']]
    .sort_values(['distinct_id', 'timestamp'])
    .copy()
)

# Calculate time difference between consecutive events for each user
df_sessions['time_since_last_event'] = df_sessions.groupby('distinct_id')['timestamp'].diff()

# Define session: 30-minute gap threshold
session_gap_threshold = pd.Timedelta(minutes=30)
df_sessions['is_new_session'] = (df_sessions['time_since_last_event'] > session_gap_threshold) | (df_sessions['time_since_last_event'].isna())
df_sessions['session_number'] = df_sessions.groupby('distinct_id')['is_new_session'].cumsum()

# Create unique session identifier
df_sessions['session_id'] = df_sessions['distinct_id'] + '_session_' + df_sessions['session_number'].astype(str)

print(f"\n✓ Identified {df_sessions['session_id'].nunique():,} unique sessions across {df_sessions['distinct_id'].nunique():,} users")

# Calculate session-level metrics
session_metrics = df_sessions.groupby('session_id').agg({
    'distinct_id': 'first',
    'timestamp': ['min', 'max', 'count']
}).reset_index()

session_metrics.columns = ['session_id', 'user_id', 'session_start', 'session_end', 'events_in_session']
session_metrics['session_length_minutes'] = (session_metrics['session_end'] - session_metrics['session_start']).dt.total_seconds() / 60
session_metrics['event_density'] = session_metrics['events_in_session'] / (session_metrics['session_length_minutes'] + 1)  # +1 to avoid division by zero

# Identify "deep work" sessions: high duration and event density
deep_work_threshold_duration = session_metrics['session_length_minutes'].quantile(0.75)
deep_work_threshold_density = session_metrics['event_density'].quantile(0.75)

session_metrics['is_deep_work'] = (
    (session_metrics['session_length_minutes'] >= deep_work_threshold_duration) & 
    (session_metrics['event_density'] >= deep_work_threshold_density)
)

print(f"\n✓ Deep work sessions threshold: {deep_work_threshold_duration:.1f}+ minutes & {deep_work_threshold_density:.1f}+ events/min")
print(f"✓ Identified {session_metrics['is_deep_work'].sum():,} deep work sessions ({session_metrics['is_deep_work'].sum() / len(session_metrics) * 100:.1f}%)")

# Calculate inter-session gaps for each user
user_session_gaps = session_metrics.sort_values(['user_id', 'session_start']).copy()
user_session_gaps['next_session_start'] = user_session_gaps.groupby('user_id')['session_start'].shift(-1)
user_session_gaps['inter_session_gap_hours'] = (user_session_gaps['next_session_start'] - user_session_gaps['session_end']).dt.total_seconds() / 3600

# Aggregate session patterns per user
session_patterns_per_user = session_metrics.groupby('user_id').agg({
    'session_id': 'count',
    'session_length_minutes': ['mean', 'median', 'std', 'max'],
    'events_in_session': ['mean', 'median', 'sum'],
    'event_density': ['mean', 'median'],
    'is_deep_work': 'sum'
}).reset_index()

session_patterns_per_user.columns = ['user_id', 'total_sessions', 
                                       'avg_session_length', 'median_session_length', 'std_session_length', 'max_session_length',
                                       'avg_events_per_session', 'median_events_per_session', 'total_events',
                                       'avg_event_density', 'median_event_density',
                                       'deep_work_sessions']

# Add inter-session gap metrics
inter_session_stats = user_session_gaps.groupby('user_id')['inter_session_gap_hours'].agg(['mean', 'median']).reset_index()
inter_session_stats.columns = ['user_id', 'avg_inter_session_gap_hours', 'median_inter_session_gap_hours']

session_patterns_per_user = session_patterns_per_user.merge(inter_session_stats, on='user_id', how='left')

# Fill NaN for users with only one session
session_patterns_per_user['avg_inter_session_gap_hours'].fillna(0, inplace=True)
session_patterns_per_user['median_inter_session_gap_hours'].fillna(0, inplace=True)

# Calculate deep work ratio
session_patterns_per_user['deep_work_ratio'] = session_patterns_per_user['deep_work_sessions'] / session_patterns_per_user['total_sessions']

print(f"\n✓ Created session patterns for {len(session_patterns_per_user):,} users with {len(session_patterns_per_user.columns)} features")
print("\nSample session pattern features:")
print(session_patterns_per_user.head(3).to_string(index=False))

print("\n" + "=" * 80)
print("SESSION PATTERN SUMMARY STATISTICS")
print("=" * 80)
print(session_patterns_per_user.describe().round(2).to_string())

print("\n✓ Session pattern analysis complete!")