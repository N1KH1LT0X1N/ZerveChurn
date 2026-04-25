import pandas as pd
import numpy as np

print("=" * 80)
print("PRIMARY SUCCESS METRICS CALCULATION")
print("=" * 80)

# Prepare base user data from user_retention
user_base = user_retention.groupby('distinct_id').agg({
    'timestamp': ['min', 'max', 'count'],
    'event': lambda x: x.nunique()
}).reset_index()

user_base.columns = ['user_id', 'first_activity', 'last_activity', 'total_events', 'unique_event_types']
user_base['first_activity'] = pd.to_datetime(user_base['first_activity'])
user_base['last_activity'] = pd.to_datetime(user_base['last_activity'])

# Calculate current date (use max date in dataset as "current")
current_date = user_base['last_activity'].max()
analysis_start_date = user_base['first_activity'].min()

print(f"\n📊 Analysis Period: {analysis_start_date.date()} to {current_date.date()}")
print(f"📊 Total Users: {len(user_base):,}")

# Calculate days since first and last activity
user_base['days_since_first'] = (current_date - user_base['first_activity']).dt.total_seconds() / (24 * 3600)
user_base['days_since_last'] = (current_date - user_base['last_activity']).dt.total_seconds() / (24 * 3600)
user_base['tenure_days'] = (user_base['last_activity'] - user_base['first_activity']).dt.total_seconds() / (24 * 3600)

print("\n" + "=" * 80)
print("METRIC 1: LONG-TERM RETENTION")
print("=" * 80)
print("Definition: Active in last 30 days AND was active 90+ days ago")

# Active in last 30 days
user_base['active_last_30d'] = user_base['days_since_last'] <= 30

# Was active 90+ days ago (user must have been around for at least 90 days)
user_base['eligible_for_retention'] = user_base['days_since_first'] >= 90

# Long-term retention: active recently AND has been around for 90+ days
user_base['long_term_retention'] = user_base['active_last_30d'] & user_base['eligible_for_retention']

print(f"\n✓ Eligible users (90+ days tenure): {user_base['eligible_for_retention'].sum():,} ({user_base['eligible_for_retention'].sum() / len(user_base) * 100:.1f}%)")
print(f"✓ Retained users (active last 30d + 90d tenure): {user_base['long_term_retention'].sum():,} ({user_base['long_term_retention'].sum() / user_base['eligible_for_retention'].sum() * 100:.1f}% of eligible)")

print("\n" + "=" * 80)
print("METRIC 2: UPGRADE CONVERSION (FREE → PAID)")
print("=" * 80)
print("Definition: Evidence of paid features usage")

# Check for credit/monetization events
credit_events = ['credit_balance_updated', 'credits_purchased', 'credit_usage_tracked']
monetization_events = user_retention[user_retention['event'].isin(credit_events)]

paid_users = monetization_events['distinct_id'].unique()
user_base['is_paid_user'] = user_base['user_id'].isin(paid_users)

print(f"\n✓ Paid users: {user_base['is_paid_user'].sum():,} ({user_base['is_paid_user'].sum() / len(user_base) * 100:.1f}%)")

print("\n" + "=" * 80)
print("METRIC 3: DEPLOYMENT SUCCESS")
print("=" * 80)
print("Definition: At least 1 project deployed")

# Check for deployment events
deployment_events = [
    'api_deployed', 'endpoint_deployed', 'model_deployed',
    'sagemaker_endpoint_deployed', 'api_route_created',
    'deployment_successful', 'canvas_published'
]

deployment_activity = user_retention[user_retention['event'].isin(deployment_events)]
deployed_users = deployment_activity['distinct_id'].unique()
user_base['has_deployment'] = user_base['user_id'].isin(deployed_users)

print(f"\n✓ Users with deployments: {user_base['has_deployment'].sum():,} ({user_base['has_deployment'].sum() / len(user_base) * 100:.1f}%)")

print("\n" + "=" * 80)
print("METRIC 4: COLLABORATION SUCCESS")
print("=" * 80)
print("Definition: 3+ shares/collaboration events")

# Collaboration events
collaboration_events = [
    'canvas_shared', 'canvas_shared_with_user', 'share_link_created',
    'comment_added', 'comment_replied', 'user_invited',
    'workspace_invite_sent', 'mention_notification'
]

collab_activity = user_retention[user_retention['event'].isin(collaboration_events)]
user_collab_counts = collab_activity.groupby('distinct_id').size().reset_index(name='collab_count')

user_base = user_base.merge(user_collab_counts, left_on='user_id', right_on='distinct_id', how='left')
user_base['collab_count'] = user_base['collab_count'].fillna(0)
user_base['collaboration_success'] = user_base['collab_count'] >= 3

print(f"\n✓ Users with 3+ collaborations: {user_base['collaboration_success'].sum():,} ({user_base['collaboration_success'].sum() / len(user_base) * 100:.1f}%)")

print("\n" + "=" * 80)
print("METRIC 5: POWER USER EMERGENCE")
print("=" * 80)
print("Definition: Top 25% engagement after 60+ days")

# Filter users with 60+ days tenure
power_user_eligible = user_base[user_base['days_since_first'] >= 60].copy()

if len(power_user_eligible) > 0:
    # Calculate engagement score: events per day of tenure
    power_user_eligible['engagement_rate'] = power_user_eligible['total_events'] / power_user_eligible['tenure_days'].clip(lower=1)
    
    # Top 25% by engagement rate
    engagement_threshold = power_user_eligible['engagement_rate'].quantile(0.75)
    power_user_eligible['is_power_user'] = power_user_eligible['engagement_rate'] >= engagement_threshold
    
    # Merge back
    user_base = user_base.merge(
        power_user_eligible[['user_id', 'is_power_user']], 
        on='user_id', 
        how='left'
    )
    user_base['is_power_user'] = user_base['is_power_user'].fillna(False)
    
    print(f"\n✓ Eligible users (60+ days): {len(power_user_eligible):,}")
    print(f"✓ Power users (top 25% engagement): {user_base['is_power_user'].sum():,} ({user_base['is_power_user'].sum() / len(power_user_eligible) * 100:.1f}% of eligible)")
    print(f"  Engagement threshold: {engagement_threshold:.2f} events/day")
else:
    user_base['is_power_user'] = False
    print("\n⚠ No users with 60+ days tenure yet")

print("\n" + "=" * 80)
print("PRIMARY METRICS SUMMARY")
print("=" * 80)

metrics_summary = {
    'Metric': [
        'Long-term Retention',
        'Upgrade Conversion',
        'Deployment Success',
        'Collaboration Success',
        'Power User Emergence'
    ],
    'Count': [
        user_base['long_term_retention'].sum(),
        user_base['is_paid_user'].sum(),
        user_base['has_deployment'].sum(),
        user_base['collaboration_success'].sum(),
        user_base['is_power_user'].sum()
    ],
    'Percentage': [
        user_base['long_term_retention'].sum() / len(user_base) * 100,
        user_base['is_paid_user'].sum() / len(user_base) * 100,
        user_base['has_deployment'].sum() / len(user_base) * 100,
        user_base['collaboration_success'].sum() / len(user_base) * 100,
        user_base['is_power_user'].sum() / len(user_base) * 100
    ]
}

primary_metrics_summary = pd.DataFrame(metrics_summary)
print(f"\n{primary_metrics_summary.to_string(index=False)}")

# Store for downstream use
user_success_metrics = user_base[[
    'user_id', 'long_term_retention', 'is_paid_user', 'has_deployment',
    'collaboration_success', 'is_power_user', 'total_events', 'tenure_days',
    'days_since_first', 'days_since_last'
]].copy()

print(f"\n✅ Primary metrics calculated for {len(user_success_metrics):,} users")