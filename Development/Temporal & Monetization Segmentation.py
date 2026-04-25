import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Temporal segmentation - adoption timing, activity patterns, consistency

# Get temporal info for each user
user_temporal = user_retention.groupby('distinct_id').agg({
    'timestamp': ['min', 'max', 'count']
}).reset_index()
user_temporal.columns = ['distinct_id', 'first_event', 'last_event', 'total_events_temp']

# Calculate time-based features
_ref_ts = pd.to_datetime(user_retention['timestamp']).max()
user_temporal['days_since_first'] = (_ref_ts - pd.to_datetime(user_temporal['first_event'])).dt.days
user_temporal['days_active_span'] = (pd.to_datetime(user_temporal['last_event']) - pd.to_datetime(user_temporal['first_event'])).dt.days + 1
user_temporal['first_event_month'] = pd.to_datetime(user_temporal['first_event']).dt.to_period('M')

# Adoption timing (early vs late adopters)
first_quartile = user_temporal['first_event'].quantile(0.25)
third_quartile = user_temporal['first_event'].quantile(0.75)

def classify_adoption(timestamp):
    if timestamp < first_quartile:
        return 'Early Adopter'
    elif timestamp > third_quartile:
        return 'Late Adopter'
    else:
        return 'Mid Adopter'

user_temporal['adoption_timing'] = user_temporal['first_event'].apply(classify_adoption)

# Activity patterns - weekend vs weekday
user_retention['timestamp_dt'] = pd.to_datetime(user_retention['timestamp'])
user_retention['dayofweek'] = user_retention['timestamp_dt'].dt.dayofweek
user_retention['is_weekend'] = user_retention['dayofweek'].isin([5, 6])

weekend_activity = user_retention.groupby(['distinct_id', 'is_weekend']).size().unstack(fill_value=0)
weekend_activity.columns = ['weekday_events', 'weekend_events']
weekend_activity['weekend_ratio'] = weekend_activity['weekend_events'] / (weekend_activity['weekday_events'] + weekend_activity['weekend_events'])

def classify_activity_pattern(ratio):
    if ratio > 0.5:
        return 'Weekend User'
    elif ratio < 0.2:
        return 'Weekday User'
    else:
        return 'Balanced User'

weekend_activity['activity_pattern'] = weekend_activity['weekend_ratio'].apply(classify_activity_pattern)

# Consistency - bursty vs consistent
user_temporal_consistency = user_retention.groupby(['distinct_id', pd.to_datetime(user_retention['timestamp']).dt.date]).size().reset_index(name='daily_events')
consistency_metrics = user_temporal_consistency.groupby('distinct_id')['daily_events'].agg(['std', 'mean']).reset_index()
consistency_metrics['consistency_score'] = consistency_metrics['std'] / (consistency_metrics['mean'] + 1)  # Lower = more consistent

def classify_consistency(score):
    if score < 2:
        return 'Consistent'
    elif score < 5:
        return 'Moderate'
    else:
        return 'Bursty'

consistency_metrics['consistency_label'] = consistency_metrics['consistency_score'].apply(classify_consistency)

# Monetization segmentation
# Check if user has credit-related events
# Private variable (_credit_events) to avoid naming conflict with Kaplan-Meier Survival block downstream
_credit_events = ['credits_used', 'credits_below_1', 'credits_below_2', 'credits_below_5']
monetization_data = user_retention[user_retention['event'].isin(_credit_events)]
users_with_credit_activity = monetization_data['distinct_id'].unique()

# Create monetization segments
user_temporal['has_credit_activity'] = user_temporal['distinct_id'].isin(users_with_credit_activity)
user_temporal['monetization_segment'] = user_temporal['has_credit_activity'].map({True: 'Credit User', False: 'Free User'})

# Combine all temporal and monetization features
temporal_segments = user_temporal.merge(
    weekend_activity[['activity_pattern', 'weekend_ratio']],
    left_on='distinct_id',
    right_index=True,
    how='left'
).merge(
    consistency_metrics[['distinct_id', 'consistency_label', 'consistency_score']],
    on='distinct_id',
    how='left'
)

# Fill missing values for users without enough data
temporal_segments['activity_pattern'] = temporal_segments['activity_pattern'].fillna('Balanced User')
temporal_segments['consistency_label'] = temporal_segments['consistency_label'].fillna('Consistent')

print(f"Temporal & Monetization segmentation completed for {len(temporal_segments)} users\n")

print("Adoption Timing Distribution:")
print(temporal_segments['adoption_timing'].value_counts())

print("\nActivity Pattern Distribution:")
print(temporal_segments['activity_pattern'].value_counts())

print("\nConsistency Distribution:")
print(temporal_segments['consistency_label'].value_counts())

print("\nMonetization Segment Distribution:")
print(temporal_segments['monetization_segment'].value_counts())

print("\nTemporal Metrics Summary:")
print(temporal_segments[['days_since_first', 'days_active_span', 'weekend_ratio', 'consistency_score']].describe())
