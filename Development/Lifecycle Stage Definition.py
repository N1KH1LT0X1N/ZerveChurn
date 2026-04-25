import pandas as pd
import numpy as np

# Load user retention data - optimize for performance
df_lifecycle = user_retention[['uuid', 'timestamp', 'event']].copy()

# Ensure timestamp is datetime
df_lifecycle['timestamp'] = pd.to_datetime(df_lifecycle['timestamp'])

# Calculate user tenure (days since first event) - vectorized approach
user_first_event = df_lifecycle.groupby('uuid')['timestamp'].min()

df_lifecycle['days_since_first_event'] = df_lifecycle.apply(
    lambda row: (row['timestamp'] - user_first_event[row['uuid']]).total_seconds() / (24 * 3600), 
    axis=1
)

# Define lifecycle stages - vectorized
def assign_lifecycle_stage(days):
    if days <= 7:
        return 'Onboarding'
    elif days <= 30:
        return 'Exploration'
    elif days <= 90:
        return 'Adoption'
    else:
        return 'Maturity'

df_lifecycle['lifecycle_stage'] = df_lifecycle['days_since_first_event'].apply(assign_lifecycle_stage)

# Calculate stage-specific metrics per user using groupby for performance
stage_metrics_list = []

for stage in ['Onboarding', 'Exploration', 'Adoption', 'Maturity']:
    stage_df = df_lifecycle[df_lifecycle['lifecycle_stage'] == stage].copy()
    
    if len(stage_df) > 0:
        stage_agg = stage_df.groupby('uuid').agg({
            'event': ['count', 'nunique'],
            'days_since_first_event': ['min', 'max'],
            'timestamp': ['min', 'max']
        }).reset_index()
        
        stage_agg.columns = ['uuid', 'event_count', 'unique_events', 'days_min', 'days_max', 'ts_min', 'ts_max']
        stage_agg['lifecycle_stage'] = stage
        stage_agg['days_in_stage'] = stage_agg['days_max'] - stage_agg['days_min']
        stage_agg['events_per_day'] = stage_agg['event_count'] / stage_agg['days_in_stage'].clip(lower=1)
        
        stage_metrics_list.append(stage_agg)

lifecycle_stage_metrics = pd.concat(stage_metrics_list, ignore_index=True)

# Calculate max tenure per user
user_max_tenure = df_lifecycle.groupby('uuid')['days_since_first_event'].max()

# Identify progression patterns
user_stages = lifecycle_stage_metrics.groupby('uuid')['lifecycle_stage'].apply(list).reset_index()
user_stages.columns = ['uuid', 'stages_reached']

user_events_summary = lifecycle_stage_metrics.groupby('uuid').agg({
    'event_count': 'sum',
    'unique_events': 'sum'
}).reset_index()

user_lifecycle_progression = user_stages.merge(user_events_summary, on='uuid')
user_lifecycle_progression['max_tenure_days'] = user_lifecycle_progression['uuid'].map(user_max_tenure)

# Classify progression success
def classify_progression(row):
    stages = row['stages_reached']
    tenure = row['max_tenure_days']
    
    if tenure > 90 and 'Maturity' in stages:
        return 'Successful_LongTerm'
    elif tenure > 30 and 'Adoption' in stages:
        return 'Successful_Adopter'
    elif tenure > 7 and 'Exploration' in stages:
        return 'Exploring'
    elif tenure <= 7:
        return 'Early_Onboarding'
    else:
        return 'Stalled'

user_lifecycle_progression['progression_status'] = user_lifecycle_progression.apply(classify_progression, axis=1)

# Calculate stage transition success rates
user_lifecycle_progression['reached_exploration'] = user_lifecycle_progression['stages_reached'].apply(
    lambda x: any(s in x for s in ['Exploration', 'Adoption', 'Maturity'])
)
user_lifecycle_progression['reached_adoption'] = user_lifecycle_progression['stages_reached'].apply(
    lambda x: any(s in x for s in ['Adoption', 'Maturity'])
)
user_lifecycle_progression['reached_maturity'] = user_lifecycle_progression['stages_reached'].apply(
    lambda x: 'Maturity' in x
)

print("=" * 70)
print("LIFECYCLE STAGE ANALYSIS")
print("=" * 70)
print(f"\nTotal Users Analyzed: {len(user_lifecycle_progression):,}")
print(f"Total Events: {len(df_lifecycle):,}")
print()

print("Stage Transition Success Rates:")
print(f"  - Onboarding → Exploration: {user_lifecycle_progression['reached_exploration'].sum() / len(user_lifecycle_progression) * 100:.1f}%")
print(f"  - Exploration → Adoption: {user_lifecycle_progression['reached_adoption'].sum() / len(user_lifecycle_progression) * 100:.1f}%")
print(f"  - Adoption → Maturity: {user_lifecycle_progression['reached_maturity'].sum() / len(user_lifecycle_progression) * 100:.1f}%")
print()

print("Progression Status Distribution:")
print(user_lifecycle_progression['progression_status'].value_counts().to_string())
print()

print("Average Metrics by Lifecycle Stage:")
stage_summary = lifecycle_stage_metrics.groupby('lifecycle_stage').agg({
    'event_count': 'mean',
    'unique_events': 'mean',
    'days_in_stage': 'mean',
    'events_per_day': 'mean'
}).round(2)
print(stage_summary.to_string())
