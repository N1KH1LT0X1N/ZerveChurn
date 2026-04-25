import pandas as pd
import numpy as np

print("=" * 80)
print("FEATURE ADOPTION TRAJECTORY ANALYSIS")
print("=" * 80)

# Define advanced features to track
advanced_features = {
    'agent_usage': ['agent_accept_suggestion', 'agent_worker_created', 'agent_worker_finished', 
                    'agent_chat_message_sent', 'agent_tool_call_create_block_tool', 
                    'agent_tool_call_run_block_tool', 'agent_tool_call_refactor_block_tool'],
    'collaboration': ['canvas_shared_with_user', 'comment_added', 'share_link_created'],
    'deployment': ['api_deployed', 'endpoint_created', 'model_deployed'],
    'data_connections': ['connection_created', 'connection_tested', 'query_executed'],
    'advanced_blocks': ['gen_ai_block_created', 'query_block_created', 'r_block_created'],
    'visualization': ['chart_created', 'plot_generated', 'visualization_customized'],
    'file_management': ['file_uploaded', 'file_downloaded', 'asset_imported'],
    'compute_config': ['compute_config_changed', 'memory_increased', 'gpu_requested'],
    'credits_usage': ['credits_used', 'addon_credits_used', 'credits_exceeded']
}

print(f"\n✓ Tracking {len(advanced_features)} advanced feature categories")

# Get first usage timestamp for each feature per user
df_adoption = user_retention.copy()

feature_adoption_data = []

for user_id in df_adoption['distinct_id'].unique():
    user_events = df_adoption[df_adoption['distinct_id'] == user_id].sort_values('timestamp')
    user_first_event = user_events['timestamp'].min()
    
    adoption_record = {'user_id': user_id, 'first_event_timestamp': user_first_event}
    
    # Track first usage of each feature category
    for feature_category, event_list in advanced_features.items():
        feature_events = user_events[user_events['event'].isin(event_list)]
        
        if len(feature_events) > 0:
            first_use = feature_events['timestamp'].min()
            days_to_adoption = (first_use - user_first_event).total_seconds() / (24 * 3600)
            usage_count = len(feature_events)
            
            adoption_record[f'{feature_category}_first_use'] = first_use
            adoption_record[f'{feature_category}_days_to_adoption'] = days_to_adoption
            adoption_record[f'{feature_category}_usage_count'] = usage_count
            adoption_record[f'{feature_category}_adopted'] = 1
        else:
            adoption_record[f'{feature_category}_first_use'] = None
            adoption_record[f'{feature_category}_days_to_adoption'] = None
            adoption_record[f'{feature_category}_usage_count'] = 0
            adoption_record[f'{feature_category}_adopted'] = 0
    
    feature_adoption_data.append(adoption_record)

feature_adoption_df = pd.DataFrame(feature_adoption_data)

print(f"\n✓ Analyzed feature adoption for {len(feature_adoption_df):,} users")

# Calculate feature adoption velocity (features adopted per day active)
df_adoption_user_lifetime = df_adoption.groupby('distinct_id').agg({
    'timestamp': ['min', 'max']
}).reset_index()
df_adoption_user_lifetime.columns = ['user_id', 'first_timestamp', 'last_timestamp']
df_adoption_user_lifetime['lifetime_days'] = (
    df_adoption_user_lifetime['last_timestamp'] - df_adoption_user_lifetime['first_timestamp']
).dt.total_seconds() / (24 * 3600)

feature_adoption_df = feature_adoption_df.merge(df_adoption_user_lifetime, on='user_id', how='left')

# Count total features adopted per user
adoption_cols = [col for col in feature_adoption_df.columns if col.endswith('_adopted')]
feature_adoption_df['total_features_adopted'] = feature_adoption_df[adoption_cols].sum(axis=1)

# Calculate adoption velocity
feature_adoption_df['adoption_velocity'] = feature_adoption_df['total_features_adopted'] / (feature_adoption_df['lifetime_days'] + 1)

# Identify feature champions (early adopters with persistence)
# Early adopter = adopted within first 7 days
# Persistence = used at least 10 times

champion_features = []
for feature_category in advanced_features.keys():
    days_col = f'{feature_category}_days_to_adoption'
    usage_col = f'{feature_category}_usage_count'
    
    if days_col in feature_adoption_df.columns:
        feature_adoption_df[f'{feature_category}_champion'] = (
            (feature_adoption_df[days_col] <= 7) & 
            (feature_adoption_df[usage_col] >= 10)
        ).astype(int)
        champion_features.append(f'{feature_category}_champion')

# Calculate total champion score
feature_adoption_df['champion_score'] = feature_adoption_df[champion_features].sum(axis=1)

print("\n" + "=" * 80)
print("FEATURE ADOPTION SUMMARY")
print("=" * 80)

for feature_category in advanced_features.keys():
    adopted_col = f'{feature_category}_adopted'
    if adopted_col in feature_adoption_df.columns:
        adoption_rate = feature_adoption_df[adopted_col].sum() / len(feature_adoption_df) * 100
        
        days_col = f'{feature_category}_days_to_adoption'
        if days_col in feature_adoption_df.columns:
            avg_days = feature_adoption_df[feature_adoption_df[adopted_col] == 1][days_col].mean()
        else:
            avg_days = None
        
        print(f"\n{feature_category.upper()}")
        print(f"  - Adoption rate: {adoption_rate:.1f}%")
        if avg_days is not None:
            print(f"  - Avg days to adoption: {avg_days:.1f}")

print("\n" + "=" * 80)
print("FEATURE ADOPTION TRAJECTORY STATISTICS")
print("=" * 80)
stats_cols = ['total_features_adopted', 'adoption_velocity', 'champion_score', 'lifetime_days']
print(feature_adoption_df[stats_cols].describe().round(2).to_string())

print("\n✓ Feature adoption trajectory analysis complete!")