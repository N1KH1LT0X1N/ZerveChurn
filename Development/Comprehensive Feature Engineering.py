import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

print("=" * 80)
print("COMPREHENSIVE FEATURE ENGINEERING FOR ML")
print("=" * 80)

# Start with user_retention DataFrame (use local _ prefix to avoid conflicts)
_df_eng = user_retention.copy()

# Get event categories from taxonomy
_event_cats = event_categories  # dict: event → category string

# ============================================================================
# CATEGORY 1: ENGAGEMENT FEATURES
# ============================================================================
print("\n📊 CATEGORY 1: ENGAGEMENT FEATURES")

engagement_features = _df_eng.groupby('distinct_id').agg(
    total_events_eng=('event', 'count'),
    unique_event_types_eng=('event', 'nunique'),
    active_days_eng=('timestamp', lambda x: pd.to_datetime(x).dt.date.nunique()),
    first_event_eng=('timestamp', 'min'),
    last_event_eng=('timestamp', 'max')
).reset_index()

# Engagement consistency (coefficient of variation per user)
_daily_events = _df_eng.groupby(
    ['distinct_id', pd.to_datetime(_df_eng['timestamp']).dt.date]
).size().reset_index(name='n')
_daily_stats = _daily_events.groupby('distinct_id')['n'].agg(
    std_events='std', mean_events='mean'
).fillna(0).reset_index()
_daily_stats['engagement_consistency_eng'] = _daily_stats['std_events'] / (_daily_stats['mean_events'] + 1)
engagement_features = engagement_features.merge(
    _daily_stats[['distinct_id', 'engagement_consistency_eng']], on='distinct_id', how='left')

# Engagement trend: ratio of second-half vs first-half events
_df_eng_ts = _df_eng[['distinct_id', 'timestamp']].copy()
_df_eng_ts['ts'] = pd.to_datetime(_df_eng_ts['timestamp'])
_user_min = _df_eng_ts.groupby('distinct_id')['ts'].min().rename('ts_min')
_user_max = _df_eng_ts.groupby('distinct_id')['ts'].max().rename('ts_max')
_df_eng_ts = _df_eng_ts.merge(_user_min, on='distinct_id').merge(_user_max, on='distinct_id')
_df_eng_ts['midpoint'] = _df_eng_ts['ts_min'] + (_df_eng_ts['ts_max'] - _df_eng_ts['ts_min']) / 2
_first_half  = _df_eng_ts[_df_eng_ts['ts'] <= _df_eng_ts['midpoint']].groupby('distinct_id').size().rename('first_half')
_second_half = _df_eng_ts[_df_eng_ts['ts'] > _df_eng_ts['midpoint']].groupby('distinct_id').size().rename('second_half')
_trend_df = pd.concat([_first_half, _second_half], axis=1).fillna(0).reset_index()
_trend_df.columns = ['distinct_id', 'first_half', 'second_half']
_trend_df['engagement_trend_eng'] = (_trend_df['second_half'] - _trend_df['first_half']) / (_trend_df['first_half'] + 1)
engagement_features = engagement_features.merge(
    _trend_df[['distinct_id', 'engagement_trend_eng']], on='distinct_id', how='left')
engagement_features['engagement_trend_eng'] = engagement_features['engagement_trend_eng'].fillna(0)

print(f"✓ Engineered {len(engagement_features.columns) - 1} engagement features")

# ============================================================================
# CATEGORY 2: WORKFLOW FEATURES
# ============================================================================
print("\n⚙️ CATEGORY 2: WORKFLOW FEATURES")

_creation_events = [e for e in _event_cats if 'create' in e.lower() or 'created' in e.lower()]
_execution_events = [e for e in _event_cats if 'run' in e.lower() or 'execution' in e.lower()]
_error_events     = [e for e in _event_cats if 'error' in e.lower() or 'failed' in e.lower()]

_wf_base = engagement_features[['distinct_id']].copy()
_wf_base['creation_count']  = _df_eng[_df_eng['event'].isin(_creation_events)].groupby('distinct_id').size().reindex(_wf_base['distinct_id']).fillna(0).values
_wf_base['execution_count'] = _df_eng[_df_eng['event'].isin(_execution_events)].groupby('distinct_id').size().reindex(_wf_base['distinct_id']).fillna(0).values
_wf_base['error_count_wf']  = _df_eng[_df_eng['event'].isin(_error_events)].groupby('distinct_id').size().reindex(_wf_base['distinct_id']).fillna(0).values
_wf_base['error_rate']                = _wf_base['error_count_wf'] / (_wf_base['execution_count'] + 1)
_wf_base['workflow_completion_rate']  = _wf_base['execution_count'] / (_wf_base['creation_count'] + 1)
_wf_base['feature_diversity_score']   = engagement_features['unique_event_types_eng'] / (engagement_features['total_events_eng'] + 1)

workflow_features = _wf_base.copy()
print(f"✓ Engineered {len(workflow_features.columns) - 1} workflow features")

# ============================================================================
# CATEGORY 3: TEMPORAL FEATURES (numeric only, drop datetime cols before fillna)
# ============================================================================
print("\n⏰ CATEGORY 3: TEMPORAL FEATURES")

# Build a purely numeric temporal feature table
_temp_num = engagement_features[['distinct_id']].copy()

# Time to first deployment (days)
_deploy_events_list = [e for e in _event_cats if 'deploy' in e.lower()]
if _deploy_events_list:
    _deploy_df_tmp = _df_eng[_df_eng['event'].isin(_deploy_events_list)]
    if len(_deploy_df_tmp) > 0:
        _first_dep = _deploy_df_tmp.groupby('distinct_id')['timestamp'].min()
        _first_dep_days = (
            pd.to_datetime(_first_dep) -
            pd.to_datetime(engagement_features.set_index('distinct_id')['first_event_eng'])
        ).dt.days.rename('time_to_first_deployment')
        _temp_num = _temp_num.merge(_first_dep_days, on='distinct_id', how='left')
    else:
        _temp_num['time_to_first_deployment'] = -1
else:
    _temp_num['time_to_first_deployment'] = -1

# Days since last activity
_current_ts = pd.to_datetime(user_retention['timestamp']).max()
_temp_num['days_since_last_activity'] = (
    _current_ts - pd.to_datetime(engagement_features['last_event_eng'])
).dt.days.values

# Weekend ratio from temporal_segments
_ts_seg = temporal_segments[['distinct_id', 'weekend_ratio']].copy()
_temp_num = _temp_num.merge(_ts_seg, on='distinct_id', how='left')

# Evening activity ratio
_df_eng_h = _df_eng[['distinct_id', 'timestamp']].copy()
_df_eng_h['_hour'] = pd.to_datetime(_df_eng_h['timestamp']).dt.hour
_evening = _df_eng_h[(_df_eng_h['_hour'] >= 18) & (_df_eng_h['_hour'] <= 23)].groupby('distinct_id').size()
_total_act = _df_eng_h.groupby('distinct_id').size()
_evening_ratio_s = (_evening / _total_act).fillna(0).rename('evening_ratio')
_temp_num = _temp_num.merge(
    _evening_ratio_s.reset_index().rename(columns={0: 'evening_ratio'}),
    on='distinct_id', how='left')

# Lifecycle stage (string, kept separately)
def _classify_lifecycle(days):
    if days <= 7:    return 'active'
    elif days <= 30: return 'at_risk'
    elif days <= 90: return 'dormant'
    else:            return 'churned'

_temp_num['lifecycle_stage'] = _temp_num['days_since_last_activity'].apply(_classify_lifecycle)

# Only fillna on numeric columns
_num_cols_tmp = _temp_num.select_dtypes(include=[np.number]).columns.tolist()
_temp_num[_num_cols_tmp] = _temp_num[_num_cols_tmp].fillna(0)

temporal_features = _temp_num.copy()
print(f"✓ Engineered {len([c for c in temporal_features.columns if c != 'distinct_id'])} temporal features")

# ============================================================================
# CATEGORY 4: COLLABORATION FEATURES
# ============================================================================
print("\n🤝 CATEGORY 4: COLLABORATION FEATURES")

_collab_events = [e for e, cat in _event_cats.items() if cat == 'Collaboration & Sharing']

collaboration_features = engagement_features[['distinct_id']].copy()
collaboration_features['share_count'] = _df_eng[_df_eng['event'].isin(_collab_events)].groupby('distinct_id').size().reindex(collaboration_features['distinct_id']).fillna(0).values
collaboration_features['unique_collaborators'] = collaboration_features['share_count']
collaboration_features['collaboration_initiation_rate'] = collaboration_features['share_count'] / (engagement_features['total_events_eng'] + 1)
collaboration_features.fillna(0, inplace=True)
print(f"✓ Engineered {len(collaboration_features.columns) - 1} collaboration features")

# ============================================================================
# CATEGORY 5: AI AGENT FEATURES
# ============================================================================
print("\n🤖 CATEGORY 5: AI AGENT FEATURES")

_agent_events_list = [e for e in _event_cats if 'agent' in e.lower()]

ai_features = engagement_features[['distinct_id']].copy()
ai_features['agent_interaction_count'] = _df_eng[_df_eng['event'].isin(_agent_events_list)].groupby('distinct_id').size().reindex(ai_features['distinct_id']).fillna(0).values

if _agent_events_list:
    _first_agent = _df_eng[_df_eng['event'].isin(_agent_events_list)].groupby('distinct_id')['timestamp'].min()
    _agent_adopt_days = (
        pd.to_datetime(_first_agent) -
        pd.to_datetime(engagement_features.set_index('distinct_id')['first_event_eng'])
    ).dt.days.fillna(-1).rename('agent_adoption_day')
    ai_features = ai_features.merge(_agent_adopt_days, on='distinct_id', how='left')
else:
    ai_features['agent_adoption_day'] = -1

ai_features['agent_reliance_score'] = ai_features['agent_interaction_count'] / (engagement_features['total_events_eng'] + 1)
_ai_num = ai_features.select_dtypes(include=[np.number]).columns.tolist()
ai_features[_ai_num] = ai_features[_ai_num].fillna(0)
print(f"✓ Engineered {len(ai_features.columns) - 1} AI agent features")

# ============================================================================
# CATEGORY 6: DERIVED FEATURES
# ============================================================================
print("\n📈 CATEGORY 6: DERIVED FEATURES")

derived_features = engagement_features[['distinct_id']].copy()
derived_features['engagement_intensity']      = engagement_features['total_events_eng'] / (engagement_features['active_days_eng'] + 1)
derived_features['feature_utilization_ratio'] = engagement_features['unique_event_types_eng'] / max(len(_event_cats), 1)
derived_features['productivity_score']         = (
    workflow_features['workflow_completion_rate'] +
    workflow_features['execution_count'] / 100 -
    workflow_features['error_rate']
)
derived_features['exploration_score'] = (
    workflow_features['feature_diversity_score'] * engagement_features['unique_event_types_eng']
)
print(f"✓ Engineered {len(derived_features.columns) - 1} derived features")

# ============================================================================
# CATEGORY 7: INTERACTION FEATURES
# ============================================================================
print("\n🔄 CATEGORY 7: INTERACTION FEATURES")

interaction_features = engagement_features[['distinct_id']].copy()
interaction_features['engagement_x_diversity']     = engagement_features['total_events_eng'] * workflow_features['feature_diversity_score']
interaction_features['deployment_x_collaboration'] = workflow_features['execution_count'] / 100 * collaboration_features['collaboration_initiation_rate']
interaction_features['agent_x_completion']         = ai_features['agent_reliance_score'] * workflow_features['workflow_completion_rate']
print(f"✓ Engineered {len(interaction_features.columns) - 1} interaction features")

# ============================================================================
# MERGE ALL FEATURES (numeric only)
# ============================================================================
print("\n" + "=" * 80)
print("MERGING ALL FEATURE CATEGORIES")
print("=" * 80)

feature_matrix = engagement_features[['distinct_id']].copy()
for _fdf in [engagement_features, workflow_features, temporal_features,
              collaboration_features, ai_features, derived_features, interaction_features]:
    _num_cols = _fdf.select_dtypes(include=[np.number]).columns.tolist()
    _to_merge = _fdf[['distinct_id'] + _num_cols]
    feature_matrix = feature_matrix.merge(_to_merge, on='distinct_id', how='left', suffixes=('', '_dup'))

feature_matrix = feature_matrix.loc[:, ~feature_matrix.columns.str.endswith('_dup')]
feature_matrix = feature_matrix.loc[:, ~feature_matrix.columns.duplicated()]
feature_matrix.fillna(0, inplace=True)
feature_matrix.replace([np.inf, -np.inf], [1e6, -1e6], inplace=True)

print(f"\n✓ Feature matrix created: {len(feature_matrix)} users × {len(feature_matrix.columns)-1} features")

# ============================================================================
# FEATURE SCALING
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE SCALING")
print("=" * 80)

_user_ids_fe = feature_matrix['distinct_id']
_feat_cols   = [c for c in feature_matrix.columns if c != 'distinct_id']

scaler_fe = StandardScaler()
_scaled   = scaler_fe.fit_transform(feature_matrix[_feat_cols])
scaled_df_fe = pd.DataFrame(_scaled, columns=_feat_cols)
scaled_df_fe.insert(0, 'distinct_id', _user_ids_fe.values)

if 'lifecycle_stage' in temporal_features.columns:
    _lc = temporal_features.set_index('distinct_id')['lifecycle_stage']
    scaled_df_fe['lifecycle_stage'] = _user_ids_fe.map(_lc).values

print(f"✓ Scaled {len(_feat_cols)} numerical features using StandardScaler")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ COMPREHENSIVE FEATURE ENGINEERING COMPLETE")
print("=" * 80)
print(f"\n📈 Matrix Dimensions:")
print(f"   • Users:          {len(feature_matrix):,}")
print(f"   • Raw Features:   {len(feature_matrix.columns) - 1}")
print(f"   • Scaled Features:{len(scaled_df_fe.columns) - 2}")

print("\n📋 Sample (first 3 rows, first 8 cols):")
print(feature_matrix[feature_matrix.columns[:8].tolist()].head(3).to_string(index=False))
print("\n✅ Ready for ML modeling!")
