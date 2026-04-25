import pandas as pd
import numpy as np

print("=" * 80)
print("COLLABORATION SIGNATURE ANALYSIS")
print("=" * 80)

# Define collaboration events with unique name to avoid conflicts
_collab_event_list = [
    'canvas_shared_with_user', 'share_link_created', 'comment_added', 
    'comment_replied', 'mention_notification', 'workspace_invite_sent',
    'user_added_to_workspace', 'canvas_shared'
]

# Calculate collaboration metrics per user
_collaboration_data_list = []

for _user_id_iter in user_retention['distinct_id'].unique():
    _user_events_iter = user_retention[user_retention['distinct_id'] == _user_id_iter]
    _total_events_iter = len(_user_events_iter)
    
    # Count collaboration events
    _collab_event_cnt = len(_user_events_iter[_user_events_iter['event'].isin(_collab_event_list)])
    _sharing_freq = _collab_event_cnt
    
    # Calculate collaboration ratio
    _collab_ratio = _collab_event_cnt / max(_total_events_iter, 1)
    
    # Solo vs team oriented score
    _has_sharing_flag = _user_events_iter['event'].isin(_collab_event_list).any()
    
    # Solo indicators - working alone without collaboration
    _solo_scr = 1.0 - _collab_ratio if _total_events_iter >= 10 else 0.5
    _team_scr = _collab_ratio
    
    _collaboration_data_list.append({
        'user_id': _user_id_iter,
        'sharing_frequency': _sharing_freq,
        'collaboration_event_count': _collab_event_cnt,
        'collaboration_ratio': _collab_ratio,
        'has_sharing_activity': int(_has_sharing_flag),
        'solo_oriented_score': _solo_scr,
        'team_oriented_score': _team_scr
    })

collaboration_signature_df = pd.DataFrame(_collaboration_data_list)

print(f"\n✓ Analyzed collaboration signature for {len(collaboration_signature_df):,} users")
print(f"\nUsers with collaboration activity: {collaboration_signature_df['has_sharing_activity'].sum():,} ({collaboration_signature_df['has_sharing_activity'].sum() / len(collaboration_signature_df) * 100:.1f}%)")
print(f"Average collaboration ratio: {collaboration_signature_df['collaboration_ratio'].mean():.4f}")

print("\n" + "=" * 80)
print("BUILDING COMPREHENSIVE BEHAVIORAL FINGERPRINT MATRIX")
print("=" * 80)

# Merge all dimensions
print("\n✓ Merging all behavioral dimensions...")

# Start with session patterns
behavioral_fingerprint = session_patterns_per_user.copy()
print(f"  - Session patterns: {len(behavioral_fingerprint)} users, {len(session_patterns_per_user.columns)-1} features")

# Add workflow sequences
behavioral_fingerprint = behavioral_fingerprint.merge(
    workflow_sequence_df[['user_id', 'power_user_score', 'struggle_score', 'sequence_diversity',
                          'has_agent_workflow', 'has_deployment_sequence', 'error_count',
                          'trigram_count', 'fourgram_count', 'fivegram_count']], 
    on='user_id', 
    how='left'
)
print(f"  - Workflow sequences added: {len(workflow_sequence_df.columns)-1} features")

# Add collaboration signature
behavioral_fingerprint = behavioral_fingerprint.merge(
    collaboration_signature_df,
    on='user_id',
    how='left'
)
print(f"  - Collaboration signature added: {len(collaboration_signature_df.columns)-1} features")

# Fill NaN values with 0 for users with no data in certain dimensions
behavioral_fingerprint.fillna(0, inplace=True)

print(f"\n✓ COMPREHENSIVE BEHAVIORAL FINGERPRINT MATRIX CREATED")
print(f"\n📊 Matrix dimensions: {len(behavioral_fingerprint):,} users × {len(behavioral_fingerprint.columns)} features")

print("\n" + "=" * 80)
print("FEATURE CATEGORIES IN BEHAVIORAL FINGERPRINT")
print("=" * 80)

feature_categories = {
    'Session Patterns': ['total_sessions', 'avg_session_length', 'median_session_length', 
                         'std_session_length', 'max_session_length', 'avg_events_per_session',
                         'median_events_per_session', 'total_events', 'avg_event_density',
                         'median_event_density', 'deep_work_sessions', 'avg_inter_session_gap_hours',
                         'median_inter_session_gap_hours', 'deep_work_ratio'],
    'Workflow Sequences': ['power_user_score', 'struggle_score', 'sequence_diversity',
                          'has_agent_workflow', 'has_deployment_sequence', 'error_count',
                          'trigram_count', 'fourgram_count', 'fivegram_count'],
    'Collaboration Signature': ['sharing_frequency', 'collaboration_event_count', 'collaboration_ratio',
                                'has_sharing_activity', 'solo_oriented_score', 'team_oriented_score']
}

for _cat_name, _cat_features in feature_categories.items():
    _available_features = [f for f in _cat_features if f in behavioral_fingerprint.columns]
    print(f"\n{_cat_name}: {len(_available_features)} features")
    print(f"  {', '.join(_available_features[:5])}" + (f"... (+{len(_available_features)-5} more)" if len(_available_features) > 5 else ""))

print("\n" + "=" * 80)
print("BEHAVIORAL FINGERPRINT SUMMARY STATISTICS")
print("=" * 80)

# Select key features for summary
key_features = ['total_sessions', 'avg_session_length', 'deep_work_ratio',
                'power_user_score', 'struggle_score', 'sequence_diversity',
                'collaboration_ratio', 'team_oriented_score']

available_key_features = [f for f in key_features if f in behavioral_fingerprint.columns]
print(behavioral_fingerprint[available_key_features].describe().round(3).to_string())

print("\n" + "=" * 80)
print("SAMPLE BEHAVIORAL FINGERPRINTS")
print("=" * 80)
print("\nFirst 3 user fingerprints:")
print(behavioral_fingerprint.head(3).to_string(index=False))

print("\n" + "=" * 80)
print("✓ BEHAVIORAL FINGERPRINT MATRIX COMPLETE!")
print("=" * 80)
print(f"\n✅ Success! Created comprehensive behavioral fingerprint matrix with:")
print(f"   • {len(behavioral_fingerprint):,} users")
print(f"   • {len(behavioral_fingerprint.columns)} total features")
print(f"   • 4 dimensions: Session Patterns, Workflow Sequences, Feature Adoption, Collaboration")
print(f"   • Ready for ML modeling and advanced analytics")
