import pandas as pd
import numpy as np
from scipy.stats import entropy

# Map events to workflow stages using the event_to_stage mapping
event_to_stage_map = workflow_mapping['event_to_stage']

# Add workflow stage column to user data
user_event_matrix_workflow = user_retention[['distinct_id', 'event']].copy()
user_event_matrix_workflow['workflow_stage'] = user_event_matrix_workflow['event'].map(event_to_stage_map)

# Filter out unmapped events
user_event_matrix_workflow = user_event_matrix_workflow[user_event_matrix_workflow['workflow_stage'].notna()]

# Calculate workflow pattern metrics per user
workflow_metrics = user_event_matrix_workflow.groupby(['distinct_id', 'workflow_stage']).size().unstack(fill_value=0)

# Calculate primary workflow patterns for each user
workflow_patterns = {}
for _user_id in workflow_metrics.index:       # private loop var
    user_workflow = workflow_metrics.loc[_user_id]
    
    # Identify primary workflow pattern based on event distribution
    total_events_wf = user_workflow.sum()
    if total_events_wf == 0:
        workflow_patterns[_user_id] = 'Inactive'
        continue
    
    workflow_pcts = user_workflow / total_events_wf
    
    # Categorize based on dominant activity using the actual stage names from mapping
    if workflow_pcts.get('Editing & Iteration', 0) > 0.4:
        workflow_patterns[_user_id] = 'Canvas-Focused'
    elif workflow_pcts.get('Exploration', 0) > 0.5:
        workflow_patterns[_user_id] = 'Notebook-Heavy'
    elif workflow_pcts.get('Deployment', 0) > 0.3:
        workflow_patterns[_user_id] = 'Deployment-Oriented'
    elif workflow_pcts.get('Collaboration', 0) > 0.2:
        workflow_patterns[_user_id] = 'Collaboration-Driven'
    elif workflow_pcts.get('AI Assistance', 0) > 0.3:
        workflow_patterns[_user_id] = 'AI-Powered'
    else:
        workflow_patterns[_user_id] = 'Mixed Workflow'

# Calculate workflow diversity using entropy
workflow_diversity = {}
for _user_id in workflow_metrics.index:       # private loop var
    user_workflow = workflow_metrics.loc[_user_id]
    if user_workflow.sum() > 0:
        # Normalize to probabilities
        probs = user_workflow[user_workflow > 0] / user_workflow.sum()
        # Calculate entropy (higher = more diverse)
        workflow_diversity[_user_id] = entropy(probs)
    else:
        workflow_diversity[_user_id] = 0

# Create workflow segmentation dataframe
workflow_segments = pd.DataFrame({
    'workflow_pattern': pd.Series(workflow_patterns),
    'workflow_diversity': pd.Series(workflow_diversity)
})

# Merge with engagement metrics for richer segmentation
workflow_segments = workflow_segments.join(engagement_metrics[['total_events']], how='left')

print(f"Workflow segmentation completed for {len(workflow_segments)} users")
print("\nWorkflow Pattern Distribution:")
print(workflow_segments['workflow_pattern'].value_counts())

print("\nWorkflow Diversity Statistics:")
print(workflow_segments['workflow_diversity'].describe())

print("\nAverage Activity by Workflow Pattern:")
pattern_activity = workflow_segments.groupby('workflow_pattern').agg({
    'total_events': 'mean',
    'workflow_diversity': 'mean'
}).round(2)
print(pattern_activity.sort_values('total_events', ascending=False))
