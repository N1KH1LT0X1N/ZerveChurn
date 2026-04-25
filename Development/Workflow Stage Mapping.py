import pandas as pd
import numpy as np

# =====================================================
# MAP EVENTS TO USER WORKFLOW STAGES
# =====================================================

print("=" * 80)
print("WORKFLOW STAGE MAPPING - User Journey Framework")
print("=" * 80)

# Get event categories from taxonomy reference
taxonomy_event_categories = taxonomy_reference['event_categories']
taxonomy_all_events = taxonomy_reference['category_summary']

# Define comprehensive workflow stages based on user progression
workflow_stages = {
    # STAGE 1: ONBOARDING (New user activation)
    'Onboarding': {
        'description': 'Initial activation, learning platform basics, completing tutorials',
        'events': [
            'canvas_onboarding_tour_start', 'canvas_onboarding_tour_introduction_step',
            'canvas_onboarding_tour_languages_step', 'canvas_onboarding_tour_code_and_variables_step',
            'canvas_onboarding_tour_running_code_step', 'canvas_onboarding_tour_finished',
            'canvas_onboarding_tour_skipped', 'skip_onboarding_form', 'user_signed_up',
            'account_activated', 'email_verified', 'welcome_email_sent'
        ]
    },
    'Exploration': {
        'description': 'First canvas creation, exploring interface, trying basic features',
        'events': [
            'pageview', '$pageview', 'canvas_viewed', 'canvas_opened', 'canvas_switched',
            'block_viewed', 'link_clicked', 'help_opened', 'documentation_viewed',
            'tutorial_accessed', 'example_opened', 'template_browsed', 'search_performed'
        ]
    },
    'Creation': {
        'description': 'Creating first blocks, writing code, setting up canvases',
        'events': [
            'canvas_created', 'block_created', 'block_create', 'python_block_created',
            'r_block_created', 'markdown_block_created', 'query_block_created',
            'gen_ai_block_created', 'canvas_autosave', 'canvas_saved', 'new_block'
        ]
    },
    'Editing & Iteration': {
        'description': 'Modifying code, refining workflows, iterating on solutions',
        'events': [
            'block_edit', 'block_edited', 'block_content_changed', 'edit_block',
            'block_name_changed', 'canvas_block_edited', 'code_edited', 'text_input_focused',
            'text_input_blurred', 'block_resize', 'block_moved', 'block_position_changed',
            'canvas_renamed', 'block_duplicate', 'block_duplicated'
        ]
    },
    'Execution': {
        'description': 'Running blocks, executing workflows, generating outputs',
        'events': [
            'block_run', 'run_block', 'block_execution_started', 'block_execution_finished',
            'block_execution_success', 'block_execution_failed', 'cell_run',
            'run_all_blocks', 'run_downstream_blocks', 'run_button_clicked', 'block_cancelled'
        ]
    },
    'AI Assistance': {
        'description': 'Using AI for code generation, suggestions, and problem solving',
        'events': [
            'agent_new_chat', 'agent_start_from_prompt', 'agent_message', 'agent_chat_opened',
            'agent_prompt_submitted', 'agent_accept_suggestion', 'agent_suggestion_accepted',
            'agent_suggestion_rejected', 'agent_worker_created', 'agent_worker_finished',
            'agent_worker_failed', 'agent_created_block', 'agent_generated_code',
            'copilot_suggestion_accepted', 'agent_edit_request_from_error',
            'agent_regenerate_from_ticket', 'agent_chat_edit_request_from_error',
            'agent_continue_request', 'agent_regenerate_ticket', 'agent_chat_message_sent',
            'agent_chat_response_received'
        ]
    },
    'Data Operations': {
        'description': 'Connecting to data sources, loading data, running queries',
        'events': [
            'file_uploaded', 'file_downloaded', 'data_imported', 'csv_imported',
            'connection_created', 'connection_tested', 'connection_configured',
            'query_executed', 'sql_query_run', 'data_fetched', 'data_loaded',
            'query_result_viewed', 'database_connected'
        ]
    },
    'Analysis & Visualization': {
        'description': 'Creating charts, running analyses, generating insights',
        'events': [
            'chart_created', 'plot_generated', 'visualization_customized',
            'block_output_viewed', 'variable_inspected', 'chart_viewed',
            'dashboard_created', 'model_trained', 'analysis_run', 'output_downloaded'
        ]
    },
    'Collaboration': {
        'description': 'Sharing work, collaborating with team members, comments',
        'events': [
            'canvas_shared', 'canvas_shared_with_user', 'share_link_created',
            'canvas_permission_changed', 'comment_added', 'comment_replied',
            'workspace_invite_sent', 'user_added_to_workspace', 'mention_notification'
        ]
    },
    'Deployment': {
        'description': 'Deploying models, creating APIs, productionizing workflows',
        'events': [
            'api_deployed', 'endpoint_created', 'model_deployed', 'app_published',
            'deployment_configured', 'container_launched', 'service_started'
        ]
    },
    'Optimization': {
        'description': 'Configuring compute, optimizing performance, advanced settings',
        'events': [
            'compute_config_changed', 'settings_opened', 'preferences_updated',
            'environment_configured', 'package_installed', 'runtime_configured',
            'performance_monitored', 'memory_increased', 'gpu_requested'
        ]
    },
    'Monitoring & Maintenance': {
        'description': 'Tracking performance, reviewing logs, maintaining workflows',
        'events': [
            'logs_viewed', 'metrics_viewed', 'health_check_performed',
            'alert_triggered', 'error_logged', 'debug_mode_enabled',
            'error_occurred', 'exception_caught', 'stack_trace_viewed'
        ]
    },
    'Active Usage': {
        'description': 'Regular platform interactions, ongoing engagement',
        'events': [
            'session_started', 'user_login', 'tab_switched', 'window_focused',
            'canvas_minimap_opened', 'keyboard_shortcut_used', 'notification_viewed'
        ]
    },
    'Monetization': {
        'description': 'Credit consumption, billing events, upgrades',
        'events': [
            'credits_used', 'credits_below_1', 'credits_below_2', 'credits_below_5',
            'credits_purchased', 'plan_upgraded', 'payment_processed',
            'subscription_changed', 'billing_viewed', 'invoice_downloaded'
        ]
    }
}

# Map all events to workflow stages
# Use _stage, _event, _stage_data, _cat as private loop vars to avoid export conflicts
stage_event_to_stage = {}
for _stage, _stage_data in workflow_stages.items():
    for _event in _stage_data['events']:
        stage_event_to_stage[_event] = _stage

# For events not explicitly mapped, use intelligent categorization
for _event in taxonomy_event_categories.keys():
    if _event not in stage_event_to_stage:
        _cat = taxonomy_event_categories[_event]
        
        if 'Onboarding' in _cat:
            stage_event_to_stage[_event] = 'Onboarding'
        elif 'AI Agent' in _cat:
            stage_event_to_stage[_event] = 'AI Assistance'
        elif 'Block' in _cat and 'Creation' in _cat:
            stage_event_to_stage[_event] = 'Creation'
        elif 'Editing' in _cat:
            stage_event_to_stage[_event] = 'Editing & Iteration'
        elif 'Execution' in _cat:
            stage_event_to_stage[_event] = 'Execution'
        elif _cat in ['Data Connections', 'Query Operations', 'File Management', 'Data Operations']:
            stage_event_to_stage[_event] = 'Data Operations'
        elif _cat in ['Visualization', 'Analysis & Modeling']:
            stage_event_to_stage[_event] = 'Analysis & Visualization'
        elif 'Collaboration' in _cat:
            stage_event_to_stage[_event] = 'Collaboration'
        elif 'Deployment' in _cat:
            stage_event_to_stage[_event] = 'Deployment'
        elif _cat in ['Compute & Resources', 'Settings & Preferences']:
            stage_event_to_stage[_event] = 'Optimization'
        elif 'Credits' in _cat or 'Billing' in _cat:
            stage_event_to_stage[_event] = 'Monetization'
        elif 'Session' in _cat or 'Navigation' in _cat:
            stage_event_to_stage[_event] = 'Active Usage'
        elif 'Error' in _cat or 'Debugging' in _cat or 'Monitoring' in _cat:
            stage_event_to_stage[_event] = 'Monitoring & Maintenance'
        else:
            stage_event_to_stage[_event] = 'Exploration'

# Get all events with their counts
all_stage_events = user_retention['event'].value_counts()
total_stage_events = len(user_retention)

# Calculate stage statistics using private loop vars
workflow_stage_stats = {}
for _event, _count in all_stage_events.items():
    _stage = stage_event_to_stage.get(_event, 'Uncategorized')
    if _stage not in workflow_stage_stats:
        workflow_stage_stats[_stage] = {'count': 0, 'events': []}
    workflow_stage_stats[_stage]['count'] += _count
    workflow_stage_stats[_stage]['events'].append((_event, _count))

# Create stage summary DataFrame using private loop vars
stage_summary_df = pd.DataFrame([
    {
        'Workflow Stage': _stage,
        'Description': workflow_stages.get(_stage, {}).get('description', 'Miscellaneous activities'),
        'Event Count': len(_stats['events']),
        'Total Occurrences': _stats['count'],
        'Percentage': (_stats['count'] / total_stage_events * 100)
    }
    for _stage, _stats in workflow_stage_stats.items()
]).sort_values('Total Occurrences', ascending=False)

print(f"\nTotal workflow stages: {len(stage_summary_df)}\n")
print(stage_summary_df.to_string(index=False))

# Define typical user progression path
progression_order = [
    'Onboarding', 'Exploration', 'Creation', 'Editing & Iteration',
    'Execution', 'AI Assistance', 'Data Operations', 'Analysis & Visualization',
    'Collaboration', 'Optimization', 'Deployment', 'Monitoring & Maintenance'
]

print("\n" + "=" * 80)
print("TYPICAL USER PROGRESSION PATH")
print("=" * 80)
print("\nStages ordered by typical user journey:\n")

for _idx, _stage in enumerate(progression_order, 1):
    if _stage in workflow_stage_stats:
        _stats = workflow_stage_stats[_stage]
        _pct = (_stats['count'] / total_stage_events * 100)
        print(f"{_idx:2d}. {_stage:30s} | {len(_stats['events']):3d} events | {_stats['count']:8,} occurrences ({_pct:5.2f}%)")

# Store workflow mapping
workflow_mapping = {
    'event_to_stage': stage_event_to_stage,
    'stage_stats': workflow_stage_stats,
    'stage_summary': stage_summary_df,
    'workflow_stages': workflow_stages,
    'progression_order': progression_order
}

print("\n" + "=" * 80)
print("✓ Workflow stage mapping completed and stored in 'workflow_mapping' dictionary")
print("=" * 80)
