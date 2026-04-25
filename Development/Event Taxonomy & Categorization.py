import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =====================================================
# 1. CATEGORIZE ALL 141 EVENTS INTO LOGICAL GROUPS
# =====================================================

# Get all unique events and their frequencies
all_events = user_retention['event'].value_counts()
total_events = len(user_retention)

print("=" * 80)
print("EVENT TAXONOMY - CATEGORIZING 141 EVENTS")
print("=" * 80)
print(f"\nTotal unique events: {len(all_events)}")
print(f"Total event occurrences: {total_events:,}\n")

# Define comprehensive event taxonomy based on user workflow and functionality
event_taxonomy = {
    # ONBOARDING & ACCOUNT MANAGEMENT
    'Onboarding & Activation': [
        'canvas_onboarding_tour_finished', 'canvas_onboarding_tour_code_and_variables_step',
        'canvas_onboarding_tour_start', 'canvas_onboarding_tour_skipped',
        'canvas_onboarding_tour_languages_step', 'canvas_onboarding_tour_running_code_step',
        'canvas_onboarding_tour_introduction_step', 'account_activated', 'user_signed_up',
        'welcome_email_sent', 'email_verified'
    ],
    
    # BLOCK CREATION & EDITING
    'Block Creation': [
        'block_create', 'block_created', 'python_block_created', 'r_block_created',
        'markdown_block_created', 'query_block_created', 'gen_ai_block_created',
        'block_duplicate', 'block_duplicated'
    ],
    
    'Block Editing': [
        'block_edit', 'block_edited', 'block_content_changed', 'block_name_changed',
        'block_description_changed', 'block_resize', 'block_moved', 'block_position_changed'
    ],
    
    # BLOCK EXECUTION & RESULTS
    'Block Execution': [
        'block_run', 'block_execution_started', 'block_execution_finished',
        'block_execution_success', 'block_execution_failed', 'block_execution_cancelled',
        'run_button_clicked', 'cell_run', 'run_all_blocks', 'run_downstream_blocks'
    ],
    
    'Block Results & Output': [
        'block_output_viewed', 'block_result_expanded', 'block_result_collapsed',
        'variable_inspected', 'output_downloaded', 'chart_viewed', 'table_viewed'
    ],
    
    # CANVAS OPERATIONS
    'Canvas Navigation': [
        'canvas_opened', 'canvas_viewed', 'canvas_switched', 'canvas_zoom_in',
        'canvas_zoom_out', 'canvas_pan', 'canvas_fit_to_screen', 'canvas_minimap_opened'
    ],
    
    'Canvas Management': [
        'canvas_created', 'canvas_renamed', 'canvas_deleted', 'canvas_duplicated',
        'canvas_saved', 'canvas_auto_saved', 'canvas_shared', 'canvas_exported'
    ],
    
    # AI AGENT INTERACTIONS
    'AI Agent - Suggestions': [
        'agent_accept_suggestion', 'agent_suggestion_accepted', 'agent_suggestion_rejected',
        'agent_suggestion_generated', 'agent_suggestion_modified', 'copilot_suggestion_accepted'
    ],
    
    'AI Agent - Workers': [
        'agent_worker_created', 'agent_worker_finished', 'agent_worker_failed',
        'agent_task_started', 'agent_task_completed', 'agent_task_cancelled'
    ],
    
    'AI Agent - Chat': [
        'agent_chat_opened', 'agent_chat_message_sent', 'agent_chat_response_received',
        'agent_prompt_submitted', 'agent_conversation_started', 'chat_message_sent'
    ],
    
    # DATA CONNECTIONS & QUERIES
    'Data Connections': [
        'connection_created', 'connection_tested', 'connection_failed', 'connection_deleted',
        'database_connected', 'api_connected', 'connection_configured'
    ],
    
    'Query Operations': [
        'query_executed', 'query_edited', 'query_failed', 'query_cancelled',
        'sql_query_run', 'data_fetched', 'data_loaded', 'query_result_viewed'
    ],
    
    # COLLABORATION
    'Collaboration & Sharing': [
        'canvas_shared_with_user', 'canvas_permission_changed', 'comment_added',
        'comment_replied', 'mention_notification', 'workspace_invite_sent',
        'user_added_to_workspace', 'share_link_created'
    ],
    
    # FILE & DATA MANAGEMENT
    'File Management': [
        'file_uploaded', 'file_downloaded', 'file_deleted', 'file_renamed',
        'folder_created', 'asset_imported', 'asset_exported'
    ],
    
    'Data Import/Export': [
        'data_imported', 'data_exported', 'csv_imported', 'excel_imported',
        'parquet_loaded', 'json_imported', 'data_source_connected'
    ],
    
    # VISUALIZATION & ANALYSIS
    'Visualization': [
        'chart_created', 'plot_generated', 'visualization_customized',
        'dashboard_created', 'graph_exported', 'chart_type_changed'
    ],
    
    'Analysis & Modeling': [
        'model_trained', 'model_evaluated', 'prediction_made', 'analysis_run',
        'statistical_test_performed', 'correlation_analyzed', 'feature_engineered'
    ],
    
    # DEPLOYMENT & PRODUCTION
    'Deployment': [
        'api_deployed', 'endpoint_created', 'model_deployed', 'app_published',
        'service_started', 'container_launched', 'deployment_configured'
    ],
    
    'Monitoring & Logs': [
        'logs_viewed', 'error_logged', 'performance_monitored', 'metrics_viewed',
        'alert_triggered', 'health_check_performed'
    ],
    
    # SETTINGS & CONFIGURATION
    'Settings & Preferences': [
        'settings_opened', 'preferences_updated', 'theme_changed', 'keyboard_shortcut_used',
        'language_changed', 'profile_updated', 'notification_settings_changed'
    ],
    
    'Compute & Resources': [
        'compute_config_changed', 'memory_increased', 'cpu_allocated',
        'gpu_requested', 'environment_configured', 'package_installed',
        'dependency_added', 'runtime_configured'
    ],
    
    # CREDITS & BILLING
    'Credits & Billing': [
        'credits_purchased', 'credits_used', 'payment_processed', 'plan_upgraded',
        'subscription_changed', 'billing_viewed', 'invoice_downloaded'
    ],
    
    # SESSION & PAGEVIEW
    'Session & Navigation': [
        'pageview', '$pageview', 'page_visited', 'session_started', 'session_ended',
        'user_login', 'user_logout', 'tab_switched', 'window_focused', 'window_blurred'
    ],
    
    # ERRORS & DEBUGGING
    'Errors & Debugging': [
        'error_occurred', 'exception_caught', 'debug_mode_enabled', 'breakpoint_set',
        'variable_inspected', 'stack_trace_viewed', 'error_resolved'
    ],
    
    # SEARCH & DISCOVERY
    'Search & Discovery': [
        'search_performed', 'search_result_clicked', 'template_browsed',
        'example_opened', 'documentation_viewed', 'help_opened', 'tutorial_accessed'
    ]
}

# Categorize all events
event_categories = {}
categorized_events = set()

for _cat, _event_list in event_taxonomy.items():
    for _ev in _event_list:
        if _ev in all_events.index:
            event_categories[_ev] = _cat
            categorized_events.add(_ev)

# Handle uncategorized events by analyzing their names
uncategorized = set(all_events.index) - categorized_events

# Auto-categorize uncategorized events based on keywords
keyword_mapping = {
    'agent': 'AI Agent - General',
    'block': 'Block Operations',
    'canvas': 'Canvas Operations',
    'query': 'Query Operations',
    'connection': 'Data Connections',
    'file': 'File Management',
    'data': 'Data Operations',
    'chart': 'Visualization',
    'run': 'Execution',
    'click': 'User Interactions',
    'view': 'Viewing & Inspection',
    'edit': 'Editing',
    'create': 'Creation',
    'delete': 'Deletion',
    'error': 'Errors & Debugging',
    'session': 'Session & Navigation',
    'user': 'User Actions',
    'credit': 'Credits & Billing',
    'deploy': 'Deployment',
    'share': 'Collaboration & Sharing'
}

for _ev2 in uncategorized:
    event_lower = _ev2.lower()
    categorized = False
    
    for keyword, _c in keyword_mapping.items():
        if keyword in event_lower:
            event_categories[_ev2] = _c
            categorized = True
            break
    
    if not categorized:
        event_categories[_ev2] = 'Other / Miscellaneous'

# Calculate category statistics
category_stats = {}
for _ev3, _cnt in all_events.items():
    _c2 = event_categories.get(_ev3, 'Uncategorized')
    if _c2 not in category_stats:
        category_stats[_c2] = {'count': 0, 'events': []}
    category_stats[_c2]['count'] += _cnt
    category_stats[_c2]['events'].append((_ev3, _cnt))

# Sort categories by frequency
category_df = pd.DataFrame([
    {
        'Category': _c3,
        'Event Count': len(_stats['events']),
        'Total Occurrences': _stats['count'],
        'Percentage': (_stats['count'] / total_events * 100)
    }
    for _c3, _stats in category_stats.items()
]).sort_values('Total Occurrences', ascending=False)

print("\n" + "=" * 80)
print("EVENT CATEGORIES - SUMMARY")
print("=" * 80)
print(f"\nTotal categories: {len(category_df)}\n")
print(category_df.to_string(index=False))

# =====================================================
# 2. IDENTIFY RARE BUT IMPORTANT EVENTS (<1%)
# =====================================================

rare_threshold = total_events * 0.01
rare_events = all_events[all_events < rare_threshold]

print("\n" + "=" * 80)
print(f"RARE BUT IMPORTANT EVENTS (< 1%, threshold: {rare_threshold:,.0f} occurrences)")
print("=" * 80)
print(f"\nTotal rare events: {len(rare_events)}")
print(f"Rare events represent {(rare_events.sum() / total_events * 100):.2f}% of all events\n")

# Identify important rare events by category - ALL variables prefixed with _ to avoid conflicts
important_rare = []
for _ev4, _ev_cnt in rare_events.items():
    _cat4 = event_categories.get(_ev4, 'Uncategorized')
    _pct = (_ev_cnt / total_events * 100)  # private: avoid conflict
    
    # Flag potentially important rare events
    importance = 'HIGH' if any(kw in _ev4.lower() for kw in ['error', 'deploy', 'credit', 'upgrade', 'payment']) else 'MEDIUM'
    
    important_rare.append({
        'Event': _ev4,
        'Category': _cat4,
        'Count': _ev_cnt,
        'Percentage': _pct,
        'Importance': importance
    })

rare_df = pd.DataFrame(important_rare).sort_values('Count', ascending=False)
print(rare_df.head(30).to_string(index=False))

# Store the taxonomy as reference dictionary
taxonomy_reference = {
    'event_categories': event_categories,
    'category_stats': category_stats,
    'category_summary': category_df,
    'rare_events': rare_df,
    'total_events': total_events,
    'total_unique_events': len(all_events)
}

print("\n" + "=" * 80)
print("✓ Event taxonomy created and stored in 'taxonomy_reference' dictionary")
print("=" * 80)
