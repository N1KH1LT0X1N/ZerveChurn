import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import numpy as np

print("=" * 80)
print("INTERACTIVE VISUALIZATIONS & SEGMENT EXPORT - PRODUCTION QUALITY AUDIT")
print("=" * 80)

# ============================================================================
# PART 1: DATA VALIDATION & PREPARATION
# ============================================================================

print("\n[VALIDATION] Checking all upstream data sources...")

# Validate engagement_metrics
if engagement_metrics is None or len(engagement_metrics) == 0:
    raise ValueError("ERROR: engagement_metrics is empty or None")

engagement_viz = engagement_metrics.reset_index(drop=True).copy()
engagement_viz['distinct_id'] = engagement_metrics.index.astype(str)
engagement_viz['segment_label'] = engagement_viz['segment_label'].astype(str).replace({'nan': 'Unknown', 'None': 'Unknown'})
engagement_viz['bubble_size'] = engagement_viz['unique_event_types'].clip(lower=1)
print(f"✓ Engagement metrics: {len(engagement_viz)} users")
print(f"  segment_label unique values: {sorted(engagement_viz['segment_label'].unique().tolist())}")

# Validate workflow_segments
if workflow_segments is None or len(workflow_segments) == 0:
    raise ValueError("ERROR: workflow_segments is empty or None")
workflow_viz = workflow_segments.reset_index(drop=False).copy()
if 'index' in workflow_viz.columns:
    workflow_viz = workflow_viz.rename(columns={'index': 'distinct_id'})
workflow_viz['distinct_id'] = workflow_viz['distinct_id'].astype(str)
workflow_viz['workflow_pattern'] = workflow_viz['workflow_pattern'].astype(str).replace({'nan': 'Unknown', 'None': 'Unknown'})
print(f"✓ Workflow segments: {len(workflow_viz)} users")

# Validate temporal_segments
if temporal_segments is None or len(temporal_segments) == 0:
    raise ValueError("ERROR: temporal_segments is empty or None")
temporal_viz = temporal_segments.reset_index(drop=True).copy()
for _col in ['adoption_timing', 'activity_pattern', 'consistency_label', 'monetization_segment']:
    if _col in temporal_viz.columns:
        temporal_viz[_col] = temporal_viz[_col].astype(str).replace({'nan': 'Unknown', 'None': 'Unknown'})
print(f"✓ Temporal segments: {len(temporal_viz)} users")
print(f"  adoption_timing:     {sorted(temporal_viz['adoption_timing'].unique().tolist())}")
print(f"  activity_pattern:    {sorted(temporal_viz['activity_pattern'].unique().tolist())}")
print(f"  consistency_label:   {sorted(temporal_viz['consistency_label'].unique().tolist())}")
print(f"  monetization_segment:{sorted(temporal_viz['monetization_segment'].unique().tolist())}")

# Zerve brand colors and theme — prefixed with _ to avoid conflicts with downstream blocks
_zerve_colors_viz = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF',
                     '#1F77B4', '#9467BD', '#8C564B', '#C49C94', '#E377C2', '#F7B6D2']
_dark_bg_viz = '#1D1D20'
_dark_text_viz = '#fbfbff'
_plot_bg_viz = '#2A2A2E'
_grid_color_viz = '#333333'

# Credit events for monetization filtering (private — avoids name conflict with KM Survival block)
_credit_events_viz = ["credits_used", "credits_below_1", "credits_below_2", "credits_below_5"]

print("\n" + "=" * 80)
print("PART 2: CREATING 5 INTERACTIVE VISUALIZATIONS")
print("=" * 80)

# ============================================================================
# VIZ 1: ENGAGEMENT SEGMENTATION SCATTER
# ============================================================================

print("\n[VIZ 1] Engagement Segmentation: User Activity Scatter Plot")

_seg_labels = sorted(engagement_viz['segment_label'].unique().tolist())
print(f"  Segments found: {_seg_labels}")

fig1 = go.Figure()
for _idx, _seg in enumerate(_seg_labels):
    _mask = engagement_viz['segment_label'] == _seg
    _d = engagement_viz[_mask]
    fig1.add_trace(go.Scatter(
        x=_d['total_events'].tolist(),
        y=_d['active_days'].tolist(),
        mode='markers',
        name=str(_seg),
        marker=dict(
            size=(_d['bubble_size'].clip(upper=200) / 5 + 4).tolist(),
            color=_zerve_colors_viz[_idx % len(_zerve_colors_viz)],
            line=dict(width=0.5, color='#444444'),
            opacity=0.75
        ),
        text=_d['distinct_id'].tolist(),
        customdata=list(zip(
            _d['session_count'].tolist(),
            _d['avg_events_per_session'].round(2).tolist(),
            _d['segment_label'].tolist()
        )),
        hovertemplate=(
            '<b>User:</b> %{text}<br>'
            '<b>Total Events:</b> %{x}<br>'
            '<b>Active Days:</b> %{y}<br>'
            '<b>Sessions:</b> %{customdata[0]}<br>'
            '<b>Avg Events/Session:</b> %{customdata[1]:.2f}<br>'
            '<b>Segment:</b> %{customdata[2]}<extra></extra>'
        )
    ))

fig1.update_xaxes(
    type='log', title_text='Total Events',
    title_font=dict(size=13, color=_dark_text_viz, family='Arial'),
    tickfont=dict(color=_dark_text_viz, size=11),
    gridcolor=_grid_color_viz, zeroline=False, showline=True, linewidth=1, linecolor=_grid_color_viz
)
fig1.update_yaxes(
    title_text='Active Days',
    title_font=dict(size=13, color=_dark_text_viz, family='Arial'),
    tickfont=dict(color=_dark_text_viz, size=11),
    gridcolor=_grid_color_viz, zeroline=False, showline=True, linewidth=1, linecolor=_grid_color_viz
)
fig1.update_layout(
    title=dict(text='<b>Engagement Segmentation</b><br><sub>User Activity Patterns & Lifecycle Status</sub>',
               font=dict(size=15, color=_dark_text_viz)),
    template='plotly_dark', paper_bgcolor=_dark_bg_viz, plot_bgcolor=_plot_bg_viz,
    font=dict(color=_dark_text_viz, family='Arial', size=12), height=700, width=1000,
    hovermode='closest', showlegend=True,
    legend=dict(bgcolor='rgba(29, 29, 32, 0.95)', bordercolor='#555555', borderwidth=1,
                font=dict(size=11))
)
fig1.show()
print("✓ VIZ 1: Created successfully")

# ============================================================================
# VIZ 2: WORKFLOW PATTERN DISTRIBUTION
# ============================================================================

print("\n[VIZ 2] Workflow Pattern Distribution: User Behavior Classification")

_wf_counts = workflow_viz['workflow_pattern'].value_counts().reset_index()
_wf_counts.columns = ['workflow_pattern', 'count']
_wf_counts = _wf_counts.sort_values('count', ascending=True)

fig2 = go.Figure(go.Bar(
    y=_wf_counts['workflow_pattern'].astype(str).tolist(),
    x=_wf_counts['count'].tolist(),
    orientation='h',
    marker=dict(
        color=_wf_counts['count'].tolist(),
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='Users', thickness=15, len=0.7)
    ),
    hovertemplate='<b>%{y}</b><br>Users: %{x}<extra></extra>'
))
fig2.update_xaxes(
    title_text='Number of Users',
    title_font=dict(size=13, color=_dark_text_viz, family='Arial'),
    tickfont=dict(color=_dark_text_viz, size=11),
    gridcolor=_grid_color_viz, zeroline=False
)
fig2.update_yaxes(
    title_text='Workflow Pattern',
    title_font=dict(size=13, color=_dark_text_viz, family='Arial'),
    tickfont=dict(color=_dark_text_viz, size=11)
)
fig2.update_layout(
    title=dict(text='<b>Workflow Pattern Distribution</b><br><sub>User Behavior Classification by Workflow Type</sub>',
               font=dict(size=15, color=_dark_text_viz)),
    template='plotly_dark', paper_bgcolor=_dark_bg_viz, plot_bgcolor=_plot_bg_viz,
    font=dict(color=_dark_text_viz, family='Arial', size=12), height=550, width=1000,
    hovermode='y unified', showlegend=False
)
fig2.show()
print("✓ VIZ 2: Created successfully")

# ============================================================================
# VIZ 3: TEMPORAL & MONETIZATION 4-PANEL GRID
# ============================================================================

print("\n[VIZ 3] Temporal & Monetization: 4-Panel Segmentation Matrix")

fig3 = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Adoption Timing', 'Activity Pattern', 'Consistency Level', 'Monetization Segment'),
    specs=[[{'type': 'bar'}, {'type': 'bar'}], [{'type': 'bar'}, {'type': 'bar'}]],
    vertical_spacing=0.16, horizontal_spacing=0.14
)

for _col_name, _col_color, _row, _col in [
    ('adoption_timing', _zerve_colors_viz[0], 1, 1),
    ('activity_pattern', _zerve_colors_viz[1], 1, 2),
    ('consistency_label', _zerve_colors_viz[2], 2, 1),
    ('monetization_segment', _zerve_colors_viz[3], 2, 2),
]:
    _counts_tmp = temporal_viz[_col_name].value_counts().sort_values(ascending=False).reset_index()
    _counts_tmp.columns = [_col_name, 'count']
    fig3.add_trace(
        go.Bar(
            x=_counts_tmp[_col_name].astype(str).tolist(),
            y=_counts_tmp['count'].tolist(),
            marker=dict(color=_col_color),
            hovertemplate='<b>%{x}</b><br>Users: %{y}<extra></extra>',
            showlegend=False
        ),
        row=_row, col=_col
    )

for _i in range(1, 3):
    for _j in range(1, 3):
        fig3.update_xaxes(tickfont=dict(color=_dark_text_viz, size=10), tickangle=-30, row=_i, col=_j)
        fig3.update_yaxes(tickfont=dict(color=_dark_text_viz, size=10), gridcolor=_grid_color_viz,
                          zeroline=False, row=_i, col=_j)

fig3.update_layout(
    title_text='<b>Temporal & Monetization Segmentation</b><br><sub>4-Dimensional User Behavior Analysis</sub>',
    title_font=dict(size=15, color=_dark_text_viz),
    template='plotly_dark', paper_bgcolor=_dark_bg_viz, plot_bgcolor=_plot_bg_viz,
    font=dict(color=_dark_text_viz, family='Arial', size=11), height=900, width=1200,
    showlegend=False, hovermode='x unified'
)
fig3.show()
print("✓ VIZ 3: Created successfully")

# ============================================================================
# VIZ 4: SESSION INTENSITY VS BEHAVIOR DIVERSITY SCATTER
# ============================================================================

print("\n[VIZ 4] Session Intensity vs Workflow Diversity Analysis")

_scatter_data = engagement_viz.copy()
_scatter_data = _scatter_data.merge(
    workflow_viz[['distinct_id', 'workflow_pattern', 'workflow_diversity']],
    on='distinct_id', how='left'
)
_scatter_data['segment_label'] = _scatter_data['segment_label'].astype(str).replace({'nan': 'Unknown', 'None': 'Unknown'})
_scatter_data['workflow_pattern'] = _scatter_data['workflow_pattern'].astype(str).replace({'nan': 'Unknown', 'None': 'Unknown'})
_scatter_data = _scatter_data.dropna(subset=['session_count', 'workflow_diversity'])
_scatter_data['bubble_size2'] = (_scatter_data['total_events'].clip(lower=1, upper=10000) / 100 + 4)
print(f"  VIZ 4 data rows: {len(_scatter_data)}")

_seg_labels4 = sorted(_scatter_data['segment_label'].unique().tolist())
fig4 = go.Figure()
for _idx, _seg in enumerate(_seg_labels4):
    _d = _scatter_data[_scatter_data['segment_label'] == _seg]
    fig4.add_trace(go.Scatter(
        x=_d['session_count'].tolist(),
        y=_d['workflow_diversity'].tolist(),
        mode='markers',
        name=str(_seg),
        marker=dict(
            size=_d['bubble_size2'].tolist(),
            color=_zerve_colors_viz[_idx % len(_zerve_colors_viz)],
            line=dict(width=0.5, color='#444444'),
            opacity=0.75
        ),
        text=_d['distinct_id'].tolist(),
        customdata=list(zip(_d['workflow_pattern'].tolist(), _d['total_events'].tolist())),
        hovertemplate=(
            '<b>User:</b> %{text}<br>'
            '<b>Sessions:</b> %{x}<br>'
            '<b>Workflow Diversity:</b> %{y:.3f}<br>'
            '<b>Pattern:</b> %{customdata[0]}<br>'
            '<b>Total Events:</b> %{customdata[1]}<extra></extra>'
        )
    ))

fig4.update_xaxes(
    type='log', title_text='Number of Sessions',
    title_font=dict(size=13, color=_dark_text_viz, family='Arial'),
    tickfont=dict(color=_dark_text_viz, size=11),
    gridcolor=_grid_color_viz, zeroline=False, showline=True, linewidth=1, linecolor=_grid_color_viz
)
fig4.update_yaxes(
    title_text='Workflow Diversity (Entropy)',
    title_font=dict(size=13, color=_dark_text_viz, family='Arial'),
    tickfont=dict(color=_dark_text_viz, size=11),
    gridcolor=_grid_color_viz, zeroline=False, showline=True, linewidth=1, linecolor=_grid_color_viz
)
fig4.update_layout(
    title=dict(text='<b>Session Intensity vs Behavior Diversity</b><br><sub>Cross-Segment Workflow Complexity Analysis</sub>',
               font=dict(size=15, color=_dark_text_viz)),
    template='plotly_dark', paper_bgcolor=_dark_bg_viz, plot_bgcolor=_plot_bg_viz,
    font=dict(color=_dark_text_viz, family='Arial', size=12), height=700, width=1000,
    hovermode='closest', showlegend=True,
    legend=dict(bgcolor='rgba(29, 29, 32, 0.95)', bordercolor='#555555', borderwidth=1,
                font=dict(size=11))
)
fig4.show()
print("✓ VIZ 4: Created successfully")

# ============================================================================
# VIZ 5: SEGMENT PROFILE HEATMAP
# ============================================================================

print("\n[VIZ 5] Segment Profile Heatmap: Normalized Metrics Comparison")

_heatmap_data = engagement_viz.dropna(
    subset=['total_events', 'active_days', 'unique_event_types', 'session_count', 'avg_events_per_session']
).copy()
_heatmap_data['segment_label'] = _heatmap_data['segment_label'].astype(str)

_segment_profiles = _heatmap_data.groupby('segment_label', sort=True).agg({
    'total_events': 'mean',
    'active_days': 'mean',
    'unique_event_types': 'mean',
    'session_count': 'mean',
    'avg_events_per_session': 'mean'
}).round(2)

_sp_norm = (_segment_profiles - _segment_profiles.min()) / (
    _segment_profiles.max() - _segment_profiles.min() + 0.0001
)

_y_labels = [str(y) for y in _sp_norm.index.tolist()]
_x_labels = list(_sp_norm.columns)
_z_vals = _sp_norm.values.tolist()
_text_vals = _segment_profiles.values.tolist()

_annotations = []
for _ri, _row_label in enumerate(_y_labels):
    for _ci, _col_label in enumerate(_x_labels):
        _annotations.append(dict(
            x=_col_label,
            y=_row_label,
            text=f"{_text_vals[_ri][_ci]:.1f}",
            showarrow=False,
            font=dict(color=_dark_text_viz, size=12)
        ))

fig5 = go.Figure(data=go.Heatmap(
    z=_z_vals,
    x=_x_labels,
    y=_y_labels,
    colorscale='Viridis',
    hovertemplate='<b>%{y}</b><br>%{x}<br>Normalized: %{z:.3f}<extra></extra>',
    colorbar=dict(title='Normalized<br>Score', thickness=15, len=0.7,
                  tickfont=dict(color=_dark_text_viz, size=11))
))

fig5.update_layout(
    title=dict(text='<b>Segment Profile Heatmap</b><br><sub>Normalized Metrics Across Engagement Segments</sub>',
               font=dict(size=15, color=_dark_text_viz)),
    template='plotly_dark', paper_bgcolor=_dark_bg_viz, plot_bgcolor=_plot_bg_viz,
    font=dict(color=_dark_text_viz, family='Arial', size=12), height=600, width=1000,
    xaxis=dict(title='<b>Metrics</b>', tickfont=dict(color=_dark_text_viz, size=11),
               tickangle=-30, side='bottom'),
    yaxis=dict(title='<b>Engagement Segment</b>', tickfont=dict(color=_dark_text_viz, size=11)),
    annotations=_annotations
)
fig5.show()
print("✓ VIZ 5: Created successfully")

# ============================================================================
# PART 5: MASTER SEGMENTATION MATRIX & EXPORT
# ============================================================================

print("\n" + "=" * 80)
print("PART 5: BUILDING MASTER SEGMENTATION MATRIX & EXPORT")
print("=" * 80 + "\n")

master_segments = engagement_viz[[
    'distinct_id', 'total_events', 'active_days', 'unique_event_types',
    'session_count', 'avg_events_per_session', 'segment_label'
]].copy()

master_segments = master_segments.merge(
    workflow_viz[['distinct_id', 'workflow_pattern', 'workflow_diversity']],
    on='distinct_id', how='left'
)

master_segments = master_segments.merge(
    temporal_viz[[
        'distinct_id', 'adoption_timing', 'activity_pattern',
        'consistency_label', 'monetization_segment', 'days_since_first',
        'days_active_span', 'consistency_score'
    ]],
    on='distinct_id', how='left'
)

for _col in ['segment_label', 'workflow_pattern', 'adoption_timing', 'activity_pattern',
             'consistency_label', 'monetization_segment']:
    if _col in master_segments.columns:
        master_segments[_col] = master_segments[_col].astype(str).replace({'nan': 'Unknown', 'None': 'Unknown'})

master_segments.rename(columns={'segment_label': 'engagement_segment',
                                 'consistency_label': 'consistency'}, inplace=True)

print(f"Master segments shape: {master_segments.shape}")
print(f"Columns: {list(master_segments.columns)}")
print(f"Total rows: {len(master_segments):,}")
_null_nonzero = master_segments.isnull().sum()
_null_nonzero = _null_nonzero[_null_nonzero > 0]
print(f"Null values: {dict(_null_nonzero) if len(_null_nonzero) > 0 else 'None ✓'}")

print("\n[EXPORT] Writing master segmentation matrix to CSV...")
master_segments.to_csv('user_segments.csv', index=False)
print("✓ Exported: user_segments.csv")

print("\n" + "=" * 80)
print("SEGMENTATION SUMMARY REPORT")
print("=" * 80)
print(f"\nTotal Users Analyzed: {len(master_segments):,}")

for _report_col, _report_label in [
    ('engagement_segment', 'ENGAGEMENT SEGMENTS'),
    ('workflow_pattern', 'WORKFLOW PATTERNS'),
    ('adoption_timing', 'ADOPTION TIMING'),
    ('activity_pattern', 'ACTIVITY PATTERNS'),
    ('consistency', 'CONSISTENCY LEVELS'),
    ('monetization_segment', 'MONETIZATION'),
]:
    _counts_report = master_segments[_report_col].value_counts()
    print(f"\n[{_report_label}] ({_counts_report.shape[0]} types):")
    for _val, _cnt in _counts_report.items():
        _pct_report = (_cnt / len(master_segments)) * 100
        print(f"    {str(_val):.<40} {_cnt:>6,} users ({_pct_report:>5.1f}%)")

print(f"\n[ENGAGEMENT METRICS SUMMARY]")
print(f"Total Events  - Mean: {master_segments['total_events'].mean():,.0f}, Max: {master_segments['total_events'].max():,.0f}")
print(f"Active Days   - Mean: {master_segments['active_days'].mean():.1f}, Max: {master_segments['active_days'].max()}")
print(f"Session Count - Mean: {master_segments['session_count'].mean():.1f}, Max: {master_segments['session_count'].max()}")

print(f"\n[SAMPLE RECORDS] First 10 users in master segmentation:")
print(master_segments.head(10).to_string(index=False))

print("\n" + "=" * 80)
print("✓✓✓ ALL VISUALIZATIONS COMPLETE & PRODUCTION-QUALITY ✓✓✓")
print("✓✓✓ MASTER SEGMENTATION MATRIX EXPORTED SUCCESSFULLY ✓✓✓")
print("=" * 80)

# ============================================================================
# NAMESPACE CLEANUP: Delete ALL variables inherited from upstream blocks
# that will conflict with the Advanced Analysis Synthesis branch when merging
# into the LTV block. Using a comprehensive list based on full variable audit.
# ============================================================================

# Full list of all potential conflict variables - everything not needed downstream
_all_conflict_vars = [
    # From Event Taxonomy & Categorization chain
    'pct', 'category', 'event', 'stage', 'label', 'color', 'cat',
    'metric', 'segment', 'window', 'name', 'status', 'keyword',
    'all_events', 'total_events', 'event_taxonomy', 'event_categories',
    'categorized_events', 'event_list', 'uncategorized', 'keyword_mapping',
    'category_stats', 'category_df', 'rare_threshold', 'rare_events',
    'important_rare', 'importance', 'rare_df', 'taxonomy_reference',
    'taxonomy_event_categories', 'taxonomy_all_events', 'workflow_stages',
    'stage_event_to_stage', 'stage_data', 'all_stage_events',
    'total_stage_events', 'workflow_stage_stats', 'stage_summary_df',
    'progression_order', 'workflow_mapping', 'stats', 'cat_summary',
    'stage_summary',
    # From Engagement Segmentation chain
    'user_id', 'cluster_id', 'cluster_data', 'user_workflow', 'workflow_pcts',
    'probs', 'total_events_wf',
    # From Composite Success Score / Validation chain
    'count', 'score', 'alt_score', 'row', 'metric_name', 'info',
    'level', 'utype', 'weight', 'segment_name', 'data', 'success_scores', 'idx',
    # From Workflow Sequence patterns
    'user_sequence_features', 'event_counts', 'power_events_count',
    'has_deployment_sequence', 'has_agent_workflow', 'has_create_run_pattern',
    'error_count', 'failed_block_runs', 'repeated_events', 'unique_sequences',
    'total_sequences', 'sequence_diversity', 'trigram_set', 'has_create_edit_run',
    'i', 'seq', 'events',
    # From Kaplan-Meier / Survival blocks
    'segment_data', 'times', 'surv_probs', 'median_idx', 'median_time',
    'engagement_levels', 'level_data', 'deployment_statuses', 'status_data',
    # From Churn Risk Scoring blocks
    'cat', 'window', 'min_days', 'max_days', 'total_churned',
]

# Suppress using locals() approach - check dir() before deleting
_deleted_vars = []
for _cv in _all_conflict_vars:
    if _cv in dir():
        try:
            exec(f'del {_cv}')
            _deleted_vars.append(_cv)
        except Exception:
            pass

print(f"✓ Namespace conflict variables cleaned up ({len(_deleted_vars)} deleted).")
