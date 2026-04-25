import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("=" * 80)
print("LTV PREDICTION & UNIT ECONOMICS")
print("=" * 80)

# ============================================================================
# PART 1: BUILD UNIFIED LTV DATASET
# ============================================================================
print("\n[1] Building unified LTV dataset from upstream sources...")

# master_segments: distinct_id, total_events, active_days, unique_event_types,
#   session_count, avg_events_per_session, engagement_segment, workflow_pattern,
#   workflow_diversity, adoption_timing, activity_pattern, consistency,
#   monetization_segment, days_since_first, days_active_span, consistency_score
#
# unified_df: user_id, success_score, success_tier, total_events,
#   days_since_last, is_power_user, is_network_hub, is_anomalous,
#   power_user_score, struggle_score, network_centrality, is_super_connector
#
# survival_with_segments: user_id, distinct_id, tenure_days, long_term_retention,
#   is_paid_user, has_deployment, collaboration_success, engagement_level, time_to_deployment

_ltv_base = master_segments.copy()

# unified_df uses user_id — rename to distinct_id for join
_unified_renamed = unified_df.rename(columns={'user_id': 'distinct_id'})
# Backfill `churn_proxy` if upstream `unified_df` predates the Apr 2026 fix.
if 'churn_proxy' not in _unified_renamed.columns and 'success_score' in _unified_renamed.columns:
    _unified_renamed = _unified_renamed.assign(
        churn_proxy=(100.0 - _unified_renamed['success_score']).clip(0, 100)
    )
_ltv_df = _ltv_base.merge(
    _unified_renamed[[
        'distinct_id', 'success_score', 'churn_proxy', 'success_tier',
        'is_power_user', 'is_network_hub', 'is_anomalous',
        'power_user_score', 'struggle_score', 'network_centrality', 'is_super_connector'
    ]],
    on='distinct_id', how='left'
)

# survival_with_segments already has BOTH user_id AND distinct_id columns
# Use distinct_id directly as the join key (no rename needed)
_surv_cols = ['distinct_id', 'tenure_days', 'long_term_retention',
              'is_paid_user', 'has_deployment', 'collaboration_success',
              'engagement_level', 'time_to_deployment']
_surv_available = [c for c in _surv_cols if c in survival_with_segments.columns]

_ltv_df = _ltv_df.merge(
    survival_with_segments[_surv_available].drop_duplicates(subset='distinct_id'),
    on='distinct_id', how='left'
)

# Fill missing values for users not in active cohort.
# Users absent from active_users_survival are treated as already churned, so
# their churn_proxy defaults to 100 (max risk → retention_probability = 0).
# `success_score` is left at 100 only for legacy display; the LTV formula
# below relies exclusively on `churn_proxy`.
_ltv_df['success_score'] = _ltv_df['success_score'].fillna(100.0)
_ltv_df['churn_proxy'] = _ltv_df['churn_proxy'].fillna(100.0)
_ltv_df['success_tier'] = _ltv_df['success_tier'].fillna('High')
_ltv_df['is_power_user'] = _ltv_df['is_power_user'].fillna(False)
_ltv_df['is_network_hub'] = _ltv_df['is_network_hub'].fillna(False)
_ltv_df['is_anomalous'] = _ltv_df['is_anomalous'].fillna(False)
_ltv_df['power_user_score'] = _ltv_df['power_user_score'].fillna(0.0)
_ltv_df['struggle_score'] = _ltv_df['struggle_score'].fillna(0.0)
_ltv_df['network_centrality'] = _ltv_df['network_centrality'].fillna(0.0)
_ltv_df['tenure_days'] = _ltv_df['tenure_days'].fillna(_ltv_df['days_since_first'])
_ltv_df['long_term_retention'] = _ltv_df['long_term_retention'].fillna(False)
_ltv_df['is_paid_user'] = _ltv_df['is_paid_user'].fillna(False)
_ltv_df['has_deployment'] = _ltv_df['has_deployment'].fillna(False)
_ltv_df['collaboration_success'] = _ltv_df['collaboration_success'].fillna(False)

print(f"  LTV dataset: {_ltv_df.shape[0]:,} users, {_ltv_df.shape[1]} columns")

# ============================================================================
# PART 2: COMPUTE BEHAVIORAL LTV SCORES
# ============================================================================
# Engagement-weighted retention LTV:
#   LTV_score = engagement_value × (1 - churn_prob) × tier_multiplier × bonuses
print("\n[2] Computing behavioral LTV scores...")

_max_events = _ltv_df['total_events'].quantile(0.99).clip(min=1)
_max_days = _ltv_df['active_days'].quantile(0.99).clip(min=1)
_max_tenure = _ltv_df['tenure_days'].quantile(0.99).clip(min=1)
_max_power = _ltv_df['power_user_score'].quantile(0.99).clip(min=0.01)
_max_sessions = _ltv_df['session_count'].quantile(0.99).clip(min=1)

_ltv_df['_ev_n'] = (_ltv_df['total_events'] / _max_events).clip(0, 1)
_ltv_df['_dy_n'] = (_ltv_df['active_days'] / _max_days).clip(0, 1)
_ltv_df['_te_n'] = (_ltv_df['tenure_days'] / _max_tenure).clip(0, 1)
_ltv_df['_pw_n'] = (_ltv_df['power_user_score'] / _max_power).clip(0, 1)
_ltv_df['_se_n'] = (_ltv_df['session_count'] / _max_sessions).clip(0, 1)

_ltv_df['ltv_engagement_score'] = (
    0.30 * _ltv_df['_ev_n'] +
    0.20 * _ltv_df['_dy_n'] +
    0.20 * _ltv_df['_te_n'] +
    0.15 * _ltv_df['_pw_n'] +
    0.10 * _ltv_df['_se_n'] +
    0.05 * _ltv_df['is_power_user'].astype(float)
).clip(0, 1)

# Use churn_proxy (= 100 - success_score) so the *high-success* users get
# the *low* churn probability they should have. The previous formula
# (success_score / 100) was inverted — see docs/repo_state_and_next_steps.md §6.
_ltv_df['churn_probability'] = (_ltv_df['churn_proxy'] / 100.0).clip(0, 1)
_ltv_df['retention_probability'] = 1.0 - _ltv_df['churn_probability']

_seg_multipliers = {'Power Users': 5.0, 'Network Hubs': 3.5, 'Anomalous': 2.5, 'Regular': 1.0}

def _get_user_tier(row):
    if row['is_power_user']:
        return 'Power Users'
    elif row['is_network_hub']:
        return 'Network Hubs'
    elif row['is_anomalous']:
        return 'Anomalous'
    return 'Regular'

_ltv_df['user_tier'] = _ltv_df.apply(_get_user_tier, axis=1)
_ltv_df['tier_multiplier'] = _ltv_df['user_tier'].map(_seg_multipliers)
_ltv_df['deployment_bonus'] = _ltv_df['has_deployment'].astype(float) * 0.3
_ltv_df['collab_bonus'] = _ltv_df['collaboration_success'].astype(float) * 0.2

_ltv_df['ltv_raw'] = (
    _ltv_df['ltv_engagement_score'] *
    _ltv_df['retention_probability'] *
    _ltv_df['tier_multiplier'] *
    (1 + _ltv_df['deployment_bonus'] + _ltv_df['collab_bonus'])
)

_max_ltv_raw = _ltv_df['ltv_raw'].quantile(0.99).clip(min=0.001)
_ltv_df['ltv_score'] = (_ltv_df['ltv_raw'] / _max_ltv_raw * 100).clip(0, 100).round(2)

def _assign_ltv_tier(score):
    if score >= 70: return 'Platinum'
    elif score >= 45: return 'Gold'
    elif score >= 20: return 'Silver'
    return 'Bronze'

_ltv_df['ltv_tier'] = _ltv_df['ltv_score'].apply(_assign_ltv_tier)

print(f"  LTV Scores — Mean: {_ltv_df['ltv_score'].mean():.1f}, Median: {_ltv_df['ltv_score'].median():.1f}, Max: {_ltv_df['ltv_score'].max():.1f}")
for _tier, _cnt in _ltv_df['ltv_tier'].value_counts().items():
    print(f"    {_tier}: {_cnt:,} ({_cnt/len(_ltv_df)*100:.1f}%)")

# ============================================================================
# PART 3: UNIT ECONOMICS
# ============================================================================
print("\n[3] Unit economics by segment...")

_seg_groups = _ltv_df.groupby('user_tier').agg(
    user_count=('distinct_id', 'count'),
    avg_ltv_score=('ltv_score', 'mean'),
    median_ltv_score=('ltv_score', 'median'),
    avg_engagement=('ltv_engagement_score', 'mean'),
    avg_retention_prob=('retention_probability', 'mean'),
    avg_churn_risk=('churn_proxy', 'mean'),
    avg_active_days=('active_days', 'mean'),
    avg_tenure=('tenure_days', 'mean'),
    avg_total_events=('total_events', 'mean'),
    pct_deployed=('has_deployment', 'mean'),
    pct_collab=('collaboration_success', 'mean'),
).reset_index().round(2)

print("\n  Unit Economics by User Tier:")
print(_seg_groups.to_string(index=False))

_eng_seg_ltv = _ltv_df.groupby('engagement_segment').agg(
    user_count=('distinct_id', 'count'),
    avg_ltv_score=('ltv_score', 'mean'),
    avg_churn_risk=('churn_proxy', 'mean'),
    avg_tenure=('tenure_days', 'mean'),
).reset_index().sort_values('avg_ltv_score', ascending=False).round(2)

print("\n  LTV by Engagement Segment:")
print(_eng_seg_ltv.to_string(index=False))

_wf_ltv = _ltv_df.groupby('workflow_pattern').agg(
    user_count=('distinct_id', 'count'),
    avg_ltv_score=('ltv_score', 'mean'),
    avg_churn_risk=('churn_proxy', 'mean'),
).reset_index().sort_values('avg_ltv_score', ascending=False).round(2)

print("\n  LTV by Workflow Pattern:")
print(_wf_ltv.to_string(index=False))

# ============================================================================
# PART 4: VISUALIZATIONS
# ============================================================================
print("\n[4] Creating visualizations...")

_bg_ltv = '#1D1D20'
_txt_ltv = '#fbfbff'
_plt_bg = '#2A2A2E'
_grd_ltv = '#333333'
_tier_order = ['Platinum', 'Gold', 'Silver', 'Bronze']
_tier_colors = ['#ffd400', '#FFB482', '#A1C9F4', '#909094']

# --- VIZ A: LTV Distribution by Tier ---
ltv_distribution_fig = go.Figure()
for _ix, _tier in enumerate(_tier_order):
    _td = _ltv_df[_ltv_df['ltv_tier'] == _tier]['ltv_score'].values
    if len(_td) > 0:
        ltv_distribution_fig.add_trace(go.Box(
            y=_td.tolist(),
            name=f"{_tier} ({len(_td):,})",
            marker=dict(color=_tier_colors[_ix], size=3),
            line=dict(color=_tier_colors[_ix]),
            boxmean='sd', fillcolor=_tier_colors[_ix], opacity=0.75
        ))

ltv_distribution_fig.update_yaxes(
    title_text='LTV Score (0–100)',
    title_font=dict(size=13, color=_txt_ltv),
    tickfont=dict(color=_txt_ltv, size=11), gridcolor=_grd_ltv, zeroline=False
)
ltv_distribution_fig.update_xaxes(tickfont=dict(color=_txt_ltv, size=11))
ltv_distribution_fig.update_layout(
    title=dict(
        text='<b>LTV Score Distribution by User Tier</b><br><sub>Behavioral LTV across Platinum / Gold / Silver / Bronze segments</sub>',
        font=dict(size=15, color=_txt_ltv)
    ),
    template='plotly_dark', paper_bgcolor=_bg_ltv, plot_bgcolor=_plt_bg,
    font=dict(color=_txt_ltv, family='Arial', size=12),
    height=550, width=900, showlegend=True,
    legend=dict(bgcolor='rgba(29,29,32,0.95)', bordercolor='#555', borderwidth=1, font=dict(size=11))
)
ltv_distribution_fig.show()
print("  ✓ VIZ A: LTV Distribution by Tier")

# --- VIZ B: LTV vs Churn Risk ---
ltv_churn_scatter_fig = go.Figure()
_tcm = {'Platinum': '#ffd400', 'Gold': '#FFB482', 'Silver': '#A1C9F4', 'Bronze': '#909094'}
for _tier in _tier_order:
    _msk = _ltv_df['ltv_tier'] == _tier
    _ds = _ltv_df[_msk].sample(min(500, _msk.sum()), random_state=42)
    ltv_churn_scatter_fig.add_trace(go.Scatter(
        x=_ds['success_score'].tolist(),
        y=_ds['ltv_score'].tolist(),
        mode='markers',
        name=_tier,
        marker=dict(color=_tcm[_tier], size=5, opacity=0.65, line=dict(width=0.4, color='#333')),
        hovertemplate=f'<b>{_tier}</b><br>Churn Risk: %{{x:.1f}}%<br>LTV Score: %{{y:.1f}}<extra></extra>'
    ))

ltv_churn_scatter_fig.update_xaxes(
    title_text='Churn Risk Score (%)',
    title_font=dict(size=13, color=_txt_ltv), tickfont=dict(color=_txt_ltv, size=11),
    gridcolor=_grd_ltv, zeroline=False
)
ltv_churn_scatter_fig.update_yaxes(
    title_text='LTV Score (0–100)',
    title_font=dict(size=13, color=_txt_ltv), tickfont=dict(color=_txt_ltv, size=11),
    gridcolor=_grd_ltv, zeroline=False
)
ltv_churn_scatter_fig.update_layout(
    title=dict(
        text='<b>LTV Score vs Churn Risk</b><br><sub>High-LTV + high-churn users = critical retention targets</sub>',
        font=dict(size=15, color=_txt_ltv)
    ),
    template='plotly_dark', paper_bgcolor=_bg_ltv, plot_bgcolor=_plt_bg,
    font=dict(color=_txt_ltv, family='Arial', size=12),
    height=600, width=950, hovermode='closest',
    legend=dict(bgcolor='rgba(29,29,32,0.95)', bordercolor='#555', borderwidth=1, font=dict(size=11))
)
ltv_churn_scatter_fig.show()
print("  ✓ VIZ B: LTV vs Churn Risk")

# --- VIZ C: LTV by Workflow Pattern ---
_wf_sorted = _wf_ltv.sort_values('avg_ltv_score', ascending=True)
ltv_workflow_fig = go.Figure(go.Bar(
    y=_wf_sorted['workflow_pattern'].astype(str).tolist(),
    x=_wf_sorted['avg_ltv_score'].tolist(),
    orientation='h',
    marker=dict(
        color=_wf_sorted['avg_ltv_score'].tolist(),
        colorscale='Viridis', showscale=True,
        colorbar=dict(title='Avg LTV', thickness=14, len=0.7, tickfont=dict(color=_txt_ltv))
    ),
    customdata=list(zip(_wf_sorted['user_count'].tolist(), _wf_sorted['avg_churn_risk'].tolist())),
    hovertemplate='<b>%{y}</b><br>Avg LTV: %{x:.1f}<br>Users: %{customdata[0]}<br>Churn Risk: %{customdata[1]:.1f}%<extra></extra>'
))
ltv_workflow_fig.update_xaxes(
    title_text='Average LTV Score',
    title_font=dict(size=13, color=_txt_ltv), tickfont=dict(color=_txt_ltv, size=11),
    gridcolor=_grd_ltv, zeroline=False
)
ltv_workflow_fig.update_yaxes(tickfont=dict(color=_txt_ltv, size=11))
ltv_workflow_fig.update_layout(
    title=dict(
        text='<b>Average LTV by Workflow Pattern</b><br><sub>Which workflows create the highest long-term value</sub>',
        font=dict(size=15, color=_txt_ltv)
    ),
    template='plotly_dark', paper_bgcolor=_bg_ltv, plot_bgcolor=_plt_bg,
    font=dict(color=_txt_ltv, family='Arial', size=12),
    height=450, width=900, showlegend=False
)
ltv_workflow_fig.show()
print("  ✓ VIZ C: LTV by Workflow Pattern")

# --- VIZ D: Unit Economics Dashboard ---
_tc = _ltv_df['ltv_tier'].value_counts().reindex(_tier_order).fillna(0)
_tr = [float(_ltv_df[_ltv_df['ltv_tier'] == _t]['retention_probability'].mean() * 100) for _t in _tier_order]
_ta = [float(_ltv_df[_ltv_df['ltv_tier'] == _t]['ltv_score'].mean()) for _t in _tier_order]

ltv_unit_economics_fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=('User Tier Distribution', 'Avg Retention % by Tier', 'Avg LTV Score by Tier'),
    specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]],
    horizontal_spacing=0.10
)

for _ci_v, (_vals, _tmpl) in enumerate([
    (_tc.values.tolist(), '<b>%{x}</b><br>Users: %{y:,}<extra></extra>'),
    (_tr, '<b>%{x}</b><br>Retention: %{y:.1f}%<extra></extra>'),
    (_ta, '<b>%{x}</b><br>Avg LTV: %{y:.1f}<extra></extra>'),
], 1):
    ltv_unit_economics_fig.add_trace(go.Bar(
        x=_tier_order, y=_vals,
        marker=dict(color=_tier_colors),
        hovertemplate=_tmpl, showlegend=False
    ), row=1, col=_ci_v)

for _ci_v in range(1, 4):
    ltv_unit_economics_fig.update_xaxes(tickfont=dict(color=_txt_ltv, size=10), row=1, col=_ci_v)
    ltv_unit_economics_fig.update_yaxes(tickfont=dict(color=_txt_ltv, size=10),
                                         gridcolor=_grd_ltv, zeroline=False, row=1, col=_ci_v)

ltv_unit_economics_fig.update_layout(
    title=dict(
        text='<b>Unit Economics Dashboard</b><br><sub>Tier composition, retention probability, and LTV scores by user segment</sub>',
        font=dict(size=15, color=_txt_ltv)
    ),
    template='plotly_dark', paper_bgcolor=_bg_ltv, plot_bgcolor=_plt_bg,
    font=dict(color=_txt_ltv, family='Arial', size=11),
    height=480, width=1200, showlegend=False
)
ltv_unit_economics_fig.show()
print("  ✓ VIZ D: Unit Economics Dashboard")

# ============================================================================
# PART 5: EXPORT & SUMMARY
# ============================================================================
print("\n[5] Exporting LTV predictions...")

ltv_predictions = _ltv_df[[
    'distinct_id', 'user_tier', 'ltv_tier', 'ltv_score',
    'ltv_engagement_score', 'retention_probability',
    'success_score', 'churn_proxy',
    'success_tier', 'engagement_segment', 'workflow_pattern',
    'total_events', 'active_days', 'tenure_days', 'is_power_user',
    'is_network_hub', 'has_deployment', 'monetization_segment'
]].copy()
ltv_predictions['ltv_engagement_score'] = ltv_predictions['ltv_engagement_score'].round(4)
ltv_predictions['retention_probability'] = ltv_predictions['retention_probability'].round(4)

print(f"  LTV predictions shape: {ltv_predictions.shape}")

ltv_summary_stats = {
    'total_users': len(ltv_predictions),
    'platinum_users': int((ltv_predictions['ltv_tier'] == 'Platinum').sum()),
    'gold_users': int((ltv_predictions['ltv_tier'] == 'Gold').sum()),
    'silver_users': int((ltv_predictions['ltv_tier'] == 'Silver').sum()),
    'bronze_users': int((ltv_predictions['ltv_tier'] == 'Bronze').sum()),
    'avg_ltv_score': round(float(ltv_predictions['ltv_score'].mean()), 2),
    'avg_retention_prob': round(float(ltv_predictions['retention_probability'].mean()), 4),
    'pct_high_churn_risk': round(float((ltv_predictions['churn_proxy'] >= 65).mean() * 100), 2),
    'top_10pct_ltv_threshold': round(float(ltv_predictions['ltv_score'].quantile(0.90)), 2),
    'segment_economics': _seg_groups.to_dict(orient='records'),
}

print("\n" + "=" * 80)
print("LTV PREDICTION SUMMARY")
print("=" * 80)
_n = ltv_summary_stats['total_users']
print(f"\n  Total Users: {_n:,}")
print(f"  Platinum (LTV ≥ 70): {ltv_summary_stats['platinum_users']:,} ({ltv_summary_stats['platinum_users']/_n*100:.1f}%)")
print(f"  Gold (LTV 45–69):    {ltv_summary_stats['gold_users']:,} ({ltv_summary_stats['gold_users']/_n*100:.1f}%)")
print(f"  Silver (LTV 20–44):  {ltv_summary_stats['silver_users']:,} ({ltv_summary_stats['silver_users']/_n*100:.1f}%)")
print(f"  Bronze (LTV < 20):   {ltv_summary_stats['bronze_users']:,} ({ltv_summary_stats['bronze_users']/_n*100:.1f}%)")
print(f"\n  Avg LTV Score: {ltv_summary_stats['avg_ltv_score']:.2f}")
print(f"  Avg Retention Prob: {ltv_summary_stats['avg_retention_prob']:.2%}")
print(f"  % High Churn Risk (≥65): {ltv_summary_stats['pct_high_churn_risk']:.1f}%")
print(f"  Top 10% LTV Threshold: {ltv_summary_stats['top_10pct_ltv_threshold']:.1f}")

# Top 20 highest-LTV users
_top_ltv = ltv_predictions.nlargest(20, 'ltv_score')[[
    'distinct_id', 'ltv_tier', 'ltv_score', 'retention_probability',
    'success_score', 'user_tier', 'engagement_segment', 'workflow_pattern', 'total_events'
]]
print(f"\n  Top 20 Highest-LTV Users:")
print(_top_ltv.to_string(index=False))

# High-value at-risk users (critical retention targets) — high LTV AND high
# churn risk. Filter on churn_proxy so the comparison direction matches
# the variable name; previous filter on success_score >= 50 picked the
# *least* at-risk among the high-LTV cohort. See docs/repo_state_and_next_steps.md §6.
ltv_at_risk = ltv_predictions[
    (ltv_predictions['ltv_score'] >= ltv_predictions['ltv_score'].quantile(0.70)) &
    (ltv_predictions['churn_proxy'] >= 50)
].sort_values('ltv_score', ascending=False)

print(f"\n  ⚠️  HIGH-VALUE AT-RISK USERS (LTV top 30% + Churn ≥50%): {len(ltv_at_risk):,}")
if len(ltv_at_risk) > 0:
    print(ltv_at_risk[['distinct_id', 'ltv_score', 'churn_proxy', 'user_tier',
                        'engagement_segment', 'total_events']].head(10).to_string(index=False))

print("\n" + "=" * 80)
print("✓✓✓ LTV PREDICTION & UNIT ECONOMICS COMPLETE ✓✓✓")
print("=" * 80)
