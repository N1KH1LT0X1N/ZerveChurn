import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go

print("="*80)
print("INTEGRATED INSIGHTS DASHBOARD")
print("="*80)

# ============================================================================
# SYNTHESIZE ALL ADVANCED ANALYSES
# ============================================================================
print("\n" + "="*80)
print("UNIFIED INSIGHTS SYNTHESIS")
print("="*80)

# Get power users from behavioral fingerprint
power_users_set = set(
    str(uid) for uid in behavioral_fingerprint[
        behavioral_fingerprint['power_user_score'] >= behavioral_fingerprint['power_user_score'].quantile(0.90)
    ]['user_id'].values
)

# Get network hubs from centrality analysis
network_hubs_set = set(
    str(uid) for uid in centrality_network_df[
        centrality_network_df['is_super_connector'] == True
    ]['user_id'].values
)

# Get anomalous users
anomalous_users_set = set(str(uid) for uid in exceptional_users_df['user_id'].values)

print(f"\n📊 KEY SEGMENTS:")
print(f"   • Power Users: {len(power_users_set):,}")
print(f"   • Network Hubs: {len(network_hubs_set):,}")
print(f"   • Anomalous Users: {len(anomalous_users_set):,}")

# Calculate overlaps
power_and_hub = power_users_set & network_hubs_set
power_and_anomaly = power_users_set & anomalous_users_set
hub_and_anomaly = network_hubs_set & anomalous_users_set
all_three = power_users_set & network_hubs_set & anomalous_users_set

print(f"\n🔄 SEGMENT OVERLAPS:")
print(f"   • Power + Hub: {len(power_and_hub):,}")
print(f"   • Power + Anomaly: {len(power_and_anomaly):,}")
print(f"   • Hub + Anomaly: {len(hub_and_anomaly):,}")
print(f"   • All Three: {len(all_three):,}")

# Churn analysis by segment
survival_segments = active_users_survival.copy()
survival_segments['user_id'] = survival_segments['user_id'].astype(str)
survival_segments['is_power'] = survival_segments['user_id'].isin(power_users_set)
survival_segments['is_hub'] = survival_segments['user_id'].isin(network_hubs_set)
survival_segments['is_anomaly'] = survival_segments['user_id'].isin(anomalous_users_set)

segment_stats = []
for seg_name, mask in [
    ('Power Users', survival_segments['is_power']),
    ('Network Hubs', survival_segments['is_hub']),
    ('Anomalies', survival_segments['is_anomaly']),
    ('Regular', ~(survival_segments['is_power'] | survival_segments['is_hub'] | survival_segments['is_anomaly']))
]:
    data = survival_segments[mask]
    if len(data) > 0:
        avg_success = float(data['success_score'].mean())
        high_tier_pct = float((data['success_score'] >= 65).sum() / len(data) * 100)
        segment_stats.append({
            'Segment': seg_name,
            'Count': int(len(data)),
            'Avg Success': round(avg_success, 2),
            'High Tier %': round(high_tier_pct, 2)
        })

segment_stats_df = pd.DataFrame(segment_stats)
print(f"\n📊 SUCCESS SCORE (LTV PROXY) BY SEGMENT:")
print(segment_stats_df.to_string(index=False))

# ============================================================================
# ACTIONABLE INSIGHTS
# ============================================================================
print("\n" + "="*80)
print("KEY ACTIONABLE INSIGHTS")
print("="*80)

# Create unified profiles
all_active = set(str(uid) for uid in active_users_survival['user_id'].values)
_integrated_unified = []

for uid in all_active:
    uch = active_users_survival[active_users_survival['user_id'].astype(str) == uid]
    if len(uch) == 0:
        continue
    uch = uch.iloc[0]
    ubeh = behavioral_fingerprint[behavioral_fingerprint['user_id'].astype(str) == uid]
    unet = centrality_network_df[centrality_network_df['user_id'].astype(str) == uid]

    # `churn_risk` should rise as churn risk rises — read from churn_proxy
    # (added in Churn Risk Scoring & Time-Based Predictions.py) instead of
    # success_score, whose direction is inverted relative to the field name.
    # Falls back to (100 - success_score) if churn_proxy is not yet present.
    _churn_risk_val = (
        float(uch['churn_proxy']) if 'churn_proxy' in uch.index
        else max(0.0, min(100.0, 100.0 - float(uch['success_score'])))
    )
    _integrated_unified.append({
        'user_id': uid,
        'churn_risk': _churn_risk_val,
        'total_events': int(uch['total_events']),
        'days_inactive': float(uch['days_since_last']),
        'is_power': uid in power_users_set,
        'is_hub': uid in network_hubs_set,
        'is_anomaly': uid in anomalous_users_set,
        'power_score': float(ubeh.iloc[0]['power_user_score']) if len(ubeh) > 0 else 0.0,
        'centrality': float(unet.iloc[0]['composite_centrality']) if len(unet) > 0 else 0.0
    })

integrated_dashboard_df = pd.DataFrame(_integrated_unified)

# Generate insights
insights = []

elite_risk = integrated_dashboard_df[
    ((integrated_dashboard_df['is_power']) | (integrated_dashboard_df['is_hub'])) &
    (integrated_dashboard_df['churn_risk'] >= 50)
]
insights.append({'priority': 'CRITICAL', 'finding': f"{len(elite_risk)} elite users at churn risk", 'action': "VIP retention program"})

anomaly_inactive = integrated_dashboard_df[
    (integrated_dashboard_df['is_anomaly']) & (integrated_dashboard_df['days_inactive'] >= 14)
]
insights.append({'priority': 'HIGH', 'finding': f"{len(anomaly_inactive)} anomalous users inactive 14+ days", 'action': "Investigate friction points"})

hubs_no_power = integrated_dashboard_df[
    (integrated_dashboard_df['is_hub']) & (~integrated_dashboard_df['is_power'])
]
insights.append({'priority': 'MEDIUM', 'finding': f"{len(hubs_no_power)} network hubs with low engagement", 'action': "Leverage their networks"})

power_isolated = integrated_dashboard_df[
    (integrated_dashboard_df['is_power']) & (~integrated_dashboard_df['is_hub']) & (integrated_dashboard_df['centrality'] < 0.1)
]
insights.append({'priority': 'MEDIUM', 'finding': f"{len(power_isolated)} power users isolated", 'action': "Encourage collaboration"})

high_potential = integrated_dashboard_df[
    (~integrated_dashboard_df['is_power']) & (~integrated_dashboard_df['is_hub']) &
    (~integrated_dashboard_df['is_anomaly']) &
    (integrated_dashboard_df['total_events'] >= float(integrated_dashboard_df['total_events'].quantile(0.75))) &
    (integrated_dashboard_df['churn_risk'] < 30)
]
insights.append({'priority': 'OPPORTUNITY', 'finding': f"{len(high_potential)} high-potential regular users", 'action': "Nurture to power status"})

print(f"\n🎯 {len(insights)} ACTIONABLE INSIGHTS:\n")
for _idx, ins in enumerate(insights, 1):
    print(f"{_idx}. [{ins['priority']}] {ins['finding']}")
    print(f"   → {ins['action']}\n")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("="*80)
print("CREATING INTEGRATED VISUALIZATIONS")
print("="*80)

# Zerve color system
_bg = '#1D1D20'
_text = '#fbfbff'
_sec = '#909094'
_colors_pal = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF', '#ffd400', '#17b26a']

# -----------------------------------------------------------------------
# VIZ 1: Segment Composition Bar Chart (matplotlib)
# -----------------------------------------------------------------------
_categories = ['Power\nUsers', 'Network\nHubs', 'Anomalous\nUsers',
                'Power+Hub', 'Power+Anomaly', 'Hub+Anomaly', 'All Three']
_values = [
    int(len(power_users_set - network_hubs_set - anomalous_users_set)),
    int(len(network_hubs_set - power_users_set - anomalous_users_set)),
    int(len(anomalous_users_set - power_users_set - network_hubs_set)),
    int(len(power_and_hub - anomalous_users_set)),
    int(len(power_and_anomaly - network_hubs_set)),
    int(len(hub_and_anomaly - power_users_set)),
    int(len(all_three))
]

fig_segments = plt.figure(figsize=(14, 6), facecolor=_bg)
_ax_seg = fig_segments.add_subplot(111)
_ax_seg.set_facecolor(_bg)

_bars_seg = _ax_seg.bar(range(len(_categories)), _values,
                        color=_colors_pal[:len(_categories)],
                        edgecolor=_text, linewidth=1.2, alpha=0.9)

for _b, _v in zip(_bars_seg, _values):
    _h = _b.get_height()
    _ax_seg.text(_b.get_x() + _b.get_width() / 2., _h + max(_values) * 0.01,
                 f'{_v:,}', ha='center', va='bottom', color=_text,
                 fontsize=10, fontweight='bold')

_ax_seg.set_xlabel('User Segments', fontsize=13, color=_text, fontweight='bold')
_ax_seg.set_ylabel('User Count', fontsize=13, color=_text, fontweight='bold')
_ax_seg.set_title('User Segment Composition & Overlaps', fontsize=16,
                   color=_text, fontweight='bold', pad=20)
_ax_seg.set_xticks(range(len(_categories)))
_ax_seg.set_xticklabels(_categories, rotation=0, ha='center', fontsize=10, color=_text)
_ax_seg.tick_params(colors=_text, labelsize=10)
for _spine in ['bottom', 'left']:
    _ax_seg.spines[_spine].set_color(_sec)
for _spine in ['top', 'right']:
    _ax_seg.spines[_spine].set_visible(False)
_ax_seg.grid(True, alpha=0.15, color=_sec, linestyle='--', axis='y')
fig_segments.tight_layout()
print("✓ Created segment composition bar chart")

# -----------------------------------------------------------------------
# VIZ 2: Sankey Diagram — User Flow Across Segments (replaces network graph)
# The Sankey shows how users in the base population distribute across
# Power / Hub / Anomaly labels, then converge to their final overlap category.
# This captures the same segment-membership information as the network graph.
# -----------------------------------------------------------------------

# Node labels (source → intermediary → sink style)
_node_labels = [
    'All Users',        # 0  — source
    'Power Users',      # 1
    'Network Hubs',     # 2
    'Anomalous',        # 3
    'Non-Labelled',     # 4
    # sink overlap categories
    'Power Only',       # 5
    'Hub Only',         # 6
    'Anomaly Only',     # 7
    'Power + Hub',      # 8
    'Power + Anomaly',  # 9
    'Hub + Anomaly',    # 10
    'All Three',        # 11
    'Regular',          # 12
]

# Compute population counts safely
_n_power = int(len(power_users_set))
_n_hub = int(len(network_hubs_set))
_n_anomaly = int(len(anomalous_users_set))
_n_all_users = int(len(behavioral_fingerprint['user_id'].unique()))
# approximate non-labelled (may overlap across sets)
_any_labelled = power_users_set | network_hubs_set | anomalous_users_set
_n_non_labelled = max(0, _n_all_users - int(len(_any_labelled)))

# Overlap sub-counts
_n_all_three = int(len(all_three))
_n_power_hub = int(len(power_and_hub - anomalous_users_set))
_n_power_anom = int(len(power_and_anomaly - network_hubs_set))
_n_hub_anom = int(len(hub_and_anomaly - power_users_set))
_n_power_only = max(0, _n_power - _n_power_hub - _n_power_anom - _n_all_three)
_n_hub_only = max(0, _n_hub - _n_power_hub - _n_hub_anom - _n_all_three)
_n_anom_only = max(0, _n_anomaly - _n_power_anom - _n_hub_anom - _n_all_three)
_n_regular = _n_non_labelled

# Sankey source→target→value lists
_src = []
_tgt = []
_val = []

# All Users → primary labels
_src += [0, 0, 0, 0]
_tgt += [1, 2, 3, 4]
_val += [_n_power, _n_hub, _n_anomaly, _n_non_labelled]

# Primary labels → overlap sinks
# Power → subgroups
if _n_power_only > 0:
    _src.append(1); _tgt.append(5); _val.append(_n_power_only)
if _n_power_hub > 0:
    _src.append(1); _tgt.append(8); _val.append(_n_power_hub)
if _n_power_anom > 0:
    _src.append(1); _tgt.append(9); _val.append(_n_power_anom)
if _n_all_three > 0:
    _src.append(1); _tgt.append(11); _val.append(_n_all_three)

# Hub → subgroups
if _n_hub_only > 0:
    _src.append(2); _tgt.append(6); _val.append(_n_hub_only)
if _n_power_hub > 0:
    _src.append(2); _tgt.append(8); _val.append(_n_power_hub)
if _n_hub_anom > 0:
    _src.append(2); _tgt.append(10); _val.append(_n_hub_anom)
if _n_all_three > 0:
    _src.append(2); _tgt.append(11); _val.append(_n_all_three)

# Anomaly → subgroups
if _n_anom_only > 0:
    _src.append(3); _tgt.append(7); _val.append(_n_anom_only)
if _n_power_anom > 0:
    _src.append(3); _tgt.append(9); _val.append(_n_power_anom)
if _n_hub_anom > 0:
    _src.append(3); _tgt.append(10); _val.append(_n_hub_anom)
if _n_all_three > 0:
    _src.append(3); _tgt.append(11); _val.append(_n_all_three)

# Non-labelled → Regular
if _n_regular > 0:
    _src.append(4); _tgt.append(12); _val.append(_n_regular)

# Node colors
_node_colors = [
    '#909094',   # All Users
    '#A1C9F4',   # Power Users
    '#FFB482',   # Network Hubs
    '#FF9F9B',   # Anomalous
    '#D0BBFF',   # Non-Labelled
    '#A1C9F4',   # Power Only
    '#FFB482',   # Hub Only
    '#FF9F9B',   # Anomaly Only
    '#8DE5A1',   # Power + Hub
    '#ffd400',   # Power + Anomaly
    '#17b26a',   # Hub + Anomaly
    '#f04438',   # All Three
    '#D0BBFF',   # Regular
]

fig_network = go.Figure(data=[go.Sankey(
    arrangement='snap',
    node=dict(
        pad=20,
        thickness=20,
        line=dict(color='rgba(251,251,255,0.3)', width=0.5),
        label=_node_labels,
        color=_node_colors,
        hovertemplate='%{label}: %{value:,} users<extra></extra>'
    ),
    link=dict(
        source=_src,
        target=_tgt,
        value=_val,
        color='rgba(144,144,148,0.25)'
    )
)])

fig_network.update_layout(
    title=dict(
        text='User Segment Flow — From Population to Overlap Categories',
        font=dict(size=17, color=_text, family='Arial'),
        x=0.5
    ),
    paper_bgcolor=_bg,
    plot_bgcolor=_bg,
    font=dict(color=_text, size=12, family='Arial'),
    height=580,
    margin=dict(l=30, r=30, t=60, b=30)
)

print("✓ Created Sankey diagram (segment flow, replaces network graph)")

# -----------------------------------------------------------------------
# VIZ 3: Churn Risk by Segment — grouped bar (matplotlib)
# -----------------------------------------------------------------------
if len(segment_stats_df) > 0:
    fig_churn = plt.figure(figsize=(12, 6), facecolor=_bg)
    _ax_churn = fig_churn.add_subplot(111)
    _ax_churn.set_facecolor(_bg)

    _seg_names = segment_stats_df['Segment'].tolist()
    _x = np.arange(len(_seg_names))
    _w = 0.35
    _avg_risks = segment_stats_df['Avg Success'].astype(float).tolist()
    _high_risks = segment_stats_df['High Tier %'].astype(float).tolist()

    _bars1 = _ax_churn.bar(_x - _w / 2, _avg_risks, _w, label='Avg Success',
                            color='#A1C9F4', alpha=0.9, edgecolor=_text, linewidth=0.8)
    _bars2 = _ax_churn.bar(_x + _w / 2, _high_risks, _w, label='High Tier %',
                            color='#FF9F9B', alpha=0.9, edgecolor=_text, linewidth=0.8)

    _ax_churn.set_xticks(_x)
    _ax_churn.set_xticklabels(_seg_names, rotation=15, ha='right', fontsize=11, color=_text)
    _ax_churn.set_ylabel('Score / Percentage', fontsize=12, color=_text, fontweight='bold')
    _ax_churn.set_title('Success Score (LTV Proxy) Profile by User Segment', fontsize=15,
                         color=_text, fontweight='bold', pad=15)
    _ax_churn.tick_params(colors=_text)
    _ax_churn.legend(fontsize=10, labelcolor=_text, facecolor=_bg, edgecolor=_sec)
    for _spine in ['bottom', 'left']:
        _ax_churn.spines[_spine].set_color(_sec)
    for _spine in ['top', 'right']:
        _ax_churn.spines[_spine].set_visible(False)
    _ax_churn.grid(True, alpha=0.12, axis='y', color=_sec, linestyle='--')
    fig_churn.tight_layout()
    print("✓ Created churn risk by segment grouped bar chart")

print("\n" + "="*80)
print("✅ INTEGRATED DASHBOARD COMPLETE")
print("="*80)
