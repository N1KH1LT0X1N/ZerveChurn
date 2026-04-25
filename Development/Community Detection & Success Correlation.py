import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from scipy import stats

print("="*60)
print("COMMUNITY DETECTION & CONNECTIVITY ANALYSIS")
print("="*60)

# ============================================================================
# RECONSTRUCT user_success_metrics from user_base (available upstream)
# Replicates exact logic from Composite Success Score & Labeling block
# ============================================================================
_ub = user_base.copy()

# Ensure boolean types are clean
for _col in ['long_term_retention', 'is_paid_user', 'has_deployment', 'collaboration_success', 'is_power_user']:
    _ub[_col] = _ub[_col].fillna(False).astype(bool)

# Weighted success score (0-100)
_weights = {
    'long_term_retention': 0.30,
    'is_paid_user': 0.25,
    'has_deployment': 0.20,
    'is_power_user': 0.15,
    'collaboration_success': 0.10,
}

_ub['success_score'] = (
    _ub['long_term_retention'].astype(float) * _weights['long_term_retention'] * 100 +
    _ub['is_paid_user'].astype(float) * _weights['is_paid_user'] * 100 +
    _ub['has_deployment'].astype(float) * _weights['has_deployment'] * 100 +
    _ub['is_power_user'].astype(float) * _weights['is_power_user'] * 100 +
    _ub['collaboration_success'].astype(float) * _weights['collaboration_success'] * 100
)

def _assign_success_label(score):
    if score <= 30:
        return 'Failed'
    elif score <= 60:
        return 'Moderate'
    else:
        return 'Successful'

_ub['success_label'] = _ub['success_score'].apply(_assign_success_label)

# Rebuild user_success_metrics locally (same schema as the original)
_local_success_metrics = _ub[[
    'user_id', 'long_term_retention', 'is_paid_user', 'has_deployment',
    'collaboration_success', 'is_power_user', 'total_events', 'tenure_days',
    'days_since_first', 'days_since_last', 'success_score', 'success_label'
]].copy().reset_index(drop=True)

print(f"✓ Reconstructed success metrics for {len(_local_success_metrics):,} users")
print(f"  Success score range: {_local_success_metrics['success_score'].min():.1f} - {_local_success_metrics['success_score'].max():.1f}")

# ============================================================================
# Community detection using Louvain-like greedy modularity optimization
# ============================================================================
def detect_communities_greedy(adjacency_list):
    """
    Simplified greedy community detection based on local modularity optimization.
    Similar to Louvain algorithm but computationally lightweight.
    """
    node_to_community = {node: i for i, node in enumerate(adjacency_list.keys())}
    communities = {i: [node] for i, node in enumerate(adjacency_list.keys())}
    
    m = sum(len(neighbors) for neighbors in adjacency_list.values()) / 2
    
    improved = True
    iteration = 0
    max_iterations = 20
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for node in adjacency_list.keys():
            current_comm = node_to_community[node]
            neighbors = adjacency_list[node]
            neighbor_comms = Counter([node_to_community[n] for n in neighbors])
            if neighbor_comms:
                best_comm, best_count = neighbor_comms.most_common(1)[0]
                if best_comm != current_comm and best_count > 1:
                    if node in communities.get(current_comm, []):
                        communities[current_comm].remove(node)
                    if not communities.get(current_comm):
                        communities.pop(current_comm, None)
                    communities.setdefault(best_comm, []).append(node)
                    node_to_community[node] = best_comm
                    improved = True
    
    # Clean up and renumber
    final_communities = {}
    comm_id = 0
    for comm_nodes in communities.values():
        if comm_nodes:
            final_communities[comm_id] = comm_nodes
            for node in comm_nodes:
                node_to_community[node] = comm_id
            comm_id += 1
    
    return node_to_community, final_communities

print("\nDetecting communities using greedy modularity optimization...")
_node_to_community, _communities = detect_communities_greedy(network_adjacency_list)

_n_communities = len(_communities)
print(f"Communities detected: {_n_communities}")

_community_sizes = [len(nodes) for nodes in _communities.values()]
print(f"\nCommunity size distribution:")
print(f"  Average: {np.mean(_community_sizes):.1f} users")
print(f"  Median:  {np.median(_community_sizes):.1f} users")
print(f"  Largest: {np.max(_community_sizes)} users")
print(f"  Smallest:{np.min(_community_sizes)} users")

_sorted_comms = sorted(_communities.items(), key=lambda x: len(x[1]), reverse=True)
print(f"\nTop 5 Largest Communities:")
for _ci, (_cid, _members) in enumerate(_sorted_comms[:5]):
    print(f"  Community {_cid}: {len(_members)} members")

# Add community assignments to centrality dataframe
_cnet_df = centrality_network_df.copy()
_cnet_df['community_id'] = _cnet_df['user_id'].map(_node_to_community)

# ============================================================================
# CONNECTIVITY vs SUCCESS CORRELATION
# ============================================================================
print(f"\n{'='*60}")
print("CONNECTIVITY vs SUCCESS CORRELATION")
print(f"{'='*60}")

# Merge centrality with reconstructed success metrics (bulletproof join)
_success_for_merge = _local_success_metrics[
    ['user_id', 'success_score', 'success_label', 'long_term_retention', 'is_power_user']
].copy()

# Enforce types
_success_for_merge['success_score'] = pd.to_numeric(_success_for_merge['success_score'], errors='coerce').fillna(0.0).astype(float)
_success_for_merge['long_term_retention'] = _success_for_merge['long_term_retention'].fillna(False).astype(bool)
_success_for_merge['is_power_user'] = _success_for_merge['is_power_user'].fillna(False).astype(bool)

_connectivity_success = _cnet_df.merge(
    _success_for_merge,
    on='user_id',
    how='left'
)

print(f"\nUsers in correlation analysis: {len(_connectivity_success):,}")
print(f"Users with success metrics: {_connectivity_success['success_score'].notna().sum():,}")

# Calculate correlations
_connectivity_measures = ['degree_centrality', 'betweenness_centrality', 'pagerank', 'composite_centrality']
_success_indicators_bool = ['long_term_retention', 'is_power_user']
_success_indicators_num = ['success_score']

print(f"\n{'Connectivity Metric':<25} | {'Success Indicator':<20} | Correlation | P-value")
print("-" * 80)

_correlation_results = []
for _conn_metric in _connectivity_measures:
    for _success_ind in _success_indicators_num + _success_indicators_bool:
        _col_data = _connectivity_success[_conn_metric].astype(float)
        if _success_ind in _success_indicators_bool:
            _suc_data = _connectivity_success[_success_ind].fillna(False).astype(float)
        else:
            _suc_data = pd.to_numeric(_connectivity_success[_success_ind], errors='coerce')
        
        _valid_mask = _col_data.notna() & _suc_data.notna()
        _valid_data = pd.DataFrame({'conn': _col_data[_valid_mask], 'suc': _suc_data[_valid_mask]})
        
        if len(_valid_data) > 10 and _valid_data['conn'].std() > 0 and _valid_data['suc'].std() > 0:
            _corr, _pval = stats.pearsonr(_valid_data['conn'], _valid_data['suc'])
            _correlation_results.append({
                'connectivity_metric': _conn_metric,
                'success_indicator': _success_ind,
                'correlation': float(_corr),
                'p_value': float(_pval),
                'significant': bool(_pval < 0.05)
            })
            _sig = "*" if _pval < 0.05 else ""
            print(f"{_conn_metric:<25} | {_success_ind:<20} | {_corr:>10.4f} | {_pval:>10.4f} {_sig}")

correlation_results_df = pd.DataFrame(_correlation_results)
print(f"\n✓ Computed {len(_correlation_results)} correlation pairs")

# ============================================================================
# SUPER CONNECTORS vs REGULAR USERS
# ============================================================================
print(f"\n{'='*60}")
print("SUPER CONNECTORS vs REGULAR USERS")
print(f"{'='*60}")

_super_conn = _connectivity_success[_connectivity_success['is_super_connector'] == True].copy()
_regular = _connectivity_success[_connectivity_success['is_super_connector'] == False].copy()

for _grp_name, _grp_df in [("Super Connectors", _super_conn), ("Regular Users", _regular)]:
    _n = len(_grp_df)
    _avg_score = float(_grp_df['success_score'].mean()) if _n > 0 else 0.0
    _ret_rate = float(_grp_df['long_term_retention'].astype(float).mean()) * 100 if _n > 0 else 0.0
    _pu_rate = float(_grp_df['is_power_user'].astype(float).mean()) * 100 if _n > 0 else 0.0
    print(f"\n{_grp_name} (n={_n}):")
    print(f"  Average Success Score:   {_avg_score:.4f}")
    print(f"  Long-term Retention Rate: {_ret_rate:.1f}%")
    print(f"  Power User Rate:          {_pu_rate:.1f}%")

# T-test
if len(_super_conn) > 1 and len(_regular) > 1:
    _sc_scores = _super_conn['success_score'].dropna().astype(float)
    _reg_scores = _regular['success_score'].dropna().astype(float)
    if len(_sc_scores) > 1 and len(_reg_scores) > 1:
        _t_stat, _t_pval = stats.ttest_ind(_sc_scores, _reg_scores)
        print(f"\nT-test (Super Connectors vs Regular Users):")
        print(f"  T-statistic: {_t_stat:.4f}")
        print(f"  P-value:     {_t_pval:.4f}")
        if _t_pval < 0.05:
            print(f"  Result: Significant difference in success scores! ✓")
        else:
            print(f"  Result: No significant difference")

# ============================================================================
# COMMUNITY-LEVEL SUCCESS ANALYSIS
# ============================================================================
_community_success = _connectivity_success.groupby('community_id').agg(
    avg_success_score=('success_score', 'mean'),
    std_success_score=('success_score', 'std'),
    n_users=('success_score', 'count'),
    retention_rate=('long_term_retention', lambda x: x.astype(float).mean()),
    power_user_rate=('is_power_user', lambda x: x.astype(float).mean())
).round(4).sort_values('avg_success_score', ascending=False)

print(f"\n{'='*60}")
print("TOP 5 MOST SUCCESSFUL COMMUNITIES")
print(f"{'='*60}")
print(_community_success.head().to_string())

# Store results for downstream use
community_assignments_df = _connectivity_success.copy()

# ============================================================================
# VISUALIZATION: Correlation Heatmap & Community Success Distribution
# ============================================================================
_bg = '#1D1D20'
_txt = '#fbfbff'
_sec = '#909094'
_colors = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF', '#1F77B4']

if len(correlation_results_df) > 0:
    # Plot 1: Correlation bar chart
    community_corr_fig, _ax_c = plt.subplots(figsize=(12, 6), facecolor=_bg)
    _ax_c.set_facecolor(_bg)
    
    _corr_for_plot = correlation_results_df[correlation_results_df['success_indicator'] == 'success_score'].copy()
    _corr_for_plot = _corr_for_plot.sort_values('correlation', ascending=True)
    
    _bar_colors = [_colors[0] if v >= 0 else '#FF9F9B' for v in _corr_for_plot['correlation']]
    _bars_c = _ax_c.barh(
        _corr_for_plot['connectivity_metric'],
        _corr_for_plot['correlation'].values,
        color=_bar_colors,
        alpha=0.85,
        edgecolor=_txt,
        linewidth=0.8
    )
    
    # Mark significance
    for _bar_item, (_idx, _row) in zip(_bars_c, _corr_for_plot.iterrows()):
        _xval = _row['correlation']
        _lbl = '*' if _row['significant'] else ''
        _ax_c.text(
            _xval + (0.002 if _xval >= 0 else -0.002),
            _bar_item.get_y() + _bar_item.get_height() / 2,
            _lbl, color='#ffd400', va='center', ha='left' if _xval >= 0 else 'right', fontsize=14
        )
    
    _ax_c.axvline(0, color=_sec, linestyle='--', linewidth=1)
    _ax_c.set_xlabel('Pearson Correlation with Success Score', color=_txt, fontsize=11)
    _ax_c.set_title('Network Centrality vs User Success Score\n(* = p < 0.05)', color=_txt, fontsize=13, weight='bold', pad=15)
    _ax_c.tick_params(colors=_txt)
    for _spine in _ax_c.spines.values():
        _spine.set_color(_sec)
    _ax_c.spines['top'].set_visible(False)
    _ax_c.spines['right'].set_visible(False)
    
    plt.tight_layout()

# Plot 2: Success score distribution - Super Connectors vs Regular
community_success_fig, _ax_s = plt.subplots(figsize=(10, 6), facecolor=_bg)
_ax_s.set_facecolor(_bg)

_grp_labels = ['Super\nConnectors', 'Regular\nUsers']
_grp_means = [
    float(_super_conn['success_score'].mean()) if len(_super_conn) > 0 else 0.0,
    float(_regular['success_score'].mean()) if len(_regular) > 0 else 0.0
]
_grp_colors = ['#ffd400', _colors[0]]

_bars_s = _ax_s.bar(_grp_labels, _grp_means, color=_grp_colors, alpha=0.85, edgecolor=_txt, linewidth=1.5, width=0.4)
for _bar_item, _mean_val in zip(_bars_s, _grp_means):
    _ax_s.text(
        _bar_item.get_x() + _bar_item.get_width() / 2,
        _bar_item.get_height() + 0.3,
        f'{_mean_val:.1f}',
        ha='center', va='bottom', color=_txt, fontsize=13, weight='bold'
    )

_ax_s.set_ylabel('Average Success Score (0-100)', color=_txt, fontsize=11)
_ax_s.set_title('Average Success Score: Super Connectors vs Regular Users', color=_txt, fontsize=13, weight='bold', pad=15)
_ax_s.tick_params(colors=_txt)
for _spine in _ax_s.spines.values():
    _spine.set_color(_sec)
_ax_s.spines['top'].set_visible(False)
_ax_s.spines['right'].set_visible(False)
_ax_s.set_ylim(0, max(_grp_means) * 1.25 + 1)

plt.tight_layout()

print(f"\n{'='*60}")
print("✅ COMMUNITY DETECTION & SUCCESS CORRELATION COMPLETE")
print(f"{'='*60}")
print(f"  Communities identified: {_n_communities}")
print(f"  Correlation pairs computed: {len(_correlation_results)}")
print(f"  Network users analyzed: {len(_connectivity_success):,}")
print(f"  Super connectors: {len(_super_conn)}")
print(f"  Regular users: {len(_regular)}")
