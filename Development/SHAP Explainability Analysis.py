
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("SHAP EXPLAINABILITY ANALYSIS — CHURN PREDICTION")
print("=" * 80)

# ============================================================================
# 1. SOURCE MODEL & FEATURES FROM CHURN EARLY WARNING SYSTEM
# ============================================================================
print("\n📦 Sourcing RF + feature matrix from Churn Early Warning System ...")

_shap_model = churn_ews_rf_model
_feature_names = list(churn_ews_feature_names)

_X_all = churn_ews_feature_matrix[_feature_names].copy().astype(float).fillna(0)
_user_ids_arr = churn_ews_feature_matrix['user_id'].values
_X_np = _X_all.values.astype(np.float64)
_X_all_scaled = _X_all  # EWS RF was trained on raw (unscaled) features

print(f"✓ Loaded EWS Random Forest model ({len(_shap_model.estimators_)} trees)")
print(f"✓ Feature matrix: {_X_all.shape}")
print(f"✓ Features: {_feature_names}")

# ============================================================================
# 2. COMPUTE TREE-BASED SHAP VALUES (from scratch using sklearn tree internals)
#    Method: Per-tree, per-sample contribution via tree path traversal.
#    Reference: Saabas (2014) — "Interpreting random forests" — TreeExplainer
#    approximation via mean change in node values along decision path.
# ============================================================================
print("\n🔢 Computing tree-path SHAP contributions for all users ...")

def _compute_tree_shap(tree, X_data):
    """
    Compute per-sample, per-feature contributions for a single decision tree.
    Uses the path-based decomposition: contribution of feature j for sample i
    is the sum of (node_value_after - node_value_before) at each split on j.
    """
    _n_samples, _n_feat = X_data.shape
    _contribs = np.zeros((_n_samples, _n_feat), dtype=np.float64)
    
    _tree_obj = tree.tree_
    # Node values: shape (n_nodes, n_outputs, n_classes)
    # For binary classification, class 1 probability at each node
    _n_samples_node = _tree_obj.n_node_samples
    _node_vals = _tree_obj.value[:, 0, 1] / _tree_obj.n_node_samples  # P(churn) at each node
    
    _node_indicator = tree.decision_path(X_data)  # sparse CSR matrix
    
    for _i in range(_n_samples):
        _node_path = _node_indicator[_i].indices
        for _k in range(len(_node_path) - 1):
            _parent = _node_path[_k]
            _child = _node_path[_k + 1]
            _feat_j = _tree_obj.feature[_parent]
            if _feat_j >= 0:  # not leaf
                _contribs[_i, _feat_j] += _node_vals[_child] - _node_vals[_parent]
    
    return _contribs

# Aggregate SHAP contributions across all trees in the forest
_n_trees = len(_shap_model.estimators_)
_shap_matrix = np.zeros((_X_np.shape[0], len(_feature_names)), dtype=np.float64)

for _t_idx, _estimator in enumerate(_shap_model.estimators_):
    if _t_idx % 50 == 0:
        print(f"  Processing tree {_t_idx+1}/{_n_trees} ...")
    _shap_matrix += _compute_tree_shap(_estimator, _X_np)

_shap_matrix /= _n_trees  # Average across trees

print(f"✓ SHAP matrix shape: {_shap_matrix.shape}")

# Churn probabilities for all users
_churn_proba_all = _shap_model.predict_proba(_X_all_scaled)[:, 1]

# ============================================================================
# 3. BUILD OUTPUT DATAFRAME: shap_values
# ============================================================================
shap_values = pd.DataFrame(
    _shap_matrix,
    columns=[f'shap_{c}' for c in _feature_names]
)
shap_values.insert(0, 'user_id', _user_ids_arr)
shap_values['churn_probability'] = _churn_proba_all

print(f"\n✅ shap_values DataFrame: {shap_values.shape[0]:,} users × {shap_values.shape[1]} columns")
print(shap_values.head(3).to_string())

# ============================================================================
# 4. PLOT 1 — GLOBAL FEATURE IMPORTANCE (mean |SHAP|)
# ============================================================================
_bg = '#1D1D20'
_text = '#fbfbff'
_sec = '#909094'
_palette = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF',
            '#ffd400', '#1F77B4', '#9467BD', '#E377C2']

_mean_abs_shap = np.abs(_shap_matrix).mean(axis=0)
_feat_imp_df = pd.DataFrame({
    'feature': _feature_names,
    'mean_abs_shap': _mean_abs_shap
}).sort_values('mean_abs_shap', ascending=True).reset_index(drop=True)

shap_importance_fig = plt.figure(figsize=(10, 6), facecolor=_bg)
_ax_imp = shap_importance_fig.add_subplot(111)
_ax_imp.set_facecolor(_bg)

_bars = _ax_imp.barh(
    _feat_imp_df['feature'],
    _feat_imp_df['mean_abs_shap'],
    color=[_palette[_i % len(_palette)] for _i in range(len(_feat_imp_df))],
    edgecolor='none', height=0.65
)
for _b in _bars:
    _w = _b.get_width()
    _ax_imp.text(_w + max(_feat_imp_df['mean_abs_shap']) * 0.01,
                 _b.get_y() + _b.get_height() / 2,
                 f'{_w:.4f}', va='center', ha='left', color=_text, fontsize=9)

_ax_imp.set_xlabel('Mean |SHAP Value| (average impact on churn probability)', color=_sec, fontsize=10)
_ax_imp.set_title('Global Feature Importance — Churn Prediction\n(Mean Absolute SHAP Values via Tree Path Decomposition)',
                   color=_text, fontsize=13, fontweight='bold', pad=15)
_ax_imp.tick_params(colors=_text, labelsize=10)
for _spine in _ax_imp.spines.values():
    _spine.set_edgecolor('#333337')
_ax_imp.set_xlim(0, _feat_imp_df['mean_abs_shap'].max() * 1.18)
_ax_imp.grid(axis='x', color='#333337', linewidth=0.5, alpha=0.6)
plt.tight_layout(pad=1.5)
print("\n✅ Global feature importance chart created")

# ============================================================================
# 5. PLOT 2 — SHAP BEESWARM (dot plot by feature + value)
# ============================================================================
_n_disp = min(500, len(_shap_matrix))
_sample_idx = np.random.choice(len(_shap_matrix), _n_disp, replace=False)
_shap_sample = _shap_matrix[_sample_idx]
_X_sample = _X_np[_sample_idx]
_order = np.argsort(_mean_abs_shap)
_cmap = plt.cm.RdBu_r

shap_beeswarm_fig = plt.figure(figsize=(11, 7), facecolor=_bg)
_ax_bee = shap_beeswarm_fig.add_subplot(111)
_ax_bee.set_facecolor(_bg)

for _yi, _fi in enumerate(_order):
    _sv = _shap_sample[:, _fi]
    _fv_raw = _X_sample[:, _fi]
    _fv_norm = (_fv_raw - _fv_raw.min()) / (np.ptp(_fv_raw) + 1e-9)
    _dot_colors = _cmap(_fv_norm)
    _jitter = np.random.uniform(-0.35, 0.35, len(_sv))
    _ax_bee.scatter(_sv, _yi + _jitter, c=_dot_colors, s=7, alpha=0.55,
                    linewidths=0, rasterized=True)

_ax_bee.set_yticks(range(len(_order)))
_ax_bee.set_yticklabels([_feature_names[_fi] for _fi in _order], color=_text, fontsize=10)
_ax_bee.axvline(0, color='#555559', linewidth=1, linestyle='--')
_ax_bee.set_xlabel('SHAP Value (impact on churn probability)', color=_sec, fontsize=10)
_ax_bee.set_title('SHAP Beeswarm — Feature Impact Distribution\n(Colored by feature value: Blue=Low, Red=High)',
                   color=_text, fontsize=13, fontweight='bold', pad=15)
_ax_bee.tick_params(axis='x', colors=_text, labelsize=9)
for _spine in _ax_bee.spines.values():
    _spine.set_edgecolor('#333337')
_ax_bee.grid(axis='x', color='#333337', linewidth=0.4, alpha=0.5)

_sm = plt.cm.ScalarMappable(cmap=_cmap, norm=plt.Normalize(0, 1))
_sm.set_array([])
_cbar = shap_beeswarm_fig.colorbar(_sm, ax=_ax_bee, fraction=0.02, pad=0.01)
_cbar.set_label('Feature Value (Low → High)', color=_sec, fontsize=9)
_cbar.ax.yaxis.set_tick_params(color=_sec)
plt.setp(_cbar.ax.yaxis.get_ticklabels(), color=_sec)
_cbar.outline.set_edgecolor('#333337')
plt.tight_layout(pad=1.5)
print("✅ Beeswarm plot created")

# ============================================================================
# 6. PLOT 3 — PER-SEGMENT SHAP SUMMARY
# ============================================================================
_shap_feat_cols = [f'shap_{f}' for f in _feature_names]
_seg_df = churn_risk_scores[['user_id', 'risk_tier']].copy()
_shap_with_seg = shap_values.merge(_seg_df, on='user_id', how='left')
_cohort_col = 'risk_tier' if 'risk_tier' in _shap_with_seg else None

if _cohort_col and _shap_with_seg[_cohort_col].notna().sum() > 50:
    _seg_rows = []
    for _coh in sorted(_shap_with_seg[_cohort_col].dropna().unique()):
        _sub = _shap_with_seg.loc[_shap_with_seg[_cohort_col] == _coh, _shap_feat_cols]
        if len(_sub) >= 5:
            _seg_rows.append({'cohort': _coh, **_sub.abs().mean().to_dict()})
    _seg_summary_df = pd.DataFrame(_seg_rows).set_index('cohort')
    _seg_summary_df.columns = _feature_names
    _seg_labels = list(_seg_summary_df.index)
else:
    # Fallback: 4 behavioral cohorts based on churn probability quantiles
    _q_labels = ['Low Risk (Q1)', 'Moderate Risk (Q2)', 'Elevated Risk (Q3)', 'High Risk (Q4)']
    _q_cuts = pd.qcut(shap_values['churn_probability'], q=4, labels=_q_labels)
    _seg_rows = []
    for _ql in _q_labels:
        _sub = shap_values.loc[_q_cuts == _ql, _shap_feat_cols]
        if len(_sub) >= 5:
            _seg_rows.append({'cohort': _ql, **_sub.abs().mean().to_dict()})
    _seg_summary_df = pd.DataFrame(_seg_rows).set_index('cohort')
    _seg_summary_df.columns = _feature_names
    _seg_labels = _q_labels

_n_coh = len(_seg_summary_df)
_seg_colors = ['#8DE5A1', '#A1C9F4', '#FFB482', '#FF9F9B', '#D0BBFF']
_x_pos = np.arange(len(_feature_names))
_bar_w = 0.8 / _n_coh

shap_segment_fig = plt.figure(figsize=(13, 5.5), facecolor=_bg)
_ax_seg = shap_segment_fig.add_subplot(111)
_ax_seg.set_facecolor(_bg)

for _ci, (_coh_nm, _row) in enumerate(_seg_summary_df.iterrows()):
    _offset = (_ci - _n_coh / 2 + 0.5) * _bar_w
    _ax_seg.bar(_x_pos + _offset, _row.values, width=_bar_w * 0.9,
                color=_seg_colors[_ci % len(_seg_colors)], label=str(_coh_nm),
                edgecolor='none', alpha=0.9)

_ax_seg.set_xticks(_x_pos)
_ax_seg.set_xticklabels(_feature_names, rotation=35, ha='right', color=_text, fontsize=9)
_ax_seg.set_ylabel('Mean |SHAP Value|', color=_sec, fontsize=10)
_ax_seg.set_title('Per-Segment SHAP Summary — Which Features Drive Churn in Each Cohort',
                   color=_text, fontsize=13, fontweight='bold', pad=15)
_ax_seg.tick_params(axis='y', colors=_text, labelsize=9)
for _spine in _ax_seg.spines.values():
    _spine.set_edgecolor('#333337')
_ax_seg.grid(axis='y', color='#333337', linewidth=0.4, alpha=0.5)
_ax_seg.legend(facecolor='#2a2a2e', edgecolor='#333337', labelcolor=_text, fontsize=9)
plt.tight_layout(pad=1.5)
print(f"✅ Per-segment SHAP chart created ({_n_coh} cohorts)")

# ============================================================================
# 7. PLOT 4 — TOP-10 HIGHEST-RISK USER INDIVIDUAL EXPLANATIONS
# ============================================================================
_top10 = shap_values.nlargest(10, 'churn_probability').reset_index(drop=True)

shap_top10_fig = plt.figure(figsize=(13, 8), facecolor=_bg)

for _ui, _urow in _top10.iterrows():
    _ax_u = shap_top10_fig.add_subplot(2, 5, _ui + 1)
    _ax_u.set_facecolor(_bg)
    _sv = np.array([_urow[f'shap_{_f}'] for _f in _feature_names])
    _colors_u = ['#FF9F9B' if _v > 0 else '#A1C9F4' for _v in _sv]
    _sorted_idx = np.argsort(np.abs(_sv))[::-1][:6]
    _ax_u.barh(
        [_feature_names[_j] for _j in _sorted_idx[::-1]],
        [_sv[_j] for _j in _sorted_idx[::-1]],
        color=[_colors_u[_j] for _j in _sorted_idx[::-1]],
        edgecolor='none', height=0.65
    )
    _ax_u.axvline(0, color='#555559', linewidth=0.8, linestyle='--')
    _uid_short = str(_urow['user_id'])[:12] + '...' if len(str(_urow['user_id'])) > 12 else str(_urow['user_id'])
    _ax_u.set_title(f"#{_ui+1}: {_uid_short}\nP(churn)={_urow['churn_probability']:.2%}",
                    color=_text, fontsize=7, fontweight='bold', pad=4)
    _ax_u.tick_params(labelsize=6.5, colors=_sec)
    for _spine in _ax_u.spines.values():
        _spine.set_edgecolor('#333337')
    _ax_u.grid(axis='x', color='#333337', linewidth=0.3, alpha=0.5)

shap_top10_fig.suptitle(
    'Individual SHAP Explanations — Top 10 Highest-Risk Users\n'
    '(Red = pushes toward churn, Blue = pushes away from churn)',
    color=_text, fontsize=11, fontweight='bold', y=1.01
)
plt.tight_layout(pad=0.8)
print(f"✅ Individual explanations for top-10 high-risk users created")

# ============================================================================
# 8. FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SHAP ANALYSIS COMPLETE")
print("=" * 80)
print(f"\n📊 Output: shap_values — {shap_values.shape[0]:,} users × {shap_values.shape[1]} columns")
print(f"\n🔑 Global feature ranking (by mean |SHAP|):")
for _rk, _row in _feat_imp_df.iloc[::-1].iterrows():
    _bar = '█' * int(_row['mean_abs_shap'] / _feat_imp_df['mean_abs_shap'].max() * 20)
    print(f"   {_row['feature']:25s} {_bar} {_row['mean_abs_shap']:.4f}")

print(f"\n⚠️  Top 10 highest-risk users avg churn probability: {_top10['churn_probability'].mean():.2%}")
print(f"\nVisualizations produced:")
print("   1. shap_importance_fig — Global feature importance (mean |SHAP|)")
print("   2. shap_beeswarm_fig   — Feature impact beeswarm distribution")
print("   3. shap_segment_fig    — Per-segment/cohort SHAP comparison")
print("   4. shap_top10_fig      — Individual waterfall-style explanations")
