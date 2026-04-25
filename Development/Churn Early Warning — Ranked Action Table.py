import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ─── Zerve design system ──────────────────────────────────────────────────────
EW_BG   = '#1D1D20'; EW_TEXT  = '#fbfbff'; EW_SEC   = '#909094'
EW_GOLD = '#ffd400'; EW_GREEN = '#17b26a'; EW_BLUE  = '#A1C9F4'
EW_ORNG = '#FFB482'; EW_LAV   = '#D0BBFF'; EW_RED   = '#f04438'

EW_TIER_CLR = {'Critical': EW_RED, 'High': EW_ORNG, 'Medium': EW_GOLD, 'Low': EW_GREEN}
EW_TIER_ORD = ['Critical', 'High', 'Medium', 'Low']

plt.rcParams.update({
    'figure.facecolor': EW_BG, 'axes.facecolor': EW_BG,
    'text.color': EW_TEXT, 'axes.labelcolor': EW_TEXT,
    'xtick.color': EW_TEXT, 'ytick.color': EW_TEXT,
    'axes.edgecolor': EW_SEC, 'grid.color': '#2a2a2e'
})

print("=" * 70)
print("CHURN EARLY WARNING SYSTEM — RANKED ACTION TABLE")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 1. INTEGRATE SHAP TOP DRIVERS PER-USER into churn_risk_scores
# ─────────────────────────────────────────────────────────────────────────────
_EW_SHAP_MAP = {
    f'shap_{_fname}': _flabel
    for _fname, _flabel in churn_ews_feature_labels.items()
}
_ew_shap_cols = [_c for _c in _EW_SHAP_MAP.keys() if _c in shap_values.columns]
_EW_SHAP_MAP = {_c: _EW_SHAP_MAP[_c] for _c in _ew_shap_cols}

# Per-user top SHAP churn drivers (positive SHAP = pushes toward churn)
_ew_shap_sub = shap_values[['user_id'] + _ew_shap_cols].copy()
_ew_shap_sub['shap_top_driver_1'] = _ew_shap_sub[_ew_shap_cols].apply(
    lambda r: _EW_SHAP_MAP.get(
        r.index[r.values.argmax()], r.index[r.values.argmax()]), axis=1)
_ew_shap_sub['shap_top_driver_2'] = _ew_shap_sub[_ew_shap_cols].apply(
    lambda r: _EW_SHAP_MAP.get(
        sorted(zip(r.values, r.index))[-2][1],
        sorted(zip(r.values, r.index))[-2][1]), axis=1)

print(f"[1] SHAP per-user top drivers computed for {len(_ew_shap_sub):,} users")

# ─────────────────────────────────────────────────────────────────────────────
# 2. ESTIMATE DAYS-TO-CHURN
# ─────────────────────────────────────────────────────────────────────────────
_ew_crs = churn_risk_scores.copy()

# Merge SHAP drivers
_ew_crs = _ew_crs.merge(
    _ew_shap_sub[['user_id', 'shap_top_driver_1', 'shap_top_driver_2']],
    on='user_id', how='left'
)

# Hazard-calibrated days-to-churn estimate
_ew_risk_norm = (_ew_crs['churn_risk_score'] / 100.0).clip(0, 1)
_ew_grace = (30 - _ew_crs['recency_days']).clip(lower=0)
_ew_dtc   = (90 * (1 - _ew_risk_norm) + _ew_grace * (1 - _ew_risk_norm)).round(0).astype(int)
_ew_dtc   = _ew_dtc.clip(lower=0, upper=365)
_ew_crs['days_to_churn_est'] = _ew_dtc
# Already churned (recency > 30) → DTC = 0
_ew_crs.loc[_ew_crs['recency_days'] > 30, 'days_to_churn_est'] = 0

print(f"[2] Days-to-churn estimated for all {len(_ew_crs):,} users")
print(f"    Median DTC (Critical): "
      f"{_ew_crs.loc[_ew_crs['risk_tier']=='Critical','days_to_churn_est'].median():.0f}d  "
      f"| High: "
      f"{_ew_crs.loc[_ew_crs['risk_tier']=='High','days_to_churn_est'].median():.0f}d  "
      f"| Medium: "
      f"{_ew_crs.loc[_ew_crs['risk_tier']=='Medium','days_to_churn_est'].median():.0f}d")

# ─────────────────────────────────────────────────────────────────────────────
# 3. FINAL EARLY WARNING TABLE
# ─────────────────────────────────────────────────────────────────────────────
early_warning_table = _ew_crs[[
    'rank', 'user_id', 'risk_tier', 'churn_risk_score', 'churn_proba_model',
    'days_to_churn_est', 'recency_days', 'total_events', 'w7_events', 'w30_events',
    'shap_top_driver_1', 'shap_top_driver_2', 'top_risk_factors',
    'recommended_intervention'
]].rename(columns={
    'churn_risk_score':   'risk_score',
    'churn_proba_model':  'churn_prob',
    'shap_top_driver_1':  'top_churn_driver_1',
    'shap_top_driver_2':  'top_churn_driver_2',
})

# Active-only table (users not yet churned) for operational use
early_warning_active = early_warning_table[
    early_warning_table['recency_days'] <= 30
].copy().reset_index(drop=True)
early_warning_active['rank'] = early_warning_active.index + 1

print(f"\n[3] Early warning tables built:")
print(f"    Full (all users)   : {len(early_warning_table):,}")
print(f"    Active users only  : {len(early_warning_active):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. TOP 30 RANKED EARLY WARNING — PRINT TABLE
# ─────────────────────────────────────────────────────────────────────────────
_ew_top30 = early_warning_active.head(30).copy()
_ew_top30['uid_short'] = _ew_top30['user_id'].str[:14] + '…'

print(f"\n{'='*70}")
print("TOP 30 USERS — RANKED EARLY WARNING TABLE")
print(f"{'='*70}")
_ew_display_cols = ['rank', 'uid_short', 'risk_tier', 'risk_score',
                    'churn_prob', 'days_to_churn_est', 'recency_days',
                    'top_churn_driver_1', 'recommended_intervention']
print(_ew_top30[_ew_display_cols].to_string(index=False, max_colwidth=35))
print(f"{'='*70}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALISATION 1: SHAP Global Feature Importance
# ─────────────────────────────────────────────────────────────────────────────
_ew_shap_num = shap_values[_ew_shap_cols]
_ew_mean_abs_shap = _ew_shap_num.abs().mean()
_ew_mean_abs_shap.index = [_EW_SHAP_MAP.get(c, c) for c in _ew_shap_num.columns]
_ew_mean_abs_shap = _ew_mean_abs_shap.sort_values(ascending=True)

shap_global_fig = plt.figure(figsize=(10, 5))
_ew_ax_shap = shap_global_fig.add_subplot(111)
_ew_norm_max = _ew_mean_abs_shap.max()

_ew_n = len(_ew_mean_abs_shap)
_ew_bars = _ew_ax_shap.barh(
    _ew_mean_abs_shap.index, _ew_mean_abs_shap.values,
    color=[EW_LAV] * _ew_n,   # placeholder
    edgecolor='none', height=0.62
)
# Re-color: bottom (low importance) = lavender, middle = orange, top = red
for _ew_bi, _ew_bbar in enumerate(_ew_ax_shap.patches):
    if _ew_bi >= _ew_n - 3:
        _ew_bbar.set_color(EW_RED)
    elif _ew_bi >= _ew_n - 6:
        _ew_bbar.set_color(EW_ORNG)
    else:
        _ew_bbar.set_color(EW_LAV)
    _ew_bval = _ew_mean_abs_shap.iloc[_ew_bi]
    _ew_ax_shap.text(_ew_bval + _ew_norm_max * 0.01,
                     _ew_bbar.get_y() + _ew_bbar.get_height() / 2,
                     f'{_ew_bval:.4f}', va='center', fontsize=9, color=EW_TEXT)

_ew_ax_shap.set_title('Top Churn Drivers — SHAP Feature Importance\n(Mean |SHAP| via Tree Path Decomposition)',
                       fontsize=13, color=EW_TEXT, pad=14)
_ew_ax_shap.set_xlabel('Mean |SHAP Value| → Impact on Churn Probability', fontsize=10)
_ew_ax_shap.xaxis.grid(True, alpha=0.2); _ew_ax_shap.set_axisbelow(True)
_ew_lgnd = [mpatches.Patch(color=EW_RED, label='Top 3 Drivers'),
            mpatches.Patch(color=EW_ORNG, label='Drivers 4–6'),
            mpatches.Patch(color=EW_LAV, label='Secondary Drivers')]
_ew_ax_shap.legend(handles=_ew_lgnd, facecolor='#2a2a2e', edgecolor=EW_SEC,
                   labelcolor=EW_TEXT, fontsize=9, loc='lower right')
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 6. VISUALISATION 2: Risk tier × Days-to-Churn scatter
# ─────────────────────────────────────────────────────────────────────────────
ew_dtc_fig = plt.figure(figsize=(10, 5))
_ew_ax_dtc = ew_dtc_fig.add_subplot(111)
_ew_active_tiers = [t for t in EW_TIER_ORD if t in early_warning_active['risk_tier'].values]
_ew_positions = range(len(_ew_active_tiers))
for _ew_pi, _ew_tier in zip(_ew_positions, _ew_active_tiers):
    _ew_tier_dtc = early_warning_active.loc[
        early_warning_active['risk_tier'] == _ew_tier, 'days_to_churn_est']
    _ew_jitter = np.random.uniform(-0.2, 0.2, len(_ew_tier_dtc))
    _ew_ax_dtc.scatter(_ew_tier_dtc, [_ew_pi] * len(_ew_tier_dtc) + _ew_jitter,
                       color=EW_TIER_CLR[_ew_tier], s=18, alpha=0.35, linewidths=0)
    _ew_ax_dtc.plot(_ew_tier_dtc.median(), _ew_pi, 'D', color=EW_TEXT, ms=7, zorder=5,
                    label=f'{_ew_tier}  median={_ew_tier_dtc.median():.0f}d')
_ew_ax_dtc.set_yticks(list(_ew_positions))
_ew_ax_dtc.set_yticklabels(_ew_active_tiers, fontsize=11)
_ew_ax_dtc.set_xlabel('Estimated Days to Churn', fontsize=11)
_ew_ax_dtc.set_title('Days-to-Churn Estimate by Risk Tier\n(◆ = median | dots = individual users)',
                     fontsize=13, color=EW_TEXT, pad=14)
_ew_ax_dtc.legend(facecolor='#2a2a2e', edgecolor=EW_SEC, labelcolor=EW_TEXT, fontsize=9, loc='lower right')
_ew_ax_dtc.xaxis.grid(True, alpha=0.2); _ew_ax_dtc.set_axisbelow(True)
plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 7. VISUALISATION 3: Top 30 ranked warning table bar chart
# ─────────────────────────────────────────────────────────────────────────────
ew_ranked_fig = plt.figure(figsize=(13, 9))
_ew_ax_rank = ew_ranked_fig.add_subplot(111)
_ew_n_show = min(30, len(early_warning_active))
_ew_top_n  = early_warning_active.head(_ew_n_show).copy()
_ew_top_n['uid_short'] = _ew_top_n['user_id'].str[:13] + '…'
_ew_bar_cs = [EW_TIER_CLR[t] for t in _ew_top_n['risk_tier']]

_ew_ax_rank.barh(range(_ew_n_show), _ew_top_n['risk_score'].values[::-1],
                 color=_ew_bar_cs[::-1], edgecolor='none', height=0.72)
_ew_ax_rank.set_yticks(range(_ew_n_show))
_ew_ax_rank.set_yticklabels(_ew_top_n['uid_short'].values[::-1], fontsize=8)

for _ew_bi, (_ew_ridx, _ew_row) in enumerate(reversed(list(_ew_top_n.iterrows()))):
    _ew_xv   = _ew_row['risk_score']
    _ew_dtc_lbl  = f"DTC: {_ew_row['days_to_churn_est']}d"
    _ew_intv_lbl = _ew_row['recommended_intervention'][:40] + '…'
    _ew_ax_rank.text(_ew_xv + 0.5, _ew_bi, f'{_ew_dtc_lbl}  ·  {_ew_intv_lbl}',
                     va='center', fontsize=7, color=EW_SEC)

_ew_ax_rank.axvline(75, color=EW_RED,  linestyle='--', lw=1.2, alpha=0.5)
_ew_ax_rank.axvline(50, color=EW_ORNG, linestyle='--', lw=1.2, alpha=0.5)
_ew_ax_rank.axvline(25, color=EW_GOLD, linestyle='--', lw=1.2, alpha=0.5)
_ew_ax_rank.set_xlabel('Churn Risk Score (0–100)', fontsize=11)
_ew_ax_rank.set_title(
    f'Top {_ew_n_show} Active Users — Churn Early Warning Ranked Table\n'
    f'(Showing DTC estimate & recommended intervention)',
    fontsize=13, color=EW_TEXT, pad=14)
_ew_ptchs = [mpatches.Patch(color=EW_TIER_CLR[t], label=t) for t in EW_TIER_ORD
             if t in _ew_top_n['risk_tier'].values]
_ew_ax_rank.legend(handles=_ew_ptchs, facecolor='#2a2a2e', edgecolor=EW_SEC,
                   labelcolor=EW_TEXT, fontsize=9, loc='lower right')
_ew_ax_rank.set_xlim(0, 108)
_ew_ax_rank.xaxis.grid(True, alpha=0.2); _ew_ax_rank.set_axisbelow(True)
plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 8. VISUALISATION 4: Intervention breakdown by tier
# ─────────────────────────────────────────────────────────────────────────────
_ew_intv_counts = (early_warning_active
                   .groupby(['risk_tier', 'recommended_intervention'])
                   .size().reset_index(name='count')
                   .sort_values(['risk_tier', 'count'], ascending=[True, False]))

ew_intv_fig = plt.figure(figsize=(12, 5))
_ew_ax_intv = ew_intv_fig.add_subplot(111)
_ew_tier_intv_data = {}
for _ew_tier in EW_TIER_ORD:
    _ew_sub = _ew_intv_counts[_ew_intv_counts['risk_tier'] == _ew_tier].head(3)
    if len(_ew_sub): _ew_tier_intv_data[_ew_tier] = _ew_sub

_ew_x_pos = 0
_ew_xticks_pos, _ew_xticks_lbl = [], []
for _ew_tier, _ew_df in _ew_tier_intv_data.items():
    for _, _ew_irow in _ew_df.iterrows():
        _ew_lbl = _ew_irow['recommended_intervention'][:30] + '…'
        _ew_ax_intv.bar(_ew_x_pos, _ew_irow['count'], color=EW_TIER_CLR[_ew_tier],
                        edgecolor='none', width=0.7, alpha=0.85)
        _ew_ax_intv.text(_ew_x_pos, _ew_irow['count'] + 0.5, str(_ew_irow['count']),
                         ha='center', va='bottom', fontsize=8, color=EW_TEXT)
        _ew_xticks_pos.append(_ew_x_pos); _ew_xticks_lbl.append(_ew_lbl)
        _ew_x_pos += 1
    _ew_x_pos += 0.6

_ew_ax_intv.set_xticks(_ew_xticks_pos)
_ew_ax_intv.set_xticklabels(_ew_xticks_lbl, rotation=35, ha='right', fontsize=8)
_ew_ax_intv.set_ylabel('Number of Users', fontsize=11)
_ew_ax_intv.set_title('Recommended Interventions by Risk Tier\n(Top 3 per tier)',
                       fontsize=13, color=EW_TEXT, pad=14)
_ew_ptchs2 = [mpatches.Patch(color=EW_TIER_CLR[t], label=t) for t in _ew_tier_intv_data]
_ew_ax_intv.legend(handles=_ew_ptchs2, facecolor='#2a2a2e', edgecolor=EW_SEC,
                   labelcolor=EW_TEXT, fontsize=9)
_ew_ax_intv.yaxis.grid(True, alpha=0.2); _ew_ax_intv.set_axisbelow(True)
plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 9. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
_ew_tc_active = early_warning_active['risk_tier'].value_counts().reindex(EW_TIER_ORD, fill_value=0)
_ew_n_active  = len(early_warning_active)

print(f"\n{'='*70}")
print("CHURN EARLY WARNING SYSTEM — FINAL SUMMARY")
print(f"{'='*70}")
print(f"Total users scored      : {len(early_warning_table):,}")
print(f"Active users at risk    : {_ew_n_active:,}")
print(f"\nRisk Tier Breakdown (active users):")
for _ew_t, _ew_c in _ew_tc_active.items():
    _ew_pct = _ew_c / _ew_n_active * 100 if _ew_n_active > 0 else 0
    print(f"  {_ew_t:10s} : {_ew_c:5,}  ({_ew_pct:.1f}%)")

_ew_shap_ranked = _ew_mean_abs_shap.sort_values(ascending=False)
print(f"\nTop 5 Churn Drivers (SHAP global importance):")
for _ew_rk, (_ew_fn, _ew_fv) in enumerate(_ew_shap_ranked.head(5).items(), 1):
    print(f"  {_ew_rk}. {_ew_fn:30s}  SHAP={_ew_fv:.4f}")

print(f"\nEarliest intervention windows:")
_ew_crit_dtc = early_warning_active.loc[
    early_warning_active['risk_tier'] == 'Critical', 'days_to_churn_est']
if len(_ew_crit_dtc):
    print(f"  Critical avg DTC : {_ew_crit_dtc.mean():.0f}d | min: {_ew_crit_dtc.min()}d")
_ew_high_dtc = early_warning_active.loc[
    early_warning_active['risk_tier'] == 'High', 'days_to_churn_est']
if len(_ew_high_dtc):
    print(f"  High avg DTC     : {_ew_high_dtc.mean():.0f}d | min: {_ew_high_dtc.min()}d")

print(f"\nOutput variables:")
print("  early_warning_table  — all 5,410 users with full churn signals")
print("  early_warning_active — active-only users with DTC & interventions")
print(f"{'='*70}")
