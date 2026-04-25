import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
_BG      = '#1D1D20'
_TEXT    = '#fbfbff'
_SEC     = '#909094'
_PALETTE = ['#A1C9F4','#FFB482','#8DE5A1','#FF9F9B','#D0BBFF',
            '#1F77B4','#9467BD','#8C564B','#C49C94','#E377C2']
_GOLD    = '#ffd400'
_GREEN   = '#17b26a'
_RED     = '#f04438'

plt.rcParams.update({
    'figure.facecolor': _BG, 'axes.facecolor': _BG,
    'axes.edgecolor': _SEC, 'text.color': _TEXT,
    'axes.labelcolor': _TEXT, 'xtick.color': _SEC,
    'ytick.color': _SEC, 'grid.color': '#333338',
    'font.family': 'sans-serif',
})

print("=" * 80)
print("CAUSAL IMPACT & ATTRIBUTION ANALYSIS  — Full Pipeline")
print("Methods: PSM-ATT | Difference-in-Differences | IPW Synthetic Control | ITS")
print("=" * 80)

# ============================================================================
# STEP 1: BUILD ANALYSIS DATASET
# ============================================================================
print("\n[STEP 1] Building causal analysis dataset...")

causal_base = survival_data[[
    'user_id', 'churned', 'active_last_30d', 'long_term_retention',
    'tenure_days', 'days_since_first', 'risk_segment', 'engagement_level'
]].copy()

causal_df = causal_base.merge(behavioral_fingerprint, on='user_id', how='inner')
causal_df = causal_df.merge(
    workflow_sequence_df, on='user_id', how='left', suffixes=('', '_wf')
)

_q75 = causal_df['days_since_first'].quantile(0.66)
_q33 = causal_df['days_since_first'].quantile(0.33)
causal_df['adoption_timing'] = np.where(
    causal_df['days_since_first'] >= _q75, 'Early Adopter',
    np.where(causal_df['days_since_first'] <= _q33, 'Late Adopter', 'Mid Adopter')
)
causal_df['retained'] = (causal_df['churned'] == 0).astype(int)

print(f"  Dataset: {causal_df.shape}  "
      f"| Retained: {causal_df['retained'].mean()*100:.1f}%  "
      f"| Churned: {(1-causal_df['retained']).mean()*100:.1f}%")

# ============================================================================
# STEP 2: TREATMENT INDICATORS  (drop degenerate T_habitual / T_collaboration)
# ============================================================================
print("\n[STEP 2] Defining binary treatment indicators...")

_p75_pwr  = causal_df['power_user_score'].quantile(0.75)
_p66_div  = causal_df['sequence_diversity'].quantile(0.66)
# Use ≥ 3 sessions to avoid near-constant T_habitual (q75 session = 1!)
_sess_thr = max(causal_df['total_sessions'].quantile(0.75), 3.0)
_p25_str  = causal_df['struggle_score'].quantile(0.25)

causal_df['T_power_user']        = (causal_df['power_user_score'] >= _p75_pwr).astype(int)
causal_df['T_deep_work']         = (causal_df['deep_work_sessions'] >= 1).astype(int)
causal_df['T_ai_adoption']       = causal_df['has_agent_workflow'].astype(int)
causal_df['T_deployment']        = causal_df['has_deployment_sequence'].astype(int)
causal_df['T_high_diversity']    = (causal_df['sequence_diversity'] >= _p66_div).astype(int)
causal_df['T_dev_loop']          = causal_df['has_create_edit_run'].astype(int)
causal_df['T_multi_session']     = (causal_df['total_sessions'] >= 3).astype(int)
causal_df['T_smooth_experience'] = (causal_df['struggle_score'] <= _p25_str).astype(int)
causal_df['T_habitual']          = (causal_df['total_sessions'] >= _sess_thr).astype(int)
causal_df['T_collaboration']     = causal_df['has_sharing_activity'].astype(int)

causal_treatments_all = {
    'T_power_user':        'Power User Behavior (top 25% power score)',
    'T_deep_work':         'Deep Work Sessions (≥1 focused session)',
    'T_ai_adoption':       'AI/Agent Workflow Adoption',
    'T_deployment':        'Deployment Sequence Completion',
    'T_high_diversity':    'High Behavioral Diversity (top 33%)',
    'T_dev_loop':          'Create-Edit-Run Dev Loop',
    'T_multi_session':     'Multi-Session Engagement (≥3 sessions)',
    'T_smooth_experience': 'Smooth Experience (low struggle score)',
    'T_habitual':          f'Habitual Usage (≥{int(_sess_thr)} sessions)',
    'T_collaboration':     'Collaboration / Sharing Activity',
}

# Filter degenerate treatments: need at least 10 in each arm
causal_treatments = {}
for _t, _d in causal_treatments_all.items():
    _n1, _n0 = causal_df[_t].sum(), (1 - causal_df[_t]).sum()
    if _n1 >= 10 and _n0 >= 10:
        causal_treatments[_t] = _d
    else:
        print(f"  ⚠ Skipping {_t}: {_n1} treated / {_n0} control (degenerate)")

print(f"\n  {len(causal_treatments)} valid treatments:")
for _t, _d in causal_treatments.items():
    _n1 = causal_df[_t].sum()
    print(f"  {_t:<25}: {_n1:>4,} treated ({_n1/len(causal_df)*100:4.1f}%) — {_d}")

# ============================================================================
# STEP 3: CONFOUNDERS & PROPENSITY SCALER
# ============================================================================
print("\n[STEP 3] Defining confounders...")
ci_confounders = [c for c in [
    'tenure_days', 'days_since_first', 'total_events', 'total_sessions',
    'avg_session_length', 'unique_event_types', 'avg_events_per_session', 'error_count'
] if c in causal_df.columns]

for _c in ci_confounders:
    causal_df[_c] = causal_df[_c].fillna(causal_df[_c].median())

_ci_scaler = StandardScaler()
_X_ci_conf = _ci_scaler.fit_transform(causal_df[ci_confounders].values)
print(f"  Confounders: {ci_confounders}")

# ============================================================================
# STEP 4: PSM + ATT + BOOTSTRAP 95% CI
# ============================================================================
print("\n[STEP 4] PSM-ATT with bootstrap 95% CI...")

ci_psm_results = []
np.random.seed(42)

for t_col, t_desc in causal_treatments.items():
    _T = causal_df[t_col].values
    _Y = causal_df['retained'].values
    _n1, _n0 = int(_T.sum()), int((1 - _T).sum())

    # Propensity score
    _lr = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
    _lr.fit(_X_ci_conf, _T)
    _ps = _lr.predict_proba(_X_ci_conf)[:, 1]

    _df_t  = pd.DataFrame({'_ps': _ps, '_T': _T, '_Y': _Y})
    _trt   = _df_t[_df_t['_T'] == 1].reset_index(drop=True)
    _ctrl  = _df_t[_df_t['_T'] == 0].reset_index(drop=True)

    # 1:1 nearest-neighbor matching without replacement
    _mc_idx, _used = [], set()
    _cps = _ctrl['_ps'].values
    for _, _tr in _trt.iterrows():
        _d = np.abs(_cps - _tr['_ps'])
        _d = np.where([i not in _used for i in range(len(_ctrl))], _d, np.inf)
        _bi = np.argmin(_d)
        if _d[_bi] < np.inf:
            _mc_idx.append(_bi); _used.add(_bi)
        else:
            _mc_idx.append(None)

    _valid = [i for i, idx in enumerate(_mc_idx) if idx is not None]
    if len(_valid) < 5:
        continue

    _mt = _trt['_Y'].iloc[_valid].values
    _mc = _ctrl['_Y'].iloc[[_mc_idx[i] for i in _valid]].values
    _att = _mt.mean() - _mc.mean()
    _, _p = mannwhitneyu(_mt, _mc, alternative='two-sided')
    _naive = _Y[_T == 1].mean() - _Y[_T == 0].mean()

    _boots = [
        np.random.choice(_mt, len(_mt), replace=True).mean() -
        np.random.choice(_mc, len(_mc), replace=True).mean()
        for _ in range(500)
    ]
    _clo, _chi = np.percentile(_boots, 2.5), np.percentile(_boots, 97.5)
    _ch = 2 * (np.arcsin(np.sqrt(max(_mt.mean(), 1e-9))) - np.arcsin(np.sqrt(max(_mc.mean(), 1e-9))))

    ci_psm_results.append(dict(
        feature=t_col.replace('T_',''), feature_description=t_desc,
        n_treated=_n1, n_control=_n0, n_matched_pairs=len(_valid),
        retention_treated=round(_mt.mean(),4), retention_control=round(_mc.mean(),4),
        naive_effect=round(_naive,4), causal_effect_size=round(_att,4),
        ci_lower=round(_clo,4), ci_upper=round(_chi,4),
        confidence_interval=f"[{_clo:.4f}, {_chi:.4f}]",
        p_value=round(_p,6), cohen_h=round(_ch,4),
        statistically_significant=bool(_p < 0.05),
    ))
    _sig = "✓" if _p < 0.05 else "✗"
    print(f"  {_sig} {t_col:<25}: ATT={_att:+.4f}  p={_p:.4f}  CI=[{_clo:.4f},{_chi:.4f}]")

ci_psm_df = (pd.DataFrame(ci_psm_results)
               .sort_values('causal_effect_size', ascending=False)
               .reset_index(drop=True))
print(f"\n✓ PSM complete: {len(ci_psm_df)} features")

# ============================================================================
# STEP 5: DIFFERENCE-IN-DIFFERENCES
# ============================================================================
print("\n[STEP 5] Difference-in-Differences (early vs late adopter cohorts)...")
causal_df['_did_group'] = (causal_df['adoption_timing'] == 'Early Adopter').astype(int)

ci_did_results = []
for _feat in ci_psm_df['feature'].apply(lambda x: 'T_' + x):
    if _feat not in causal_df.columns:
        continue
    _tg  = causal_df[causal_df['_did_group'] == 1]
    _cg  = causal_df[causal_df['_did_group'] == 0]
    _y_t1 = _tg[_tg[_feat] == 1]['retained'].mean()
    _y_t0 = _tg[_tg[_feat] == 0]['retained'].mean()
    _y_c1 = _cg[_cg[_feat] == 1]['retained'].mean()
    _y_c0 = _cg[_cg[_feat] == 0]['retained'].mean()
    _did  = (_y_t1 - _y_t0) - (_y_c1 - _y_c0)
    _g1   = _tg[_tg[_feat] == 1]['retained'].values
    _g2   = _cg[_cg[_feat] == 1]['retained'].values
    _dp   = mannwhitneyu(_g1, _g2, alternative='two-sided')[1] if (len(_g1) >= 5 and len(_g2) >= 5) else np.nan
    ci_did_results.append(dict(feature=_feat.replace('T_',''),
                               did_estimate=round(_did,4),
                               p_value_did=round(_dp,6) if not np.isnan(_dp) else np.nan))
    print(f"  DiD {_feat:<25}: {_did:+.4f} (p={_dp:.4f})")

ci_did_df = pd.DataFrame(ci_did_results)

# ============================================================================
# STEP 6: IPW SYNTHETIC CONTROL COUNTERFACTUALS
# ============================================================================
print("\n[STEP 6] IPW Synthetic Control — counterfactual retention estimation...")
ci_sc_results = []

for t_col, _ in causal_treatments.items():
    _T_s = causal_df[t_col].values
    _Y_s = causal_df['retained'].values

    _lr_s = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
    _lr_s.fit(_X_ci_conf, _T_s)
    _ps_s = _lr_s.predict_proba(_X_ci_conf)[:, 1]

    _ctrl_m = _T_s == 0
    _wts    = np.clip(_ps_s[_ctrl_m] / (1 - _ps_s[_ctrl_m] + 1e-9), 0, 10)
    _cf     = np.average(_Y_s[_ctrl_m], weights=_wts)
    _fact   = _Y_s[_T_s == 1].mean()
    _lift   = _fact - _cf

    ci_sc_results.append(dict(
        feature=t_col.replace('T_',''),
        factual_retention=round(_fact,4),
        counterfactual_retention=round(_cf,4),
        causal_lift=round(_lift,4),
        relative_lift_pct=round((_lift / max(_cf,1e-9)) * 100, 2),
    ))
    print(f"  SC {t_col:<25}: factual={_fact:.3f}  counterfactual={_cf:.3f}  lift={_lift:+.4f}")

ci_sc_df = pd.DataFrame(ci_sc_results)

# ============================================================================
# STEP 7: INTERRUPTED TIME SERIES (ITS) — Platform Level, OLS Segmented Reg
# ============================================================================
print("\n[STEP 7] Interrupted Time Series — product change events (OLS segmented regression)...")

# Build daily aggregate from df_mom (full event log)
_its_daily = (
    df_mom.groupby('date').size().reset_index(name='platform_events')
)
_its_daily['date'] = pd.to_datetime(_its_daily['date'])
_its_daily = _its_daily.sort_values('date').reset_index(drop=True)
_its_daily['t'] = np.arange(len(_its_daily))

# Compute daily active users (distinct users)
_its_dau = (
    df_mom.groupby('date')['distinct_id'].nunique().reset_index(name='dau')
)
_its_dau['date'] = pd.to_datetime(_its_dau['date'])
_its_daily = _its_daily.merge(_its_dau, on='date', how='left')

date_min = _its_daily['date'].min()
date_max = _its_daily['date'].max()
date_mid = date_min + (date_max - date_min) / 2
print(f"  Time range: {date_min.date()} → {date_max.date()}")

# Define 3 product change interventions at natural data breakpoints
# Period midpoints based on data span (Sep 2025 – Feb 2026)
_its_interventions = [
    {'name': 'AI Agent Launch',     'date': pd.Timestamp('2025-10-01'),
     'description': 'agent_* events first appear at scale (~Oct 2025)'},
    {'name': 'Onboarding Revamp',   'date': pd.Timestamp('2025-11-15'),
     'description': 'canvas_onboarding_tour events spike mid-period'},
    {'name': 'Deep Work Features',  'date': pd.Timestamp('2026-01-01'),
     'description': 'block_run / execution patterns accelerate into 2026'},
]

# Keep only interventions within data range
_its_interventions = [
    iv for iv in _its_interventions
    if date_min <= iv['date'] <= date_max
]
print(f"  Interventions to analyze: {len(_its_interventions)}")

ci_its_results = []

for _iv in _its_interventions:
    _d0   = _iv['date']
    _idx0 = _its_daily[_its_daily['date'] >= _d0].index.min()
    if _idx0 is None or _idx0 < 5 or _idx0 > len(_its_daily) - 5:
        print(f"  ⚠  '{_iv['name']}' insufficient data around {_d0.date()}")
        continue

    _n       = len(_its_daily)
    _D       = (_its_daily['t'] >= _idx0).astype(int).values   # post-intervention dummy
    _t_post  = np.where(_D == 1, _its_daily['t'].values - _idx0, 0)  # time since intervention
    _y       = _its_daily['dau'].fillna(0).values.astype(float)

    # Segmented OLS: y = b0 + b1*t + b2*D + b3*(t_post)
    _X_its = np.column_stack([np.ones(_n), _its_daily['t'].values, _D, _t_post])
    _beta, _resid, _, _ = np.linalg.lstsq(_X_its, _y, rcond=None)

    # Level change = b2, slope change = b3
    _level_change = _beta[2]
    _slope_change = _beta[3]

    # Counterfactual: no intervention (b2=0, b3=0)
    _y_hat_fact = _X_its @ _beta
    _X_cf       = _X_its.copy(); _X_cf[:, 2] = 0; _X_cf[:, 3] = 0
    _y_hat_cf   = _X_cf @ _beta

    # Effect at end of post period
    _n_post     = int(_D.sum())
    _post_data  = _y[_D == 1]
    _cf_post    = _y_hat_cf[_D == 1]
    _avg_effect = _post_data.mean() - _cf_post.mean()

    # Bootstrap p-value for level change
    np.random.seed(42)
    _null_effects = []
    for _ in range(500):
        _perm_d = np.zeros(_n, dtype=int)
        _perm_idx = np.random.randint(5, _n - 5)
        _perm_d[_perm_idx:] = 1
        _perm_t_post = np.where(_perm_d == 1, _its_daily['t'].values - _perm_idx, 0)
        _X_p = np.column_stack([np.ones(_n), _its_daily['t'].values, _perm_d, _perm_t_post])
        _b_p, _, _, _ = np.linalg.lstsq(_X_p, _y, rcond=None)
        _null_effects.append(_b_p[2])
    _p_its = float(np.mean(np.abs(_null_effects) >= abs(_level_change)))

    _cilo = np.percentile(_null_effects, 2.5)
    _cihi = np.percentile(_null_effects, 97.5)

    ci_its_results.append(dict(
        intervention=_iv['name'],
        intervention_date=str(_d0.date()),
        pre_avg_dau=round(_y[_D == 0].mean(), 1),
        post_avg_dau=round(_y[_D == 1].mean(), 1),
        level_change=round(_level_change, 2),
        slope_change=round(_slope_change, 4),
        avg_causal_effect=round(_avg_effect, 2),
        ci_lower=round(_cilo, 2),
        ci_upper=round(_cihi, 2),
        p_value=round(_p_its, 4),
        is_significant=bool(_p_its < 0.05),
        # store for plotting
        _y=_y, _y_hat_fact=_y_hat_fact, _y_hat_cf=_y_hat_cf,
        _D=_D, _dates=_its_daily['date'].values, _d0=_d0,
    ))
    _s = "✓" if _p_its < 0.05 else "✗"
    print(f"  {_s} ITS '{_iv['name']}' ({_d0.date()}): "
          f"level_Δ={_level_change:+.1f} DAU  p={_p_its:.4f}")

ci_its_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')}
                           for r in ci_its_results])

# ============================================================================
# STEP 8: ASSEMBLE MASTER CAUSAL RESULTS
# ============================================================================
print("\n[STEP 8] Assembling causal_impact_results + causal_effects dataframe...")

causal_impact_results = (
    ci_psm_df
    .merge(ci_sc_df[['feature','factual_retention','counterfactual_retention',
                      'causal_lift','relative_lift_pct']], on='feature', how='left')
    .merge(ci_did_df[['feature','did_estimate']].rename(
               columns={'did_estimate':'did_causal_estimate'}),
           on='feature', how='left')
)

def _consensus(row):
    _ests = [row['causal_effect_size'], row['causal_lift']]
    if pd.notna(row.get('did_causal_estimate')):
        _ests.append(row['did_causal_estimate'])
    return round(np.mean(_ests), 4)

causal_impact_results['consensus_causal_effect'] = causal_impact_results.apply(_consensus, axis=1)
causal_impact_results['causal_rank_score'] = (
    np.abs(causal_impact_results['consensus_causal_effect']) *
    (-np.log10(causal_impact_results['p_value'].clip(1e-10)))
).round(4)
causal_impact_results['confounding_bias'] = (
    causal_impact_results['naive_effect'] - causal_impact_results['causal_effect_size']
).round(4)
causal_impact_results = (causal_impact_results
    .sort_values('causal_rank_score', ascending=False)
    .reset_index(drop=True))

# ─── REQUIRED OUTPUT: causal_effects ──────────────────────────────────────
causal_effects = causal_impact_results[[
    'feature', 'feature_description',
    'causal_effect_size', 'confidence_interval', 'p_value',
    'statistically_significant'
]].rename(columns={
    'causal_effect_size':       'avg_treatment_effect',
    'statistically_significant':'is_causal',
}).reset_index(drop=True)

print(f"\n{'='*95}")
print("CAUSAL EFFECTS — ranked by average treatment effect (PSM ATT)")
print(f"{'='*95}")
print(f"\n{'Rk':<3} {'Feature':<22} {'PSM ATT':>9} {'IPW Lift':>10} {'DiD':>9} "
      f"{'Consensus':>11} {'p-val':>10} {'CI':<22} Sig")
print("─" * 100)
for _ri, _row in causal_impact_results.iterrows():
    _sig = "✓***" if _row['p_value'] < 0.001 else (
           "✓**"  if _row['p_value'] < 0.01  else (
           "✓*"   if _row['p_value'] < 0.05  else "✗   "))
    _did_s = f"{_row['did_causal_estimate']:+.4f}" if pd.notna(_row.get('did_causal_estimate')) else "  N/A "
    print(f"{_ri+1:<3} {_row['feature']:<22} {_row['causal_effect_size']:>+.4f}  "
          f"{_row['causal_lift']:>+.4f}   {_did_s}  {_row['consensus_causal_effect']:>+.4f}   "
          f"{_row['p_value']:>10.6f}  {_row['confidence_interval']:<22} {_sig}")

print("\n[CAUSAL vs NAIVE — Confounding Bias]")
print(f"{'Feature':<22} {'Naive':>9} {'Causal':>9} {'Bias':>9}  Interpretation")
print("─" * 68)
for _, _r in causal_impact_results.iterrows():
    _b = _r['confounding_bias']
    _int = "Overstated" if _b > 0.01 else ("Understated" if _b < -0.01 else "Minimal bias")
    print(f"{_r['feature']:<22} {_r['naive_effect']:>+.4f}   {_r['causal_effect_size']:>+.4f}   {_b:>+.4f}  {_int}")

print("\n[IPW SYNTHETIC CONTROL — What-If Counterfactuals]")
print(f"{'Feature':<22} {'Actual':>8} {'Without Feature':>17} {'Causal Lift':>13}  Relative%")
print("─" * 72)
for _, _r in causal_impact_results.iterrows():
    print(f"{_r['feature']:<22} {_r['retention_treated']:>7.2%}  {_r['counterfactual_retention']:>8.2%}        "
          f"{_r['causal_lift']:>+.4f}     {_r['relative_lift_pct']:>+.1f}%")

if not ci_its_df.empty:
    print("\n[INTERRUPTED TIME SERIES — Platform Change Events]")
    print(f"{'Intervention':<28} {'Date':>12} {'Δ Level DAU':>13} {'Slope Δ':>9} {'p-val':>8}  Sig")
    print("─" * 80)
    for _, _r in ci_its_df.iterrows():
        _s = "✓" if _r['is_significant'] else "✗"
        print(f"{_r['intervention']:<28} {_r['intervention_date']:>12} "
              f"{_r['level_change']:>+12.1f}  {_r['slope_change']:>+8.4f}  "
              f"{_r['p_value']:>7.4f}  {_s}")

_sig_cnt = causal_impact_results['statistically_significant'].sum()
_top     = causal_impact_results.iloc[0]
print(f"\n{'='*95}")
print(f"✓ CONCLUSION: {_sig_cnt}/{len(causal_impact_results)} features have statistically significant causal effects")
print(f"  #1 Causal Driver: '{_top['feature_description']}'")
print(f"     PSM ATT: {_top['causal_effect_size']:+.4f} | IPW Lift: {_top['causal_lift']:+.4f} "
      f"| Consensus: {_top['consensus_causal_effect']:+.4f} | p={_top['p_value']:.4f}")
print(f"  Retention with feature: {_top['retention_treated']:.1%}  |  "
      f"Without (counterfactual): {_top['counterfactual_retention']:.1%}")
print(f"{'='*95}")

# ============================================================================
# VIZ 1: Causal Effects Forest Plot
# ============================================================================
print("\nGenerating visualizations...")

_n  = len(causal_impact_results)
_y  = np.arange(_n)
_fe = causal_impact_results['causal_effect_size'].values
_cl = causal_impact_results['ci_lower'].values
_cu = causal_impact_results['ci_upper'].values
_is_sig = causal_impact_results['statistically_significant'].values
_feat_names = causal_impact_results['feature'].values

causal_forest_fig, _ax = plt.subplots(figsize=(11, max(5, _n * 0.65 + 1.5)))
_ax.set_facecolor(_BG)

for _i in range(_n):
    _c = _GREEN if (_fe[_i] > 0 and _is_sig[_i]) else (_RED if (_fe[_i] < 0 and _is_sig[_i]) else _SEC)
    _ax.plot([_cl[_i], _cu[_i]], [_i, _i], color=_c, lw=2.2, zorder=2)
    _ax.scatter([_fe[_i]], [_i], color=_c, s=80, zorder=3)
    _ax.text(_cu[_i] + 0.005, _i, f" {_fe[_i]:+.3f}", va='center', color=_c, fontsize=8.5)

_ax.axvline(0, color=_SEC, lw=1, ls='--', alpha=0.7)
_ax.set_yticks(_y)
_ax.set_yticklabels(_feat_names, fontsize=9.5, color=_TEXT)
_ax.set_xlabel('Causal Effect on Retention (ATT)', color=_TEXT)
_ax.set_title('PSM Causal Effects Forest Plot\n(Green = sig. positive, Red = sig. negative, Gray = not significant)',
              color=_TEXT, pad=12)
_ax.invert_yaxis()
_ax.grid(axis='x', alpha=0.25, color='#333338')
plt.tight_layout()

# ============================================================================
# VIZ 2: Counterfactual Comparison — Actual vs What-If Retention
# ============================================================================
_pos_n = len(causal_impact_results)
causal_counterfactual_fig, _ax2 = plt.subplots(figsize=(12, max(5, _pos_n * 0.7 + 1.5)))
_ax2.set_facecolor(_BG)

_feats   = causal_impact_results['feature'].values
_actual  = causal_impact_results['retention_treated'].values * 100
_cf_ret  = causal_impact_results['counterfactual_retention'].values * 100
_xpos    = np.arange(_pos_n)

_bw = 0.35
_b1 = _ax2.bar(_xpos - _bw/2, _actual, _bw, color=_PALETTE[0], label='Actual (treated)', alpha=0.9)
_b2 = _ax2.bar(_xpos + _bw/2, _cf_ret,  _bw, color=_PALETTE[1], label='Counterfactual (without feature)', alpha=0.9)

_ax2.set_xticks(_xpos)
_ax2.set_xticklabels(_feats, rotation=40, ha='right', fontsize=9, color=_TEXT)
_ax2.set_ylabel('30-Day Retention Rate (%)', color=_TEXT)
_ax2.set_title('Counterfactual Analysis: Actual vs What-If Retention\n(IPW Synthetic Control)',
               color=_TEXT, pad=12)
_ax2.legend(facecolor='#2a2a2e', labelcolor=_TEXT, edgecolor=_SEC, fontsize=9)
_ax2.grid(axis='y', alpha=0.2)
# Annotate lifts
for _i, (_a, _c, _f) in enumerate(zip(_actual, _cf_ret, _feats)):
    _delta = _a - _c
    _col = _GREEN if _delta > 0 else _RED
    _ax2.text(_i, max(_a, _c) + 0.8, f"{_delta:+.1f}pp", ha='center', va='bottom',
              fontsize=8, color=_col, fontweight='bold')
plt.tight_layout()

# ============================================================================
# VIZ 3: Confusion-bias Heatmap — Naive vs Causal
# ============================================================================
causal_bias_fig, _ax3 = plt.subplots(figsize=(11, max(4, _pos_n * 0.55 + 1.5)))
_ax3.set_facecolor(_BG)
_naive_v  = causal_impact_results['naive_effect'].values * 100
_causal_v = causal_impact_results['causal_effect_size'].values * 100
_bias_v   = _naive_v - _causal_v

_xp3 = np.arange(_pos_n)
_bw3 = 0.28
_ax3.bar(_xp3 - _bw3, _naive_v,  _bw3, color=_PALETTE[4], label='Naive Correlation', alpha=0.85)
_ax3.bar(_xp3,        _causal_v, _bw3, color=_PALETTE[0], label='Causal (PSM ATT)',  alpha=0.85)
_ax3.bar(_xp3 + _bw3, _bias_v,   _bw3, color=_PALETTE[3], label='Confounding Bias',  alpha=0.85)

_ax3.axhline(0, color=_SEC, lw=1, ls='--', alpha=0.5)
_ax3.set_xticks(_xp3)
_ax3.set_xticklabels(_feats, rotation=40, ha='right', fontsize=9, color=_TEXT)
_ax3.set_ylabel('Effect on Retention (%)', color=_TEXT)
_ax3.set_title('Causal vs Naive Correlation — Confounding Bias Detection', color=_TEXT, pad=12)
_ax3.legend(facecolor='#2a2a2e', labelcolor=_TEXT, edgecolor=_SEC, fontsize=9)
_ax3.grid(axis='y', alpha=0.2)
plt.tight_layout()

# ============================================================================
# VIZ 4: ITS — DAU Time Series with Interventions & Counterfactual Curves
# ============================================================================
if ci_its_results:
    _n_its = len(ci_its_results)
    causal_its_fig, _its_axes = plt.subplots(_n_its, 1, figsize=(14, 5 * _n_its),
                                              facecolor=_BG, squeeze=False)

    for _ii, (_r, _axs) in enumerate(zip(ci_its_results, _its_axes.flatten())):
        _axs.set_facecolor(_BG)
        _dates_plot = pd.to_datetime(_r['_dates'])
        _y_plot     = _r['_y']
        _yf         = _r['_y_hat_fact']
        _yc         = _r['_y_hat_cf']
        _D_plot     = _r['_D']
        _d0         = _r['_d0']

        # Raw DAU
        _axs.plot(_dates_plot, _y_plot, color=_PALETTE[0], lw=1.2,
                  alpha=0.6, label='Daily Active Users', zorder=1)
        # Fitted
        _axs.plot(_dates_plot, _yf, color=_PALETTE[0], lw=2.2,
                  alpha=0.9, label='Fitted (factual)', zorder=3)
        # Counterfactual
        _axs.plot(_dates_plot, _yc, color=_PALETTE[1], lw=2, ls='--',
                  alpha=0.9, label='Counterfactual (no intervention)', zorder=3)

        # Shade post-intervention region
        _post_dates = _dates_plot[_D_plot == 1]
        if len(_post_dates) > 0:
            _axs.axvspan(_post_dates[0], _post_dates[-1], alpha=0.07, color=_PALETTE[2])

        # Intervention line
        _axs.axvline(_d0, color=_GOLD, lw=2, ls=':', label=f"Intervention: {_r['intervention']}")

        # Fill between factual and CF
        _axs.fill_between(_dates_plot, _yf, _yc,
                          where=(_D_plot == 1), alpha=0.20, color=_GREEN, label='Causal lift area')

        _sig_str = f"✓ p={_r['p_value']:.3f}" if _r['is_significant'] else f"✗ p={_r['p_value']:.3f}"
        _axs.set_title(
            f"ITS: {_r['intervention']}  |  Level Δ={_r['level_change']:+.1f} DAU  {_sig_str}",
            color=_TEXT, pad=10, fontsize=11)
        _axs.set_ylabel('Daily Active Users', color=_TEXT)
        _axs.legend(facecolor='#2a2a2e', labelcolor=_TEXT, edgecolor=_SEC, fontsize=8.5)
        _axs.grid(alpha=0.2)

    causal_its_fig.suptitle('Interrupted Time Series Analysis — Platform DAU',
                             color=_TEXT, fontsize=13, y=1.01)
    plt.tight_layout()
else:
    causal_its_fig = None

print("\n✅ All causal analysis complete!")
print(f"\ncausal_effects shape: {causal_effects.shape}")
print(causal_effects[['feature','avg_treatment_effect','confidence_interval','p_value','is_causal']].to_string())
