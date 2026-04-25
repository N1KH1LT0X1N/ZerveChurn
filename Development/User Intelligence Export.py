import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL: SUPPRESS CONFLICTING VARIABLES from parallel upstream branches
# These variables exist in multiple source blocks and cause execution to fail.
# We delete them immediately before any business logic runs.
# ─────────────────────────────────────────────────────────────────────────────

_conflict_list = [
    # From Event Taxonomy & Categorization chain
    'category', 'event', 'event_list', 'categorized_events', 'event_taxonomy',
    'event_categories', 'uncategorized', 'keyword', 'keyword_mapping',
    'category_stats', 'category_df', 'rare_threshold', 'rare_events',
    'important_rare', 'rare_df', 'taxonomy_reference', 'taxonomy_event_categories',
    'taxonomy_all_events', 'all_events', 'total_events', 'event_lower',
    'categorized', 'pct', 'importance',
    
    # From Workflow Stage Mapping chain
    'stage', 'stage_data', 'stage_event_to_stage', 'workflow_stages',
    'all_stage_events', 'total_stage_events', 'workflow_stage_stats',
    'stage_summary_df', 'progression_order', 'workflow_mapping',
    
    # Generic loop/iteration variables
    'label', 'color', 'metric', 'segment', 'window', 'name', 'cat',
    'status', 'stats',
]

for _cv in _conflict_list:
    try:
        exec(f'del {_cv}')
    except NameError:
        pass

# Force delete the most critical ones directly
for _var in ['category', 'event', 'stage', 'label', 'total_events', 'all_events']:
    try:
        exec(f'del {_var}')
    except:
        pass

print("=" * 70)
print("USER INTELLIGENCE EXPORT — Comprehensive Enriched Profile")
print("=" * 70)

# ─── 1. BUILD USER-LEVEL CHURN SCORES FROM EARLY WARNING SYSTEM ──────────────
_crs = churn_risk_scores[[
    'user_id', 'churn_risk_score', 'risk_tier',
    'churn_proba_model', 'top_risk_factors', 'recommended_intervention',
    'recency_days', 'total_events'
]].copy()
print(f"\n[1] churn_risk_scores   : {len(_crs):,} users")

# ─── 2. PULL SEGMENT SIGNALS FROM MASTER SEGMENTS ────────────────────────────
_seg = master_segments[[
    'distinct_id', 'engagement_segment', 'workflow_pattern',
    'adoption_timing', 'activity_pattern', 'consistency', 'monetization_segment'
]].copy()
_seg = _seg.rename(columns={'distinct_id': 'user_id'})
print(f"[2] master_segments     : {len(_seg):,} users")

# ─── 3. BEHAVIORAL DNA CLUSTER FROM K-MEANS ──────────────────────────────────
_bclust = engagement_viz[['distinct_id', 'cluster', 'segment_label']].copy()
_bclust = _bclust.rename(columns={
    'distinct_id':   'user_id',
    'cluster':       'kmeans_cluster_id',
    'segment_label': 'kmeans_label'
})
print(f"[3] behavioral clusters : {len(_bclust):,} users")

# ─── 4. ACQUISITION COHORT & LTV SIGNALS FROM PARQUET ────────────────────────
# Read minimal columns from parquet to derive cohort + LTV proxy
_raw = pd.read_parquet('user_retention.parquet',
                       columns=['distinct_id', 'timestamp'])
_raw['timestamp'] = pd.to_datetime(_raw['timestamp'])
_ref_dt = _raw['timestamp'].max()

_user_agg = _raw.groupby('distinct_id').agg(
    first_ts=('timestamp', 'min'),
    last_ts=('timestamp', 'max'),
    total_n=('timestamp', 'count')
).reset_index()
_user_agg = _user_agg.rename(columns={'distinct_id': 'user_id'})
_user_agg['acq_cohort'] = _user_agg['first_ts'].dt.to_period('M').astype(str)
_user_agg['recency_from_raw'] = (_ref_dt - _user_agg['last_ts']).dt.total_seconds() / 86400
print(f"[4] user cohorts        : {len(_user_agg):,} users")

# ─── 5. MERGE INTO MASTER TABLE ───────────────────────────────────────────────
print("\n[5] Merging into master user intelligence table...")
_ui = _crs.merge(_seg, on='user_id', how='outer')
_ui = _ui.merge(_bclust, on='user_id', how='left')
_ui = _ui.merge(_user_agg[['user_id', 'acq_cohort', 'total_n', 'recency_from_raw']],
                on='user_id', how='left')
print(f"   Merged shape: {_ui.shape}")

# ─── 6. BUILD BEHAVIORAL CLUSTER LABEL (DNA) ──────────────────────────────────
def _behavioral_cluster_label(row):
    _wf  = str(row.get('workflow_pattern', '') or '')
    _eng = str(row.get('engagement_segment', '') or '')
    _act = str(row.get('activity_pattern', '') or '')
    _con = str(row.get('consistency', '') or '')
    _k   = int(row.get('kmeans_cluster_id', 0) or 0)

    if _eng == 'Power Users':
        if 'AI' in _wf:       return f'C{_k}: AI Power Builder'
        if 'Collab' in _wf:   return f'C{_k}: Collaborative Expert'
        if 'Deploy' in _wf:   return f'C{_k}: Deployment Specialist'
        if 'Notebook' in _wf: return f'C{_k}: Notebook Champion'
        return f'C{_k}: Platform Expert'
    if 'AI' in _wf:           cluster_name = 'AI Enthusiast'
    elif 'Collab' in _wf:     cluster_name = 'Team Collaborator'
    elif 'Deploy' in _wf:     cluster_name = 'Builder / Deployer'
    elif 'Notebook' in _wf:   cluster_name = 'Notebook Learner'
    else:                      cluster_name = 'General Explorer'
    if _con == 'Consistent' and _act == 'Weekday User': cluster_name += ' (Regular)'
    elif _act == 'Weekend User':                         cluster_name += ' (Casual)'
    return f'C{_k}: {cluster_name}'

_ui['behavioral_cluster'] = _ui.apply(_behavioral_cluster_label, axis=1)

# ─── 7. COMPUTE LTV ESTIMATE ─────────────────────────────────────────────────
_BASE_MO = 30.0
_DISC_R  = 0.10 / 12

def _compute_ltv(row):
    _cr_raw  = float(row.get('churn_risk_score') or 50.0)
    _paid    = str(row.get('monetization_segment', '') or '').lower()
    _eng     = str(row.get('engagement_segment', '') or '')
    _wf      = str(row.get('workflow_pattern', '') or '')
    _rec     = float(row.get('recency_days') or row.get('recency_from_raw') or 30.0)

    # Churned → LTV = 0
    if _rec > 90 and _cr_raw >= 75:
        return 0.0

    # Monthly revenue proxy
    _mo = _BASE_MO
    if 'paid' in _paid or 'credit' in _paid:  _mo *= 1.80
    if _eng == 'Power Users':                  _mo *= 1.60
    if 'Deploy' in _wf:                        _mo *= 1.30
    if 'Collab' in _wf:                        _mo *= 1.20
    if 'AI' in _wf:                            _mo *= 1.10

    # Survival probability → expected tenure
    _surv_p = max(0.02, 1.0 - _cr_raw / 100.0)
    _lam    = -np.log(max(_surv_p, 0.01)) / 90
    _exp_mo = min((1.0 / _lam) / 30.0, 24.0) if _lam > 0 else 24.0

    # Discounted LTV
    _n   = max(_exp_mo, 0.0)
    _ltv = _mo * (1 - (1 / (1 + _DISC_R)) ** _n) / _DISC_R if _n > 0 else 0.0
    return round(_ltv, 2)

_ui['LTV_estimate'] = _ui.apply(_compute_ltv, axis=1)
print(f"[6] LTV computed for all {len(_ui):,} users")

# ─── 8. ASSEMBLE FINAL EXPORT COLUMNS ────────────────────────────────────────
_ui['cohort']            = _ui['acq_cohort'].fillna('Unknown')
_ui['churn_probability'] = (_ui['churn_proba_model']
                             .fillna(_ui['churn_risk_score'].fillna(50.0) / 100.0)
                             .clip(0, 1).round(4))
_ui['segment']           = _ui['engagement_segment'].fillna('Unknown')
_ui['risk_tier_col']     = _ui['risk_tier'].fillna('Unknown')
_ui['top_churn_reason']  = _ui['top_risk_factors'].fillna('Insufficient activity data')
_ui['recommended_action'] = _ui['recommended_intervention'].fillna(
    'Monitor & nurture: standard product newsletter')

print(f"\n[7] Final enrichment complete: {_ui.shape}")

# ─── 9. CREATE FINAL EXPORT DATAFRAME ────────────────────────────────────────
user_intelligence_export = pd.DataFrame({
    'user_id':            _ui['user_id'],
    'segment':            _ui['segment'],
    'churn_probability':  _ui['churn_probability'],
    'risk_tier':          _ui['risk_tier_col'],
    'cohort':             _ui['cohort'],
    'behavioral_cluster': _ui['behavioral_cluster'].fillna('C0: General Explorer'),
    'LTV_estimate':       _ui['LTV_estimate'].fillna(0.0).round(2),
    'top_churn_reason':   _ui['top_churn_reason'],
    'recommended_action': _ui['recommended_action'],
})

user_intelligence_export = (user_intelligence_export
                             .dropna(subset=['user_id'])
                             .reset_index(drop=True))

print(f"\n[8] Export shape: {user_intelligence_export.shape}")
_nulls = user_intelligence_export.isnull().sum()
print(f"    Null counts: {_nulls[_nulls > 0].to_dict() or 'None'}")

# ─── 10. WRITE CSV ────────────────────────────────────────────────────────────
user_intelligence_export.to_csv('user_intelligence_export.csv', index=False)
_sz = len(user_intelligence_export.to_csv(index=False).encode())
print(f"\n✅ user_intelligence_export.csv saved → "
      f"{len(user_intelligence_export):,} rows | {_sz:,.0f} bytes")

# ─── 11. SUMMARY STATISTICS ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EXPORT SUMMARY STATISTICS")
print("=" * 70)

_exp = user_intelligence_export
print(f"\n📊 Total Users Exported : {len(_exp):,}")

print(f"\n── Risk Tier Distribution ──────────────────────────────────────────")
for _tv in ['Critical', 'High', 'Medium', 'Low', 'Unknown']:
    _tc = (_exp['risk_tier'] == _tv).sum()
    if _tc > 0:
        print(f"   {_tv:.<20} {_tc:>5,}  ({_tc/len(_exp)*100:.1f}%)")

print(f"\n── Avg Churn Probability per Segment ────────────────────────────────")
_seg_churn = (_exp.groupby('segment')['churn_probability']
              .agg(['mean', 'count'])
              .rename(columns={'mean': 'avg_churn_prob', 'count': 'n_users'})
              .sort_values('avg_churn_prob', ascending=False).reset_index())
for _, _r in _seg_churn.iterrows():
    print(f"   {str(_r['segment']):.<28} n={_r['n_users']:>5,}  "
          f"avg_churn={_r['avg_churn_prob']:.4f}")

print(f"\n── Cohort Distribution ─────────────────────────────────────────────")
for _cv, _cc in _exp['cohort'].value_counts().sort_index().items():
    print(f"   {str(_cv):.<20} {_cc:>5,}  ({_cc/len(_exp)*100:.1f}%)")

print(f"\n── Behavioral Cluster Distribution ─────────────────────────────────")
for _bv, _bc in _exp['behavioral_cluster'].value_counts().head(12).items():
    print(f"   {str(_bv):.<40} {_bc:>5,}  ({_bc/len(_exp)*100:.1f}%)")

print(f"\n── LTV Estimate ($) ─────────────────────────────────────────────────")
_ltv = _exp['LTV_estimate']
_la  = _ltv[_ltv > 0]
print(f"   All users  : mean=${_ltv.mean():.2f} | median=${_ltv.median():.2f} | "
      f"portfolio=${_ltv.sum():,.0f}")
if len(_la) > 0:
    print(f"   Active only: mean=${_la.mean():.2f} | "
          f"median=${_la.median():.2f} | p90=${_la.quantile(0.9):.2f}")

print(f"\n── Churn Probability ───────────────────────────────────────────────")
_cp = _exp['churn_probability']
print(f"   mean={_cp.mean():.4f} | median={_cp.median():.4f} | "
      f"p90={_cp.quantile(0.90):.4f} | p99={_cp.quantile(0.99):.4f}")

print(f"\n── Sample Export (first 5 rows) ─────────────────────────────────────")
print(user_intelligence_export.head(5).to_string(index=False))

print("\n" + "=" * 70)
print("✅ USER INTELLIGENCE EXPORT COMPLETE")
print("=" * 70)