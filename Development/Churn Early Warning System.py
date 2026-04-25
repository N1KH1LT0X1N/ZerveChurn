import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

BG = '#1D1D20'; TEXT = '#fbfbff'; SEC = '#909094'
GOLD = '#ffd400'; GREEN = '#17b26a'; BLUE = '#A1C9F4'
ORANGE = '#FFB482'; LAVENDER = '#D0BBFF'; RED_WARN = '#f04438'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG,
    'text.color': TEXT, 'axes.labelcolor': TEXT,
    'xtick.color': TEXT, 'ytick.color': TEXT,
    'axes.edgecolor': SEC, 'grid.color': '#2a2a2e'
})

print("=" * 70)
print("CHURN EARLY WARNING SYSTEM — Gradient Boosting Ensemble")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD RAW DATA & ENGINEER CHURN FEATURES
# ─────────────────────────────────────────────────────────────────────────────
_cews_raw = pd.read_parquet('user_retention.parquet',
                            columns=['distinct_id', 'event', 'timestamp'])
_cews_raw['timestamp'] = pd.to_datetime(_cews_raw['timestamp'])
_ref = _cews_raw['timestamp'].max()
print(f"\n[1] {len(_cews_raw):,} events loaded | ref date: {_ref.date()}")

_cews_u = _cews_raw.groupby('distinct_id').agg(
    first_ts=('timestamp', 'min'), last_ts=('timestamp', 'max'),
    total_n=('event', 'count'), uniq_n=('event', 'nunique'),
).reset_index()
_cews_u['tenure_d']   = (_ref - _cews_u['first_ts']).dt.total_seconds() / 86400
_cews_u['recency_d']  = (_ref - _cews_u['last_ts']).dt.total_seconds() / 86400
_cews_u['daily_rate'] = _cews_u['total_n'] / _cews_u['tenure_d'].clip(lower=1)

# Rolling 7d / 30d engagement
_cews_daily = (_cews_raw.assign(date=_cews_raw['timestamp'].dt.date)
               .groupby(['distinct_id', 'date']).size()
               .reset_index(name='dn'))
_cews_daily['date'] = pd.to_datetime(_cews_daily['date'])
_cews_daily = _cews_daily.sort_values(['distinct_id', 'date'])

def _cews_roll(grp):
    g = grp.set_index('date')['dn']
    r7  = g.rolling('7D',  min_periods=1).sum().iloc[-1]
    r30 = g.rolling('30D', min_periods=1).sum().iloc[-1]
    r7p = g.rolling('7D',  min_periods=1).sum().iloc[-min(8, len(g))]
    v7s = g.rolling('7D',  min_periods=1).sum().diff()
    return pd.Series({'w7': r7, 'w30': r30,
                      'vel': r7 - r7p, 'decel': int((v7s < 0).sum())})

_cews_roll_df = _cews_daily.groupby('distinct_id').apply(_cews_roll).reset_index()

_cews_sess = (_cews_raw.assign(_d=_cews_raw['timestamp'].dt.date)
              .drop_duplicates(['distinct_id', '_d'])
              .groupby('distinct_id').size().rename('sess_d').reset_index())

_cews_f = (_cews_u
           .merge(_cews_roll_df, on='distinct_id', how='left')
           .merge(_cews_sess,    on='distinct_id', how='left'))

for _c in ['w7', 'w30', 'vel', 'decel']:
    _cews_f[_c] = _cews_f[_c].fillna(0)
_cews_f['sess_d'] = _cews_f['sess_d'].fillna(1)

# Churn-specific engineered features
_cews_f['eng_decay'] = np.where(
    _cews_f['daily_rate'] * 7 > 0,
    np.clip((_cews_f['daily_rate'] * 7 - _cews_f['w7']) / (_cews_f['daily_rate'] * 7), 0, 1),
    0.5
)
_cews_f['sess_freq']    = _cews_f['sess_d'] / _cews_f['tenure_d'].clip(lower=1)
_cews_f['rec_score']    = np.clip(_cews_f['recency_d'] / 90, 0, 1)
_cews_f['feat_brd']     = _cews_f['uniq_n'] / _cews_f['uniq_n'].max()
_cews_f['w7_ratio']     = _cews_f['w7'] / (_cews_f['daily_rate'] * 7 + 0.001)
_cews_f['w30_drop']     = np.clip(1 - _cews_f['w30'] / (_cews_f['daily_rate'] * 30 + 0.001), 0, 1)

N = len(_cews_f)
print(f"[1] Feature engineering complete: {N:,} users, 14 churn signals")

# ─────────────────────────────────────────────────────────────────────────────
# 2. CHURN LABEL  (30-day inactivity = churned)
# ─────────────────────────────────────────────────────────────────────────────
_cews_f['churned'] = (_cews_f['recency_d'] > 30).astype(int)
_cr = _cews_f['churned'].mean()
print(f"\n[2] Churn rate: {_cr:.1%}  ({_cews_f['churned'].sum():,}/{N:,})")

# ─────────────────────────────────────────────────────────────────────────────
# 3. ENSEMBLE MODEL: GradientBoosting + RandomForest + HistGradientBoosting
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.ensemble import (GradientBoostingClassifier,
                               RandomForestClassifier,
                               HistGradientBoostingClassifier)
from sklearn.preprocessing import StandardScaler

_FEATS = ['tenure_d', 'total_n', 'uniq_n', 'daily_rate',
          'w7', 'w30', 'vel', 'decel',
          'eng_decay', 'sess_freq', 'rec_score', 'feat_brd',
          'w7_ratio', 'w30_drop']

_Xc = _cews_f[_FEATS].fillna(0).astype(float)
_yc = _cews_f['churned']
_Xtr, _Xte, _ytr, _yte = train_test_split(
    _Xc, _yc, test_size=0.2, random_state=42, stratify=_yc)

_cw = {0: 1.0, 1: ((_ytr == 0).sum() / max((_ytr == 1).sum(), 1))}

# Model 1: Gradient Boosting (XGBoost-equivalent)
_gb = GradientBoostingClassifier(n_estimators=150, max_depth=4,
    learning_rate=0.08, subsample=0.8, random_state=42)
_gb.fit(_Xtr, _ytr)
_p_gb = _gb.predict_proba(_Xte)[:, 1]
_auc_gb = roc_auc_score(_yte, _p_gb)

# Model 2: HistGradientBoosting (LightGBM-equivalent)
_hgb = HistGradientBoostingClassifier(max_iter=150, max_depth=4,
    learning_rate=0.08, random_state=42, class_weight='balanced')
_hgb.fit(_Xtr, _ytr)
_p_hgb = _hgb.predict_proba(_Xte)[:, 1]
_auc_hgb = roc_auc_score(_yte, _p_hgb)

# Model 3: Random Forest (diversity)
_rf = RandomForestClassifier(n_estimators=150, max_depth=8,
    class_weight='balanced', random_state=42, n_jobs=-1)
_rf.fit(_Xtr, _ytr)
_p_rf = _rf.predict_proba(_Xte)[:, 1]
_auc_rf = roc_auc_score(_yte, _p_rf)

# Weighted ensemble: GB 0.40, HGB 0.40, RF 0.20
_p_ens = 0.40 * _p_gb + 0.40 * _p_hgb + 0.20 * _p_rf
_auc_ens = roc_auc_score(_yte, _p_ens)
_f1_ens  = f1_score(_yte, (_p_ens >= 0.5).astype(int))

print(f"\n[3] Model Performance (test set, 80/20 split):")
print(f"    GradientBoosting (XGB-equiv) AUC: {_auc_gb:.4f}")
print(f"    HistGradBoost    (LGB-equiv) AUC: {_auc_hgb:.4f}")
print(f"    RandomForest                 AUC: {_auc_rf:.4f}")
print(f"    Ensemble weighted 40/40/20   AUC: {_auc_ens:.4f}  |  F1: {_f1_ens:.4f}")

# Score all users
_pa_gb  = _gb.predict_proba(_Xc)[:, 1]
_pa_hgb = _hgb.predict_proba(_Xc)[:, 1]
_pa_rf  = _rf.predict_proba(_Xc)[:, 1]
_pa_ens = 0.40 * _pa_gb + 0.40 * _pa_hgb + 0.20 * _pa_rf

# ─────────────────────────────────────────────────────────────────────────────
# 4. COMPOSITE RISK SCORE  (60% model | 25% engagement decay | 15% recency)
# ─────────────────────────────────────────────────────────────────────────────
_rec_pen = np.clip((_cews_f['recency_d'] - 14) / 60, 0, 1).values
_comp    = 0.60 * _pa_ens + 0.25 * _cews_f['eng_decay'].values + 0.15 * _rec_pen
_cews_f  = _cews_f.copy()
_cews_f['model_prob'] = _pa_ens
_cews_f['risk_score'] = (_comp * 100).round(1)
print(f"\n[4] Composite scores: 60% ensemble | 25% engagement decay | 15% recency")

# ─────────────────────────────────────────────────────────────────────────────
# 5. RISK TIER LABELS
# ─────────────────────────────────────────────────────────────────────────────
def _tier(s):
    if s >= 75: return 'Critical'
    elif s >= 50: return 'High'
    elif s >= 25: return 'Medium'
    return 'Low'

_cews_f['risk_tier'] = _cews_f['risk_score'].apply(_tier)
_TO = ['Critical', 'High', 'Medium', 'Low']
_TC = _cews_f['risk_tier'].value_counts().reindex(_TO, fill_value=0)
print(f"\n[5] Risk tier distribution:")
for _t, _c in _TC.items():
    print(f"    {_t:10s}: {_c:5,}  ({_c/N*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 6. FEATURE IMPORTANCE & TOP RISK FACTORS PER USER
# ─────────────────────────────────────────────────────────────────────────────
_fi_gb  = dict(zip(_FEATS, _gb.feature_importances_))
_fi_hgb_arr = getattr(_hgb, 'feature_importances_', np.ones(len(_FEATS)) / len(_FEATS))
_fi_hgb = dict(zip(_FEATS, _fi_hgb_arr))
_fi_rf  = dict(zip(_FEATS, _rf.feature_importances_))
_fi_ens = {f: 0.40 * _fi_gb[f] + 0.40 * _fi_hgb[f] + 0.20 * _fi_rf[f] for f in _FEATS}
_fi_srt = sorted(_fi_ens.items(), key=lambda x: x[1], reverse=True)

_fl = {
    'eng_decay':  'Engagement decay rate',
    'rec_score':  'Recency score (inactivity)',
    'w30_drop':   '30-day event drop rate',
    'vel':        'Engagement velocity',
    'decel':      'Deceleration periods',
    'w7':         'Last-7d event count',
    'w30':        'Last-30d event count',
    'w7_ratio':   '7d vs expected ratio',
    'daily_rate': 'Historical daily rate',
    'sess_freq':  'Session frequency',
    'tenure_d':   'Tenure (days)',
    'total_n':    'Total events',
    'uniq_n':     'Feature breadth',
    'feat_brd':   'Norm. feature breadth',
}

_fi3 = [f for f, _ in _fi_srt[:3]]
_cews_f['top_risk_factors'] = (
    ' | '.join([_fl.get(f, f) for f in _fi3])
)

# ─────────────────────────────────────────────────────────────────────────────
# 7. RECOMMENDED INTERVENTIONS (rule-based, data-driven)
# ─────────────────────────────────────────────────────────────────────────────
def _ews_intv(row):
    _t = row['risk_tier']; _r = row['recency_d']
    _d = row['eng_decay']; _v = row['vel']; _dc = row['decel']
    if _t == 'Critical':
        if _r > 30:    return "Win-back campaign: personalised email + feature highlight"
        elif _dc >= 3: return "Urgent outreach: account health check + success manager call"
        return "In-app nudge: value reminder & quick-win template"
    elif _t == 'High':
        if _d > 0.7:   return "Engagement drip: showcase underused power features"
        elif _v < 0:   return "Re-activation: share advanced templates & tutorials"
        return "Proactive check-in: offer live onboarding or office hours"
    elif _t == 'Medium':
        if _dc >= 2:   return "Habit-building nudge: daily 5-min workflow challenge"
        return "Personalised tips based on usage pattern"
    return "Monitor & nurture: standard product newsletter"

_cews_f['recommended_intervention'] = _cews_f.apply(_ews_intv, axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# 8. PUBLIC OUTPUT VARIABLES
# ─────────────────────────────────────────────────────────────────────────────
churn_risk_scores = (
    _cews_f[['distinct_id', 'risk_score', 'risk_tier', 'model_prob',
              'eng_decay', 'recency_d', 'total_n', 'w7', 'w30',
              'top_risk_factors', 'recommended_intervention']]
    .rename(columns={
        'distinct_id':   'user_id',
        'risk_score':    'churn_risk_score',
        'model_prob':    'churn_proba_model',
        'eng_decay':     'engagement_decay',
        'recency_d':     'recency_days',
        'total_n':       'total_events',
        'w7':            'w7_events',
        'w30':           'w30_events',
    })
    .sort_values('churn_risk_score', ascending=False)
    .reset_index(drop=True)
)
churn_risk_scores['rank'] = churn_risk_scores.index + 1
top_20_at_risk       = churn_risk_scores.head(20).copy()
top_100_intervention = churn_risk_scores.head(100).copy()

print(f"\n[8] churn_risk_scores: {churn_risk_scores.shape}")
print(f"    Downstream vars: churn_risk_scores | top_20_at_risk | top_100_intervention")

churn_ews_feature_names = list(_FEATS)
churn_ews_feature_matrix = (
    _cews_f[['distinct_id'] + _FEATS]
    .rename(columns={'distinct_id': 'user_id'})
    .reset_index(drop=True)
)
churn_ews_rf_model = _rf
churn_ews_gb_model = _gb
churn_ews_hgb_model = _hgb
churn_ews_ensemble_weights = {'gb': 0.40, 'hgb': 0.40, 'rf': 0.20}
churn_ews_feature_labels = dict(_fl)

print(f"    Public exports for SHAP/explainability:")
print(f"      churn_ews_feature_names   ({len(churn_ews_feature_names)} feats)")
print(f"      churn_ews_feature_matrix  {churn_ews_feature_matrix.shape}")
print(f"      churn_ews_rf_model        (RandomForest, {churn_ews_rf_model.n_estimators} trees)")
print(f"      churn_ews_gb_model / churn_ews_hgb_model / churn_ews_ensemble_weights / churn_ews_feature_labels")

# ─────────────────────────────────────────────────────────────────────────────
# 9. VISUALISATIONS (5 charts) — use _ax prefix for local axes vars
# ─────────────────────────────────────────────────────────────────────────────
_TC_COLORS = {'Critical': RED_WARN, 'High': ORANGE, 'Medium': GOLD, 'Low': GREEN}

# VIZ 1: Risk tier distribution
churn_tier_dist_fig, _ax1 = plt.subplots(figsize=(10, 5))
_tv = [_TC[t] for t in _TO]; _tc2 = [_TC_COLORS[t] for t in _TO]
_bs = _ax1.bar(_TO, _tv, color=_tc2, edgecolor='none', width=0.55)
for _b, _v in zip(_bs, _tv):
    _ax1.text(_b.get_x() + _b.get_width()/2, _b.get_height() + 8,
             f'{_v:,}\n({_v/N*100:.1f}%)', ha='center', va='bottom',
             fontsize=11, color=TEXT, fontweight='bold')
_ax1.set_title('Churn Risk Tier Distribution — All Users', fontsize=14, color=TEXT, pad=14)
_ax1.set_ylabel('Number of Users', fontsize=11)
_ax1.set_ylim(0, max(_tv) * 1.25)
_ax1.yaxis.grid(True, alpha=0.25); _ax1.set_axisbelow(True)
plt.tight_layout(); plt.show()

# VIZ 2: Score distribution with tier thresholds
churn_score_dist_fig, _ax2 = plt.subplots(figsize=(11, 5))
_ax2.hist(churn_risk_scores['churn_risk_score'], bins=40, color=BLUE, edgecolor=BG, alpha=0.85)
_ax2.axvline(75, color=RED_WARN, linestyle='--', lw=1.4, label='Critical (≥75)')
_ax2.axvline(50, color=ORANGE,   linestyle='--', lw=1.4, label='High (≥50)')
_ax2.axvline(25, color=GOLD,     linestyle='--', lw=1.4, label='Medium (≥25)')
_ax2.set_title('Composite Churn Risk Score Distribution', fontsize=14, color=TEXT, pad=14)
_ax2.set_xlabel('Churn Risk Score (0–100)', fontsize=11)
_ax2.set_ylabel('Number of Users', fontsize=11)
_ax2.legend(facecolor='#2a2a2e', edgecolor=SEC, labelcolor=TEXT, fontsize=9)
_ax2.yaxis.grid(True, alpha=0.25); _ax2.set_axisbelow(True)
plt.tight_layout(); plt.show()

# VIZ 3: Feature importance
churn_feat_imp_fig, _ax3 = plt.subplots(figsize=(11, 6))
_fn = [_fl.get(f, f) for f, _ in _fi_srt[:10]]
_fv = [v for _, v in _fi_srt[:10]]
_fnorm = np.array(_fv) / max(_fv) * 100
_fc2 = [BLUE if i < 3 else ORANGE if i < 6 else LAVENDER for i in range(len(_fnorm))]
_fb = _ax3.barh(range(len(_fn)), _fnorm[::-1], color=_fc2[::-1], edgecolor='none', height=0.65)
_ax3.set_yticks(range(len(_fn))); _ax3.set_yticklabels(_fn[::-1], fontsize=10)
_ax3.set_xlabel('Relative Importance (%)', fontsize=11)
_ax3.set_title('Top 10 Churn Feature Importances\n(GradientBoosting + HistGB + RandomForest Ensemble)',
              fontsize=13, color=TEXT, pad=14)
for _b2, _v2 in zip(_fb, _fnorm[::-1]):
    _ax3.text(_v2 + 0.5, _b2.get_y() + _b2.get_height()/2, f'{_v2:.1f}%', va='center', fontsize=9, color=TEXT)
_ax3.xaxis.grid(True, alpha=0.2); _ax3.set_axisbelow(True)
plt.tight_layout(); plt.show()

# VIZ 4: Top 20 at-risk users
churn_top20_fig, _ax4 = plt.subplots(figsize=(13, 7))
_t20 = top_20_at_risk.copy()
_t20['usr'] = _t20['user_id'].str[:12] + '…'
_t20c = [_TC_COLORS[t] for t in _t20['risk_tier']]
_ax4.barh(range(20), _t20['churn_risk_score'][::-1].values,
         color=_t20c[::-1], edgecolor='none', height=0.7)
_ax4.set_yticks(range(20)); _ax4.set_yticklabels(_t20['usr'][::-1].values, fontsize=8.5)
_ax4.set_xlabel('Churn Risk Score (0–100)', fontsize=11)
_ax4.set_title('Top 20 At-Risk Users — Churn Risk Scores', fontsize=14, color=TEXT, pad=14)
_ax4.axvline(75, color=RED_WARN, linestyle='--', lw=1.2, alpha=0.6)
_ax4.axvline(50, color=ORANGE,   linestyle='--', lw=1.2, alpha=0.6)
_ptchs = [mpatches.Patch(color=_TC_COLORS[t], label=t) for t in _TO]
_ax4.legend(handles=_ptchs, facecolor='#2a2a2e', edgecolor=SEC,
           labelcolor=TEXT, fontsize=9, loc='lower right')
_ax4.xaxis.grid(True, alpha=0.2); _ax4.set_axisbelow(True)
plt.tight_layout(); plt.show()

# VIZ 5: Engagement Decay vs Model Probability
churn_scatter_fig, _ax5 = plt.subplots(figsize=(10, 6))
_sm = churn_risk_scores.sample(min(2000, N), random_state=42)
_sc = _ax5.scatter(_sm['churn_proba_model'], _sm['engagement_decay'],
                  c=_sm['churn_risk_score'], cmap='RdYlGn_r', s=20, alpha=0.6, vmin=0, vmax=100)
_cb = plt.colorbar(_sc, ax=_ax5)
_cb.set_label('Churn Risk Score', color=TEXT, fontsize=10)
_cb.ax.yaxis.set_tick_params(color=TEXT)
plt.setp(_cb.ax.yaxis.get_ticklabels(), color=TEXT)
_ax5.set_title('Engagement Decay vs Ensemble Churn Probability\n(colour = Composite Risk Score)',
              fontsize=13, color=TEXT, pad=14)
_ax5.set_xlabel('Ensemble Churn Probability', fontsize=11)
_ax5.set_ylabel('Engagement Decay Score (0–1)', fontsize=11)
_ax5.yaxis.grid(True, alpha=0.2); _ax5.xaxis.grid(True, alpha=0.2); _ax5.set_axisbelow(True)
plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 10. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CHURN EARLY WARNING SYSTEM — FINAL SUMMARY")
print("=" * 70)
print(f"Total users scored   : {N:,}")
print(f"Ensemble ROC-AUC     : {_auc_ens:.4f}  |  F1: {_f1_ens:.4f}")
print(f"30-day churn rate    : {_cr:.1%}")
for _t in _TO:
    _c2 = _TC[_t]
    print(f"  {_t:10s}        : {_c2:,}  ({_c2/N*100:.1f}%)")
print(f"\nTop 3 features: {', '.join([_fl.get(f, f) for f, _ in _fi_srt[:3]])}")
print(f"\nTop 10 at-risk users:")
print(top_20_at_risk[['rank', 'user_id', 'churn_risk_score', 'risk_tier',
                       'engagement_decay', 'recency_days', 'top_risk_factors']
                      ].head(10).to_string(index=False))
print("=" * 70)
print("\nOutput variables: churn_risk_scores | top_20_at_risk | top_100_intervention")
