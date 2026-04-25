
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── DESIGN TOKENS ───────────────────────────────────────────────────────────
_BE_BG   = '#1D1D20'
_BE_TEXT = '#fbfbff'
_BE_SEC  = '#909094'
_BE_CLR  = ['#A1C9F4','#FFB482','#8DE5A1','#FF9F9B','#D0BBFF',
             '#1F77B4','#9467BD','#C49C94','#E377C2','#ffd400']

# ─── 1. BUILD MASTER USER TABLE ──────────────────────────────────────────────
be_base = workflow_sequence_df[['user_id']].copy()

be_base = be_base.merge(
    session_patterns_per_user[['user_id','total_sessions','avg_inter_session_gap_hours',
                                'median_inter_session_gap_hours','deep_work_ratio']],
    on='user_id', how='left')

be_base = be_base.merge(
    workflow_sequence_df[['user_id','total_event_count','repeated_events_count',
                           'sequence_diversity','power_user_score','struggle_score']],
    on='user_id', how='left')

_be_mom = engagement_momentum_df.rename(columns={'distinct_id':'user_id'})
be_base = be_base.merge(
    _be_mom[['user_id','current_7d_activity','current_30d_activity',
              'momentum_change','accel_periods','decel_periods']],
    on='user_id', how='left')

# Day-of-week preference from daily_counts
_be_dc = daily_counts.rename(columns={'distinct_id':'user_id'}).copy()
_be_dc['date'] = pd.to_datetime(_be_dc['date'])
_be_dc['dow']  = _be_dc['date'].dt.dayofweek   # 0=Mon…6=Sun

_be_dow_w    = _be_dc.groupby(['user_id','dow'])['daily_events'].sum().reset_index()
_be_best_dow = _be_dow_w.loc[_be_dow_w.groupby('user_id')['daily_events'].idxmax(),
                               ['user_id','dow']].rename(columns={'dow':'best_dow'})
_DAY_MAP  = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',
              4:'Friday',5:'Saturday',6:'Sunday'}
_be_best_dow['optimal_engagement_day'] = _be_best_dow['best_dow'].map(_DAY_MAP)

_be_sp = _be_dc.groupby('user_id').agg(
    n_active_days=('date','nunique'),
    first_day=('date','min'),
    last_day=('date','max')
).reset_index()
_be_sp['days_span']     = (_be_sp['last_day'] - _be_sp['first_day']).dt.days.clip(lower=1)
_be_sp['activity_rate'] = _be_sp['n_active_days'] / (_be_sp['days_span'] / 7)

def _freq(r):
    if r >= 5:   return 'daily'
    if r >= 2.5: return '3x_week'
    if r >= 1:   return 'weekly'
    if r >= 0.5: return 'biweekly'
    return 'monthly'

_be_sp['optimal_engagement_frequency'] = _be_sp['activity_rate'].apply(_freq)

be_base = be_base.merge(_be_best_dow[['user_id','optimal_engagement_day']], on='user_id', how='left')
be_base = be_base.merge(
    _be_sp[['user_id','optimal_engagement_frequency','n_active_days','days_span']],
    on='user_id', how='left')

be_base['optimal_engagement_day'].fillna('Tuesday', inplace=True)
be_base['optimal_engagement_frequency'].fillna('monthly', inplace=True)

be_base = be_base.merge(
    active_users_survival[['user_id','success_score','tenure_days','churned',
                             'days_since_first']],
    on='user_id', how='left')

be_base = be_base.merge(
    user_success_metrics[['user_id','total_events','long_term_retention']],
    on='user_id', how='left')

print(f"Master table: {len(be_base)} users × {be_base.shape[1]} columns")
print(f"Users with success_score: {be_base['success_score'].notna().sum()}")

# ─── 2. HABIT SCORE (0–100) ──────────────────────────────────────────────────
def _mm(s, lo_q=0.01, hi_q=0.99):
    lo = s.quantile(lo_q); hi = s.quantile(hi_q)
    return ((s.clip(lo, hi) - lo) / (hi - lo + 1e-9)).clip(0, 1)

be_base['_cv']  = (be_base['avg_inter_session_gap_hours'] /
                    (be_base['median_inter_session_gap_hours'].replace(0, np.nan) + 1e-3)
                   ).fillna(0)
be_base['_reg'] = (1 - _mm(be_base['_cv'], hi_q=0.95)).clip(0, 1)
be_base['_sf']  = _mm(be_base['total_sessions'].fillna(0))
be_base['_asd'] = (be_base['n_active_days'].fillna(0) /
                    be_base['days_span'].fillna(1).clip(lower=1)).clip(0, 1)
be_base['_rep'] = (be_base['repeated_events_count'].fillna(0) /
                    be_base['total_event_count'].fillna(1).clip(lower=1)).clip(0, 1)
be_base['_ltr'] = be_base['long_term_retention'].fillna(False).astype(float)

be_base['habit_score'] = (
    0.30 * be_base['_reg'] + 0.20 * be_base['_sf'] + 0.25 * be_base['_asd'] +
    0.15 * be_base['_rep'] + 0.10 * be_base['_ltr']
) * 100

print(f"\nHabit score → mean={be_base['habit_score'].mean():.1f}  "
      f"median={be_base['habit_score'].median():.1f}  std={be_base['habit_score'].std():.1f}")

# ─── 3. LOSS AVERSION SCORE (0–100) ──────────────────────────────────────────
be_base['_inv'] = _mm(be_base['tenure_days'].fillna(0) *
                       np.log1p(be_base['total_event_count'].fillna(0)))

_be_tp = be_base['accel_periods'].fillna(0) + be_base['decel_periods'].fillna(0) + 1
be_base['_rev'] = (be_base['decel_periods'].fillna(0) / _be_tp).clip(0, 1)
be_base['_str'] = _mm(be_base['struggle_score'].fillna(0))
be_base['_chf'] = be_base['churned'].fillna(1)
be_base['_anc'] = (be_base['success_score'].fillna(0) / 100) * (1 - be_base['_chf'])

be_base['loss_aversion_score'] = (
    0.30 * be_base['_inv'] + 0.20 * be_base['_rev'] +
    0.25 * be_base['_str'] + 0.25 * be_base['_anc']
) * 100

print(f"Loss aversion → mean={be_base['loss_aversion_score'].mean():.1f}  "
      f"median={be_base['loss_aversion_score'].median():.1f}  "
      f"std={be_base['loss_aversion_score'].std():.1f}")

# ─── 4. ASSEMBLE OUTPUT ───────────────────────────────────────────────────────
behavioral_economics_scores = be_base[[
    'user_id', 'habit_score', 'loss_aversion_score',
    'optimal_engagement_day', 'optimal_engagement_frequency'
]].copy().reset_index(drop=True)

behavioral_economics_scores['habit_score']         = \
    behavioral_economics_scores['habit_score'].round(2)
behavioral_economics_scores['loss_aversion_score'] = \
    behavioral_economics_scores['loss_aversion_score'].round(2)

assert behavioral_economics_scores['habit_score'].isna().sum() == 0, \
    "habit_score contains nulls"
assert behavioral_economics_scores['loss_aversion_score'].isna().sum() == 0, \
    "loss_aversion_score contains nulls"
assert behavioral_economics_scores['optimal_engagement_day'].isna().sum() == 0, \
    "optimal_engagement_day contains nulls"
assert behavioral_economics_scores['optimal_engagement_frequency'].isna().sum() == 0, \
    "optimal_engagement_frequency contains nulls"

print(f"\n{'─'*60}")
print(f"behavioral_economics_scores: {behavioral_economics_scores.shape}")
print(f"All {len(behavioral_economics_scores):,} users scored on all dimensions ✓")
print(f"\nEngagement Day distribution:")
print(behavioral_economics_scores['optimal_engagement_day'].value_counts().to_string())
print(f"\nEngagement Frequency distribution:")
print(behavioral_economics_scores['optimal_engagement_frequency'].value_counts().to_string())
print(f"\nSample (first 10 rows):")
print(behavioral_economics_scores.head(10).to_string(index=False))

# ─── 5. VISUALISATION (all chart objects scoped within function to avoid conflicts) ─
def _plot_be_distributions(df, bg, txt, sec, clr):
    """Plot behavioral economics score distributions in a 2×2 grid."""
    _fig, _grid = plt.subplots(2, 2, figsize=(14, 10))
    _fig.suptitle('Behavioral Economics Scoring — Distributions',
                   fontsize=16, color=txt, fontweight='bold')
    _fig.patch.set_facecolor(bg)
    plt.rcParams.update({'figure.facecolor': bg, 'axes.facecolor': bg,
                          'text.color': txt, 'axes.labelcolor': txt,
                          'xtick.color': sec, 'ytick.color': sec,
                          'axes.edgecolor': '#444'})

    _p = _grid[0, 0]
    _p.set_facecolor(bg)
    _p.hist(df['habit_score'], bins=50, color=clr[0], alpha=0.85, edgecolor='none')
    _p.axvline(df['habit_score'].median(), color=clr[9], lw=2, ls='--',
                label=f"Median {df['habit_score'].median():.1f}")
    _p.set_title('Habit Score Distribution', color=txt, fontsize=12)
    _p.set_xlabel('Habit Score (0–100)', color=sec)
    _p.set_ylabel('Users', color=sec)
    _p.legend(fontsize=9, labelcolor=txt, facecolor='#2a2a30')

    _p2 = _grid[0, 1]
    _p2.set_facecolor(bg)
    _p2.hist(df['loss_aversion_score'], bins=50, color=clr[1], alpha=0.85, edgecolor='none')
    _p2.axvline(df['loss_aversion_score'].median(), color=clr[9], lw=2, ls='--',
                 label=f"Median {df['loss_aversion_score'].median():.1f}")
    _p2.set_title('Loss Aversion Score Distribution', color=txt, fontsize=12)
    _p2.set_xlabel('Loss Aversion Score (0–100)', color=sec)
    _p2.set_ylabel('Users', color=sec)
    _p2.legend(fontsize=9, labelcolor=txt, facecolor='#2a2a30')

    _p3 = _grid[1, 0]
    _p3.set_facecolor(bg)
    _day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    _day_ct = df['optimal_engagement_day'].value_counts().reindex(_day_order, fill_value=0)
    _bars3 = _p3.bar(_day_ct.index, _day_ct.values,
                      color=[clr[i % len(clr)] for i in range(7)], alpha=0.85)
    _p3.set_title('Optimal Engagement Day', color=txt, fontsize=12)
    _p3.set_xlabel('Day of Week', color=sec)
    _p3.set_ylabel('Users', color=sec)
    _p3.tick_params(axis='x', rotation=35)
    for _bar in _bars3:
        _p3.text(_bar.get_x() + _bar.get_width()/2, _bar.get_height() + 8,
                  f'{int(_bar.get_height()):,}', ha='center', va='bottom',
                  fontsize=8, color=txt)

    _p4 = _grid[1, 1]
    _p4.set_facecolor(bg)
    _freq_order = ['daily','3x_week','weekly','biweekly','monthly']
    _freq_ct = df['optimal_engagement_frequency'].value_counts().reindex(_freq_order, fill_value=0)
    _bars4 = _p4.bar(_freq_ct.index, _freq_ct.values,
                      color=[clr[i % len(clr)] for i in range(5)], alpha=0.85)
    _p4.set_title('Optimal Engagement Frequency', color=txt, fontsize=12)
    _p4.set_xlabel('Recommended Frequency', color=sec)
    _p4.set_ylabel('Users', color=sec)
    for _bar in _bars4:
        _p4.text(_bar.get_x() + _bar.get_width()/2, _bar.get_height() + 8,
                  f'{int(_bar.get_height()):,}', ha='center', va='bottom',
                  fontsize=8, color=txt)

    plt.tight_layout()
    return _fig

behavioral_economics_fig = _plot_be_distributions(
    behavioral_economics_scores, _BE_BG, _BE_TEXT, _BE_SEC, _BE_CLR)
behavioral_economics_fig.show()

print("\n✅ Behavioral Economics Scoring complete.")
print(f"   behavioral_economics_scores: {behavioral_economics_scores.shape}")
