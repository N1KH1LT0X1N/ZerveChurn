import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

print("=" * 70)
print("WEEKLY DELTA METRICS COMPUTATION")
print("=" * 70)

# ── Load master segments from CSV
_master_segs = pd.read_csv('user_segments.csv')
print(f"Loaded master_segments: {len(_master_segs)} users")

# ── Reference dates
_latest_ts = pd.to_datetime(user_retention['timestamp']).max()
_week_end   = _latest_ts.normalize()
_week_start = _week_end - timedelta(days=7)
_prev_start = _week_start - timedelta(days=7)
_prev_end   = _week_start

print(f"\nDataset reference date: {_latest_ts.date()}")
print(f"This week:  {_week_start.date()} → {_week_end.date()}")
print(f"Last week:  {_prev_start.date()} → {_prev_end.date()}")

_events_df = user_retention[['distinct_id', 'timestamp', 'event']].copy()
_events_df['timestamp'] = pd.to_datetime(_events_df['timestamp'])

_this_week_events = _events_df[(_events_df['timestamp'] >= _week_start) & (_events_df['timestamp'] < _week_end)]
_last_week_events = _events_df[(_events_df['timestamp'] >= _prev_start) & (_events_df['timestamp'] < _prev_end)]

_this_active_ids = set(_this_week_events['distinct_id'].unique())
_last_active_ids = set(_last_week_events['distinct_id'].unique())
_new_at_risk_ids = _this_active_ids - _last_active_ids
_recovered_ids   = _last_active_ids - _this_active_ids
_retained_ids    = _this_active_ids & _last_active_ids

# ── Churn and segment data from Advanced Analysis Synthesis
_udf = unified_df.copy()
# Backfill `churn_proxy` if an older `unified_df` (pre-Apr-2026 fix) didn't carry it.
if 'churn_proxy' not in _udf.columns and 'success_score' in _udf.columns:
    _udf['churn_proxy'] = (100.0 - _udf['success_score']).clip(0, 100)
# Filter on churn_proxy so "high risk" actually means high churn risk
# (previous filter on success_score >= 50 selected the *most* successful
# users). See docs/repo_state_and_next_steps.md §6.
_high_risk_df = _udf[_udf['churn_proxy'] >= 50]
_elite_at_risk_df = _high_risk_df[_high_risk_df['is_power_user'] | _high_risk_df['is_network_hub']]

_power_set   = power_users_set
_hub_set     = network_hubs_set
_anomaly_set = anomalous_users_set

# ── 1. CHURN RISK MOVEMENT
churn_risk_delta = {
    "total_active_users_this_week": int(len(_this_active_ids)),
    "total_active_users_last_week": int(len(_last_active_ids)),
    "delta_active_users":           int(len(_this_active_ids) - len(_last_active_ids)),
    "delta_pct":                    round((len(_this_active_ids) - len(_last_active_ids)) / max(len(_last_active_ids), 1) * 100, 1),
    "new_at_risk_users":            int(len(_new_at_risk_ids)),
    "recovered_users":              int(len(_recovered_ids)),
    "retained_users":               int(len(_retained_ids)),
    "high_risk_count":              int(len(_high_risk_df)),
    "elite_at_risk_count":          int(len(_elite_at_risk_df)),
    "avg_success_score":            round(float(_udf['success_score'].mean()), 2),
    "high_tier_pct":                round(float((_udf['success_score'] >= 50).mean() * 100), 2),
}
print("\n📊 CHURN RISK MOVEMENT:")
for k, v in churn_risk_delta.items():
    print(f"  {k}: {v}")

# ── 2. COHORT SHIFTS — compute momentum inline from user_retention
_mom_df = user_retention[['distinct_id', 'timestamp']].copy()
_mom_df['timestamp'] = pd.to_datetime(_mom_df['timestamp'])
_mom_df['date'] = _mom_df['timestamp'].dt.date
_daily_counts = _mom_df.groupby(['distinct_id', 'date']).size().reset_index(name='daily_events')
_daily_counts['date'] = pd.to_datetime(_daily_counts['date'])
_daily_counts = _daily_counts.sort_values(['distinct_id', 'date'])

_mom_user_stats = (
    _daily_counts.groupby('distinct_id')['daily_events']
    .agg(total_mom='sum', last_7d_events='last', momentum_change='std')
    .reset_index()
)
_accelerating_cnt = int((_mom_user_stats['momentum_change'] > 0).sum())
_decelerating_cnt = int((_mom_user_stats['momentum_change'] < 0).sum())
_stable_cnt = int((_mom_user_stats['momentum_change'] == 0).sum())
_insufficient_cnt = int(_mom_user_stats['momentum_change'].isna().sum())
_churn_risk_mom = int(((_mom_user_stats['last_7d_events'] < 2) & (_mom_user_stats['total_mom'] < 10)).sum())

_events_this = int(len(_this_week_events))
_events_last = int(len(_last_week_events))
_events_delta_wow = _events_this - _events_last
_events_delta_pct = round(_events_delta_wow / max(_events_last, 1) * 100, 1)

_sess_this = _this_week_events.groupby('distinct_id').size()
_sess_last = _last_week_events.groupby('distinct_id').size()
_avg_sess_this = round(float(_sess_this.mean()), 2) if len(_sess_this) > 0 else 0
_avg_sess_last = round(float(_sess_last.mean()), 2) if len(_sess_last) > 0 else 0
_avg_sess_delta = round(_avg_sess_this - _avg_sess_last, 2)

cohort_shifts = {
    "total_events_this_week":       _events_this,
    "total_events_last_week":       _events_last,
    "events_delta":                 _events_delta_wow,
    "events_delta_pct":             _events_delta_pct,
    "avg_events_per_user_this":     _avg_sess_this,
    "avg_events_per_user_last":     _avg_sess_last,
    "avg_events_delta":             _avg_sess_delta,
    "users_accelerating":           _accelerating_cnt,
    "users_decelerating":           _decelerating_cnt,
    "users_stable":                 _stable_cnt,
    "users_insufficient_data":      _insufficient_cnt,
    "users_at_momentum_churn_risk": _churn_risk_mom,
}
print("\n📈 COHORT SHIFTS:")
for k, v in cohort_shifts.items():
    print(f"  {k}: {v}")

# ── 3. RETENTION CHANGES
_this_30d_start = _week_end - timedelta(days=30)
_prior_30d = _events_df[(_events_df['timestamp'] >= _this_30d_start) & (_events_df['timestamp'] < _week_start)]
_prior_30d_ids = set(_prior_30d['distinct_id'].unique())
_retained_30d  = _this_active_ids & _prior_30d_ids
_retention_rate = round(len(_retained_30d) / max(len(_prior_30d_ids), 1) * 100, 1)

_lt_retention_cnt = int(user_base['long_term_retention'].sum())
_lt_retention_pct = round(_lt_retention_cnt / max(len(user_base), 1) * 100, 1)

_power_this_week = _this_active_ids & _power_set
_power_last_week = _last_active_ids & _power_set
_power_retained  = _power_this_week & _power_last_week
_power_churn_wk  = _power_last_week - _power_this_week

retention_changes = {
    "retention_rate_30d_pct":       _retention_rate,
    "users_retained_30d":           int(len(_retained_30d)),
    "long_term_retention_count":    _lt_retention_cnt,
    "long_term_retention_pct":      _lt_retention_pct,
    "power_users_active_this_week": int(len(_power_this_week)),
    "power_users_active_last_week": int(len(_power_last_week)),
    "power_users_retained_wow":     int(len(_power_retained)),
    "power_users_went_silent":      int(len(_power_churn_wk)),
    "total_churned_all_time":       int(total_churned),
    "high_risk_alert_users":        int(len(high_risk_alerts)),
}
print("\n🔄 RETENTION CHANGES:")
for k, v in retention_changes.items():
    print(f"  {k}: {v}")

# ── 4. TOP MOVERS
_top_users_this = (
    _this_week_events.groupby('distinct_id').size()
    .reset_index(name='events_this_week')
    .sort_values('events_this_week', ascending=False)
    .head(10)
)
_top_users_last_df = (
    _last_week_events.groupby('distinct_id').size()
    .reset_index(name='events_last_week')
)
_top_movers = _top_users_this.merge(_top_users_last_df, on='distinct_id', how='left')
_top_movers['events_last_week'] = _top_movers['events_last_week'].fillna(0).astype(int)
_top_movers['wow_change'] = _top_movers['events_this_week'] - _top_movers['events_last_week']
_top_movers['is_power_user'] = _top_movers['distinct_id'].isin(_power_set)
_top_movers['is_network_hub'] = _top_movers['distinct_id'].isin(_hub_set)
top_movers_delta = _top_movers.to_dict(orient='records')

print(f"\n🚀 TOP MOVERS (this week):")
for _tm in top_movers_delta[:5]:
    print(f"  User {str(_tm['distinct_id'])[:12]}... | Events: {_tm['events_this_week']} | WoW Δ: {_tm['wow_change']:+d} | Power: {_tm['is_power_user']}")

# ── 5. SEGMENT SPOTLIGHTS
_segment_this = _this_week_events.groupby('distinct_id').size().reset_index(name='events')
_segment_this = _segment_this.merge(
    _master_segs[['distinct_id', 'engagement_segment', 'workflow_pattern']],
    on='distinct_id', how='left'
)
_seg_spotlight = (
    _segment_this.groupby('engagement_segment')['events']
    .agg(['count', 'mean', 'sum'])
    .round(1)
    .reset_index()
    .rename(columns={'count': 'active_users', 'mean': 'avg_events', 'sum': 'total_events'})
    .sort_values('total_events', ascending=False)
)
segment_spotlights_data = _seg_spotlight.to_dict(orient='records')

print(f"\n🎯 SEGMENT SPOTLIGHTS:")
for _ss in segment_spotlights_data:
    print(f"  [{_ss.get('engagement_segment', 'N/A')}] Users: {_ss['active_users']} | Avg Events: {_ss['avg_events']} | Total: {_ss['total_events']}")

# ── Final metrics JSON for GenAI prompt
weekly_insights_metrics = {
    "report_date": str(_week_end.date()),
    "week_range": f"{_week_start.date()} to {_week_end.date()}",
    "prior_week_range": f"{_prev_start.date()} to {_prev_end.date()}",
    "churn_risk_movement": churn_risk_delta,
    "cohort_shifts": cohort_shifts,
    "retention_changes": retention_changes,
    "top_movers": top_movers_delta,
    "segment_spotlights": segment_spotlights_data,
    "model_performance": {
        "best_model": str(best_model_name),
        "test_accuracy": round(float(test_acc), 4),
        "test_f1": round(float(test_f1), 4),
        "test_roc_auc": round(float(test_roc_auc), 4),
    },
    "user_base_summary": {
        "total_users": int(total_users),
        "power_users": int(len(_power_set)),
        "network_hubs": int(len(_hub_set)),
        "anomalous_users": int(len(_anomaly_set)),
        "elite_at_risk": int(len(_elite_at_risk_df)),
        "high_potential": int(len(regular_high_engagement)),
    },
}

_metrics_json = json.dumps(weekly_insights_metrics, indent=2, default=str)
print(f"\n✅ Metrics JSON compiled — {len(_metrics_json)} chars")
print("\nPreview (first 800 chars):")
print(_metrics_json[:800])
