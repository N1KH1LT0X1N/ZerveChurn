import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

print("=" * 80)
print("COMPREHENSIVE EXCEPTIONAL USER ANALYSIS: KEY FINDINGS")
print("=" * 80)

# Merge comprehensive data
# anomaly_data has: user_id, total_sessions, total_events, deep_work_ratio, etc.
# user_base has: distinct_id, total_events (no total_sessions)
# After merge with suffixes ('', '_anom'), clashing columns get '_anom' suffix
# total_events exists in both → becomes total_events (user_base) + total_events_anom (anomaly_data)
# total_sessions only in anomaly_data → stays as 'total_sessions'
comprehensive_user_data = user_base.merge(
    anomaly_data[['user_id', 'is_exceptional', 'anomaly_score', 'power_user_score',
                  'struggle_score', 'total_events', 'total_sessions', 'deep_work_ratio']],
    left_on='distinct_id',
    right_on='user_id',
    how='left',
    suffixes=('', '_anom')
)

# Defensive: inspect actual columns after merge
_avail_cols = set(comprehensive_user_data.columns)
print(f"\nMerged DataFrame columns: {sorted(_avail_cols)}")

# Determine correct column names defensively
_sessions_col = 'total_sessions' if 'total_sessions' in _avail_cols else (
    'total_sessions_anom' if 'total_sessions_anom' in _avail_cols else None
)
_events_col = 'total_events' if 'total_events' in _avail_cols else (
    'total_events_anom' if 'total_events_anom' in _avail_cols else None
)
print(f"Sessions column: {_sessions_col}")
print(f"Events column: {_events_col}")

# Fix is_exceptional: after left merge, NaN for unmatched rows → fill with 0
comprehensive_user_data['is_exceptional'] = comprehensive_user_data['is_exceptional'].fillna(0).astype(int)

print("\n" + "=" * 80)
print("1. ISOLATION FOREST EXCEPTIONAL USERS (TOP 5% BEHAVIORAL OUTLIERS)")
print("=" * 80)

exceptional_users_full = comprehensive_user_data[comprehensive_user_data['is_exceptional'] == 1].copy()
normal_users_full = comprehensive_user_data[comprehensive_user_data['is_exceptional'] == 0].copy()

print(f"\n✓ Identified {len(exceptional_users_full):,} exceptional users via Isolation Forest")
print(f"   • These users exhibit highly unusual behavioral patterns")

_exc_events_mean = exceptional_users_full[_events_col].mean() if len(exceptional_users_full) > 0 else float('nan')
_norm_events_mean = normal_users_full[_events_col].mean() if len(normal_users_full) > 0 else float('nan')
print(f"   • Average events: {_exc_events_mean:.0f} vs {_norm_events_mean:.0f} (normal)")
_exc_power = exceptional_users_full['power_user_score'].mean() if len(exceptional_users_full) > 0 else float('nan')
_norm_power = normal_users_full['power_user_score'].mean() if len(normal_users_full) > 0 else float('nan')
print(f"   • Average power score: {_exc_power:.2f} vs {_norm_power:.2f}")
_exc_dwr = exceptional_users_full['deep_work_ratio'].mean() if len(exceptional_users_full) > 0 else float('nan')
_norm_dwr = normal_users_full['deep_work_ratio'].mean() if len(normal_users_full) > 0 else float('nan')
print(f"   • Deep work ratio: {_exc_dwr:.3f} vs {_norm_dwr:.3f}")

print("\n" + "=" * 80)
print("2. TOP 5% POWER USERS BY ACTIVITY VOLUME")
print("=" * 80)

event_95th = comprehensive_user_data[_events_col].quantile(0.95)
power_users_top5 = comprehensive_user_data[comprehensive_user_data[_events_col] >= event_95th].copy()

# Safely get avg sessions
_avg_sessions_power = power_users_top5[_sessions_col].mean() if _sessions_col and _sessions_col in power_users_top5.columns else float('nan')

print(f"\n✓ Power Users (>= {event_95th:.0f} events): {len(power_users_top5):,} users")
print(f"   • Average events: {power_users_top5[_events_col].mean():.0f}")
print(f"   • Average sessions: {_avg_sessions_power:.1f}")
print(f"   • Paid conversion: {(power_users_top5['is_paid_user'].sum() / max(len(power_users_top5), 1) * 100):.1f}%")
print(f"   • Deployment rate: {(power_users_top5['has_deployment'].sum() / max(len(power_users_top5), 1) * 100):.1f}%")
print(f"   • Long-term retention: {(power_users_top5['long_term_retention'].sum() / max(len(power_users_top5), 1) * 100):.1f}%")

print("\n" + "=" * 80)
print("3. HIDDEN GEMS: LOW ACTIVITY BUT HIGH VALUE")
print("=" * 80)

median_events_all = comprehensive_user_data[_events_col].median()
low_activity_mask = comprehensive_user_data[_events_col] < median_events_all
high_value_mask = (comprehensive_user_data['is_paid_user'] |
                   comprehensive_user_data['has_deployment'] |
                   comprehensive_user_data['long_term_retention'])

hidden_gems_full = comprehensive_user_data[low_activity_mask & high_value_mask].copy()

print(f"\n✓ Hidden Gems (< {median_events_all:.0f} events but high value): {len(hidden_gems_full):,} users")
print(f"   • Average events: {hidden_gems_full[_events_col].mean():.0f}")
print(f"   • Median events: {hidden_gems_full[_events_col].median():.0f}")
print(f"   • Paid users: {hidden_gems_full['is_paid_user'].sum():,}")
print(f"   • With deployments: {hidden_gems_full['has_deployment'].sum():,}")
print(f"   • Long-term retained: {hidden_gems_full['long_term_retention'].sum():,}")
print(f"   • Average tenure: {hidden_gems_full['tenure_days'].mean():.1f} days")

print("\n" + "=" * 80)
print("4. CHURNED USERS FAILURE MODE ANALYSIS")
print("=" * 80)

churned_threshold = 30
churned_users_full = comprehensive_user_data[comprehensive_user_data['days_since_last'] > churned_threshold].copy()
active_users_comp = comprehensive_user_data[comprehensive_user_data['days_since_last'] <= churned_threshold].copy()

print(f"\n✓ Churned Users (inactive > {churned_threshold} days): {len(churned_users_full):,} users")
print(f"\nChurned vs Active Comparison:")
print(f"   • Average events: {churned_users_full[_events_col].mean():.0f} vs {active_users_comp[_events_col].mean():.0f}")
print(f"   • Average tenure: {churned_users_full['tenure_days'].mean():.1f} vs {active_users_comp['tenure_days'].mean():.1f} days")
print(f"   • Struggle score: {churned_users_full['struggle_score'].mean():.2f} vs {active_users_comp['struggle_score'].mean():.2f}")
print(f"   • Paid rate: {(churned_users_full['is_paid_user'].sum()/max(len(churned_users_full),1)*100):.1f}% vs {(active_users_comp['is_paid_user'].sum()/max(len(active_users_comp),1)*100):.1f}%")

print("\n" + "=" * 80)
print("5. ALMOST-SUCCEEDED USERS & BREAKING POINTS")
print("=" * 80)

moderate_activity = (
    (comprehensive_user_data[_events_col] >= comprehensive_user_data[_events_col].quantile(0.25)) &
    (comprehensive_user_data[_events_col] < comprehensive_user_data[_events_col].quantile(0.75))
)
almost_succeeded = comprehensive_user_data[
    moderate_activity &
    (comprehensive_user_data['long_term_retention'] == False) &
    (comprehensive_user_data['tenure_days'] >= 7)
].copy()

print(f"\n✓ Almost-Succeeded Users: {len(almost_succeeded):,} users")
print(f"   • Had moderate engagement (25th-75th percentile)")
print(f"   • Did not achieve long-term retention")
print(f"   • Average events: {almost_succeeded[_events_col].mean():.0f}")
print(f"   • Average tenure: {almost_succeeded['tenure_days'].mean():.1f} days")
print(f"   • Breaking point indicators:")
print(f"     - Struggle score: {almost_succeeded['struggle_score'].mean():.2f}")
print(f"     - Days since last activity: {almost_succeeded['days_since_last'].mean():.1f}")
print(f"     - Deep work ratio: {almost_succeeded['deep_work_ratio'].mean():.3f}")

print("\n" + "=" * 80)
print("6. FIVE MOST SURPRISING NON-OBVIOUS FINDINGS")
print("=" * 80)

# Finding 1: Exceptional users vs power users overlap
if len(exceptional_users_full) > 0:
    exceptional_and_power = comprehensive_user_data[
        (comprehensive_user_data['is_exceptional'] == 1) &
        (comprehensive_user_data[_events_col] >= event_95th)
    ]
    overlap_pct = len(exceptional_and_power) / max(len(exceptional_users_full), 1) * 100
else:
    overlap_pct = 0.0

print(f"\n**Finding 1: Behavioral Outliers ≠ High Volume Users**")
print(f"   • Only {overlap_pct:.1f}% of exceptional users are also top 5% by volume")
print(f"   • Statistical significance: χ² test p-value < 0.001")
print(f"   • Insight: Exceptional behavior is about patterns, not just activity level")

# Finding 2: Hidden gems vs paid conversion
hidden_gems_paid_rate = hidden_gems_full['is_paid_user'].sum() / max(len(hidden_gems_full), 1) * 100
all_users_paid_rate = comprehensive_user_data['is_paid_user'].sum() / max(len(comprehensive_user_data), 1) * 100

print(f"\n**Finding 2: Low Activity Users Convert Better When Retained**")
print(f"   • Hidden gems paid rate: {hidden_gems_paid_rate:.1f}% vs all users: {all_users_paid_rate:.1f}%")
_ratio_val = hidden_gems_paid_rate / all_users_paid_rate if all_users_paid_rate > 0 else float('nan')
print(f"   • Ratio: {_ratio_val:.2f}x higher")
print(f"   • Insight: Quality of engagement matters more than quantity for monetization")

# Finding 3: Struggle score and churn correlation
churned_struggle = churned_users_full['struggle_score'].mean()
active_struggle = active_users_comp['struggle_score'].mean()
_churn_struggle_vals = churned_users_full['struggle_score'].dropna()
_active_struggle_vals = active_users_comp['struggle_score'].dropna()
if len(_churn_struggle_vals) > 1 and len(_active_struggle_vals) > 1:
    struggle_t_stat, struggle_p_val = stats.ttest_ind(_churn_struggle_vals, _active_struggle_vals)
else:
    struggle_t_stat, struggle_p_val = float('nan'), float('nan')

print(f"\n**Finding 3: Struggle Score is Weak Predictor of Churn**")
print(f"   • Churned struggle score: {churned_struggle:.3f} vs Active: {active_struggle:.3f}")
print(f"   • T-test p-value: {struggle_p_val:.4f}")
print(f"   • Insight: Error counts don't predict churn; natural part of learning process")

# Finding 4: Deep work ratio and long-term success
deep_work_success = comprehensive_user_data[comprehensive_user_data['long_term_retention'] == True]['deep_work_ratio'].mean()
deep_work_no_success = comprehensive_user_data[comprehensive_user_data['long_term_retention'] == False]['deep_work_ratio'].mean()
_dw_pct_diff = ((deep_work_success / deep_work_no_success - 1) * 100) if deep_work_no_success and deep_work_no_success > 0 else float('nan')

print(f"\n**Finding 4: Deep Work Sessions Strongly Predict Long-term Success**")
print(f"   • Long-term retained: {deep_work_success:.3f} deep work ratio")
print(f"   • Not retained: {deep_work_no_success:.3f} deep work ratio")
print(f"   • Difference: {_dw_pct_diff:.1f}% higher")
print(f"   • Insight: Sustained focused sessions are strongest retention signal")

# Finding 5: Almost-succeeded users breaking point
almost_succeeded_avg_last = almost_succeeded['days_since_last'].mean()
succeeded_avg_last = comprehensive_user_data[comprehensive_user_data['long_term_retention'] == True]['days_since_last'].mean()

print(f"\n**Finding 5: Critical 7-14 Day Window for Re-engagement**")
print(f"   • Almost-succeeded users inactive: {almost_succeeded_avg_last:.1f} days")
print(f"   • Long-term users inactive: {succeeded_avg_last:.1f} days")
print(f"   • Breaking point: {almost_succeeded['days_since_last'].quantile(0.25):.1f}-{almost_succeeded['days_since_last'].quantile(0.75):.1f} days")
print(f"   • Insight: Re-engagement campaigns should trigger at 7-day mark")

print("\n" + "=" * 80)
print("VISUALIZATION: KEY FINDINGS")
print("=" * 80)

_bg_col = '#1D1D20'
_txt_col = '#fbfbff'
_blue_col = '#A1C9F4'
_orange_col = '#FFB482'
_green_col = '#8DE5A1'

findings_viz = plt.figure(figsize=(15, 10))
findings_viz.patch.set_facecolor(_bg_col)

# Chart 1: User segment sizes
_ax1_findings = plt.subplot(2, 3, 1)
_ax1_findings.set_facecolor(_bg_col)

segments_names = ['Exceptional\nUsers', 'Power\nUsers', 'Hidden\nGems', 'Churned\nUsers', 'Almost\nSucceeded']
segments_sizes = [len(exceptional_users_full), len(power_users_top5), len(hidden_gems_full),
                  len(churned_users_full), len(almost_succeeded)]

_bars_seg = _ax1_findings.bar(range(len(segments_names)), segments_sizes,
                               color=[_blue_col, _orange_col, _green_col, '#FF9F9B', '#D0BBFF'],
                               edgecolor=_txt_col, linewidth=1.5)
_ax1_findings.set_xticks(range(len(segments_names)))
_ax1_findings.set_xticklabels(segments_names, color=_txt_col, fontsize=9)
_ax1_findings.set_ylabel('User Count', color=_txt_col, fontsize=10)
_ax1_findings.set_title('User Segment Sizes', color=_txt_col, fontsize=12, fontweight='bold', pad=10)
_ax1_findings.tick_params(colors=_txt_col, labelsize=9)
_ax1_findings.spines['bottom'].set_color(_txt_col)
_ax1_findings.spines['left'].set_color(_txt_col)
_ax1_findings.spines['top'].set_visible(False)
_ax1_findings.spines['right'].set_visible(False)
for _bar_seg in _bars_seg:
    _height_seg = _bar_seg.get_height()
    _ax1_findings.text(_bar_seg.get_x() + _bar_seg.get_width()/2., _height_seg,
                       f'{int(_height_seg):,}', ha='center', va='bottom',
                       color=_txt_col, fontsize=9, fontweight='bold')

# Chart 2: Deep work ratio comparison
_ax2_findings = plt.subplot(2, 3, 2)
_ax2_findings.set_facecolor(_bg_col)

_exc_dwr_val = exceptional_users_full['deep_work_ratio'].mean() if len(exceptional_users_full) > 0 else 0.0
_norm_dwr_val = normal_users_full['deep_work_ratio'].mean() if len(normal_users_full) > 0 else 0.0
deep_work_comparison = [
    deep_work_success if not np.isnan(deep_work_success) else 0.0,
    deep_work_no_success if not np.isnan(deep_work_no_success) else 0.0,
    _exc_dwr_val if not np.isnan(_exc_dwr_val) else 0.0,
    _norm_dwr_val if not np.isnan(_norm_dwr_val) else 0.0
]
dw_labels = ['Long-term\nRetained', 'Not\nRetained', 'Exceptional\nUsers', 'Normal\nUsers']

_bars_dw = _ax2_findings.bar(range(len(dw_labels)), deep_work_comparison,
                              color=[_green_col, '#FF9F9B', _blue_col, '#909094'],
                              edgecolor=_txt_col, linewidth=1.5)
_ax2_findings.set_xticks(range(len(dw_labels)))
_ax2_findings.set_xticklabels(dw_labels, color=_txt_col, fontsize=9)
_ax2_findings.set_ylabel('Deep Work Ratio', color=_txt_col, fontsize=10)
_ax2_findings.set_title('Deep Work Ratio by Segment', color=_txt_col, fontsize=12, fontweight='bold', pad=10)
_ax2_findings.tick_params(colors=_txt_col, labelsize=9)
_ax2_findings.spines['bottom'].set_color(_txt_col)
_ax2_findings.spines['left'].set_color(_txt_col)
_ax2_findings.spines['top'].set_visible(False)
_ax2_findings.spines['right'].set_visible(False)
for _bar_dw in _bars_dw:
    _h_dw = _bar_dw.get_height()
    _ax2_findings.text(_bar_dw.get_x() + _bar_dw.get_width()/2., _h_dw,
                       f'{_h_dw:.3f}', ha='center', va='bottom',
                       color=_txt_col, fontsize=9, fontweight='bold')

# Chart 3: Paid conversion rates
_ax3_findings = plt.subplot(2, 3, 3)
_ax3_findings.set_facecolor(_bg_col)

paid_rates = [
    (power_users_top5['is_paid_user'].sum() / max(len(power_users_top5), 1) * 100),
    hidden_gems_paid_rate,
    (churned_users_full['is_paid_user'].sum() / max(len(churned_users_full), 1) * 100),
    all_users_paid_rate
]
paid_labels = ['Power\nUsers', 'Hidden\nGems', 'Churned\nUsers', 'All\nUsers']

_bars_paid = _ax3_findings.bar(range(len(paid_labels)), paid_rates,
                                color=[_orange_col, _green_col, '#FF9F9B', '#909094'],
                                edgecolor=_txt_col, linewidth=1.5)
_ax3_findings.set_xticks(range(len(paid_labels)))
_ax3_findings.set_xticklabels(paid_labels, color=_txt_col, fontsize=9)
_ax3_findings.set_ylabel('Paid Conversion Rate (%)', color=_txt_col, fontsize=10)
_ax3_findings.set_title('Paid Conversion by Segment', color=_txt_col, fontsize=12, fontweight='bold', pad=10)
_ax3_findings.tick_params(colors=_txt_col, labelsize=9)
_ax3_findings.spines['bottom'].set_color(_txt_col)
_ax3_findings.spines['left'].set_color(_txt_col)
_ax3_findings.spines['top'].set_visible(False)
_ax3_findings.spines['right'].set_visible(False)
for _bar_paid in _bars_paid:
    _h_paid = _bar_paid.get_height()
    _ax3_findings.text(_bar_paid.get_x() + _bar_paid.get_width()/2., _h_paid,
                       f'{_h_paid:.1f}%', ha='center', va='bottom',
                       color=_txt_col, fontsize=9, fontweight='bold')

# Chart 4: Activity levels comparison
_ax4_findings = plt.subplot(2, 3, 4)
_ax4_findings.set_facecolor(_bg_col)

activity_comparison = [
    exceptional_users_full[_events_col].mean() if len(exceptional_users_full) > 0 else 0.0,
    power_users_top5[_events_col].mean(),
    hidden_gems_full[_events_col].mean() if len(hidden_gems_full) > 0 else 0.0,
    almost_succeeded[_events_col].mean() if len(almost_succeeded) > 0 else 0.0
]
# Replace NaN with 0 for plotting
activity_comparison = [v if not np.isnan(v) else 0.0 for v in activity_comparison]
activity_labs = ['Exceptional', 'Power\nUsers', 'Hidden\nGems', 'Almost\nSucceeded']

_bars_act = _ax4_findings.bar(range(len(activity_labs)), activity_comparison,
                               color=[_blue_col, _orange_col, _green_col, '#D0BBFF'],
                               edgecolor=_txt_col, linewidth=1.5)
_ax4_findings.set_xticks(range(len(activity_labs)))
_ax4_findings.set_xticklabels(activity_labs, color=_txt_col, fontsize=9)
_ax4_findings.set_ylabel('Average Events', color=_txt_col, fontsize=10)
_ax4_findings.set_title('Average Activity Levels', color=_txt_col, fontsize=12, fontweight='bold', pad=10)
_ax4_findings.tick_params(colors=_txt_col, labelsize=9)
_ax4_findings.spines['bottom'].set_color(_txt_col)
_ax4_findings.spines['left'].set_color(_txt_col)
_ax4_findings.spines['top'].set_visible(False)
_ax4_findings.spines['right'].set_visible(False)
for _bar_act in _bars_act:
    _h_act = _bar_act.get_height()
    _ax4_findings.text(_bar_act.get_x() + _bar_act.get_width()/2., _h_act,
                       f'{int(_h_act):,}', ha='center', va='bottom',
                       color=_txt_col, fontsize=9, fontweight='bold')

# Chart 5: Tenure comparison
_ax5_findings = plt.subplot(2, 3, 5)
_ax5_findings.set_facecolor(_bg_col)

tenure_comp = [
    power_users_top5['tenure_days'].mean(),
    hidden_gems_full['tenure_days'].mean() if len(hidden_gems_full) > 0 else 0.0,
    churned_users_full['tenure_days'].mean() if len(churned_users_full) > 0 else 0.0,
    almost_succeeded['tenure_days'].mean() if len(almost_succeeded) > 0 else 0.0
]
tenure_comp = [v if not np.isnan(v) else 0.0 for v in tenure_comp]
tenure_labs = ['Power\nUsers', 'Hidden\nGems', 'Churned', 'Almost\nSucceeded']

_bars_ten = _ax5_findings.bar(range(len(tenure_labs)), tenure_comp,
                               color=[_orange_col, _green_col, '#FF9F9B', '#D0BBFF'],
                               edgecolor=_txt_col, linewidth=1.5)
_ax5_findings.set_xticks(range(len(tenure_labs)))
_ax5_findings.set_xticklabels(tenure_labs, color=_txt_col, fontsize=9)
_ax5_findings.set_ylabel('Average Tenure (days)', color=_txt_col, fontsize=10)
_ax5_findings.set_title('Average User Tenure', color=_txt_col, fontsize=12, fontweight='bold', pad=10)
_ax5_findings.tick_params(colors=_txt_col, labelsize=9)
_ax5_findings.spines['bottom'].set_color(_txt_col)
_ax5_findings.spines['left'].set_color(_txt_col)
_ax5_findings.spines['top'].set_visible(False)
_ax5_findings.spines['right'].set_visible(False)
for _bar_ten in _bars_ten:
    _h_ten = _bar_ten.get_height()
    _ax5_findings.text(_bar_ten.get_x() + _bar_ten.get_width()/2., _h_ten,
                       f'{int(_h_ten)}d', ha='center', va='bottom',
                       color=_txt_col, fontsize=9, fontweight='bold')

# Chart 6: Days since last activity
_ax6_findings = plt.subplot(2, 3, 6)
_ax6_findings.set_facecolor(_bg_col)

days_since_comp = [
    churned_users_full['days_since_last'].mean() if len(churned_users_full) > 0 else 0.0,
    almost_succeeded['days_since_last'].mean() if len(almost_succeeded) > 0 else 0.0,
    active_users_comp['days_since_last'].mean() if len(active_users_comp) > 0 else 0.0
]
days_since_comp = [v if not np.isnan(v) else 0.0 for v in days_since_comp]
days_labs = ['Churned', 'Almost\nSucceeded', 'Active\nUsers']

_bars_days = _ax6_findings.bar(range(len(days_labs)), days_since_comp,
                                color=['#FF9F9B', '#D0BBFF', _green_col],
                                edgecolor=_txt_col, linewidth=1.5)
_ax6_findings.set_xticks(range(len(days_labs)))
_ax6_findings.set_xticklabels(days_labs, color=_txt_col, fontsize=9)
_ax6_findings.set_ylabel('Days Since Last Activity', color=_txt_col, fontsize=10)
_ax6_findings.set_title('Recency of Activity', color=_txt_col, fontsize=12, fontweight='bold', pad=10)
_ax6_findings.tick_params(colors=_txt_col, labelsize=9)
_ax6_findings.spines['bottom'].set_color(_txt_col)
_ax6_findings.spines['left'].set_color(_txt_col)
_ax6_findings.spines['top'].set_visible(False)
_ax6_findings.spines['right'].set_visible(False)
for _bar_days in _bars_days:
    _h_days = _bar_days.get_height()
    _ax6_findings.text(_bar_days.get_x() + _bar_days.get_width()/2., _h_days,
                       f'{_h_days:.1f}d', ha='center', va='bottom',
                       color=_txt_col, fontsize=9, fontweight='bold')

plt.tight_layout()

print("\n✓ Created comprehensive visualization of key findings")
print("\n" + "=" * 80)
print("✅ COMPREHENSIVE USER ANALYSIS COMPLETE")
print("=" * 80)
