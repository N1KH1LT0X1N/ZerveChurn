import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Prepare time-based engagement data using 'distinct_id' as user key
df_mom = user_retention[['distinct_id', 'timestamp']].copy()
df_mom['timestamp'] = pd.to_datetime(df_mom['timestamp'])
df_mom['date'] = df_mom['timestamp'].dt.date

# Daily event counts per user
daily_counts = df_mom.groupby(['distinct_id', 'date']).size().reset_index(name='daily_events')
daily_counts['date'] = pd.to_datetime(daily_counts['date'])
daily_counts = daily_counts.sort_values(['distinct_id', 'date'])

# Vectorised 7d and 30d rolling per user using groupby
def rolling_stats(group):
    g = group.set_index('date').sort_index()
    r7  = g['daily_events'].rolling('7D', min_periods=1).sum()
    r30 = g['daily_events'].rolling('30D', min_periods=1).sum()
    v7  = r7.diff()
    accel = (v7 > 0).sum()
    decel = (v7 < 0).sum()
    # momentum trend: compare last 7d vs prior 7d
    if len(r7) >= 2:
        recent  = r7.iloc[-1]
        prev    = r7.iloc[-min(8, len(r7))]
        mc = recent - prev
        mt = 'Accelerating' if mc > 0 else ('Decelerating' if mc < 0 else 'Stable')
    else:
        recent = r7.iloc[-1] if len(r7) > 0 else 0
        mc = 0
        mt = 'Insufficient_Data'
    # churn risk: last 3 periods declining
    recent_v = v7.iloc[-3:] if len(v7) >= 3 else v7
    churn_risk = (recent_v < 0).sum() >= 2
    return pd.Series({
        'total_events': g['daily_events'].sum(),
        'current_7d_activity': recent,
        'current_30d_activity': r30.iloc[-1] if len(r30) > 0 else 0,
        'momentum_change': mc,
        'momentum_trend': mt,
        'accel_periods': accel,
        'decel_periods': decel,
        'churn_risk': churn_risk,
    })

engagement_momentum_df = daily_counts.groupby('distinct_id').apply(rolling_stats).reset_index()

# ── Print summary
print("=" * 70)
print("ENGAGEMENT MOMENTUM TRACKING")
print("=" * 70)
print(f"\nTotal Users Analyzed: {len(engagement_momentum_df):,}")
print()
print("Momentum Trend Distribution:")
print(engagement_momentum_df['momentum_trend'].value_counts().to_string())
print()
print("Churn Risk Analysis:")
at_risk = engagement_momentum_df['churn_risk'].sum()
print(f"  - Users at churn risk:  {at_risk:,} ({at_risk / len(engagement_momentum_df) * 100:.1f}%)")
print(f"  - Users not at risk:    {(~engagement_momentum_df['churn_risk']).sum():,}")
print()
print("Average Metrics by Momentum Trend:")
momentum_summary = engagement_momentum_df.groupby('momentum_trend').agg({
    'total_events': 'mean',
    'current_7d_activity': 'mean',
    'current_30d_activity': 'mean',
    'churn_risk': 'sum'
}).round(2)
print(momentum_summary.to_string())
print()
print("Engagement Velocity Patterns:")
print(f"  - Mean acceleration periods: {engagement_momentum_df['accel_periods'].mean():.1f}")
print(f"  - Mean deceleration periods: {engagement_momentum_df['decel_periods'].mean():.1f}")

# ── Visualisation (all styling vars private with _ prefix)
_bg_col  = '#1D1D20'
_txt_col = '#fbfbff'
_sec_col = '#909094'
_mom_colors = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF']

momentum_viz_fig = plt.figure(figsize=(14, 10), facecolor=_bg_col)
axes_list = [momentum_viz_fig.add_subplot(2, 2, i+1) for i in range(4)]
for _mom_ax in axes_list:
    _mom_ax.set_facecolor(_bg_col)

_mom_ax1, _mom_ax2, _mom_ax3, _mom_ax4 = axes_list

# 1. Momentum Trend Distribution
trend_counts = engagement_momentum_df['momentum_trend'].value_counts()
_tc_labels = trend_counts.index.astype(str).tolist()
_tc_vals   = trend_counts.values.tolist()
_mom_ax1.barh(range(len(_tc_labels)), _tc_vals, color=_mom_colors[0])
_mom_ax1.set_yticks(range(len(_tc_labels)))
_mom_ax1.set_yticklabels(_tc_labels, color=_txt_col)
_mom_ax1.set_xlabel('Number of Users', color=_txt_col)
_mom_ax1.set_title('User Distribution by Momentum Trend', color=_txt_col, fontsize=12, pad=15)
_mom_ax1.tick_params(colors=_txt_col)
for _sp in ['top', 'right']: _mom_ax1.spines[_sp].set_visible(False)
_mom_ax1.spines['bottom'].set_color(_sec_col)
_mom_ax1.spines['left'].set_color(_sec_col)

# 2. Churn Risk by Activity Level
_bins = [0, 5, 15, 50, float('inf')]
_labs = ['Very Low', 'Low', 'Medium', 'High']
engagement_momentum_df['activity_bin'] = pd.cut(
    engagement_momentum_df['current_7d_activity'], bins=_bins, labels=_labs)
churn_by_activity = engagement_momentum_df.groupby('activity_bin', observed=True)['churn_risk'].apply(
    lambda x: x.sum() / len(x) * 100 if len(x) > 0 else 0)
_cb_labels = churn_by_activity.index.astype(str).tolist()
_cb_vals   = churn_by_activity.values.tolist()
_mom_ax2.bar(range(len(_cb_labels)), _cb_vals, color=_mom_colors[3])
_mom_ax2.set_xticks(range(len(_cb_labels)))
_mom_ax2.set_xticklabels(_cb_labels, color=_txt_col)
_mom_ax2.set_ylabel('Churn Risk %', color=_txt_col)
_mom_ax2.set_title('Churn Risk by 7-Day Activity Level', color=_txt_col, fontsize=12, pad=15)
_mom_ax2.tick_params(colors=_txt_col)
for _sp in ['top', 'right']: _mom_ax2.spines[_sp].set_visible(False)
_mom_ax2.spines['bottom'].set_color(_sec_col)
_mom_ax2.spines['left'].set_color(_sec_col)

# 3. Acceleration vs Deceleration scatter
_mom_ax3.scatter(engagement_momentum_df['accel_periods'].tolist(),
                  engagement_momentum_df['decel_periods'].tolist(),
                  alpha=0.4, c=_mom_colors[2], s=20)
_mom_ax3.set_xlabel('Acceleration Periods', color=_txt_col)
_mom_ax3.set_ylabel('Deceleration Periods', color=_txt_col)
_mom_ax3.set_title('User Engagement Volatility', color=_txt_col, fontsize=12, pad=15)
_mom_ax3.tick_params(colors=_txt_col)
for _sp in ['top', 'right']: _mom_ax3.spines[_sp].set_visible(False)
_mom_ax3.spines['bottom'].set_color(_sec_col)
_mom_ax3.spines['left'].set_color(_sec_col)

# 4. Momentum Change Distribution
_mc = engagement_momentum_df['momentum_change']
_mc_clipped = _mc[_mc != 0].clip(-20, 20).tolist()
_mom_ax4.hist(_mc_clipped, bins=30, color=_mom_colors[4], edgecolor='none', alpha=0.8)
_mom_ax4.axvline(0, color=_txt_col, linestyle='--', alpha=0.5, linewidth=1)
_mom_ax4.set_xlabel('Momentum Change (events)', color=_txt_col)
_mom_ax4.set_ylabel('Number of Users', color=_txt_col)
_mom_ax4.set_title('Distribution of Momentum Changes', color=_txt_col, fontsize=12, pad=15)
_mom_ax4.tick_params(colors=_txt_col)
for _sp in ['top', 'right']: _mom_ax4.spines[_sp].set_visible(False)
_mom_ax4.spines['bottom'].set_color(_sec_col)
_mom_ax4.spines['left'].set_color(_sec_col)

plt.tight_layout()
print("\n✓ Momentum tracking visualizations created.")
