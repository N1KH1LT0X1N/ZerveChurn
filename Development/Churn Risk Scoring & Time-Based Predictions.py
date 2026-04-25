import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

print("=" * 80)
print("SUCCESS SCORING (LTV PROXY) & TIME-BASED CHURN-WINDOW PREDICTIONS")
print("=" * 80)

# ============================================================================
# USE BEST MODEL FROM ENSEMBLE TO SCORE ALL ACTIVE USERS
# ============================================================================
print("\n🎯 CALCULATING SUCCESS (LTV-PROXY) SCORES FOR ALL ACTIVE USERS")
print("=" * 80)

# Prepare features for scoring
active_users_survival = survival_data[survival_data['churned'] == 0].copy()

scoring_features_active = ['total_events', 'tenure_days', 'days_since_first', 'days_since_last']

# Convert boolean to int for modeling
X_score_active = active_users_survival[scoring_features_active].copy()

# Use best model to predict P(High-Value) probability (rebranded as success score)
active_success_proba = best_model_obj.predict_proba(scaler_prep.transform(X_score_active))[:, 1]

# Scale to 0-100
success_scores_active = (active_success_proba * 100).round(2)

# Add to dataframe
active_users_survival['success_score'] = success_scores_active

# Derived churn-risk proxy: a high success score implies low churn risk and
# vice versa, so flipping the 0-100 scale gives a column whose semantics
# match every "risk_threshold >= X" comparison downstream. Several consumers
# (LTV, Weekly Delta, Integrated Dashboard, Hybrid GNN, Advanced Analysis)
# previously read `success_score` with churn-risk thresholds — that's the
# inversion residual called out in docs/pipeline_deep_dive.md §3.2 and
# docs/repo_state_and_next_steps.md §6. Migrating those call sites to use
# `churn_proxy` makes the comparison direction match the variable name.
active_users_survival['churn_proxy'] = (100.0 - active_users_survival['success_score']).clip(0, 100)

# Categorize success tier
def _categorize_success_tier(score):
    if score < 30:
        return 'Low'
    elif score < 65:
        return 'Medium'
    else:
        return 'High'

active_users_survival['success_tier'] = active_users_survival['success_score'].apply(_categorize_success_tier)

print(f"✓ Calculated success scores for {len(active_users_survival):,} active users")
print(f"\n📊 SUCCESS SCORE DISTRIBUTION")
print(f"   • Mean: {active_users_survival['success_score'].mean():.2f}")
print(f"   • Median: {active_users_survival['success_score'].median():.2f}")
print(f"   • Min: {active_users_survival['success_score'].min():.2f}")
print(f"   • Max: {active_users_survival['success_score'].max():.2f}")

success_tier_dist_active = active_users_survival['success_tier'].value_counts()
for cat in ['Low', 'Medium', 'High']:
    count = success_tier_dist_active.get(cat, 0)
    pct = (count / len(active_users_survival)) * 100
    print(f"   • {cat}: {count:,} ({pct:.1f}%)")

# ============================================================================
# TIME-BASED CHURN PREDICTIONS (30/60/90 DAYS)
# ============================================================================
print("\n📅 30/60/90-DAY CHURN PREDICTIONS")
print("=" * 80)

# For time-based predictions, we'll use a simpler heuristic approach
# combined with the churn risk score

def predict_churn_window(row):
    """Predict when a user is likely to churn based on their profile.
    Uses `churn_proxy` (= 100 - success_score) so high values correctly
    signal high churn risk — fixes the inversion flagged in
    docs/repo_state_and_next_steps.md §6.
    """
    risk = row['churn_proxy']
    days_inactive = row['days_since_last']

    # High risk + already inactive
    if risk >= 65 and days_inactive >= 14:
        return '30_days'
    elif risk >= 50 and days_inactive >= 7:
        return '60_days'
    elif risk >= 30:
        return '90_days'
    else:
        return 'beyond_90_days'

active_users_survival['predicted_churn_window'] = active_users_survival.apply(predict_churn_window, axis=1)

churn_window_dist = active_users_survival['predicted_churn_window'].value_counts()
print(f"\n🔮 PREDICTED CHURN TIMING")
for window in ['30_days', '60_days', '90_days', 'beyond_90_days']:
    count = churn_window_dist.get(window, 0)
    pct = (count / len(active_users_survival)) * 100
    print(f"   • {window.replace('_', ' ').title()}: {count:,} ({pct:.1f}%)")

# ============================================================================
# IDENTIFY CRITICAL VULNERABILITY WINDOWS
# ============================================================================
print("\n⚠️  CRITICAL VULNERABILITY WINDOWS")
print("=" * 80)

# Analyze churn timing from historical data
churned_users_survival = survival_data[survival_data['churned'] == 1].copy()

# Group by time windows
vulnerability_windows = [
    ('0-7 days', 0, 7),
    ('8-14 days', 8, 14),
    ('15-30 days', 15, 30),
    ('31-60 days', 31, 60),
    ('61-90 days', 61, 90),
    ('90+ days', 91, 999)
]

window_churn_counts = []
for name, min_days, max_days in vulnerability_windows:
    count = len(churned_users_survival[
        (churned_users_survival['tenure_days'] >= min_days) & 
        (churned_users_survival['tenure_days'] <= max_days)
    ])
    window_churn_counts.append((name, count))

total_churned = len(churned_users_survival)
print(f"\n📊 HISTORICAL CHURN TIMING (Total Churned: {total_churned:,})")
for name, count in window_churn_counts:
    pct = (count / total_churned) * 100 if total_churned > 0 else 0
    print(f"   • {name}: {count:,} ({pct:.1f}%)")

# Identify highest risk period
max_churn_window = max(window_churn_counts, key=lambda x: x[1])
print(f"\n🚨 HIGHEST CHURN RISK PERIOD: {max_churn_window[0]}")

# ============================================================================
# EARLY WARNING ALERTS
# ============================================================================
print("\n🚨 EARLY WARNING ALERTS: HIGH-RISK ACTIVE USERS")
print("=" * 80)

# Identify users needing immediate attention.
# Filter on `churn_proxy >= 70` (i.e. success_score <= 30) so that "high risk"
# really means high churn-risk; the previous `success_score >= 70` filter was
# selecting the *most successful* users as alerts. See docs/repo_state_and_next_steps.md §6.
high_risk_alerts = active_users_survival[
    (active_users_survival['churn_proxy'] >= 70) &
    (active_users_survival['days_since_last'] >= 7)
].sort_values('churn_proxy', ascending=False)

print(f"\n⚠️  {len(high_risk_alerts)} USERS REQUIRE IMMEDIATE ATTENTION")
if len(high_risk_alerts) > 0:
    print(f"   • Average Churn-Risk Proxy: {high_risk_alerts['churn_proxy'].mean():.1f}%")
    print(f"   • Average Days Inactive: {high_risk_alerts['days_since_last'].mean():.1f}")
    print(f"   • Avg Total Events: {high_risk_alerts['total_events'].mean():.0f}")

# Zerve design colors
bg_color = '#1D1D20'
text_color = '#fbfbff'
secondary_text = '#909094'

# Visualization: Risk Score Distribution
risk_dist_fig = plt.figure(figsize=(14, 7), facecolor=bg_color)
risk_ax = plt.gca()
risk_ax.set_facecolor(bg_color)

risk_ax.hist(active_users_survival['success_score'], bins=20, color='#FFB482', 
             edgecolor='#fbfbff', alpha=0.8, linewidth=1.5)
risk_ax.axvline(30, color='#8DE5A1', linestyle='--', linewidth=2, label='Low Tier Threshold')
risk_ax.axvline(65, color='#FF9F9B', linestyle='--', linewidth=2, label='High Tier Threshold')

risk_ax.set_xlabel('Success Score (0-100, LTV proxy)', fontsize=13, color=text_color, fontweight='bold')
risk_ax.set_ylabel('Number of Active Users', fontsize=13, color=text_color, fontweight='bold')
risk_ax.set_title('Success Score Distribution for Active Users', 
                  fontsize=16, color=text_color, fontweight='bold', pad=20)
risk_ax.tick_params(colors=text_color, labelsize=11)
risk_ax.spines['bottom'].set_color(secondary_text)
risk_ax.spines['left'].set_color(secondary_text)
risk_ax.spines['top'].set_visible(False)
risk_ax.spines['right'].set_visible(False)
risk_ax.legend(loc='best', frameon=True, facecolor=bg_color, edgecolor=secondary_text,
               fontsize=11, labelcolor=text_color)
risk_ax.grid(True, alpha=0.15, color=secondary_text, linestyle='--', axis='y')
plt.tight_layout()

print("\n✅ SUCCESS SCORING & CHURN-WINDOW PREDICTIONS COMPLETE")
print("=" * 80)
