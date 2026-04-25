import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 80)
print("SURVIVAL ANALYSIS: DATA PREPARATION")
print("=" * 80)

# Define churn/censoring
current_date_analysis = pd.Timestamp('2025-12-09')  # Analysis date
churn_threshold_days = 30  # No activity in 30 days = churned

print(f"\n📅 Analysis Date: {current_date_analysis}")
print(f"⏱️  Churn Threshold: {churn_threshold_days} days of inactivity")

# Create survival dataset
survival_data = user_base.copy()

# Calculate time to event (days from first activity to last activity or censoring)
survival_data['time_to_event'] = survival_data['tenure_days']

# Determine churn status (1 = churned/event occurred, 0 = censored/still active)
survival_data['churned'] = (survival_data['days_since_last'] > churn_threshold_days).astype(int)

# For active users, time to event is from first activity to analysis date
survival_data.loc[survival_data['churned'] == 0, 'time_to_event'] = survival_data.loc[survival_data['churned'] == 0, 'days_since_first']

# Remove users with zero or negative time
survival_data = survival_data[survival_data['time_to_event'] > 0].copy()

# Define risk segments based on behavioral features
def assign_risk_segment(row):
    """Assign risk segment based on user characteristics"""
    risk_score = 0
    
    # Low engagement
    if row['total_events'] < 10:
        risk_score += 2
    
    # Recent inactivity
    if row['days_since_last'] > 7:
        risk_score += 2
    
    # Short tenure
    if row['tenure_days'] < 7:
        risk_score += 1
    
    # No paid/deployment/collaboration
    if not row['is_paid_user']:
        risk_score += 1
    if not row['has_deployment']:
        risk_score += 1
    if not row['collaboration_success']:
        risk_score += 1
    
    if risk_score >= 5:
        return 'High Risk'
    elif risk_score >= 3:
        return 'Medium Risk'
    else:
        return 'Low Risk'

survival_data['risk_segment'] = survival_data.apply(assign_risk_segment, axis=1)

# Create segmentation variables for stratified analysis
survival_data['user_type'] = 'Free User'
survival_data.loc[survival_data['is_paid_user'], 'user_type'] = 'Paid User'

survival_data['engagement_level'] = 'Low'
survival_data.loc[survival_data['total_events'] >= 50, 'engagement_level'] = 'High'
survival_data.loc[(survival_data['total_events'] >= 10) & (survival_data['total_events'] < 50), 'engagement_level'] = 'Medium'

survival_data['deployment_status'] = 'No Deployment'
survival_data.loc[survival_data['has_deployment'], 'deployment_status'] = 'Has Deployment'

# Calculate time to first deployment and first upgrade (if applicable)
# For now, we'll use has_deployment as a binary indicator
survival_data['time_to_deployment'] = np.nan
survival_data.loc[survival_data['has_deployment'], 'time_to_deployment'] = survival_data.loc[survival_data['has_deployment'], 'tenure_days']

print("\n📊 SURVIVAL COHORT SUMMARY")
print("=" * 80)
print(f"Total Users: {len(survival_data):,}")
print(f"Churned Users: {survival_data['churned'].sum():,} ({survival_data['churned'].mean()*100:.1f}%)")
print(f"Active Users (Censored): {(1-survival_data['churned']).sum():,} ({(1-survival_data['churned'].mean())*100:.1f}%)")

print("\n🎯 RISK SEGMENT DISTRIBUTION")
risk_dist = survival_data['risk_segment'].value_counts()
for segment in ['High Risk', 'Medium Risk', 'Low Risk']:
    count = risk_dist.get(segment, 0)
    pct = (count / len(survival_data)) * 100
    print(f"   • {segment}: {count:,} ({pct:.1f}%)")

print("\n📈 ENGAGEMENT LEVEL DISTRIBUTION")
engagement_dist = survival_data['engagement_level'].value_counts()
for level in ['Low', 'Medium', 'High']:
    count = engagement_dist.get(level, 0)
    pct = (count / len(survival_data)) * 100
    churn_rate = survival_data[survival_data['engagement_level'] == level]['churned'].mean() * 100
    print(f"   • {level}: {count:,} ({pct:.1f}%) - Churn Rate: {churn_rate:.1f}%")

print("\n💼 USER TYPE DISTRIBUTION")
user_type_dist = survival_data['user_type'].value_counts()
for utype in ['Free User', 'Paid User']:
    count = user_type_dist.get(utype, 0)
    pct = (count / len(survival_data)) * 100
    churn_rate = survival_data[survival_data['user_type'] == utype]['churned'].mean() * 100
    print(f"   • {utype}: {count:,} ({pct:.1f}%) - Churn Rate: {churn_rate:.1f}%")

print("\n📊 TIME TO EVENT STATISTICS")
print(f"   • Mean: {survival_data['time_to_event'].mean():.2f} days")
print(f"   • Median: {survival_data['time_to_event'].median():.2f} days")
print(f"   • Min: {survival_data['time_to_event'].min():.2f} days")
print(f"   • Max: {survival_data['time_to_event'].max():.2f} days")

print("\n✅ SURVIVAL DATA PREPARATION COMPLETE")
print("=" * 80)
