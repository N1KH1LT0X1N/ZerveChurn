import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

print("=" * 80)
print("SURVIVAL ANALYSIS: SIMPLIFIED K-M & COX REGRESSION")
print("=" * 80)

# Zerve design colors
bg_color = '#1D1D20'
text_color = '#fbfbff'
secondary_text = '#909094'

# ============================================================================
# MANUAL KAPLAN-MEIER IMPLEMENTATION
# ============================================================================
print("\n📈 COMPUTING KAPLAN-MEIER SURVIVAL CURVES")
print("=" * 80)

def compute_kaplan_meier(time, event):
    """Compute Kaplan-Meier survival estimates"""
    # Sort by time
    _df_km = pd.DataFrame({'time': time, 'event': event}).sort_values('time')
    
    # Get unique event times
    unique_times = _df_km['time'].unique()
    
    survival_probs = []
    cum_survival = 1.0
    n_at_risk = len(_df_km)
    
    for t in unique_times:
        # Events and censored at this time
        at_time = _df_km[_df_km['time'] == t]
        n_events = at_time['event'].sum()
        n_at_time = len(at_time)
        
        if n_at_risk > 0 and n_events > 0:
            cum_survival *= (n_at_risk - n_events) / n_at_risk
        
        survival_probs.append(cum_survival)
        n_at_risk -= n_at_time
    
    return unique_times, np.array(survival_probs)

# ============================================================================
# 1. KAPLAN-MEIER BY RISK SEGMENT
# ============================================================================
kmf_risk_fig = plt.figure(figsize=(14, 8), facecolor=bg_color)
kmf_risk_ax = plt.gca()
kmf_risk_ax.set_facecolor(bg_color)

segments = ['Low Risk', 'Medium Risk', 'High Risk']
colors_risk = ['#8DE5A1', '#FFB482', '#FF9F9B']

for segment, color in zip(segments, colors_risk):
    segment_data = survival_data[survival_data['risk_segment'] == segment]
    if len(segment_data) > 0:
        times, surv_probs = compute_kaplan_meier(segment_data['time_to_event'].values, 
                                                  segment_data['churned'].values)
        kmf_risk_ax.step(times, surv_probs, where='post', color=color, linewidth=2.5, label=segment)
        
        # Median survival (50% survival)
        median_idx = np.where(surv_probs <= 0.5)[0]
        median_time = times[median_idx[0]] if len(median_idx) > 0 else np.nan
        print(f"   • {segment}: Median Survival = {median_time:.2f} days, N = {len(segment_data)}")

kmf_risk_ax.set_xlabel('Days Since First Activity', fontsize=13, color=text_color, fontweight='bold')
kmf_risk_ax.set_ylabel('Survival Probability (Retention)', fontsize=13, color=text_color, fontweight='bold')
kmf_risk_ax.set_title('Kaplan-Meier Survival Curves by Risk Segment', 
                       fontsize=16, color=text_color, fontweight='bold', pad=20)
kmf_risk_ax.tick_params(colors=text_color, labelsize=11)
kmf_risk_ax.spines['bottom'].set_color(secondary_text)
kmf_risk_ax.spines['left'].set_color(secondary_text)
kmf_risk_ax.spines['top'].set_visible(False)
kmf_risk_ax.spines['right'].set_visible(False)
kmf_risk_ax.legend(loc='best', frameon=True, facecolor=bg_color, edgecolor=secondary_text, 
                   fontsize=11, labelcolor=text_color)
kmf_risk_ax.grid(True, alpha=0.15, color=secondary_text, linestyle='--')
plt.tight_layout()

# ============================================================================
# 2. KAPLAN-MEIER BY ENGAGEMENT LEVEL
# ============================================================================
kmf_engagement_fig = plt.figure(figsize=(14, 8), facecolor=bg_color)
kmf_engagement_ax = plt.gca()
kmf_engagement_ax.set_facecolor(bg_color)

engagement_levels = ['Low', 'Medium', 'High']
colors_engagement = ['#FF9F9B', '#FFB482', '#8DE5A1']

for level, color in zip(engagement_levels, colors_engagement):
    level_data = survival_data[survival_data['engagement_level'] == level]
    if len(level_data) > 0:
        times, surv_probs = compute_kaplan_meier(level_data['time_to_event'].values, 
                                                  level_data['churned'].values)
        kmf_engagement_ax.step(times, surv_probs, where='post', color=color, linewidth=2.5, 
                                label=f'{level} Engagement')
        
        median_idx = np.where(surv_probs <= 0.5)[0]
        median_time = times[median_idx[0]] if len(median_idx) > 0 else np.nan
        print(f"   • {level}: Median Survival = {median_time:.2f} days, N = {len(level_data)}")

kmf_engagement_ax.set_xlabel('Days Since First Activity', fontsize=13, color=text_color, fontweight='bold')
kmf_engagement_ax.set_ylabel('Survival Probability (Retention)', fontsize=13, color=text_color, fontweight='bold')
kmf_engagement_ax.set_title('Kaplan-Meier Survival Curves by Engagement Level', 
                            fontsize=16, color=text_color, fontweight='bold', pad=20)
kmf_engagement_ax.tick_params(colors=text_color, labelsize=11)
kmf_engagement_ax.spines['bottom'].set_color(secondary_text)
kmf_engagement_ax.spines['left'].set_color(secondary_text)
kmf_engagement_ax.spines['top'].set_visible(False)
kmf_engagement_ax.spines['right'].set_visible(False)
kmf_engagement_ax.legend(loc='best', frameon=True, facecolor=bg_color, edgecolor=secondary_text,
                         fontsize=11, labelcolor=text_color)
kmf_engagement_ax.grid(True, alpha=0.15, color=secondary_text, linestyle='--')
plt.tight_layout()

# ============================================================================
# 3. KAPLAN-MEIER BY DEPLOYMENT STATUS
# ============================================================================
kmf_deployment_fig = plt.figure(figsize=(14, 8), facecolor=bg_color)
kmf_deployment_ax = plt.gca()
kmf_deployment_ax.set_facecolor(bg_color)

deployment_statuses = ['Has Deployment', 'No Deployment']
colors_deployment = ['#8DE5A1', '#FFB482']

for status, color in zip(deployment_statuses, colors_deployment):
    status_data = survival_data[survival_data['deployment_status'] == status]
    if len(status_data) > 0:
        times, surv_probs = compute_kaplan_meier(status_data['time_to_event'].values, 
                                                  status_data['churned'].values)
        kmf_deployment_ax.step(times, surv_probs, where='post', color=color, linewidth=2.5, label=status)
        
        median_idx = np.where(surv_probs <= 0.5)[0]
        median_time = times[median_idx[0]] if len(median_idx) > 0 else np.nan
        print(f"   • {status}: Median Survival = {median_time:.2f} days, N = {len(status_data)}")

kmf_deployment_ax.set_xlabel('Days Since First Activity', fontsize=13, color=text_color, fontweight='bold')
kmf_deployment_ax.set_ylabel('Survival Probability (Retention)', fontsize=13, color=text_color, fontweight='bold')
kmf_deployment_ax.set_title('Kaplan-Meier Survival Curves by Deployment Status', 
                            fontsize=16, color=text_color, fontweight='bold', pad=20)
kmf_deployment_ax.tick_params(colors=text_color, labelsize=11)
kmf_deployment_ax.spines['bottom'].set_color(secondary_text)
kmf_deployment_ax.spines['left'].set_color(secondary_text)
kmf_deployment_ax.spines['top'].set_visible(False)
kmf_deployment_ax.spines['right'].set_visible(False)
kmf_deployment_ax.legend(loc='best', frameon=True, facecolor=bg_color, edgecolor=secondary_text,
                         fontsize=11, labelcolor=text_color)
kmf_deployment_ax.grid(True, alpha=0.15, color=secondary_text, linestyle='--')
plt.tight_layout()

print("\n✅ KAPLAN-MEIER ANALYSIS COMPLETE")
print("=" * 80)
