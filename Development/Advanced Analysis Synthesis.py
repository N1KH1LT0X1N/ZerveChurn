import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("="*80)
print("ADVANCED ANALYSIS SYNTHESIS & UNIFIED INSIGHTS")
print("="*80)

# Suppress any inherited conflicting variables from upstream branches
# by immediately overwriting them with private _ versions
_category = None  # suppress inherited `category` from Collaboration Signature chain

# ============================================================================
# 1. IDENTIFY OVERLAPS: POWER USERS vs NETWORK HUBS vs ANOMALIES
# ============================================================================
print("\n" + "="*80)
print("1. OVERLAP ANALYSIS: POWER USERS, NETWORK HUBS, ANOMALIES")
print("="*80)

# Get power users from behavioral fingerprint
power_users_set = set(behavioral_fingerprint[
    behavioral_fingerprint['power_user_score'] >= behavioral_fingerprint['power_user_score'].quantile(0.90)
]['user_id'].values)

# Get network hubs from centrality analysis
network_hubs_set = set(centrality_network_df[
    centrality_network_df['is_super_connector'] == True
]['user_id'].values)

# Get anomalous users
anomalous_users_set = set(exceptional_users_df['user_id'].values)

print(f"\n📊 USER SEGMENT SIZES:")
print(f"   • Power Users (top 10%): {len(power_users_set):,}")
print(f"   • Network Hubs (top 10%): {len(network_hubs_set):,}")
print(f"   • Anomalous Users (5%): {len(anomalous_users_set):,}")

# Calculate overlaps
power_and_hub = power_users_set & network_hubs_set
power_and_anomaly = power_users_set & anomalous_users_set
hub_and_anomaly = network_hubs_set & anomalous_users_set
all_three = power_users_set & network_hubs_set & anomalous_users_set

print(f"\n🔄 OVERLAPS:")
print(f"   • Power Users + Network Hubs: {len(power_and_hub):,}")
print(f"   • Power Users + Anomalies: {len(power_and_anomaly):,}")
print(f"   • Network Hubs + Anomalies: {len(hub_and_anomaly):,}")
print(f"   • ALL THREE: {len(all_three):,}")

# Unique to each category
unique_power = power_users_set - network_hubs_set - anomalous_users_set
unique_hubs = network_hubs_set - power_users_set - anomalous_users_set
unique_anomaly = anomalous_users_set - power_users_set - network_hubs_set

print(f"\n🎯 UNIQUE TO EACH:")
print(f"   • Only Power Users: {len(unique_power):,}")
print(f"   • Only Network Hubs: {len(unique_hubs):,}")
print(f"   • Only Anomalies: {len(unique_anomaly):,}")

# ============================================================================
# 2. CHURN PATTERNS: How do anomalies differ?
# ============================================================================
print("\n" + "="*80)
print("2. CHURN PATTERNS BY USER SEGMENT")
print("="*80)

# Add segment labels to survival data
survival_with_segments = active_users_survival.copy()
survival_with_segments['is_power_user_segment'] = survival_with_segments['user_id'].isin(power_users_set)
survival_with_segments['is_network_hub'] = survival_with_segments['user_id'].isin(network_hubs_set)
survival_with_segments['is_anomalous'] = survival_with_segments['user_id'].isin(anomalous_users_set)

# Analyze churn risk by segment
segment_churn_analysis = []

for segment_name, segment_mask in [
    ('Power Users', survival_with_segments['is_power_user_segment']),
    ('Network Hubs', survival_with_segments['is_network_hub']),
    ('Anomalous Users', survival_with_segments['is_anomalous']),
    ('Regular Users', ~(survival_with_segments['is_power_user_segment'] |
                        survival_with_segments['is_network_hub'] |
                        survival_with_segments['is_anomalous']))
]:
    segment_data = survival_with_segments[segment_mask]
    if len(segment_data) > 0:
        segment_churn_analysis.append({
            'Segment': segment_name,
            'Count': len(segment_data),
            'Avg Success': segment_data['success_score'].mean(),
            'High Tier %': (segment_data['success_score'] >= 65).sum() / len(segment_data) * 100,
            'Avg Days Inactive': segment_data['days_since_last'].mean(),
            'Avg Total Events': segment_data['total_events'].mean()
        })

segment_churn_df = pd.DataFrame(segment_churn_analysis)
print(f"\n📊 SUCCESS SCORE (LTV PROXY) BY SEGMENT:")
print(segment_churn_df.to_string(index=False))

# ============================================================================
# 3. CROSS-REFERENCE ALL ANALYSES
# ============================================================================
print("\n" + "="*80)
print("3. COMPREHENSIVE USER PROFILES")
print("="*80)

# Create unified user profile
all_active_users = set(active_users_survival['user_id'].values)
unified_profiles = []

for _uid in all_active_users:
    # Get churn data
    user_churn = active_users_survival[active_users_survival['user_id'] == _uid].iloc[0]

    # Check behavioral data
    _ubeh = behavioral_fingerprint[behavioral_fingerprint['user_id'] == _uid]
    if len(_ubeh) > 0:
        _ubeh = _ubeh.iloc[0]
        power_score = _ubeh['power_user_score']
        struggle_score = _ubeh['struggle_score']
    else:
        power_score = 0
        struggle_score = 0

    # Check network centrality
    _unet = centrality_network_df[centrality_network_df['user_id'] == _uid]
    if len(_unet) > 0:
        composite_centrality = _unet.iloc[0]['composite_centrality']
        is_super_connector = _unet.iloc[0]['is_super_connector']
    else:
        composite_centrality = 0
        is_super_connector = False

    unified_profiles.append({
        'user_id': _uid,
        'success_score': user_churn['success_score'],
        # churn_proxy: low success → high churn risk. Carried through
        # unified_df so downstream consumers (Weekly Delta Metrics,
        # Integrated Dashboard, etc.) can use a correctly-oriented threshold.
        'churn_proxy': max(0.0, min(100.0, 100.0 - float(user_churn['success_score']))),
        'success_tier': user_churn['success_tier'],
        'total_events': user_churn['total_events'],
        'days_since_last': user_churn['days_since_last'],
        'is_power_user': _uid in power_users_set,
        'is_network_hub': _uid in network_hubs_set,
        'is_anomalous': _uid in anomalous_users_set,
        'power_user_score': power_score,
        'struggle_score': struggle_score,
        'network_centrality': composite_centrality,
        'is_super_connector': is_super_connector
    })

unified_df = pd.DataFrame(unified_profiles)

print(f"\n✓ Created unified profiles for {len(unified_df):,} active users")
print(f"\nProfile includes:")
print(f"   • Churn risk metrics")
print(f"   • Power user indicators")
print(f"   • Network centrality scores")
print(f"   • Anomaly detection flags")

# ============================================================================
# 4. ACTIONABLE INSIGHTS
# ============================================================================
print("\n" + "="*80)
print("4. KEY ACTIONABLE INSIGHTS")
print("="*80)

actionable_insights = []

# Insight 1: Elite users at risk.
# Use churn_proxy >= 50 (i.e. success_score <= 50) so we actually catch
# elite users whose churn risk is high — previous filter on success_score >= 50
# was selecting the *least* at-risk elites. See docs/repo_state_and_next_steps.md §6.
elite_at_risk = unified_df[
    ((unified_df['is_power_user']) | (unified_df['is_network_hub'])) &
    (unified_df['churn_proxy'] >= 50)
]
if len(elite_at_risk) > 0:
    actionable_insights.append({
        'priority': 'CRITICAL',
        'insight': f"{len(elite_at_risk)} high-value users (power users/network hubs) show churn risk ≥50%",
        'action': "Implement VIP retention program with personalized outreach"
    })

# Insight 2: Anomalous behavior patterns
anomaly_low_activity = unified_df[
    (unified_df['is_anomalous']) &
    (unified_df['days_since_last'] >= 14)
]
if len(anomaly_low_activity) > 0:
    actionable_insights.append({
        'priority': 'HIGH',
        'insight': f"{len(anomaly_low_activity)} anomalous users have been inactive 14+ days",
        'action': "Investigate anomaly causes - potential product friction or success completion"
    })

# Insight 3: Network hubs not power users
hubs_not_power = unified_df[
    (unified_df['is_network_hub']) &
    (~unified_df['is_power_user'])
]
if len(hubs_not_power) > 0:
    actionable_insights.append({
        'priority': 'MEDIUM',
        'insight': f"{len(hubs_not_power)} network hubs have low power user scores",
        'action': "Target these connectors to amplify engagement through their networks"
    })

# Insight 4: Power users with low centrality
power_no_network = unified_df[
    (unified_df['is_power_user']) &
    (~unified_df['is_network_hub']) &
    (unified_df['network_centrality'] < 0.1)
]
if len(power_no_network) > 0:
    actionable_insights.append({
        'priority': 'MEDIUM',
        'insight': f"{len(power_no_network)} power users operate in isolation (low network centrality)",
        'action': "Encourage collaboration features to increase stickiness and value"
    })

# Insight 5: Regular users with high engagement
regular_high_engagement = unified_df[
    (~unified_df['is_power_user']) &
    (~unified_df['is_network_hub']) &
    (~unified_df['is_anomalous']) &
    (unified_df['total_events'] >= unified_df['total_events'].quantile(0.75)) &
    (unified_df['success_score'] < 30)
]
if len(regular_high_engagement) > 0:
    actionable_insights.append({
        'priority': 'OPPORTUNITY',
        'insight': f"{len(regular_high_engagement)} regular users show high engagement and low churn risk",
        'action': "Nurture to power user status through feature education and advanced capabilities"
    })

print(f"\n🎯 GENERATED {len(actionable_insights)} ACTIONABLE INSIGHTS:\n")
for _ins_i, insight_item in enumerate(actionable_insights, 1):
    print(f"{_ins_i}. [{insight_item['priority']}]")
    print(f"   📊 Finding: {insight_item['insight']}")
    print(f"   💡 Action: {insight_item['action']}\n")

# Store for dashboard
actionable_insights_list = actionable_insights

# ============================================================================
# 5. SUCCESS DISTRIBUTION VISUALIZATION
# ============================================================================
_bg_c   = '#1D1D20'
_txt_c  = '#fbfbff'
_clr_c  = ['#A1C9F4','#FFB482','#8DE5A1']

success_dist_fig = plt.figure(figsize=(10, 5), facecolor=_bg_c)
_sd_ax = success_dist_fig.add_subplot(1, 2, 1)
_sd_ax.set_facecolor(_bg_c)
_sd_labels = ['Successful', 'Not Successful']
_sd_counts = [
    user_success_metrics['success_label'].eq('Successful').sum(),
    user_success_metrics['success_label'].ne('Successful').sum()
]
_sd_bars = _sd_ax.bar(_sd_labels, _sd_counts, color=[_clr_c[2], _clr_c[0]])
for _bar in _sd_bars:
    _sd_ax.text(_bar.get_x() + _bar.get_width()/2, _bar.get_height() + 5,
                f'{int(_bar.get_height()):,}', ha='center', va='bottom',
                fontsize=10, color=_txt_c)
_sd_ax.set_title('Success Label Distribution', color=_txt_c)
_sd_ax.tick_params(colors=_txt_c)

_sd_ax2 = success_dist_fig.add_subplot(1, 2, 2)
_sd_ax2.set_facecolor(_bg_c)
_alt_dist = user_success_metrics['alternative_label'].value_counts()
_sd_ax2.bar(_alt_dist.index, _alt_dist.values, color=[_clr_c[_i % len(_clr_c)] for _i in range(len(_alt_dist))])
_sd_ax2.set_title('Alternative Label Distribution', color=_txt_c)
_sd_ax2.tick_params(colors=_txt_c, axis='x', rotation=30)

plt.tight_layout()
print("\n✅ ADVANCED ANALYSIS SYNTHESIS COMPLETE")

# ============================================================================
# CLEAN UP: Suppress loop variables that would conflict with other branches
# in the merged namespace when downstream blocks have two upstream sources.
# These are all scalars/strings from loop iterations — they serve no 
# downstream purpose and MUST be made private to prevent merge conflicts.
# ============================================================================
# Suppress inherited loop-variable leftovers from this block's upstream chains
# that overlap with the Interactive Visualizations & Segment Export branch
try:
    del pct
except NameError:
    pass
try:
    del label
except NameError:
    pass
try:
    del metric
except NameError:
    pass
try:
    del weight
except NameError:
    pass
try:
    del score
except NameError:
    pass
try:
    del segment
except NameError:
    pass
try:
    del color
except NameError:
    pass
try:
    del status
except NameError:
    pass
try:
    del window
except NameError:
    pass
try:
    del name
except NameError:
    pass
try:
    del cat
except NameError:
    pass
try:
    del alt_score
except NameError:
    pass
try:
    del count
except NameError:
    pass
try:
    del level
except NameError:
    pass
try:
    del utype
except NameError:
    pass

print("✓ Namespace cleanup complete — conflict variables suppressed.")
