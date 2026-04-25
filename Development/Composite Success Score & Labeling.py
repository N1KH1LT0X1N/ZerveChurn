import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=" * 80)
print("COMPOSITE SUCCESS SCORE CALCULATION")
print("=" * 80)

# Define metric weights based on business value
# Higher weights for metrics that demonstrate clear business value
weights = {
    'long_term_retention': 0.30,    # Highest - shows sustained value
    'is_paid_user': 0.25,            # High - direct revenue
    'has_deployment': 0.20,          # Medium-high - shows product usage depth
    'is_power_user': 0.15,           # Medium - shows engagement potential
    'collaboration_success': 0.10     # Lower - nice-to-have feature
}

print("\n📊 Metric Weights:")
for _metric_name, _metric_weight in weights.items():
    print(f"  • {_metric_name}: {_metric_weight:.0%}")

# Calculate weighted success score (0-100 scale)
success_scores = []

for _, _row in user_success_metrics.iterrows():
    _score = 0
    _score += weights['long_term_retention'] * 100 if _row['long_term_retention'] else 0
    _score += weights['is_paid_user'] * 100 if _row['is_paid_user'] else 0
    _score += weights['has_deployment'] * 100 if _row['has_deployment'] else 0
    _score += weights['is_power_user'] * 100 if _row['is_power_user'] else 0
    _score += weights['collaboration_success'] * 100 if _row['collaboration_success'] else 0
    
    success_scores.append(_score)

user_success_metrics['success_score'] = success_scores

print(f"\n✓ Success scores calculated for {len(user_success_metrics):,} users")
print(f"\nScore Distribution:")
print(user_success_metrics['success_score'].describe().round(2))

# Assign success labels based on score thresholds
def assign_success_label(score):
    if score <= 30:
        return 'Failed'
    elif score <= 60:
        return 'Moderate'
    else:
        return 'Successful'

user_success_metrics['success_label'] = user_success_metrics['success_score'].apply(assign_success_label)

print("\n" + "=" * 80)
print("SUCCESS LABEL DISTRIBUTION")
print("=" * 80)

label_counts = user_success_metrics['success_label'].value_counts()
label_pcts = user_success_metrics['success_label'].value_counts(normalize=True) * 100

print("\nLabel Distribution:")
for _lbl in ['Failed', 'Moderate', 'Successful']:
    _cnt = label_counts.get(_lbl, 0)
    _lpct = label_pcts.get(_lbl, 0)
    print(f"  • {_lbl:12s}: {_cnt:5,d} ({_lpct:5.1f}%)")

# Calculate average metrics by success label
print("\n" + "=" * 80)
print("AVERAGE CHARACTERISTICS BY SUCCESS LABEL")
print("=" * 80)

success_characteristics = user_success_metrics.groupby('success_label').agg({
    'long_term_retention': 'mean',
    'is_paid_user': 'mean',
    'has_deployment': 'mean',
    'collaboration_success': 'mean',
    'is_power_user': 'mean',
    'total_events': 'mean',
    'tenure_days': 'mean',
    'success_score': 'mean'
})

success_characteristics = success_characteristics.reindex(['Failed', 'Moderate', 'Successful'])
success_characteristics['long_term_retention'] = success_characteristics['long_term_retention'] * 100
success_characteristics['is_paid_user'] = success_characteristics['is_paid_user'] * 100
success_characteristics['has_deployment'] = success_characteristics['has_deployment'] * 100
success_characteristics['collaboration_success'] = success_characteristics['collaboration_success'] * 100
success_characteristics['is_power_user'] = success_characteristics['is_power_user'] * 100

print("\n" + success_characteristics.round(2).to_string())

print("\n" + "=" * 80)
print("METRIC CONTRIBUTION ANALYSIS")
print("=" * 80)

# Calculate how each metric contributes to overall success
for _metric in ['long_term_retention', 'is_paid_user', 'has_deployment', 'collaboration_success', 'is_power_user']:
    successful_with_metric = len(user_success_metrics[(user_success_metrics['success_label'] == 'Successful') & (user_success_metrics[_metric] == True)])
    total_successful = len(user_success_metrics[user_success_metrics['success_label'] == 'Successful'])
    
    if total_successful > 0:
        _contribution = successful_with_metric / total_successful * 100
        print(f"\n{_metric}:")
        print(f"  • {successful_with_metric:,} / {total_successful:,} successful users have this metric ({_contribution:.1f}%)")

print("\n" + "=" * 80)
print("VISUALIZATION: SUCCESS LABEL DISTRIBUTION")
print("=" * 80)

# Create professional visualization
bg_color = '#1D1D20'
text_color = '#fbfbff'
secondary_text = '#909094'
zerve_colors = ['#FF9F9B', '#FFB482', '#8DE5A1']

success_dist_fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg_color)
ax.set_facecolor(bg_color)

# Create bar chart
labels = ['Failed', 'Moderate', 'Successful']
counts = [label_counts.get(_lbl, 0) for _lbl in labels]
colors = zerve_colors

bars = ax.bar(labels, counts, color=colors, alpha=0.8, edgecolor=text_color, linewidth=1.5)

# Add value labels on bars
for bar in bars:
    _height = bar.get_height()
    _bar_pct = _height / len(user_success_metrics) * 100
    ax.text(bar.get_x() + bar.get_width()/2., _height,
            f'{int(_height):,}\n({_bar_pct:.1f}%)',
            ha='center', va='bottom', color=text_color, fontsize=11, weight='bold')

ax.set_xlabel('Success Label', color=text_color, fontsize=12, weight='bold')
ax.set_ylabel('Number of Users', color=text_color, fontsize=12, weight='bold')
ax.set_title('User Success Distribution (0-100 Score Scale)', 
             color=text_color, fontsize=14, weight='bold', pad=20)

ax.tick_params(colors=text_color, labelsize=10)
ax.spines['bottom'].set_color(secondary_text)
ax.spines['left'].set_color(secondary_text)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2, color=secondary_text, linestyle='--')

plt.tight_layout()
print("\n✓ Distribution visualization created")

print("\n" + "=" * 80)
print("✅ COMPOSITE SUCCESS SCORE COMPLETE")
print("=" * 80)
print(f"\n📊 Final Dataset: {len(user_success_metrics):,} users with success scores and labels")
print(f"📊 Score range: {user_success_metrics['success_score'].min():.1f} - {user_success_metrics['success_score'].max():.1f}")
print(f"📊 Mean score: {user_success_metrics['success_score'].mean():.1f}")
print(f"📊 Median score: {user_success_metrics['success_score'].median():.1f}")
