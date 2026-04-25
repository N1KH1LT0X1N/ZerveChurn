import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Use event_categories from upstream: {event_name: category_string}
feat_cats = event_categories  # dict: event → category

# Work on a lightweight subset of columns
df_feat = user_retention[['distinct_id', 'timestamp', 'event']].copy()
df_feat['timestamp'] = pd.to_datetime(df_feat['timestamp'])
df_feat['feature_category'] = df_feat['event'].map(feat_cats)

# Drop rows where category is NaN
df_feat = df_feat.dropna(subset=['feature_category'])

# ── Per-user first event date (vectorised)
user_start = df_feat.groupby('distinct_id')['timestamp'].min().rename('user_start')
df_feat = df_feat.merge(user_start, on='distinct_id')
df_feat['days_since_start'] = (df_feat['timestamp'] - df_feat['user_start']).dt.total_seconds() / 86400

# ── Feature adoption summary per user
feature_evolution_df = df_feat.groupby('distinct_id').agg(
    total_features_used=('feature_category', 'nunique'),
    total_days_active=('days_since_start', 'max'),
).reset_index()

# Weekly feature diversity: week number per event row
df_feat['week'] = (df_feat['days_since_start'] // 7).astype(int)
weekly_div = df_feat.groupby(['distinct_id', 'week'])['feature_category'].nunique().reset_index()
weekly_div_agg = weekly_div.groupby('distinct_id')['feature_category'].agg(
    avg_weekly_feature_diversity='mean',
    max_weekly_feature_diversity='max'
).reset_index()
feature_evolution_df = feature_evolution_df.merge(weekly_div_agg, on='distinct_id', how='left')
feature_evolution_df[['avg_weekly_feature_diversity', 'max_weekly_feature_diversity']] = \
    feature_evolution_df[['avg_weekly_feature_diversity', 'max_weekly_feature_diversity']].fillna(0)

# ── Gateway features (vectorised)
unique_categories = df_feat['feature_category'].unique().tolist()
user_features_set = df_feat.groupby('distinct_id')['feature_category'].apply(set)

gateway_analysis = []
for fc in unique_categories:
    users_with = df_feat[df_feat['feature_category'] == fc]['distinct_id'].unique()
    users_without = feature_evolution_df[~feature_evolution_df['distinct_id'].isin(users_with)]['distinct_id']
    
    with_eng = feature_evolution_df[feature_evolution_df['distinct_id'].isin(users_with)]['total_features_used'].mean()
    without_eng = feature_evolution_df[feature_evolution_df['distinct_id'].isin(users_without)]['total_features_used'].mean() \
        if len(users_without) > 0 else 0
    
    gateway_analysis.append({
        'feature': fc,
        'users_count': len(users_with),
        'avg_features_used_with': with_eng,
        'avg_features_used_without': without_eng if not np.isnan(without_eng) else 0,
        'engagement_lift': with_eng - (without_eng if not np.isnan(without_eng) else 0)
    })

gateway_features_df = pd.DataFrame(gateway_analysis).sort_values('engagement_lift', ascending=False)

# ── Adoption sequences (vectorised, sample 1000 users)
first_use_by_feature = df_feat.groupby(['distinct_id', 'feature_category'])['days_since_start'].min().reset_index()
first_use_by_feature.columns = ['distinct_id', 'feature_category', 'first_day']

sample_users = feature_evolution_df['distinct_id'].head(1000)
seq_data = first_use_by_feature[first_use_by_feature['distinct_id'].isin(sample_users)]
seq_data_sorted = seq_data.sort_values(['distinct_id', 'first_day'])

adoption_sequences_list = []
for uid, grp in seq_data_sorted.groupby('distinct_id'):
    seq = grp['feature_category'].tolist()[:5]
    if len(seq) >= 2:
        adoption_sequences_list.append(' → '.join(seq))

from collections import Counter
common_sequences = Counter(adoption_sequences_list).most_common(15)

# ── Output
print("=" * 70)
print("FEATURE ADOPTION EVOLUTION")
print("=" * 70)
print(f"\nTotal Users Analyzed: {len(feature_evolution_df):,}")
print(f"Unique Feature Categories: {df_feat['feature_category'].nunique()}")
print()
print("Average Feature Adoption Metrics:")
print(f"  - Mean features used per user:    {feature_evolution_df['total_features_used'].mean():.2f}")
print(f"  - Median features used per user:  {feature_evolution_df['total_features_used'].median():.0f}")
print(f"  - Mean weekly feature diversity:  {feature_evolution_df['avg_weekly_feature_diversity'].mean():.2f}")
print()
print("Top Gateway Features (Highest Engagement Lift):")
print(gateway_features_df.head(10)[['feature', 'users_count', 'engagement_lift']].to_string(index=False))
print()
if common_sequences:
    print("Most Common Feature Adoption Sequences:")
    for seq, cnt in common_sequences[:10]:
        print(f"  {cnt:3d} users: {seq}")

# ── Visualisation
bg_col  = '#1D1D20'
txt_col = '#fbfbff'
sec_col = '#909094'
zerve_colors = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF',
                '#1F77B4', '#9467BD', '#8C564B', '#C49C94', '#E377C2']

# Chart 1: Feature diversity distribution
feature_adoption_chart = plt.figure(figsize=(12, 5), facecolor=bg_col)
feature_adoption_chart.patch.set_facecolor(bg_col)

ax_a = feature_adoption_chart.add_subplot(1, 2, 1)
ax_a.set_facecolor(bg_col)
counts_vals = feature_evolution_df['total_features_used'].value_counts().sort_index()
ax_a.bar(counts_vals.index.astype(str).tolist(), counts_vals.values.tolist(),
         color=zerve_colors[0], edgecolor='none', alpha=0.85)
ax_a.set_xlabel('# Feature Categories Used', color=txt_col, fontsize=11)
ax_a.set_ylabel('# Users', color=txt_col, fontsize=11)
ax_a.set_title('Feature Adoption Distribution', color=txt_col, fontsize=13, fontweight='bold', pad=12)
ax_a.tick_params(colors=txt_col)
for sp in ['top', 'right']: ax_a.spines[sp].set_visible(False)
ax_a.spines['bottom'].set_color(sec_col)
ax_a.spines['left'].set_color(sec_col)

# Chart 2: Top gateway features
ax_b = feature_adoption_chart.add_subplot(1, 2, 2)
ax_b.set_facecolor(bg_col)
top_gw = gateway_features_df.head(10)
gw_labels = [str(f)[:20] for f in top_gw['feature'].tolist()]
gw_vals   = top_gw['engagement_lift'].tolist()
ax_b.barh(range(len(gw_labels)), gw_vals, color=zerve_colors[2], edgecolor='none', alpha=0.85)
ax_b.set_yticks(range(len(gw_labels)))
ax_b.set_yticklabels(gw_labels, color=txt_col, fontsize=9)
ax_b.set_xlabel('Engagement Lift (# additional features used)', color=txt_col, fontsize=10)
ax_b.set_title('Top Gateway Features', color=txt_col, fontsize=13, fontweight='bold', pad=12)
ax_b.tick_params(colors=txt_col)
for sp in ['top', 'right']: ax_b.spines[sp].set_visible(False)
ax_b.spines['bottom'].set_color(sec_col)
ax_b.spines['left'].set_color(sec_col)

plt.tight_layout()
print("\n✓ Feature adoption visualizations created.")
