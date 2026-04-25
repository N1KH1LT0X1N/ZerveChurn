import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch

# =====================================================
# CREATE HIERARCHICAL VISUALIZATION OF EVENT TAXONOMY
# =====================================================

print("=" * 80)
print("CREATING HIERARCHICAL VISUALIZATIONS")
print("=" * 80)

# Extract data from taxonomy and workflow mapping
cat_summary = taxonomy_reference['category_summary']
stage_summary = workflow_mapping['stage_summary']

# Set Zerve design system colors
bg_color = '#1D1D20'
text_color = '#fbfbff'
zerve_colors = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF',
                '#1F77B4', '#9467BD', '#8C564B', '#C49C94', '#E377C2', '#F7B6D2']

# =====================================================
# VISUALIZATION 1: WORKFLOW STAGE FUNNEL
# =====================================================

workflow_funnel = plt.figure(figsize=(14, 10))
workflow_funnel.patch.set_facecolor(bg_color)
ax1 = workflow_funnel.add_subplot(111)
ax1.set_facecolor(bg_color)

# Get stages in progression order
stages_ordered = []
for stage in workflow_mapping['progression_order']:
    if stage in workflow_mapping['stage_stats']:
        stats = workflow_mapping['stage_stats'][stage]
        stages_ordered.append({
            'stage': stage,
            'count': stats['count'],
            'events': len(stats['events']),
            '_pct': (stats['count'] / workflow_mapping['stage_stats'][stage]['count'] * 100) if stats['count'] > 0 else 0
        })

stage_names = [s['stage'] for s in stages_ordered]
stage_counts = [s['count'] for s in stages_ordered]
stage_events = [s['events'] for s in stages_ordered]

y_positions = np.arange(len(stage_names))
bars = ax1.barh(y_positions, stage_counts, color=zerve_colors[:len(stage_names)])

# Add value labels
for i, (count, events) in enumerate(zip(stage_counts, stage_events)):
    ax1.text(count + max(stage_counts)*0.01, i, f'{count:,} ({events} events)', 
             va='center', ha='left', color=text_color, fontsize=11, fontweight='bold')

ax1.set_yticks(y_positions)
ax1.set_yticklabels(stage_names, color=text_color, fontsize=11)
ax1.set_xlabel('Event Occurrences', color=text_color, fontsize=12, fontweight='bold')
ax1.set_title('User Workflow Progression - Event Distribution by Stage', 
              color=text_color, fontsize=14, fontweight='bold', pad=20)
ax1.tick_params(colors=text_color)
ax1.spines['bottom'].set_color(text_color)
ax1.spines['left'].set_color(text_color)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='x', alpha=0.2, color=text_color)

plt.tight_layout()
workflow_funnel.show()

# =====================================================
# VISUALIZATION 2: EVENT CATEGORY TREEMAP (Simulated)
# =====================================================

category_treemap = plt.figure(figsize=(16, 10))
category_treemap.patch.set_facecolor(bg_color)
ax2 = category_treemap.add_subplot(111)
ax2.set_facecolor(bg_color)

# Get top 15 categories
top_cats = cat_summary.nlargest(15, 'Total Occurrences')

# Create horizontal stacked bars showing event distribution
y_pos = np.arange(len(top_cats))
bars = ax2.barh(y_pos, top_cats['Total Occurrences'], 
                color=[zerve_colors[i % len(zerve_colors)] for i in range(len(top_cats))])

# Add labels with category name, count, and percentage (use _pct to avoid conflict)
for i, (idx, row) in enumerate(top_cats.iterrows()):
    count = row['Total Occurrences']
    _pct = row['Percentage']  # private: not exported to downstream blocks
    events = row['Event Count']
    label = f"{row['Category']}: {count:,} ({_pct:.1f}%) - {events} events"
    ax2.text(count/2, i, label, va='center', ha='center', 
             color='#1D1D20', fontsize=10, fontweight='bold')

ax2.set_yticks([])
ax2.set_xlabel('Event Occurrences', color=text_color, fontsize=12, fontweight='bold')
ax2.set_title('Event Categories - Top 15 by Volume', 
              color=text_color, fontsize=14, fontweight='bold', pad=20)
ax2.tick_params(colors=text_color)
ax2.spines['bottom'].set_color(text_color)
ax2.spines['left'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
category_treemap.show()

# =====================================================
# VISUALIZATION 3: CATEGORY-TO-STAGE MAPPING
# =====================================================

# Create mapping between categories and stages
cat_to_stage_mapping = {}
for event, category in taxonomy_reference['event_categories'].items():
    stage = workflow_mapping['event_to_stage'].get(event, 'Uncategorized')
    key = (category, stage)
    cat_to_stage_mapping[key] = cat_to_stage_mapping.get(key, 0) + 1

# Get top combinations
top_mappings = sorted(cat_to_stage_mapping.items(), key=lambda x: x[1], reverse=True)[:20]

mapping_viz = plt.figure(figsize=(14, 10))
mapping_viz.patch.set_facecolor(bg_color)
ax3 = mapping_viz.add_subplot(111)
ax3.set_facecolor(bg_color)

labels = [f"{cat} → {stage}" for (cat, stage), _ in top_mappings]
values = [count for _, count in top_mappings]

y_pos = np.arange(len(labels))
bars = ax3.barh(y_pos, values, color=[zerve_colors[i % len(zerve_colors)] for i in range(len(values))])

# Add value labels
for i, val in enumerate(values):
    ax3.text(val + max(values)*0.01, i, f'{val}', 
             va='center', ha='left', color=text_color, fontsize=10, fontweight='bold')

ax3.set_yticks(y_pos)
ax3.set_yticklabels(labels, color=text_color, fontsize=9)
ax3.set_xlabel('Number of Events', color=text_color, fontsize=12, fontweight='bold')
ax3.set_title('Category → Workflow Stage Mapping (Top 20)', 
              color=text_color, fontsize=14, fontweight='bold', pad=20)
ax3.tick_params(colors=text_color)
ax3.spines['bottom'].set_color(text_color)
ax3.spines['left'].set_color(text_color)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.grid(axis='x', alpha=0.2, color=text_color)

plt.tight_layout()
mapping_viz.show()

# =====================================================
# VISUALIZATION 4: EVENT FREQUENCY DISTRIBUTION
# =====================================================

all_event_counts = user_retention['event'].value_counts()

frequency_dist = plt.figure(figsize=(14, 8))
frequency_dist.patch.set_facecolor(bg_color)

# Plot 1: Distribution of event frequencies (log scale)
ax4a = frequency_dist.add_subplot(121)
ax4a.set_facecolor(bg_color)

bins = [1, 10, 100, 1000, 10000, 100000, 200000]
hist_data = ax4a.hist(all_event_counts.values, bins=bins, color='#A1C9F4', 
                       edgecolor=text_color, linewidth=1.5)
ax4a.set_xscale('log')
ax4a.set_xlabel('Event Frequency (log scale)', color=text_color, fontsize=11, fontweight='bold')
ax4a.set_ylabel('Number of Event Types', color=text_color, fontsize=11, fontweight='bold')
ax4a.set_title('Event Frequency Distribution', color=text_color, fontsize=12, fontweight='bold')
ax4a.tick_params(colors=text_color)
ax4a.spines['bottom'].set_color(text_color)
ax4a.spines['left'].set_color(text_color)
ax4a.spines['top'].set_visible(False)
ax4a.spines['right'].set_visible(False)
ax4a.grid(alpha=0.2, color=text_color)

# Plot 2: Top 20 events
ax4b = frequency_dist.add_subplot(122)
ax4b.set_facecolor(bg_color)

top20_events = all_event_counts.head(20)
y_pos = np.arange(len(top20_events))
bars = ax4b.barh(y_pos, top20_events.values, 
                 color=[zerve_colors[i % len(zerve_colors)] for i in range(len(top20_events))])

ax4b.set_yticks(y_pos)
ax4b.set_yticklabels(top20_events.index, color=text_color, fontsize=9)
ax4b.set_xlabel('Occurrences', color=text_color, fontsize=11, fontweight='bold')
ax4b.set_title('Top 20 Events by Volume', color=text_color, fontsize=12, fontweight='bold')
ax4b.tick_params(colors=text_color)
ax4b.spines['bottom'].set_color(text_color)
ax4b.spines['left'].set_color(text_color)
ax4b.spines['top'].set_visible(False)
ax4b.spines['right'].set_visible(False)
ax4b.invert_yaxis()

plt.tight_layout()
frequency_dist.show()

print("\n✓ Created 4 hierarchical visualizations:")
print("  1. Workflow Stage Funnel (progression path)")
print("  2. Category Distribution (top 15)")
print("  3. Category-to-Stage Mapping (top 20)")
print("  4. Event Frequency Distribution")
print("=" * 80)
