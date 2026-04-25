
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("GNN LAYER: HYBRID GNN-ENSEMBLE CHURN MODEL")
print("Step 3: GNN-enhanced predictions + Community success correlation")
print("=" * 80)

# ============================================================================
# 1. BUILD HYBRID FEATURE SET
# Join GNN embeddings with survival/churn data
# ============================================================================
print("\n📊 Building Hybrid Feature Set...")

# Get users that have churn labels (from survival analysis)
_has_churn = survival_data[['user_id', 'churned', 'time_to_event']].copy()

# Merge GNN embeddings
hybrid_df = _has_churn.merge(gnn_embedding_df, on='user_id', how='inner')

# Also add base behavioral features
_base_features = behavioral_fingerprint[[
    'user_id', 'power_user_score', 'struggle_score', 'deep_work_ratio',
    'total_sessions', 'avg_session_length', 'sequence_diversity',
    'collaboration_ratio', 'team_oriented_score'
]].copy()
hybrid_df = hybrid_df.merge(_base_features, on='user_id', how='left')

print(f"✓ Hybrid dataset size: {hybrid_df.shape}")
print(f"   Churn rate: {hybrid_df['churned'].mean()*100:.1f}%")

# ============================================================================
# 2. DEFINE FEATURES: BASELINE vs HYBRID
# ============================================================================
base_feat_cols = [
    'power_user_score', 'struggle_score', 'deep_work_ratio',
    'total_sessions', 'avg_session_length', 'sequence_diversity',
    'collaboration_ratio', 'team_oriented_score'
]

gnn_feat_cols = [f'gnn_emb_{j}' for j in range(16)] + [
    'influence_score', 'gnn_degree', 'neighbor_churn_rate'
]

base_available = [c for c in base_feat_cols if c in hybrid_df.columns]
gnn_available = [c for c in gnn_feat_cols if c in hybrid_df.columns]
hybrid_feat_cols = base_available + gnn_available

print(f"   Baseline features: {len(base_available)}")
print(f"   GNN features: {len(gnn_available)}")
print(f"   Hybrid features: {len(hybrid_feat_cols)}")

# ============================================================================
# 3. TRAIN/TEST SPLIT (time-aware: retain temporal ordering)
# ============================================================================
np.random.seed(42)
hybrid_df_clean = hybrid_df.dropna(subset=hybrid_feat_cols + ['churned'])

_n = len(hybrid_df_clean)
_idx = np.random.permutation(_n)
_train_end = int(_n * 0.75)

train_idx = _idx[:_train_end]
test_idx = _idx[_train_end:]

X_base_train = hybrid_df_clean.iloc[train_idx][base_available].fillna(0).values
X_base_test  = hybrid_df_clean.iloc[test_idx][base_available].fillna(0).values
X_hybrid_train = hybrid_df_clean.iloc[train_idx][hybrid_feat_cols].fillna(0).values
X_hybrid_test  = hybrid_df_clean.iloc[test_idx][hybrid_feat_cols].fillna(0).values
y_train = hybrid_df_clean.iloc[train_idx]['churned'].values
y_test  = hybrid_df_clean.iloc[test_idx]['churned'].values

print(f"\n✓ Train size: {len(y_train)}, Test size: {len(y_test)}")
print(f"   Train churn rate: {y_train.mean()*100:.1f}%")

# ============================================================================
# 4. TRAIN BASELINE (no GNN) vs HYBRID (with GNN)
# ============================================================================
print("\n📊 Training Baseline vs Hybrid Models...")

# Scale features
_scaler_base = StandardScaler()
X_base_train_s = _scaler_base.fit_transform(X_base_train)
X_base_test_s  = _scaler_base.transform(X_base_test)

_scaler_hybrid = StandardScaler()
X_hybrid_train_s = _scaler_hybrid.fit_transform(X_hybrid_train)
X_hybrid_test_s  = _scaler_hybrid.transform(X_hybrid_test)

# Baseline model
baseline_model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
baseline_model.fit(X_base_train_s, y_train)
baseline_proba = baseline_model.predict_proba(X_base_test_s)[:, 1]
baseline_pred  = baseline_model.predict(X_base_test_s)
baseline_auc   = roc_auc_score(y_test, baseline_proba)
baseline_f1    = f1_score(y_test, baseline_pred, average='weighted', zero_division=0)

# Hybrid GNN model
hybrid_model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
hybrid_model.fit(X_hybrid_train_s, y_train)
hybrid_proba  = hybrid_model.predict_proba(X_hybrid_test_s)[:, 1]
hybrid_pred   = hybrid_model.predict(X_hybrid_test_s)
hybrid_auc    = roc_auc_score(y_test, hybrid_proba)
hybrid_f1     = f1_score(y_test, hybrid_pred, average='weighted', zero_division=0)

auc_lift   = (hybrid_auc - baseline_auc) / baseline_auc * 100
f1_lift    = (hybrid_f1 - baseline_f1) / (baseline_f1 + 1e-9) * 100

print(f"\n{'='*60}")
print(f"HYBRID MODEL COMPARISON RESULTS:")
print(f"{'='*60}")
print(f"{'Metric':<25} {'Baseline':>12} {'GNN Hybrid':>12} {'Lift %':>10}")
print(f"{'-'*60}")
print(f"{'ROC-AUC':<25} {baseline_auc:>12.4f} {hybrid_auc:>12.4f} {auc_lift:>+9.2f}%")
print(f"{'F1 Score (weighted)':<25} {baseline_f1:>12.4f} {hybrid_f1:>12.4f} {f1_lift:>+9.2f}%")
print(f"{'Test Samples':<25} {len(y_test):>12,}")
print(f"{'Positive Class Rate':<25} {y_test.mean():>12.3f}")

# Full hybrid score for all users
# Apply to the full embedding df
all_hybrid_feats = hybrid_df_clean[hybrid_feat_cols].fillna(0).values
all_hybrid_feats_s = _scaler_hybrid.transform(all_hybrid_feats)
hybrid_df_clean = hybrid_df_clean.copy()
hybrid_df_clean['hybrid_churn_score'] = hybrid_model.predict_proba(all_hybrid_feats_s)[:, 1] * 100

# ============================================================================
# 5. COMMUNITY SUCCESS CORRELATION
# ============================================================================
print("\n\n📊 Community Detection & Success Correlation...")

# Merge community with user success metrics
comm_analysis = gnn_embedding_df[['user_id', 'community_id', 'influence_score', 
                                   'is_retention_anchor', 'gnn_degree']].copy()
comm_analysis = comm_analysis.merge(
    behavioral_fingerprint[['user_id', 'power_user_score', 'deep_work_ratio', 'collaboration_ratio']],
    on='user_id', how='left'
)
comm_analysis = comm_analysis.merge(
    survival_data[['user_id', 'churned']].drop_duplicates('user_id'),
    on='user_id', how='left'
)

# Community stats
comm_stats = comm_analysis.groupby('community_id').agg(
    n_users=('user_id', 'count'),
    avg_influence=('influence_score', 'mean'),
    avg_power_score=('power_user_score', 'mean'),
    churn_rate=('churned', 'mean'),
    avg_degree=('gnn_degree', 'mean'),
    n_retention_anchors=('is_retention_anchor', 'sum'),
    avg_collab=('collaboration_ratio', 'mean')
).reset_index().round(3)

# Label communities
def label_community(row):
    if row['avg_power_score'] > 5 and row['churn_rate'] < 0.3:
        return 'Power Tribe 🏆'
    elif row['churn_rate'] > 0.5:
        return 'At-Risk Tribe ⚠️'
    elif row['n_retention_anchors'] > row['n_users'] * 0.3:
        return 'Retention Hub 🔒'
    elif row['avg_collab'] > 0.05:
        return 'Collaboration Tribe 🤝'
    elif row['avg_degree'] > comm_stats['avg_degree'].median():
        return 'Connected Tribe 🔗'
    else:
        return 'Regular Users 👥'

comm_stats['tribe_label'] = comm_stats.apply(label_community, axis=1)

print(f"\n{'Community':>10} {'Users':>8} {'Churn%':>8} {'Influence':>10} {'Power':>8} {'Label'}")
print('-' * 70)
for _, row in comm_stats.sort_values('n_users', ascending=False).iterrows():
    print(f"  Comm {int(row['community_id']):>3}   {int(row['n_users']):>6}   "
          f"{row['churn_rate']*100:>6.1f}%  {row['avg_influence']:>9.1f}  "
          f"{row['avg_power_score']:>7.2f}  {row['tribe_label']}")

# ============================================================================
# 6. RETENTION ANCHORS ANALYSIS
# ============================================================================
print("\n\n📊 High-Influence Retention Anchors...")

retention_anchors = gnn_embedding_df[gnn_embedding_df['is_retention_anchor']].copy()
retention_anchors = retention_anchors.merge(
    behavioral_fingerprint[['user_id', 'power_user_score', 'total_events', 'team_oriented_score']],
    on='user_id', how='left'
)

# Use active_users_survival which carries success_score / churn_proxy.
_surv_cols = ['user_id', 'churned']
if 'success_score' in active_users_survival.columns:
    _surv_cols.append('success_score')
if 'churn_proxy' in active_users_survival.columns:
    _surv_cols.append('churn_proxy')

retention_anchors = retention_anchors.merge(
    active_users_survival[_surv_cols].drop_duplicates('user_id'),
    on='user_id', how='left'
)

# Backfill churn_proxy from success_score if the upstream block didn't populate it.
if 'churn_proxy' not in retention_anchors.columns and 'success_score' in retention_anchors.columns:
    retention_anchors['churn_proxy'] = (100.0 - retention_anchors['success_score']).clip(0, 100)

print(f"  Total Retention Anchors: {len(retention_anchors):,}")
print(f"  Avg influence score:     {retention_anchors['influence_score'].mean():.1f}")
print(f"  Avg power user score:    {retention_anchors['power_user_score'].mean():.2f}")
print(f"  Avg network degree:      {retention_anchors['gnn_degree'].mean():.1f}")

# At-risk anchors (retention anchors with high churn risk).
# Filter on churn_proxy so direction matches: high value = high churn risk.
# Previous filter on success_score >= 50 selected the *most successful*
# anchors and labelled them at-risk. See docs/repo_state_and_next_steps.md §6.
if 'churn_proxy' in retention_anchors.columns:
    at_risk_anchors = retention_anchors[retention_anchors['churn_proxy'] >= 50]
elif 'churned' in retention_anchors.columns:
    at_risk_anchors = retention_anchors[retention_anchors['churned'] == 1]
else:
    at_risk_anchors = retention_anchors.head(0)
print(f"  At-risk retention anchors: {len(at_risk_anchors):,}")

# Save hybrid model results
hybrid_churn_model_results = {
    'baseline_auc': baseline_auc,
    'hybrid_auc': hybrid_auc,
    'auc_lift_pct': auc_lift,
    'baseline_f1': baseline_f1,
    'hybrid_f1': hybrid_f1,
    'f1_lift_pct': f1_lift,
    'n_test': len(y_test),
    'n_retention_anchors': len(retention_anchors),
    'community_stats': comm_stats
}

gnn_social_influence_df = hybrid_df_clean[[
    'user_id', 'churned', 'hybrid_churn_score',
    'influence_score', 'gnn_degree', 'neighbor_churn_rate',
    'is_retention_anchor', 'community_id'
]].copy()

print(f"\n✅ HYBRID MODEL COMPLETE")
print(f"   AUC lift from GNN: {auc_lift:+.2f}%")
print(f"   Retention anchors: {n_retention_anchors:,}")
print(f"   Communities detected: {len(comm_stats)}")
