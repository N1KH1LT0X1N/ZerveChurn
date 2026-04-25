
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("GNN LAYER: SOCIAL & BEHAVIORAL INFLUENCE MODELING")
print("Step 1: User Similarity Graph Construction")
print("=" * 80)

# ============================================================================
# PREPARE NODE FEATURES FROM BEHAVIORAL FINGERPRINT
# ============================================================================
# Use the full behavioral fingerprint for all users
gnn_behavior_df = behavioral_fingerprint.copy()
gnn_users = gnn_behavior_df['user_id'].values
n_gnn_users = len(gnn_users)
print(f"\n✓ Total users for GNN: {n_gnn_users:,}")

# Create user index mapping
gnn_user_to_idx = {uid: i for i, uid in enumerate(gnn_users)}

# Node feature matrix: behavioral features normalized
gnn_feature_cols = [
    'total_sessions', 'avg_session_length', 'avg_events_per_session',
    'total_events', 'deep_work_ratio', 'power_user_score', 'struggle_score',
    'sequence_diversity', 'has_agent_workflow', 'has_deployment_sequence',
    'error_count', 'trigram_count', 'collaboration_ratio', 'team_oriented_score',
    'sharing_frequency'
]

gnn_available_features = [c for c in gnn_feature_cols if c in gnn_behavior_df.columns]
gnn_X_raw = gnn_behavior_df[gnn_available_features].fillna(0).values.astype(float)

gnn_scaler = StandardScaler()
gnn_X_scaled = gnn_scaler.fit_transform(gnn_X_raw)

print(f"✓ Node feature matrix: {gnn_X_scaled.shape}")

# ============================================================================
# CONSTRUCT BEHAVIORAL SIMILARITY GRAPH
# ============================================================================
print("\n📊 Building Behavioral Similarity Graph...")

# Use k-NN on behavioral features to find similar users
# For efficiency on 5,410 users, use sampled cosine similarity + threshold
# We'll use batch cosine similarity to build sparse graph

# Batch similarity computation to avoid OOM
BATCH_SIZE = 500
SIM_THRESHOLD = 0.85    # Only connect users above this similarity
MAX_NEIGHBORS = 10       # Cap connections per user for sparsity

gnn_adj_list = defaultdict(set)
gnn_edge_weights = {}

total_edges_added = 0
for batch_start in range(0, n_gnn_users, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, n_gnn_users)
    batch_X = gnn_X_scaled[batch_start:batch_end]
    
    # Compute similarity of this batch vs all users
    sim_matrix = cosine_similarity(batch_X, gnn_X_scaled)
    
    for local_i, global_i in enumerate(range(batch_start, batch_end)):
        sims = sim_matrix[local_i].copy()
        sims[global_i] = -1  # exclude self
        
        # Get top-k neighbors above threshold
        top_k_idx = np.argsort(sims)[::-1][:MAX_NEIGHBORS * 2]
        
        for j in top_k_idx:
            if sims[j] >= SIM_THRESHOLD and j != global_i:
                uid_i = gnn_users[global_i]
                uid_j = gnn_users[j]
                if uid_j not in gnn_adj_list[uid_i]:
                    gnn_adj_list[uid_i].add(uid_j)
                    gnn_adj_list[uid_j].add(uid_i)
                    edge_key = (min(global_i, j), max(global_i, j))
                    gnn_edge_weights[edge_key] = float(sims[j])
                    total_edges_added += 1
                if len(gnn_adj_list[uid_i]) >= MAX_NEIGHBORS:
                    break

print(f"✓ Behavioral similarity edges: {total_edges_added:,}")
print(f"   Threshold: {SIM_THRESHOLD}, Max neighbors: {MAX_NEIGHBORS}")

# ============================================================================
# ADD CO-ENGAGEMENT EDGES (users with same event type patterns)
# ============================================================================
print("\n📊 Adding Co-engagement Pattern Edges...")

# Users sharing top-3 events form weak co-engagement ties
# Proxy via trigram similarity: users with overlapping trigram counts
co_engagement_edges = 0
if 'trigram_count' in gnn_behavior_df.columns and 'fivegram_count' in gnn_behavior_df.columns:
    # Bin users by activity level as a proxy for co-engagement
    gnn_behavior_df['activity_bin'] = pd.qcut(
        gnn_behavior_df['total_events'].clip(0, gnn_behavior_df['total_events'].quantile(0.99)),
        q=20, labels=False, duplicates='drop'
    )
    
    for _bin_id, _group in gnn_behavior_df.groupby('activity_bin'):
        _uids = _group['user_id'].values
        if len(_uids) > 1:
            # Connect high-activity pairs within same bin
            _high_power = _group[_group['power_user_score'] >= _group['power_user_score'].median()]
            for _u1 in _high_power['user_id'].values[:5]:  # limit to avoid too many edges
                for _u2 in _uids[:10]:
                    if _u1 != _u2 and _u2 not in gnn_adj_list[_u1]:
                        gnn_adj_list[_u1].add(_u2)
                        gnn_adj_list[_u2].add(_u1)
                        i1, i2 = gnn_user_to_idx[_u1], gnn_user_to_idx[_u2]
                        gnn_edge_weights[(min(i1,i2), max(i1,i2))] = 0.6  # moderate weight
                        co_engagement_edges += 1

print(f"✓ Co-engagement edges added: {co_engagement_edges:,}")

# ============================================================================
# BUILD EDGE LIST FOR GNN
# ============================================================================
gnn_edge_src = []
gnn_edge_dst = []
gnn_edge_w = []

for (i, j), w in gnn_edge_weights.items():
    gnn_edge_src.append(i)
    gnn_edge_dst.append(j)
    gnn_edge_w.append(w)
    gnn_edge_src.append(j)
    gnn_edge_dst.append(i)
    gnn_edge_w.append(w)

gnn_edge_src = np.array(gnn_edge_src, dtype=np.int32)
gnn_edge_dst = np.array(gnn_edge_dst, dtype=np.int32)
gnn_edge_w = np.array(gnn_edge_w, dtype=np.float32)

# Graph stats
gnn_degree = np.zeros(n_gnn_users, dtype=int)
for i in range(n_gnn_users):
    gnn_degree[i] = len(gnn_adj_list[gnn_users[i]])

print(f"\n📊 GRAPH STATISTICS:")
print(f"   Nodes (users): {n_gnn_users:,}")
print(f"   Unique edges:  {len(gnn_edge_weights):,}")
print(f"   Avg degree:    {gnn_degree.mean():.2f}")
print(f"   Max degree:    {gnn_degree.max()}")
print(f"   Min degree:    {gnn_degree.min()}")
print(f"   % Isolated:    {(gnn_degree == 0).mean()*100:.1f}%")
print(f"   Density:       {len(gnn_edge_weights) / (n_gnn_users*(n_gnn_users-1)/2) * 100:.3f}%")

# Save graph structure for downstream GNN training
gnn_graph = {
    'node_features': gnn_X_scaled,
    'adj_list': dict(gnn_adj_list),
    'edge_src': gnn_edge_src,
    'edge_dst': gnn_edge_dst,
    'edge_weights': gnn_edge_w,
    'edge_weight_map': gnn_edge_weights,
    'user_to_idx': gnn_user_to_idx,
    'idx_to_user': {i: u for u, i in gnn_user_to_idx.items()},
    'users': gnn_users,
    'feature_cols': gnn_available_features,
    'degree': gnn_degree
}

print(f"\n✅ Graph construction complete. {n_gnn_users:,} nodes, {len(gnn_edge_weights):,} edges.")
