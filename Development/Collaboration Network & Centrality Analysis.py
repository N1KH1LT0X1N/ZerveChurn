import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Since there are no explicit collaboration sharing events in the data, 
# we construct an inferred collaboration network from behavioral similarity
# (k-NN cosine similarity on the behavioral fingerprint features).

print("="*60)
print("COLLABORATION NETWORK ANALYSIS")
print("="*60)

# Get user base and behavioral fingerprint
users_collab_data = collaboration_signature_df.copy()
users_behavior = behavioral_fingerprint.copy()

print(f"\nTotal users in analysis: {len(users_collab_data):,}")
print(f"Users with collaboration signature: {users_collab_data['has_sharing_activity'].sum():,}")
print(f"Team-oriented users (score > 0): {(users_collab_data['team_oriented_score'] > 0).sum():,}")

# Get high-activity users from behavioral fingerprint
high_activity_users = users_behavior[users_behavior['total_events'] >= 50].copy()
print(f"High-activity users (50+ events): {len(high_activity_users):,}")

# Sample a manageable subset for centrality computation
np.random.seed(42)
sample_size = min(500, len(high_activity_users))
sampled_users = high_activity_users.sample(n=sample_size, random_state=42)['user_id'].values
n_users = len(sampled_users)

# Pull behavioral features for the sample (engagement + workflow + collaboration signals)
_behav_cols = [
    'total_sessions', 'avg_events_per_session', 'total_events', 'deep_work_ratio',
    'power_user_score', 'struggle_score', 'sequence_diversity',
    'collaboration_ratio', 'team_oriented_score', 'sharing_frequency'
]
_behav_available = [c for c in _behav_cols if c in users_behavior.columns]
_sample_behav = (users_behavior.set_index('user_id')
                 .loc[sampled_users, _behav_available]
                 .fillna(0)
                 .astype(float))

# Standardize + cosine similarity; mask self-loops
_scaler_cn = StandardScaler()
_sample_X = _scaler_cn.fit_transform(_sample_behav.values)
_sim = cosine_similarity(_sample_X)
np.fill_diagonal(_sim, -np.inf)

# k-NN graph: each user links to top-k most-similar peers above a threshold
KNN_MAX_NEIGHBORS = 8
KNN_SIM_THRESHOLD = 0.70

adjacency_list = defaultdict(list)
edge_list = []
_edge_set = set()

for i in range(n_users):
    sims_i = _sim[i]
    top_idx = np.argsort(sims_i)[::-1][:KNN_MAX_NEIGHBORS]
    for j in top_idx:
        if sims_i[j] < KNN_SIM_THRESHOLD:
            continue
        u_i = sampled_users[i]
        u_j = sampled_users[j]
        edge = tuple(sorted([u_i, u_j]))
        if edge in _edge_set:
            continue
        _edge_set.add(edge)
        edge_list.append(edge)
        adjacency_list[u_i].append(u_j)
        adjacency_list[u_j].append(u_i)

# Ensure every sampled user appears in the adjacency list, even if isolated,
# so downstream centrality metrics cover the full sample.
for _u in sampled_users:
    _ = adjacency_list[_u]

print(f"\nNetwork Statistics:")
print(f"Nodes (users): {len(adjacency_list):,}")
print(f"Edges (collaboration links): {len(edge_list):,}")
print(f"Edge construction: k-NN cosine similarity (k={KNN_MAX_NEIGHBORS}, threshold={KNN_SIM_THRESHOLD:.2f})")
print(f"Behavioral features used: {', '.join(_behav_available)}")

# Calculate network density
possible_edges = n_users * (n_users - 1) / 2
network_density = len(edge_list) / possible_edges if possible_edges > 0 else 0
print(f"Network density: {network_density:.6f}")

# Calculate centrality metrics manually

# 1. DEGREE CENTRALITY - number of connections
degree_centrality = {}
for user in adjacency_list:
    degree_centrality[user] = len(set(adjacency_list[user]))

# 2. BETWEENNESS CENTRALITY (simplified) - how often user is on shortest paths
# Full calculation is complex, so we approximate based on degree and position
betweenness_centrality = {}
for user in adjacency_list:
    # Approximate: users with higher degree and connections to diverse groups have higher betweenness
    neighbors = set(adjacency_list[user])
    second_order = set()
    for neighbor in neighbors:
        second_order.update(adjacency_list[neighbor])
    reach = len(second_order)
    betweenness_centrality[user] = reach * len(neighbors) / n_users if n_users > 0 else 0

# 3. PAGERANK (simplified eigenvector centrality)
# Iterative algorithm: importance = sum of importance of neighbors
pagerank = {user: 1.0/n_users for user in adjacency_list}
damping = 0.85
n_iterations = 20

for _ in range(n_iterations):
    new_pagerank = {}
    for user in adjacency_list:
        rank_sum = sum(pagerank[neighbor] / len(adjacency_list[neighbor]) 
                      for neighbor in adjacency_list[user])
        new_pagerank[user] = (1 - damping) / n_users + damping * rank_sum
    pagerank = new_pagerank

# Normalize centrality metrics
max_degree = max(degree_centrality.values()) if degree_centrality else 1
max_between = max(betweenness_centrality.values()) if betweenness_centrality else 1

degree_centrality_norm = {u: v/max_degree for u, v in degree_centrality.items()}
betweenness_centrality_norm = {u: v/max_between for u, v in betweenness_centrality.items()}

# Create centrality dataframe
centrality_records = []
for user in adjacency_list:
    centrality_records.append({
        'user_id': user,
        'degree_centrality': degree_centrality_norm[user],
        'betweenness_centrality': betweenness_centrality_norm[user],
        'pagerank': pagerank[user],
        'num_connections': degree_centrality[user]
    })

centrality_network_df = pd.DataFrame(centrality_records)
centrality_network_df['composite_centrality'] = (
    centrality_network_df['degree_centrality'] + 
    centrality_network_df['betweenness_centrality'] + 
    centrality_network_df['pagerank']
) / 3

print(f"\n{'='*60}")
print("CENTRALITY METRICS SUMMARY")
print(f"{'='*60}")
print(f"\nDegree Centrality (connections):")
print(f"  Mean: {centrality_network_df['degree_centrality'].mean():.4f}")
print(f"  Median: {centrality_network_df['degree_centrality'].median():.4f}")
print(f"  Max: {centrality_network_df['degree_centrality'].max():.4f}")

print(f"\nBetweenness Centrality (bridging):")
print(f"  Mean: {centrality_network_df['betweenness_centrality'].mean():.4f}")
print(f"  Median: {centrality_network_df['betweenness_centrality'].median():.4f}")
print(f"  Max: {centrality_network_df['betweenness_centrality'].max():.4f}")

print(f"\nPageRank (influence):")
print(f"  Mean: {centrality_network_df['pagerank'].mean():.6f}")
print(f"  Median: {centrality_network_df['pagerank'].median():.6f}")
print(f"  Max: {centrality_network_df['pagerank'].max():.6f}")

# Identify super connectors (top 10%)
threshold_90 = centrality_network_df['composite_centrality'].quantile(0.90)
centrality_network_df['is_super_connector'] = centrality_network_df['composite_centrality'] >= threshold_90

n_super = centrality_network_df['is_super_connector'].sum()
print(f"\n{'='*60}")
print(f"SUPER CONNECTORS (Top 10%): {n_super} users")
print(f"{'='*60}")

top_connectors = centrality_network_df.nlargest(10, 'composite_centrality')
print("\nTop 10 Super Connectors:")
for idx, row in top_connectors.iterrows():
    print(f"  User: {row['user_id'][:16]}... | Connections: {row['num_connections']:2d} | Composite Score: {row['composite_centrality']:.4f}")

# Store for downstream use
network_adjacency_list = adjacency_list
network_edge_list = edge_list
