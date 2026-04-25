
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("GNN LAYER: GRAPHSAGE TRAINING & SOCIAL INFLUENCE EMBEDDINGS")
print("Step 2: GraphSAGE (PyTorch, trained) + Social Influence Scoring")
print("=" * 80)

# ============================================================================
# GRAPHSAGE IMPLEMENTATION (PyTorch, trainable)
# GraphSAGE: h_v^k = L2norm(LeakyReLU(W_self · h_v^{k-1} + W_neigh · mean({h_u^{k-1}})))
# Note: LeakyReLU (α=0.1) is used instead of plain ReLU to avoid the dead-unit
# collapse that otherwise pushes negative-sample dot products and pos_auc
# downward over the course of training (observed on Colab session 3 with
# plain ReLU + lr=5e-3: pair_auc 0.893 → 0.315 over 30 epochs).
# ============================================================================

class GraphSAGELayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_self = nn.Linear(d_in, d_out, bias=False)
        self.W_neigh = nn.Linear(d_in, d_out, bias=False)
        self.act = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, H, A_norm):
        H_neigh = torch.sparse.mm(A_norm, H)
        out = self.W_self(H) + self.W_neigh(H_neigh)
        out = self.act(out)
        out = F.normalize(out, p=2, dim=1)
        return out


class GraphSAGE2(nn.Module):
    def __init__(self, d_in, d_h1, d_h2):
        super().__init__()
        self.l1 = GraphSAGELayer(d_in, d_h1)
        self.l2 = GraphSAGELayer(d_h1, d_h2)

    def forward(self, H, A_norm):
        H1 = self.l1(H, A_norm)
        H2 = self.l2(H1, A_norm)
        return H1, H2

# ============================================================================
# BUILD SPARSE ROW-NORMALIZED ADJACENCY
# ============================================================================
_adj = gnn_graph['adj_list']
_u2i = gnn_graph['user_to_idx']
_users = gnn_graph['users']

_rows = []
_cols = []
_deg = np.zeros(n_gnn_users, dtype=np.float32)
for i, uid in enumerate(_users):
    neighs = _adj.get(uid, set())
    neigh_idx = [_u2i[nb] for nb in neighs if nb in _u2i]
    _deg[i] = max(len(neigh_idx), 1)
    for j in neigh_idx:
        _rows.append(i)
        _cols.append(j)

if len(_rows) > 0:
    _A_indices = torch.tensor([_rows, _cols], dtype=torch.long)
    _A_values = torch.tensor([1.0 / _deg[r] for r in _rows], dtype=torch.float32)
else:
    _A_indices = torch.zeros((2, 0), dtype=torch.long)
    _A_values = torch.zeros(0, dtype=torch.float32)
_A_norm = torch.sparse_coo_tensor(_A_indices, _A_values, (n_gnn_users, n_gnn_users)).coalesce()

# ============================================================================
# TRAINING: UNSUPERVISED LINK PREDICTION WITH NEGATIVE SAMPLING
# ============================================================================
torch.manual_seed(42)
np.random.seed(42)

d_in = gnn_X_scaled.shape[1]
d_h1 = 32
d_h2 = 16

model = GraphSAGE2(d_in, d_h1, d_h2)
# lr lowered from 5e-3 → 1e-3 (Colab session 3 stabilization fix — see
# class GraphSAGELayer comment above and docs/pipeline_deep_dive.md §7.5).
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

H_in = torch.tensor(gnn_X_scaled, dtype=torch.float32)

# Positive edge pairs (unique, undirected)
_pos_src = []
_pos_dst = []
for (i, j) in gnn_graph['edge_weight_map'].keys():
    _pos_src.append(i)
    _pos_dst.append(j)
pos_src_t = torch.tensor(_pos_src, dtype=torch.long)
pos_dst_t = torch.tensor(_pos_dst, dtype=torch.long)
n_pos = len(_pos_src)

print(f"✓ GraphSAGE architecture: {d_in} → {d_h1} → {d_h2}")
print(f"  Positive training edges: {n_pos:,}")
print(f"  Nodes: {n_gnn_users:,}  Avg degree: {_deg.mean():.2f}")

N_EPOCHS = 30
BATCH_SIZE = min(8192, n_pos) if n_pos > 0 else 0
N_NEG_PER_POS = 1

print("\n📊 Training GraphSAGE (unsupervised link prediction)...")
for epoch in range(N_EPOCHS):
    model.train()
    if n_pos == 0:
        print(f"  epoch {epoch+1}: no positive edges; skipping training")
        break

    # Shuffle positive edges each epoch
    perm = torch.randperm(n_pos)
    epoch_loss = 0.0
    n_batches = 0
    for batch_start in range(0, n_pos, BATCH_SIZE):
        batch_idx = perm[batch_start:batch_start + BATCH_SIZE]
        b_pos_src = pos_src_t[batch_idx]
        b_pos_dst = pos_dst_t[batch_idx]
        b_n = b_pos_src.shape[0]

        # Negative sampling: random destinations per source
        b_neg_dst = torch.randint(0, n_gnn_users, (b_n * N_NEG_PER_POS,), dtype=torch.long)
        b_neg_src = b_pos_src.repeat_interleave(N_NEG_PER_POS)

        optimizer.zero_grad()
        _, H2 = model(H_in, _A_norm)

        pos_score = (H2[b_pos_src] * H2[b_pos_dst]).sum(dim=1)
        neg_score = (H2[b_neg_src] * H2[b_neg_dst]).sum(dim=1)

        loss = -F.logsigmoid(pos_score).mean() - F.logsigmoid(-neg_score).mean()
        loss.backward()
        optimizer.step()

        epoch_loss += float(loss.item())
        n_batches += 1

    avg_loss = epoch_loss / max(n_batches, 1)
    if epoch % 5 == 0 or epoch == N_EPOCHS - 1:
        model.eval()
        with torch.no_grad():
            _, H2_eval = model(H_in, _A_norm)
            pos_s = (H2_eval[pos_src_t] * H2_eval[pos_dst_t]).sum(dim=1)
            rand_dst = torch.randint(0, n_gnn_users, (n_pos,), dtype=torch.long)
            neg_s = (H2_eval[pos_src_t] * H2_eval[rand_dst]).sum(dim=1)
            pos_auc = float(((pos_s.unsqueeze(1) > neg_s.unsqueeze(0)).float().mean()).item()) if n_pos > 0 else float('nan')
        print(f"  epoch {epoch+1:2d}/{N_EPOCHS}  loss={avg_loss:.4f}  pos_score_mean={float(pos_s.mean()):.3f}  neg_score_mean={float(neg_s.mean()):.3f}  pair_auc={pos_auc:.3f}")

# ============================================================================
# FORWARD PASS: FINAL EMBEDDINGS
# ============================================================================
print("\n📊 Computing final embeddings from trained model...")

model.eval()
with torch.no_grad():
    H1_t, H2_t = model(H_in, _A_norm)

# Materialize as numpy arrays so downstream numpy code is unchanged
H0 = gnn_X_scaled.astype(np.float32)
H1 = H1_t.cpu().numpy().astype(np.float32)
gnn_embeddings = H2_t.cpu().numpy().astype(np.float32)
print(f"  Layer 1 output shape: {H1.shape}")
print(f"  Layer 2 (embeddings) shape: {gnn_embeddings.shape}")

# ============================================================================
# SOCIAL INFLUENCE SCORING
# Measure how much each user's engagement "pulls" their neighbors
# ============================================================================
print("\n📊 Computing Social Influence Scores...")

# Pull behavioral signal columns
_bfp = gnn_behavior_df.set_index('user_id')
_active_col = 'power_user_score'  # proxy for retention/engagement

gnn_influence_scores = {}
for i, uid in enumerate(_users):
    neighbors = list(_adj.get(uid, set()))
    if not neighbors:
        gnn_influence_scores[uid] = 0.0
        continue
    
    neigh_indices = [_u2i[nb] for nb in neighbors if nb in _u2i]
    if not neigh_indices:
        gnn_influence_scores[uid] = 0.0
        continue
    
    # Influence = correlation between THIS user's power_user_score
    # and their neighbors' average retention
    uid_score = _bfp.loc[uid, _active_col] if uid in _bfp.index else 0
    neigh_scores = []
    for ni in neigh_indices:
        nuid = _users[ni]
        if nuid in _bfp.index:
            neigh_scores.append(_bfp.loc[nuid, _active_col])
    
    if neigh_scores:
        avg_neigh = np.mean(neigh_scores)
        # Influence = user's score * (avg neighbor score / global avg)
        global_avg = _bfp[_active_col].mean()
        ratio = avg_neigh / (global_avg + 1e-9)
        # Weight by degree (highly connected users have more influence)
        degree_weight = np.log1p(len(neighbors)) / np.log1p(gnn_degree.max())
        gnn_influence_scores[uid] = float(uid_score * ratio * degree_weight)
    else:
        gnn_influence_scores[uid] = 0.0

gnn_influence_arr = np.array([gnn_influence_scores[uid] for uid in _users])

# Normalize to [0, 100]
_inf_min, _inf_max = gnn_influence_arr.min(), gnn_influence_arr.max()
gnn_influence_norm = (gnn_influence_arr - _inf_min) / (_inf_max - _inf_min + 1e-9) * 100

# ============================================================================
# CHURN CO-OCCURRENCE: Do connected users churn together?
# ============================================================================
print("\n📊 Analyzing Churn Co-occurrence in Graph...")

# Get churn status for users we have it for
_surv_set = set(active_users_survival['user_id'].values)
_churned_set = set(survival_data[survival_data['churned'] == 1]['user_id'].values)

churn_concordance_scores = []
for i, uid in enumerate(_users):
    neighbors = list(_adj.get(uid, set()))
    neigh_in_surv = [nb for nb in neighbors if nb in _surv_set]
    
    if not neigh_in_surv:
        continue
    
    uid_churned = 1 if uid in _churned_set else 0
    neigh_churned = [1 if nb in _churned_set else 0 for nb in neigh_in_surv]
    
    if neigh_churned:
        concordance = sum(neigh_churned) / len(neigh_churned)
        churn_concordance_scores.append({
            'user_id': uid,
            'churned': uid_churned,
            'neighbor_churn_rate': concordance,
            'n_neighbors': len(neigh_in_surv)
        })

churn_concordance_df = pd.DataFrame(churn_concordance_scores)
if len(churn_concordance_df) > 0:
    corr = churn_concordance_df['churned'].corr(churn_concordance_df['neighbor_churn_rate'])
    print(f"  ✓ Churn concordance correlation: {corr:.4f}")
    print(f"  ✓ Avg neighbor churn rate (churned users): {churn_concordance_df[churn_concordance_df['churned']==1]['neighbor_churn_rate'].mean():.3f}")
    print(f"  ✓ Avg neighbor churn rate (retained users): {churn_concordance_df[churn_concordance_df['churned']==0]['neighbor_churn_rate'].mean():.3f}")

# ============================================================================
# BUILD PER-USER GNN EMBEDDING DATAFRAME
# ============================================================================
gnn_embedding_df = pd.DataFrame({
    'user_id': _users,
    'influence_score': gnn_influence_norm,
    **{f'gnn_emb_{j}': gnn_embeddings[:, j] for j in range(d_h2)}
})

# Add degree info
gnn_embedding_df['gnn_degree'] = gnn_degree

# Add churn concordance
if len(churn_concordance_df) > 0:
    gnn_embedding_df = gnn_embedding_df.merge(
        churn_concordance_df[['user_id', 'neighbor_churn_rate']],
        on='user_id', how='left'
    )
    gnn_embedding_df['neighbor_churn_rate'] = gnn_embedding_df['neighbor_churn_rate'].fillna(0)
else:
    gnn_embedding_df['neighbor_churn_rate'] = 0.0

# Identify "churn influencers" — high influence + high neighbor churn rate
gnn_embedding_df['is_churn_influencer'] = (
    (gnn_embedding_df['influence_score'] >= np.percentile(gnn_influence_norm, 85)) &
    (gnn_embedding_df['neighbor_churn_rate'] >= 0.3)
)

# Identify "retention anchors" — high influence + low neighbor churn rate
gnn_embedding_df['is_retention_anchor'] = (
    (gnn_embedding_df['influence_score'] >= np.percentile(gnn_influence_norm, 75)) &
    (gnn_embedding_df['neighbor_churn_rate'] <= 0.2)
)

n_churn_influencers = gnn_embedding_df['is_churn_influencer'].sum()
n_retention_anchors = gnn_embedding_df['is_retention_anchor'].sum()

print(f"\n📊 SOCIAL INFLUENCE RESULTS:")
print(f"   ✓ Churn Influencers (high influence + neighbor churn): {n_churn_influencers:,}")
print(f"   ✓ Retention Anchors (high influence + neighbor retained): {n_retention_anchors:,}")
print(f"   ✓ Top influence score: {gnn_influence_norm.max():.1f}")
print(f"   ✓ Mean influence score: {gnn_influence_norm.mean():.1f}")

# ============================================================================
# COMMUNITY DETECTION: Label Propagation (numpy)
# ============================================================================
print("\n📊 Running Community Detection (Label Propagation)...")

N_COMMUNITIES_TARGET = 8
MAX_LP_ITERS = 30

# Initialize each node as its own community
community_labels = np.arange(n_gnn_users, dtype=np.int32)

for _iter in range(MAX_LP_ITERS):
    prev = community_labels.copy()
    # Randomized order
    _order = np.random.permutation(n_gnn_users)
    for i in _order:
        uid = _users[i]
        neighbors = list(_adj.get(uid, set()))
        neigh_indices = [_u2i[nb] for nb in neighbors if nb in _u2i]
        if neigh_indices:
            neigh_labels = community_labels[neigh_indices]
            # Majority vote
            vals, cnts = np.unique(neigh_labels, return_counts=True)
            community_labels[i] = vals[cnts.argmax()]
    
    # Check convergence
    changed = (community_labels != prev).sum()
    if changed == 0:
        print(f"  Converged at iteration {_iter+1}")
        break

# Remap community IDs (many small → merge into K communities)
unique_comms, comm_sizes = np.unique(community_labels, return_counts=True)
n_raw_communities = len(unique_comms)
print(f"  Raw communities detected: {n_raw_communities}")

# Keep only large communities; merge small ones via KMeans on embeddings
if n_raw_communities > N_COMMUNITIES_TARGET:
    kmeans_comm = KMeans(n_clusters=N_COMMUNITIES_TARGET, random_state=42, n_init=10)
    # Use embeddings for community assignment
    gnn_community_ids = kmeans_comm.fit_predict(gnn_embeddings)
else:
    gnn_community_ids = community_labels

unique_final, final_sizes = np.unique(gnn_community_ids, return_counts=True)
print(f"  Final communities: {len(unique_final)}")
for cid, csize in sorted(zip(unique_final, final_sizes), key=lambda x: -x[1]):
    print(f"    Community {cid}: {csize:,} users")

gnn_embedding_df['community_id'] = gnn_community_ids

print(f"\n✅ GraphSAGE embeddings complete.")
print(f"   Embedding shape: {gnn_embeddings.shape}")
print(f"   gnn_embedding_df shape: {gnn_embedding_df.shape}")
print(f"   Columns: {list(gnn_embedding_df.columns[:8])}...")
