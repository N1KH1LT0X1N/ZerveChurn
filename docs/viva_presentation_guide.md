# ZerveChurn Project — Viva/Presentation Guide

## Project Overview

**Objective**: Build comprehensive user behavior analytics pipeline for churn prediction, segmentation, and behavioral intelligence.

**Dataset**: 409,287 events from 5,410 users across 141 event types (Sep-Dec 2025, 98 days).

**Platform**: Zerve.ai visual canvas (67 blocks, 78 edges, 7 stages).

---

## Technical Architecture

### 7-Stage Pipeline
1. **Ingestion & EDA** (13 blocks): Data loading, quality checks, temporal coverage
2. **Event Semantics** (3 blocks): 22 categories, 14 workflow stages
3. **Segmentation** (8 blocks): K-means engagement, workflow patterns, temporal/monetization
4. **Behavioral Features** (9 blocks): Session patterns, n-grams, 70+ engineered features
5. **Modeling** (12 blocks): Success ensemble, churn EWS, survival, LTV, GNN
6. **Graph/GNN** (5 blocks): Collaboration network, GraphSAGE, hybrid model
7. **Reporting** (16 blocks): SHAP, causal analysis, dashboards, exports

### Execution Model
- Shared namespace across blocks (Zerve runtime)
- S3 state bucket for caching
- Local replication via `scripts/run_canvas_locally.py`

---

## Key Models

### Success Ensemble
- **Label**: `alternative_label` (High Value based on events/retention/power/tenure)
- **Features**: 4 numeric (total_events, tenure_days, days_since_first, days_since_last)
- **Models**: RF, GB, AdaBoost, LR, Voting, Stacking
- **Performance**: 99.51% accuracy, 0.9952 ROC-AUC
- **Caveat**: Label leakage - features are inputs to label generation. True generalization AUC: 0.9414 (future-holdout)

### Churn Early Warning System
- **Separate genuine churn model** (not success ensemble)
- **Features**: 14 churn-specific engineered features (tenure, recency, rolling windows, velocity, decay)
- **Label**: `recency_d > 30`
- **Ensemble**: GB (40%) + HGB (40%) + RF (20%)
- **Score**: `0.6×model + 0.25×decay + 0.15×recency`
- **Tiers**: Critical, High, Medium, Low

### Survival Analysis
- **Method**: Hand-rolled Kaplan-Meier (not lifelines)
- **Data**: time_to_event, churned (days_since_last > 30), risk_segment
- **Output**: Survival curves by segment, median survival times

### GNN (GraphSAGE)
- **Graph**: 5,410 nodes, ~40K edges (kNN on 15 behavioral features)
- **Training**: PyTorch, 2 layers (d_in→32→16), Adam lr=1e-3, LeakyReLU(0.1), 30 epochs
- **Loss**: Unsupervised link prediction, pair_auc 0.900→0.909
- **Hybrid Model**: Baseline + GNN embeddings (AUC 0.8824 vs 0.8834 baseline, minimal lift)

---

## Key Findings

### Engagement Polarization
- Top 5% users: 544 events avg
- Median user: 5 events avg
- Deep work sessions: 8.3% (8.4+ min, 2.0+ events/min)

### Feature Adoption
- Basic Blocks: 89% (0.5 days)
- Data Operations: 67% (2.1 days)
- Visualization: 54% (4.2 days)
- AI Features: 32% (7.8 days)
- **Major drop-off**: 67% → 54% after data operations

### Workflow Funnel
- Onboarding → Exploration: retained
- Exploration → Creation: **40% drop-off** (critical activation gap)
- Creation → Execution: retained

### Retention Signals
- 3+ sessions in week 1 → 80% retention
- Full workflow completion → 90% retention
- 4+ feature categories → 85% retention
- Deep work sessions → 75% retention

---

## Critical Limitations

### Data Limitations
- **No monetization events** (credits_purchased, payment_processed) → `is_paid_user = 0` for all users
- **No deployment events** (api_deployed, model_deployed) → `has_deployment = 0` for all users
- **No collaboration events** (canvas_shared, comment_added) → `collaboration_success = 0` for all users
- All metrics based on these families are structurally zero

### Model Limitations
- **Label leakage**: Success ensemble features are inputs to label generation
- **Inverted semantics**: `success_score` (formerly `churn_risk_score`) - 11 call sites still treat high success as high churn risk
- **Two unreconciled LTVs**: Behavioral (0-100) and Dollar ($) computed independently
- **GNN minimal lift**: Hybrid model shows −0.12 pp AUC vs baseline

### Reporting Limitations
- **Hand-coded literals**: Reports contain numbers like "642 paid users" that are not computed from data
- **17 duplicate reports**: Block re-runs regenerate identical content
- **LLM dependency**: Weekly insights block requires AWS Bedrock, no fallback

---

## Open Issues

### High Priority
1. Inverted `success_score` semantics (11 call sites need migration to `churn_proxy`)
2. Hand-coded report literals (reports don't reflect actual data)
3. Unicode encoding failures (~25 blocks fail without PYTHONUTF8=1)
4. NumPy dtype rejection (2 files with `select_dtypes(include=['object', 'str'])`)

### Medium Priority
1. Leaked f-string filename (`user_behavior_analytics_report_{report_date.replace('-', '')}.md`)
2. Duplicate block (`Engagement Forecast per Segment (Copy).py`)
3. Matplotlib date overflow in forecast visualization
4. Memory error under pressure in session analysis
5. LLM block no fallback

### Fixed (Apr 2026)
✅ Boolean leakage features removed from success ensemble
✅ `churn_risk_score` renamed to `success_score`
✅ `pd.Timestamp.now()` replaced with dataset max timestamp
✅ Behavioral similarity in Collaboration Network (kNN)
✅ GraphSAGE actually trained (PyTorch implementation)
✅ SHAP on EWS RandomForest
✅ Future-holdout retrain (4-feature baseline, AUC 0.9414)
✅ Rich-features holdout (33-feature variant, AUC 0.9168)

---

## Future Work

### Technical Debt (1-2 weeks)
1. Fix inverted `success_score` call sites
2. Add UTF-8 stdio wrapper to `run_canvas_locally.py`
3. Fix NumPy dtype rejection
4. Add LLM block fallback
5. Make reporting blocks data-driven
6. Fix matplotlib date overflow

### Methodological (1-2 months)
1. Implement truly leakage-free label (future holdout only)
2. Cross-validate on temporal splits
3. Explore alternative feature sets (reduce correlation)
4. Expand GNN to full user base
5. Reconcile two LTV computations

### Product Insights
1. Target Exploration → Creation gap (40% drop-off)
2. Implement week-1 session frequency nudges
3. Build actual paid user identification (requires real payment events)
4. Team onboarding flow for collaboration

---

## Presentation Structure (20 min)

1. **Title & Overview** (1 min): Objective, dataset, platform
2. **Architecture** (2 min): 7-stage DAG, execution model
3. **Data Pipeline** (2 min): Taxonomy, segmentation, features
4. **Modeling** (3 min): Success ensemble, EWS, survival, GNN
5. **Key Findings** (3 min): Engagement, adoption, workflow, retention
6. **Limitations** (2 min): Missing events, leakage, inverted semantics
7. **Future Work** (2 min): Technical debt, methodology, product
8. **Q&A** (3 min)

---

## Common Q&A

**Q: Why two churn models?**
A: Success ensemble predicts "High Value" user (99.51% accuracy but label-leaky). EWS is genuine churn model on 14 engineered features with independent label derivation.

**Q: What's the real churn prediction accuracy?**
A: Success ensemble 99.51% reflects label leakage. Future-holdout retrain shows true generalization AUC of 0.9414. EWS is the credible churn model (features and label independently derived).

**Q: Why is GNN lift so small?**
A: Hybrid model shows −0.12 pp AUC vs baseline. Likely due to small sample (4,593 rows) or limited graph signal. Graph is behavioral-similarity based, not ground-truth collaboration.

**Q: Why are paid user metrics zero?**
A: Dataset has no monetization events (credits_purchased, payment_processed). All paid user metrics are structurally zero. Report numbers like "642 paid users" are hand-coded literals, not computed from data.

**Q: What's the difference between the two LTVs?**
A: Behavioral LTV (0-100 score) is weighted composite of engagement metrics. Dollar LTV ($) uses discounted annuity with $30/month base. They're computed independently and not reconciled.

**Q: How does the pipeline run locally?**
A: `scripts/run_canvas_locally.py` topologically sorts canvas DAG, executes blocks in shared namespace, captures matplotlib figures, and supports checkpointing. Mimics Zerve runtime behavior.

**Q: What are the main blockers for clean local execution?**
A: Unicode encoding (need PYTHONUTF8=1), NumPy dtype rejection, matplotlib date overflow, memory pressure, LLM block dependency. All documented in `docs/repo_state_and_next_steps.md`.

**Q: How was the GNN verified?**
A: `scripts/build_gnn_colab_notebook.py` generates self-contained Colab notebook. Verified on Colab: pair_auc 0.900→0.909, hybrid AUC lift +0.15% (small but positive).

**Q: What's the biggest insight from the analysis?**
A: Deep work session ratio is the strongest leading indicator of long-term retention - stronger than total events or time on platform. Also, 40% drop-off at Exploration → Creation is the critical activation gap.

**Q: How would you improve this project?**
A: Fix technical debt (inverted semantics, hand-coded reports), implement leakage-free labels, expand GNN to full user base, add real monetization events, reconcile LTVs, make reporting data-driven.
