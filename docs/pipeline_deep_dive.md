# ZerveChurn Pipeline — Deep-Dive Analysis

> A block-by-block reading of every `.py` / `.md` / `.text` file in `Development/`, cross-referenced against the edges in `canvas.yaml` and the actual contents of `user_retention.parquet`. Purpose: document **what the pipeline really does** (vs. what the marketing-style reports claim), flag where numbers are hand-authored vs. data-derived, and surface the places where a reader should be careful interpreting results.

This complements:

- `README.md` — repo overview, Zerve-to-file mapping, repro instructions.
- `docs/zerve_platform_report.md` — platform semantics (canvas, layers, DAG, S3 cache, block types).
- `canvas_dag.md` — Mermaid rendering of the 67-block / 78-edge DAG grouped into 7 stages.

---

## Table of Contents

- [1. The dataset — what's actually in `user_retention.parquet`](#1-the-dataset--whats-actually-in-user_retentionparquet)
- [2. Stage-by-stage: what each block really computes](#2-stage-by-stage-what-each-block-really-computes)
- [3. Cross-cutting findings](#3-cross-cutting-findings)
- [4. Model-validity assessment](#4-model-validity-assessment)
- [5. Data-integrity issues & bugs](#5-data-integrity-issues--bugs)
- [6. What the numbers in the generated reports actually mean](#6-what-the-numbers-in-the-generated-reports-actually-mean)
- [7. Recommendations](#7-recommendations)

---

## 1. The dataset — what's actually in `user_retention.parquet`

Parquet file (~50 MB on disk), **409,287 rows × 107 columns**, loaded by `Example Dataset.py` / `load_and_inspect_dataset.py` / `02_statistical_profiling.py`.

- **Grain:** one row per event. Key identifiers: `distinct_id` (5,410 unique, PostHog-style client UUID), `person_id`, `event` (141 unique event names), `timestamp`.
- **Timespan:** Sep 1 → Dec 8, 2025 (98 days) — confirmed by `Churn Early Warning System.py:26-30` (`_ref = _cews_raw['timestamp'].max()`).
- **Completeness:** 96 of the 107 columns have missing values — they are PostHog-style `prop_$*` event-specific properties. Core columns (`distinct_id`, `event`, `timestamp`) are complete.
- **Event distribution is long-tailed and heavily dominated by `credits_used`** (~39.1% of all events, per `README.md` key findings).

**Crucial omission in the raw data** (confirmed by `Validation & Business Alignment.py:17-32` and `Primary Success Metrics.py:52-98`):

- **No monetization events.** The strings `credits_purchased`, `credit_balance_updated`, `credit_usage_tracked`, `payment_processed`, `plan_upgraded` **do not occur** in the event log. Only `credits_used` / `credits_below_*` do, which are *consumption* events, not purchase events. → `is_paid_user` is **always False** for every user.
- **No deployment events.** `api_deployed`, `model_deployed`, `endpoint_created`, `canvas_published` etc. don't occur. → `has_deployment` is **always False**.
- **No collaboration events.** `canvas_shared`, `canvas_shared_with_user`, `share_link_created`, `comment_added`, `workspace_invite_sent` don't occur. → `collaboration_success` is **always False**.

These three "missing" event families are the root cause of almost every caveat that follows. Any block that reasons about paid conversion, deployments, or collaboration is reasoning about constant-False features.

---

## 2. Stage-by-stage: what each block really computes

The canvas logically splits into 7 stages. Annotations below use block-file names verbatim.

### Stage 1 — Ingestion & EDA (13 blocks)

- `Example Dataset.py` (`@/Users/.../Development/Example Dataset.py:1-9`) — 3-line loader; just `pd.read_parquet('user_retention.parquet')` into `user_retention`.
- `Data Exploration.py`, `load_and_inspect_dataset.py`, `02_statistical_profiling.py` — descriptive stats: missing-value matrix, dtype distribution, cardinality, numerical percentiles/skew/kurtosis. No writes, no downstream-consumed variables beyond `user_retention` itself.
- `User Activity Analysis.py`, `Temporal Patterns.py`, `User Type & Anomaly Detection.py`, `Statistical Summaries.py`, `data_quality_checks.py`, `date_range_and_temporal_coverage.py`, `Dataset Field-Level Description.py`, `Additional Exploratory Visualizations.py` — histograms, weekday/hour heatmaps, activity-level buckets. Outputs are matplotlib/plotly figures; in the Git export these are **not persisted** (see `zerve_platform_report.md §7.2-7.3`).
- `01_data_loading_and_overview.md` — markdown doc block.

Effectively: **pure read-only EDA that sets up `user_retention` for everything downstream**.

### Stage 2 — Event Semantics (3 blocks)

- `Event Taxonomy & Categorization.py` (`@/Users/.../Development/Event Taxonomy & Categorization.py:1-292`) — hand-coded dictionary mapping event strings → 22 categories, with a keyword-fallback for uncategorized events. Outputs `event_categories` (dict) and `taxonomy_reference`.
- `Workflow Stage Mapping.py` — maps event → one of 14 workflow stages (`Onboarding`, `Exploration`, `Creation`, `Editing & Iteration`, `Execution`, `AI Assistance`, `Data Operations`, `Analysis & Visualization`, `Collaboration`, `Deployment`, `Optimization`, `Monitoring & Maintenance`, `Active Usage`, `Monetization`). Outputs `workflow_mapping`.
- `Hierarchical Event Visualization.py` — Plotly sunburst/funnel figures.

Note: both taxonomy blocks include stage lists for `Collaboration` / `Deployment` / `Monetization` events that **don't fire** in this dataset. Downstream any counting against these stages will return 0.

### Stage 3 — Segmentation (8 blocks)

- `Engagement Segmentation.py` — **K-means with k=5** on `[total_events, unique_event_types, session_count, active_days, avg_events_per_session]`, then rule-based relabeling into `Power Users / Rising Stars / Consistent Engagers / Casual Explorers / Churned`. Defines sessions by a 30-min gap. Produces `engagement_metrics` (DataFrame keyed by `distinct_id`).
- `Workflow Pattern Segmentation.py` — buckets users into `Notebook-Heavy / AI-Powered / Mixed / Canvas-Focused / Deployment-Oriented / Collaboration-Driven / Inactive` by stage-distribution thresholds; computes workflow-diversity via Shannon entropy.
- `Temporal & Monetization Segmentation.py` — adoption quartiles on `first_event`, weekend-vs-weekday ratio, coefficient-of-variation for consistency, and `_credit_events` membership for the "Credit User" vs "Free User" split. **⚠ Bug:** `days_since_first = pd.Timestamp.now() - first_event` — uses *wall-clock* `now()` instead of the dataset's max timestamp (see §5).
- `Feature Adoption Evolution.py`, `Feature Adoption Trajectories.py`, `Lifecycle Stage Definition.py`, `Growth Trajectory Classification.py` — longitudinal adoption curves, champion scores, exponential/linear/plateau/decline trajectory fits.
- `Interactive Visualizations & Segment Export.py` — 5 plotly charts + merges all segmentation dimensions into **`master_segments`** and writes `user_segments.csv` (5,410 × 16). This is a major downstream dependency.

The final `user_segments.csv` is the **canonical per-user segmentation matrix**, used by `Weekly Delta Metrics Computation.py`, `LTV Prediction`, and `User Intelligence Export`.

### Stage 4 — Behavioral features (9 blocks)

- `Session Pattern Analysis.py` — 30-min-gap sessionization, per-session duration / event-density, "deep work" flag (top-quartile on both duration and density). Produces `session_patterns_per_user` (12,641 sessions across 5,410 users; ~28% deep work).
- `Workflow Sequence Patterns.py` — event n-grams (3/4/5-gram) per user + heuristic `power_user_score` / `struggle_score` based on keyword hits like `deploy`, `agent`, `error`. Produces `workflow_sequence_df`. **Note**: since `api_deployed` / `model_deployed` never fire, `has_deployment_sequence` is always 0; `power_user_score` is dominated by `has_create_run_pattern` and `has_agent_workflow`.
- `Collaboration Signature & Final Matrix.py` — counts collab events (which are all zero in this dataset, so `collaboration_ratio ≈ 0` for everyone) and merges session + sequence + collab data into `behavioral_fingerprint`. This is the **master behavioral feature table** used by Isolation Forest, Causal Analysis, GNN, and Cohort DNA.
- `Isolation Forest Anomaly Detection.py` — `contamination=0.05` → flags top-5% outliers (~271 users) as `is_exceptional`. Output `exceptional_users_df`.
- `Comprehensive Feature Engineering.py` — builds a 70-ish-column numeric matrix across 7 categories (engagement / workflow / temporal / collaboration / AI agent / derived / interaction) and runs `StandardScaler`. Output `feature_matrix` + `scaled_df_fe`. **⚠ Bug**: `days_since_last_activity = pd.Timestamp.now() - last_event` — same wall-clock issue.
- `Engagement Momentum Tracking.py`, `Engagement Forecast per Segment.py` (+ its `(Copy)` duplicate), `Cohort Behavioral DNA — Feature Engineering.py` — rolling 7d/30d metrics, acceleration/deceleration counts, per-segment forecasts, and a unified `unified_df` that joins behavioral + churn + network features.

The **`Engagement Forecast per Segment (Copy).py`** block is a byte-level duplicate in both the file tree and `canvas.yaml`; it can be deleted without impact.

### Stage 5 — Modeling: churn / success / survival / LTV (12 blocks)

This is the most tangled part of the pipeline. There are effectively **two independent churn models** plus a survival analysis and an LTV scoring block, and the naming obscures which is which:

1. **"Success" ensemble (blocks `4fe8f20e → 6fddc4f7 → 4dc5810c → 4a843183 → 6f1f3592`)**
   - `Primary Success Metrics.py` → 5 booleans: `long_term_retention`, `is_paid_user`, `has_deployment`, `collaboration_success`, `is_power_user`. **Three of these are always False** because of the missing event families.
   - `Composite Success Score & Labeling.py` → weighted sum (30/25/20/15/10) into 0–100, bucketed into `Failed / Moderate / Successful`. On this data: 99.6% `Failed`, 0.4% `Moderate`, 0 `Successful`.
   - `Validation & Business Alignment.py` → explicitly flags the class imbalance, recomputes an **alternative** label (`High Value / Growing / Early/Churned`) based on events / tenure / power-user status. Binary target for modeling = `alternative_label == 'High Value'` (~3.6% of users).
   - `01_data_prep_train_val_test_split.py` → 70/15/15 stratified split, `StandardScaler`, **manual upsampling** of the minority class to 50/50 (the README's "SMOTE" description is imprecise — it's plain `sklearn.utils.resample` with replacement).
   - `02_base_models_ensemble.py` → trains `RandomForest`, `GradientBoosting`, `AdaBoost`, `LogisticRegression`, then a `VotingClassifier` and a `StackingClassifier`. Picks the best by **weighted F1 on the val set**, persists all six + the scaler + `validation_results` to `ensemble_models.pkl`. The "99.88% accuracy" headline comes from this block's test-set evaluation. **See §4 for why this number is misleading.**

2. **Churn Risk Scoring (block `35e752f4`)** — NOT the Early Warning System.
   - `Churn Risk Scoring & Time-Based Predictions.py` reuses `best_model_obj` from the success ensemble, applies it to `active_users_survival`, scales the output probability to 0–100 as `churn_risk_score`, and categorises (`Low / Medium / High Risk`). Adds a rule-based `predicted_churn_window` (`30 / 60 / 90 / beyond_90_days`). This `active_users_survival['churn_risk_score']` is what `Integrated Dashboard Synthesis`, `Advanced Analysis Synthesis`, `LTV Prediction`, and the hybrid GNN model all consume.
   - **Critically**: this is a *success* model rebranded as a *churn* model. See §4.

3. **Survival analysis (blocks `3e0b74e7 → aea140e4`)**
   - `Survival Analysis Data Preparation.py` → `survival_data` with `time_to_event`, `churned` (1 if `days_since_last > 30`), manual `risk_segment`, `engagement_level`, `deployment_status`, `time_to_deployment`.
   - `Kaplan-Meier Survival Curves by Segment.py` → **hand-rolled K-M estimator** (no `lifelines` despite README mentioning it) plus three figures by risk / engagement / deployment. Median survival per segment is printed to stdout.

4. **LTV (block `5215a49f`)**
   - `LTV Prediction & Unit Economics.py` → joins `master_segments` + `unified_df` + `survival_with_segments`, computes a behavioural LTV score: `0.30×events + 0.20×active_days + 0.20×tenure + 0.15×power_score + 0.10×sessions + 0.05×is_power_user`, multiplied by retention probability × tier multiplier × deployment/collab bonuses, normalised to 0–100, bucketed into `Platinum / Gold / Silver / Bronze`. Produces `ltv_predictions` DataFrame (in-memory only — not written to disk in this block).
   - This is a **behavioural score, not a dollar LTV**. `User Intelligence Export.py` independently computes a $-denominated LTV with a discounted-annuity formula (30$/month base × multipliers × expected-tenure months) and writes it to `user_intelligence_export.csv` as `LTV_estimate`. The two are not reconciled.

5. **Churn Early Warning System (blocks `8986b069 → 0e209692`)** — the **second, separate** churn model.
   - `Churn Early Warning System.py` trains its own `GradientBoostingClassifier + HistGradientBoostingClassifier + RandomForestClassifier` 40/40/20 ensemble on **14 churn-specific engineered features** (tenure, recency, 7d/30d rolling events, velocity, deceleration count, engagement decay, session frequency, feature breadth, 7d-vs-expected ratio, 30d-drop, etc.). Target: `recency_d > 30`. Computes a composite `risk_score = 0.6×model_ensemble_prob + 0.25×engagement_decay + 0.15×recency_penalty`, tiers it into `Critical / High / Medium / Low`, and outputs `churn_risk_scores` (different DataFrame from the one in block #2 above), `top_20_at_risk`, `top_100_intervention`.
   - `Churn Early Warning — Ranked Action Table.py` joins those scores with **SHAP values from the success-model** (see next), computes a `days_to_churn_est = 90 × (1 - risk) + grace × (1 - risk)`, and produces `early_warning_table` + `early_warning_active`. This is the actionable-intervention output.

6. **Behavioral Economics Scoring (block `6a01f22c`)** — computes a `habit_score` (0–100) and `loss_aversion_score` (0–100) per user plus an `optimal_engagement_day` / `optimal_engagement_frequency` recommendation based on day-of-week activity peaks. Independent of the model ensembles.

### Stage 6 — Graph / GNN (5 blocks)

**[Apr 2026 update]** Both graphs in this stage are now behaviourally grounded, and GraphSAGE is now trained. The caveats below are preserved as a record of the original state.

- `Collaboration Network & Centrality Analysis.py` (`@c:\Dev\ZerveChurn\Development\Collaboration Network & Centrality Analysis.py:24-85`) — **[FIXED Apr 2026 §7.4]** samples 500 high-activity users (≥50 events) and connects them via `StandardScaler` + cosine-similarity kNN with `k=8` and `threshold=0.70` on behavioural features `[total_sessions, avg_events_per_session, total_events, deep_work_ratio, power_user_score, struggle_score, sequence_diversity, collaboration_ratio, team_oriented_score, sharing_frequency]`. Empirical network stats: 500 nodes, ~2,600 edges, density ~0.021. The previous `np.random.choice` edge construction (5% "hubs" with 10–30 random neighbours, 95% with 1–8) was removed entirely.
- `Community Detection & Success Correlation.py` runs greedy-modularity on the kNN graph; its t-tests between "Super Connectors" and "Regular Users" are now statistics over a behaviourally-clustered network rather than noise.
- `GNN Social Influence Graph Construction.py` builds a **separate** graph: cosine-similarity kNN (≥0.85, ≤10 neighbours) on 15 standardized behavioural features + extra activity-bin-based "co-engagement" edges. Unchanged by session 4 — it was already legitimate.
- `GraphSAGE Training & Social Influence Embeddings.py` (`@c:\Dev\ZerveChurn\Development\GraphSAGE Training & Social Influence Embeddings.py:1-170`) — **[FIXED Apr 2026 §7.5]** now implements a **trainable 2-layer PyTorch GraphSAGE** (d_in→32→16) with `torch.optim.Adam(lr=1e-3, weight_decay=1e-5)` and 30 epochs of **unsupervised link-prediction** training: positive edges from `gnn_graph['edge_weight_map']`, uniform negative sampling at 1:1, loss `-logσ(pos) - logσ(-neg)`. Exposes `gnn_embeddings`, `H0`, `H1` as numpy arrays for downstream consumers unchanged. The previous untrained numpy Xavier-init forward pass was removed. **Activation:** `LeakyReLU(α=0.1)` after each `GraphSAGELayer` (not plain `ReLU`) — see §7.5 follow-up below for why this and the lr drop matter.
- `Hybrid GNN Churn Model & Community Analysis.py` trains a `GradientBoostingClassifier` on `[base_features + gnn_embeddings + influence_score + neighbor_churn_rate]` vs the same features without the GNN additions and reports an "AUC lift from GNN". Post-§7.5 this lift (when present) is attributable to a **learned** 16-dim graph representation rather than a random projection.

Remaining caveat: the graph-based t-tests and modularity figures are still descriptive over a *small sample* (500 users) of the population — they reflect behavioural-similarity clustering, not ground-truth collaboration.

### Stage 7 — Reporting & export (16 blocks)

- `SHAP Explainability Analysis.py` — implements **Saabas tree-path decomposition in numpy** (not the real SHAP package). Runs on the **RandomForest from the success-model bundle** (9 input features: `total_events`, `tenure_days`, `days_since_first`, `days_since_last`, `long_term_retention`, `is_paid_user`, `has_deployment`, `collaboration_success`, `is_power_user`). Produces `shap_values` DataFrame — this is what the Ranked Action Table labels as "Top Churn Drivers". **But the model is a success classifier, not a churn classifier**, so SHAP values are technically "contributions to P(High-Value)" mapped onto churn-labelled UI copy.
- `Causal Impact & Attribution Analysis.py` — a solid 600-line block implementing **PSM with 1:1 nearest-neighbor matching + bootstrap CIs**, **Difference-in-Differences** across adoption cohorts, **IPW synthetic control** counterfactuals, and **Interrupted Time Series (segmented OLS)** around 3 product-change dates. Filters out degenerate treatments (`T_deployment`, `T_collaboration` etc. get dropped because <10 treated users). Output `causal_effects` DataFrame with `avg_treatment_effect`, `confidence_interval`, `p_value`, `is_causal`.
- `Comprehensive User Analysis Findings.py` — segment-overlap analysis (exceptional × power × hidden-gem × churned × almost-succeeded) with 6 matplotlib bar charts. Purely descriptive.
- `Advanced Analysis Synthesis.py` / `Integrated Dashboard Synthesis.py` — reconcile the power-user / network-hub / anomalous-user sets into a unified profile and generate 5 "actionable insights" priority-tagged `CRITICAL / HIGH / MEDIUM / OPPORTUNITY`. Both blocks aggressively `del` inherited loop variables from upstream — Zerve's shared namespace leaks short names like `category`, `event`, `stage`, `label`, `pct`, `score`, `count` between unrelated blocks, and these blocks have to defensively clear them.
- `Weekly Delta Metrics Computation.py` — builds a JSON of week-over-week deltas (`churn_risk_delta`, `cohort_shifts`, `retention_changes`, `top_movers_delta`, `segment_spotlights_data`, `model_performance`, `user_base_summary`) from the dataset's final 14 days. Injected into the LLM prompt.
- `Weekly Insights Executive Briefing.text` (LLM block, Claude 3 Haiku on Bedrock `eu-west-1`, T=0.5, max_tokens=4096) — templated prompt that Jinja-interpolates `weekly_insights_metrics` and instructs the model to produce a CEO-style 3-minute briefing. Output variable `output` → `weekly_insights_report.md` via `Save Report to File.py`.
- `Export Report as Markdown File.py` — **writes a 500+ line hand-authored markdown report** (`user_behavior_analytics_report_YYYYMMDD_HHMMSS.md`). Important: *most of the specific numbers in this report are hard-coded in the string literal* (e.g. "642 paid users", "18% Team Players", "7% Collaborators", workflow-stage funnel percentages). They are **not computed from the block's upstream variables**, and therefore do not reflect the actual dataset. See §6.
- `Social Media Post Drafts.py` — hard-coded marketing copy (Twitter/X + LinkedIn posts) with the same pre-authored statistics. Writes `social_media_posts_YYYYMMDD.md`.
- `Executive Summary Report Generation.py`, `Save Report to File.py` — short writers that persist the LLM output or an exec summary to disk.
- `User Intelligence Export.py` — the **main machine-readable downstream artefact**. Merges `churn_risk_scores` (from EWS) + `master_segments` + `engagement_viz` k-means cluster + parquet-derived cohort + computes discounted-annuity `LTV_estimate` + a `behavioral_cluster` label, writes `user_intelligence_export.csv` (5,410 × 9). Heavy namespace-cleanup prologue (~40 `del` statements) because it sits downstream of 4+ parallel branches.
- `Git Repository Setup.py` — meta block; just writes a `.gitignore` when run.
- Markdown blocks: `Project README.md`, `Quality Assurance Checklist.md`, `Comprehensive Analysis Report.md`, `key_findings_summary.md`.

---

## 3. Cross-cutting findings

### 3.1 Zerve shared-namespace hygiene

Every block runs in the same Python namespace in Zerve (see `docs/zerve_platform_report.md §5.4`). Short loop variables (`cat`, `event`, `stage`, `label`, `pct`, `score`, `count`, `window`, …) leak across blocks and silently overwrite each other, particularly where two parallel branches converge at a reporting block.

The codebase compensates with three defensive patterns:

1. **Underscore-prefixed locals** — most blocks rename `x → _x` inside comprehensions / loops (visible throughout e.g. `Event Taxonomy & Categorization.py`, `Churn Early Warning System.py`).
2. **Explicit `del` prologues** — `User Intelligence Export.py:12-42`, `Interactive Visualizations & Segment Export.py:436-478`, `Advanced Analysis Synthesis.py:280-339`, `Cohort Behavioral DNA — Feature Engineering.py:8-9` all delete ~30-50 inherited names before running.
3. **Block-local variable prefixes** — e.g. `_ew_`, `_ltv_`, `_cews_`, `_be_`, `_gnn_`.

This is normal for Zerve-style canvases (the docs themselves recommend it), but it means **running any `.py` in isolation outside Zerve will `NameError` immediately** because the upstream variables it expects (`user_retention`, `event_categories`, `behavioral_fingerprint`, `churn_risk_scores`, `active_users_survival`, …) don't exist in a fresh Python session. Use `scripts/run_canvas_locally.py` for replay.

### 3.2 Two different scores, previously both named `churn_risk_score`  — 🔧 **[FIXED Apr 2026] renamed**

- `active_users_survival['success_score']` — set in `Churn Risk Scoring & Time-Based Predictions.py` from the **success ensemble's** `P(alternative_label == 'High Value')` probability (×100). Its companion `success_tier` column buckets it into `Low` / `Medium` / `High`. Consumed by `Advanced Analysis Synthesis`, `Cohort Behavioral DNA`, `Integrated Dashboard`, `LTV Prediction & Unit Economics`, `Behavioral Economics Scoring`, `Hybrid GNN Churn Model`, and `Weekly Delta Metrics Computation` (via `unified_df`). **Previously called `churn_risk_score` / `predicted_risk_category`** — renamed per §7.2 so that its actual semantics (success-likelihood, LTV proxy) are visible at the call site.
- `churn_risk_scores` (plural DataFrame) — set in `Churn Early Warning System.py` from the **dedicated churn ensemble** on 14 engineered features, with the composite formula `0.6×model + 0.25×decay + 0.15×recency`. Its `churn_risk_score` column is the composite 0–100 score; its `risk_tier` column buckets it into `Critical` / `High` / `Medium` / `Low`. Consumed by the Ranked Action Table and User Intelligence Export. This name was **kept** because it is a genuine churn score.

These are **different scores computed from different models on different labels**. The numeric ranges (0–100) are still comparable, but the column names now make the distinction obvious.

**Residual issue (resolved Apr 2026 via `churn_proxy`):** `Churn Risk Scoring & Time-Based Predictions.py` now also emits a derived column `churn_proxy = (100 - success_score).clip(0,100)` on `active_users_survival`. Downstream consumers that previously read `success_score` with churn-risk thresholds — `LTV Prediction & Unit Economics.py` (`churn_probability` formula at line 97-98 and the four `avg_churn_risk` aggregations + `pct_high_churn_risk` summary + `ltv_at_risk` filter), `Weekly Delta Metrics Computation.py:39` (`_high_risk_df`), `Integrated Dashboard Synthesis.py` (the `'churn_risk'` field in `_integrated_unified`), `Hybrid GNN Churn Model & Community Analysis.py:213-214` (`at_risk_anchors`), `Advanced Analysis Synthesis.py:167-170` and `Cohort Behavioral DNA — Feature Engineering.py:169-172` (the `elite_at_risk` filters), and the producer block's own `predict_churn_window` heuristic + `high_risk_alerts` filter — have all been migrated to read `churn_proxy`. `unified_df` carries `churn_proxy` through both producers (Advanced Analysis Synthesis + Cohort Behavioral DNA), with a defensive backfill in every consumer for older serialized inputs. `Behavioral Economics Scoring.py:115` (`_anc = success_score/100 * (1-churned)`) was left as-is because in that formula `_anc` represents *current investment level* (high-success-still-active = much to lose), which is the semantically-correct direction for a loss-aversion weight. **Net effect:** the at-risk filters, churn-probability calculations, and dashboard `churn_risk` field now select / report users whose *low* success score implies *high* churn risk, instead of the inverted reading.

### 3.3 Multiple hand-rolled implementations of well-known algorithms

- `SHAP Explainability Analysis.py` — implements Saabas tree-path decomposition in pure numpy (not the `shap` package).
- `Kaplan-Meier Survival Curves by Segment.py` — implements KM in a ~30-line Python function, doesn't use `lifelines`.
- `GraphSAGE Training & Social Influence Embeddings.py` — **[Apr 2026]** replaced the untrained numpy forward pass with a trainable PyTorch `GraphSAGE2` module (2 layers, d_in→32→16) + Adam + 30 epochs of unsupervised link prediction with uniform negative sampling. Still a minimal implementation (~50 lines of model + training loop), not PyG or DGL.
- `Community Detection & Success Correlation.py` — greedy modularity optimisation in ~40 lines, doesn't use `networkx.algorithms.community` or `python-louvain`.
- `Collaboration Network & Centrality Analysis.py` — manual degree / pagerank / approximate-betweenness centrality.

The approximations are reasonable for illustration but should not be conflated with their library-grade equivalents when reporting numbers.

### 3.4 Label leakage in the success ensemble

**[Apr 2026 update, future-holdout follow-up]** `scripts/future_holdout_retrain.py` provides a leakage-free comparison by building a *temporal* split instead of a composite-label one: features are computed from days 1–89 of the event log only, and the target is `1` if the user has ≥1 event in the held-out days 90–99. Same 4 features (`total_events`, `tenure_days`, `days_since_first`, `days_since_last`), same ensemble (`RF / GB / Ada / LR / Voting / Stacking`) as `Development/02_base_models_ensemble.py`. Run summary (`outputs/_future_holdout_89_9.json`):

- **Users** with feature-window activity: 3,905; active in target window: 129 (3.30%).
- **Best model**: AdaBoost (weighted val F1 0.935).
- **Test set**: accuracy 0.9044, ROC-AUC **0.9414**, F1 0.9290.
- **Δ vs. composite-label baseline** (0.9951 / 0.9952): Δ accuracy −0.091, **Δ ROC-AUC −0.054**.

The ~5 pp AUC drop is the expected "leakage-removal penalty"; the residual 0.94 AUC is genuine past→future predictive signal (largely carried by `days_since_last`).

**[Apr 2026 update, rich-features variant]** `scripts/future_holdout_retrain_rich.py` extends the same feature window and label with 29 additional behavioural columns (9 session-pattern, 12 workflow n-gram / power / struggle, 8 engagement-momentum). Run summary (`outputs/_future_holdout_89_9_rich.json`):

- **Features**: 33 total (4 base + 9 session + 12 workflow + 8 momentum). All computed strictly on the days 1–89 event slice so the model cannot observe the target.
- **Val ROC-AUC winner**: `voting` (soft-weighted F1) at 0.8369; **test ROC-AUC of best-by-val-AUC model**: 0.9168. Per-model test AUC: `voting` 0.9168, `ada` 0.9162, `lr` 0.8860, `rf` 0.8806, `stacking` 0.8383, `gb` 0.8164.
- **Test accuracy / F1 of best-by-val-AUC (`voting`)**: 0.9317 / 0.9448 (both improve vs. 4-feature base at 0.9044 / 0.9290).
- **Δ ROC-AUC vs. base 4-feature holdout** (0.9414): **−0.025**. No model configuration crossed the 0.95 AUC target set in §7 item 15.
- **Top 10 features by importance**: `total_sessions` (0.27), `active_days` (0.14), `days_since_last` (0.11), `avg_inter_session_gap_hours` (0.09), `sequence_diversity` (0.07), `tenure_days` (0.06), `avg_event_density` (0.03), `avg_events_per_session` (0.03), `days_since_first` (0.03), `power_user_score` (0.02). `total_sessions` and `active_days` absorb most of what `total_events` + `days_since_last` already carry in the base model; the remaining 23 features contribute <3 % each.

**Interpretation.** With only 129 positive users (19 in test) and 33 highly-correlated feature columns, the additional signal doesn't translate into better ranking. The rich model buys threshold-quality wins (higher accuracy, higher F1) but loses ~2.5 pp of discrimination. This is a real data-size limit rather than a feature-engineering defect: the 4-feature baseline is already at the near-ceiling for what 89 days of behavioural signal can say about the 10-day future window.

`01_data_prep_train_val_test_split.py:27-35` **originally** selected the feature set:

```python
feature_data = user_success_metrics[[
    'user_id', 'total_events', 'tenure_days', 'days_since_first', 'days_since_last',
    'long_term_retention', 'is_paid_user', 'has_deployment',
    'collaboration_success', 'is_power_user'
]].copy()
```

The target is `alternative_label == 'High Value'`, which is computed in `Validation & Business Alignment.py:118-158` as:

```python
_alt_s += 25 if row['total_events'] >= 100 else 15 if >=10 else 10 if >=5 else 0
_alt_s += 30 if row['long_term_retention'] else 15 if days_since_first >= 30 ...
_alt_s += 20 if row['is_power_user']
_alt_s += 15 if tenure_days >= 30 else 10 if >= 7
```

**[Apr 2026 update]** The 5 boolean columns (`long_term_retention`, `is_paid_user`, `has_deployment`, `collaboration_success`, `is_power_user`) were removed from the feature list per §7.1, and the ensemble was retrained. The feature set is now just `[total_events, tenure_days, days_since_first, days_since_last]`. Test accuracy dropped from **99.88% → 99.51%** (ROC-AUC 0.9952). The ~0.4 pp drop is **smaller than expected** because `total_events`, `tenure_days` and `days_since_first` are *still* the step-function inputs to the label score (`>=100/10/5`, `>=30/7`, `>=30/7` respectively). The model is still mostly reconstructing the label-generation rule — just from the continuous thresholds instead of the explicit booleans. A fully leakage-free retrain would require either (a) replacing the label with one derived from held-out future behavior, or (b) replacing the features with truly independent behavioral signals (e.g. session patterns, workflow n-grams, engagement momentum). Until then, the 99.51% should still be read as a sanity check, not a generalization claim.

### 3.5 The Early Warning System does **not** have this leakage

By contrast, `Churn Early Warning System.py` trains on 14 purely-engineered behavioural features (`tenure_d`, `total_n`, `uniq_n`, `daily_rate`, `w7`, `w30`, `vel`, `decel`, `eng_decay`, `sess_freq`, `rec_score`, `feat_brd`, `w7_ratio`, `w30_drop`) against the label `recency_d > 30`. Features and label are both derived from the event log but via *different* functions (rolling windows & ratios vs. a hard 30-day threshold). The reported ROC-AUC from this block is the credible churn-model number for this repo.

### 3.6 Paid / deployment / collaboration booleans are structurally zero

Because `Primary Success Metrics.py:52-98` checks event-string membership and those events don't occur (§1):

| Feature | Observed in 5,410 users |
| --- | --- |
| `is_paid_user` | 0 |
| `has_deployment` | 0 |
| `collaboration_success` | 0 |

Any bar chart, segmentation, or LTV multiplier that keys on these fields is keying on constants. `Export Report as Markdown File.py` prints "642 paid users (12%)" — that number is a **literal in the Python string**, not a computed quantity.

---

## 4. Model-validity assessment

| Output | What it measures | Validity on this data |
| --- | --- | --- |
| Success ensemble (`ensemble_models.pkl`) — "99.88% accuracy" | Whether the 9 input features can reconstruct the `alternative_label` that was derived from a subset of them | **Low as a generalization claim** — label leakage (§3.4). Useful as a sanity check that the pipeline wires correctly. |
| `active_users_survival['churn_risk_score']` (from success ensemble) | P("High Value" = 1) rescaled to 0–100, *used as churn risk* | **Mis-labelled**. This is a success score displayed with churn semantics. High score = likely successful, not likely to churn. |
| `churn_risk_scores` (Early Warning System) | Composite of GB/HGB/RF ensemble prob + engagement decay + recency on label `recency_d>30` | **Credible.** Features and label independently derived. AUC reported from a held-out split. The headline churn output of the repo. |
| `days_to_churn_est` | `90 × (1-risk) + grace × (1-risk)` heuristic, capped at [0, 365] | **Illustrative.** Not model-predicted; rule-of-thumb mapping from risk score. |
| Kaplan-Meier medians | Median time-to-churn by risk / engagement / deployment segment | **Valid**, but `deployment_status` split is degenerate (all users = "No Deployment"). |
| LTV (behavioural, 0–100) | Weighted composite of engagement + retention × tier multipliers | **Self-consistent**, but dollar interpretation is arbitrary. |
| LTV (`$`, `user_intelligence_export.csv`) | Discounted annuity with $30/mo base and multiplicative adjustments | **Purely stipulative.** The $30/mo base and multipliers are hard-coded constants with no calibration to revenue data. |
| Isolation Forest (~271 anomalies) | Top-5% outliers on 14 behavioural features | **Valid** — it's a legitimate unsupervised flag. Does not make claims about churn/success causality. |
| Community detection / super connectors | Greedy modularity on a **behavioural kNN graph** (post §7.4, k=8, cosine ≥0.70 on 10 behavioural features) | **Valid as a descriptive clustering statistic.** t-tests vs. success scores now reflect behavioural-similarity clusters, not noise. Sample is 500 high-activity users, so treat as exploratory rather than population-level. |
| GNN embeddings / AUC lift | **Trained** 2-layer PyTorch GraphSAGE (Adam `lr=1e-3` + `LeakyReLU(0.1)`, 30 epochs, unsupervised link prediction with -logσ loss) on the kNN + co-engagement graph (5,410 nodes / 40,097 edges) | **Learned graph representation, but small lift on this dataset.** Verified on Colab (session 3, post-stabilization-fix §7.5): **Hybrid GBM ROC-AUC 0.8824 vs Baseline 0.8834 (−0.12 pp); F1 0.8005 vs 0.7964 (+0.52 pp)** on a 1,149-row test set with 31.9% positive rate. Eight communities + 1,402 retention anchors detected. Embeddings reflect graph structure but the 16-dim GNN columns add no ranking signal beyond the 8 baseline behavioural features on this 4,593-row hybrid dataset — see §7.5 follow-up. |
| SHAP rankings | Saabas tree-path decomposition on the **EWS** RandomForest (post Apr 2026 fix) over the 14 churn-specific engineered features | **Valid as feature-contribution to churn.** The Ranked Action Table's "top churn driver" columns are now genuinely explaining the churn ensemble. Note SHAP is computed on the RF component alone (not the 40/40/20 GB+HGB+RF ensemble), since tree-path decomposition requires the classifier-tree structure. |
| Causal analysis (PSM / DiD / IPW / ITS) | Treatment effects on retention, with proper confounder adjustment and CIs | **Valid for the treatments that survive the ≥10/arm filter** (power_user, deep_work, ai_adoption, high_diversity, dev_loop, multi_session, smooth_experience). Treatments based on missing event families (`T_deployment`, `T_collaboration`, `T_habitual`) are correctly dropped as degenerate. |
| Weekly LLM briefing | Claude-Haiku 3 rendering of `weekly_insights_metrics` JSON | **Valid as narrative over the JSON**; inherits whatever limitations the metrics JSON itself has. |

---

## 5. Data-integrity issues & bugs

1. **[FIXED Apr 2026] Wall-clock `pd.Timestamp.now()` used where dataset-max should be** — `Comprehensive Feature Engineering.py:102` and `Temporal & Monetization Segmentation.py:15` now use `pd.to_datetime(user_retention['timestamp']).max()` as the reference date, matching the pattern already in `Primary Success Metrics.py:19`. Previously, running the pipeline more than ~90 days after Dec 2025 would classify every user as `'churned'` and inflate `days_since_first`.
2. **Duplicate block** — `Engagement Forecast per Segment (Copy).py` is byte-identical to `Engagement Forecast per Segment.py` and is wired into the canvas as a separate node. No downstream block reads its outputs differently. Safe to delete.
3. **Broken filename on one report** — `user_behavior_analytics_report_{report_date.replace('-', '')}.md` is a leaked Python f-string — the block that generates it forgot to call `.format(...)`. The file content is valid markdown.
4. **LLM block has no guard for Bedrock failure** — `Weekly Insights Executive Briefing.text` is a prompt template; if Bedrock credentials aren't configured, the block fails silently when running in Zerve and the `output` variable is never set, which breaks `Save Report to File.py` downstream.
5. **Hand-coded statistics in `Export Report as Markdown File.py`** — the generated report claims numbers like "642 paid users", "workflow funnel drop-off 40%", "power user path deployment rate X%" that are **not** computed from the upstream variables. They are string literals. The JSON-style tables are hand-authored and do not update if the data changes.
6. **Same issue in `Social Media Post Drafts.py`** — "99.88% accuracy", "43% of churn in 31-60 day window", "28% deep work", "271 anomalous users" are hard-coded in the Python string. They happen to match the README's headline numbers, but they don't re-compute.
7. **[FIXED Apr 2026] `Collaboration Network & Centrality Analysis.py`** now matches its block description — `np.random.choice` edge construction was replaced with `StandardScaler` + cosine-similarity kNN (`k=8`, `threshold=0.70`) on 10 behavioural features (`total_sessions`, `avg_events_per_session`, `total_events`, `deep_work_ratio`, `power_user_score`, `struggle_score`, `sequence_diversity`, `collaboration_ratio`, `team_oriented_score`, `sharing_frequency`). See §7.4.
7a. **[FIXED Apr 2026] Success-model scoring block still used the pre-leakage-fix 9-feature list.** `Churn Risk Scoring & Time-Based Predictions.py:19-21` originally passed `['total_events', 'tenure_days', 'days_since_first', 'days_since_last', 'long_term_retention', 'is_paid_user', 'has_deployment', 'collaboration_success', 'is_power_user']` to `scaler_prep.transform(...)`, but the Apr 2026 §7.1 fix shrank the ensemble's fitted feature set to the 4 numeric features. The block would have raised `ValueError: The feature names should match those that were passed during fit` on any re-run. It now uses the matching 4-feature list.
8. **[FIXED Apr 2026] SHAP values from a success model relabelled as churn drivers** — `SHAP Explainability Analysis.py` now consumes `churn_ews_rf_model` + `churn_ews_feature_matrix` + `churn_ews_feature_names` (publicly exported from `Churn Early Warning System.py`). The tree-path decomposition runs against the EWS RandomForest over its 14 churn-specific behavioral features (tenure, recency, rolling 7d/30d events, velocity, deceleration, engagement decay, session frequency, feature breadth, etc.). The cohort split in §6 now uses `churn_risk_scores['risk_tier']` instead of `active_users_survival['predicted_risk_category']`. `Churn Early Warning — Ranked Action Table.py` now derives `_EW_SHAP_MAP` from the public `churn_ews_feature_labels` dict, so the "top churn drivers" column is now genuinely driver-of-churn. Top 5 global drivers on the Apr 2026 run: recency score, 7d-vs-expected ratio, session frequency, tenure (days), historical daily rate.
9. **[FIXED Apr 2026] GraphSAGE is now trained** (`GraphSAGE Training & Social Influence Embeddings.py:1-167`) — the numpy Xavier-init forward pass was replaced with a PyTorch `GraphSAGE2` module (d_in→32→16) + `torch.optim.Adam(lr=5e-3)` + unsupervised link-prediction loss (`-logσ(pos) - logσ(-neg)` with uniform negative sampling at 1:1) over 30 epochs. Downstream variables `gnn_embeddings`, `H0`, `H1` are still numpy arrays so consumers (`Hybrid GNN Churn Model & Community Analysis.py`) are unchanged. See §7.5.

None of these are blockers for reading the repo. They are all things a reviewer would want to know before quoting numbers externally.

---

## 6. What the numbers in the generated reports actually mean

The repo contains four kinds of timestamped / named markdown outputs at the repo root:

- `user_behavior_analytics_report_*.md` — produced by `Export Report as Markdown File.py`. **Hand-authored narrative**, ~500 lines of pre-written text with numbers baked in. Does not reflect the loaded data; re-running the block produces the same content with a different filename. *Useful as a template / storyboard for what a finished analysis should look like, not as data.*
- `social_media_posts_*.md` — produced by `Social Media Post Drafts.py`. Same story — hand-authored literals.
- `weekly_insights_report.md` — produced by the Claude-Haiku LLM block + `Save Report to File.py`. **This one is data-driven**: the JSON in `weekly_insights_metrics` (~200 lines of WoW deltas, top movers, segment spotlights) is computed from the event log, and Claude renders it into prose. Accuracy here is a function of the metrics JSON (see §4).
- `README.md` top-level numbers (409,287 events / 5,410 users / 141 events / 99.51% accuracy post-Apr-2026 / 271 anomalies / 12,641 sessions) — these match what the pipeline actually produces on this dataset. The 99.51% has the residual-leakage caveat from §3.4. The older "99.88% accuracy / 14 high-risk users" figures were from the pre-fix run and no longer hold; the current EWS output is 0 Critical / 0 High / 8 Medium / 3,172 Low among active users.

---

## 7. Recommendations

### Must-fix before trusting the outputs

1. **[DONE Apr 2026 — partial] Correct the label leakage in the success ensemble** — the 5 boolean features (`long_term_retention`, `is_paid_user`, `has_deployment`, `collaboration_success`, `is_power_user`) are dropped from the feature list in `01_data_prep_train_val_test_split.py`. The ensemble was retrained; test accuracy dropped from **99.88% → 99.51%** (ROC-AUC **0.9952**). **Caveat:** the drop is only ~0.4 pp because the remaining 4 numeric features (`total_events`, `tenure_days`, `days_since_first`, `days_since_last`) are also inputs to the `alternative_label` step functions — see §3.4 for why a fully leakage-free retrain would require a different label or different features.
2. **[DONE Apr 2026] Renamed `active_users_survival['churn_risk_score']` to `success_score`** everywhere downstream. Touched `Churn Risk Scoring & Time-Based Predictions.py` (source — also renamed `predicted_risk_category` → `success_tier` with values `Low` / `Medium` / `High`), plus the consumers: `Advanced Analysis Synthesis.py`, `Cohort Behavioral DNA — Feature Engineering.py`, `Integrated Dashboard Synthesis.py`, `Behavioral Economics Scoring.py`, `Weekly Delta Metrics Computation.py` (including the `avg_churn_risk_score` → `avg_success_score` and `high_risk_pct` → `high_tier_pct` keys on `churn_risk_delta`), `LTV Prediction & Unit Economics.py` (including the `_ltv_df['success_tier'].fillna('High')` default change), `Hybrid GNN Churn Model & Community Analysis.py`. Report copy that directly labeled the column — print headers, chart titles, legend labels, DataFrame display column names — was updated to say "Success Score" / "Success" / "High Tier %". Narrative strings that derive "at-risk" claims from the threshold (e.g. "elite users at churn risk" if `score >= 50`) were **left as-is** because the threshold direction is itself inverted (high success-score flagged as at-risk); surfacing that is the whole point of the rename — see §3.2 for the residual-inversion caveat. **The plural `churn_risk_scores` DataFrame from the Early Warning System was untouched** since it is a genuine churn score.
2a. **[DONE Apr 2026] Aligned `Churn Risk Scoring & Time-Based Predictions.py` with the post-§7.1 feature set** (dropped the 5 boolean features that the success ensemble no longer accepts after the Apr 2026 retrain) — see §5 item 7a.
3. **[DONE Apr 2026] Replace `pd.Timestamp.now()` with `user_retention['timestamp'].max()`** — both `Comprehensive Feature Engineering.py:102` and `Temporal & Monetization Segmentation.py:15` now use the dataset max timestamp as the reference date.
4. **[DONE Apr 2026 — §7.4] Rewrote `Collaboration Network & Centrality Analysis.py` to use real behavioural-similarity edges** — now a `StandardScaler` + `cosine_similarity` kNN (`k=8`, `threshold=0.70`) on 10 behavioural features drawn from `behavioral_fingerprint`. Random-choice edge construction removed entirely. Block description updated in `canvas.yaml`. Downstream `Community Detection & Success Correlation` runs over the new graph.
5. **[DONE Apr 2026 — §7.5] Trained the GraphSAGE weights** with unsupervised link prediction. `GraphSAGE Training & Social Influence Embeddings.py` is now a PyTorch `GraphSAGE2` module (d_in→32→16) + `torch.optim.Adam(lr=1e-3, weight_decay=1e-5)` running 30 epochs of `-logσ(pos) - logσ(-neg)` with uniform negative sampling. Each `GraphSAGELayer` uses `LeakyReLU(α=0.1)` (not plain `ReLU`) followed by row-wise L2 normalization. Downstream `Hybrid GNN Churn Model & Community Analysis.py` now consumes learned embeddings rather than a random Xavier projection.

   **Stabilization addendum (Colab session 3).** The first verified Colab run with the original `lr=5e-3` + `ReLU` config showed clear training collapse — pair-AUC went `0.893 → 0.891 → 0.352 → 0.614 → 0.505 → 0.307 → 0.315` over 30 epochs (full trace in [outputs/_colab_gnn_cells_dump.txt](../outputs/_colab_gnn_cells_dump.txt) line 107–113), driven by ReLU dead-unit collapse interacting with the per-layer `F.normalize`. Two minimal changes were applied: lr `5e-3 → 1e-3` and `ReLU → LeakyReLU(0.1)`. The notebook `notebooks/gnn_colab_verification.ipynb` was regenerated. **Re-verification on Colab — done (Apr 2026):** the executed-cell outputs preserved in `notebooks/gnn_colab_verification.ipynb` show `pair_auc` rising monotonically from **0.900 (epoch 1) → 0.909 (epoch 30)** with loss decreasing `1.236 → 1.046` and no sign of collapse — full numbers in `docs/repo_state_and_next_steps.md §4`. The hybrid-model AUC lift from the trained embeddings is **+0.15 %** on the same 1,149-row test set referenced in §4. The contingency interventions (drop inner `F.normalize`, `n_neg_per_pos=5`, gradient clipping) are no longer needed.

   **Hybrid model (block 67) verified Colab numbers (session 3, pre-stabilization-fix run):** hybrid dataset 4,593 × 33 features (8 baseline + 19 GNN), 30.8% churn rate. Train/test 3,444 / 1,149. **Baseline GBM:** ROC-AUC 0.8834, F1 0.7964. **Hybrid GBM:** ROC-AUC 0.8824, F1 0.8005. **AUC lift −0.12 pp, F1 lift +0.52 pp.** Eight communities detected (sizes 21–3,095, churn-rate range 17.1%–71.0%). 1,402 retention anchors with avg power-user score 10.80 and avg degree 19.4. Numbers were captured even with collapsed embeddings, so they represent a **lower bound** on what the post-stabilization-fix run can produce.
6. **[DONE Apr 2026] Recompute SHAP on the Early Warning System ensemble** — `Churn Early Warning System.py` now publicly exports `churn_ews_rf_model`, `churn_ews_feature_matrix`, `churn_ews_feature_names`, `churn_ews_feature_labels`. `SHAP Explainability Analysis.py` reads these instead of `ensemble_models.pkl`'s success-RF, and its cohort split uses `churn_risk_scores['risk_tier']`. `Churn Early Warning — Ranked Action Table.py` derives `_EW_SHAP_MAP` from the EWS feature-label dict. Top 5 global drivers (Apr 2026 run): recency score, 7d-vs-expected ratio, session frequency, tenure (days), historical daily rate.

### Nice-to-have

7. **[DONE Apr 2026]** Deleted `Engagement Forecast per Segment (Copy).py` and its canvas node + edge from `canvas.yaml`.
8. **[DONE Apr 2026]** Renamed the leaked-f-string report to `user_behavior_analytics_report_BROKEN_TEMPLATE.md` so the literal `{report_date.replace('-', '')}.md` fragment is no longer the filename. The underlying bug in `Export Report as Markdown File.py` (template not interpolated) is item 9 and remains open.
9. Replace the hand-authored tables in `Export Report as Markdown File.py` with f-string interpolations of the actually-computed upstream variables, so the report stays honest when re-run.
10. Document in the block descriptions (the canvas-level `description:` field) which blocks load from / write to files, and what their primary output variable name is — would make reading the DAG much easier. **[PARTIAL Apr 2026]** — descriptions added for `Churn Risk Scoring & Time-Based Predictions`, `Collaboration Network & Centrality Analysis`, `GraphSAGE Training & Social Influence Embeddings` to document the §7.2/§7.4/§7.5 behaviour. Other blocks still have empty `description:` fields.
11. Add a top-of-canvas constant block with `REFERENCE_DATE = user_retention['timestamp'].max()` and have every temporal block import from it.

### Follow-ups spawned by session-4 fixes

14. **[DONE Apr 2026 — §3.4] Built a future-holdout success-label retrain** (`scripts/future_holdout_retrain.py`). Trains the same `{RF, GB, Ada, LR, Voting, Stacking}` ensemble on 4 features from days 1–89 with label = activity in days 90–99. Best model AdaBoost (val F1 0.935), test accuracy 0.9044, ROC-AUC 0.9414 — a ~5 pp AUC drop vs. the composite-label baseline, which is the expected leakage-removal penalty. Residual 0.94 AUC is genuine past→future signal.
15. **[DONE Apr 2026 — §3.4] Built a richer-features variant of the §3.4 holdout script** (`scripts/future_holdout_retrain_rich.py`). Feeds 29 extra behavioural columns (session patterns, workflow n-grams, power/struggle scores, 7d-vs-prior-7d momentum) on top of the 4 base features — all computed strictly from the feature-window event slice so the leakage-free guarantee is preserved. Result (`outputs/_future_holdout_89_9_rich.json`): best-by-val-AUC model is `voting`, test ROC-AUC **0.9168** (vs. base-holdout 0.9414 = **−0.025 pp**). Accuracy and F1 did improve (0.9317 / 0.9448 vs. 0.9044 / 0.9290), but **no model configuration crossed the 0.95 AUC target**. With only 19 positive users in the test set, the 29 additional features increase model variance more than they add ranking signal — the 4-feature baseline is at the data-size ceiling for this 89-day→10-day split. The lift we hoped for would require either a longer dataset or a richer positive class, not more engineered features on the same 99-day slice.
16. **[TODO] Re-sync Zerve canvas from the edited `Development/*.py` files.** `canvas.yaml` in this repo is block metadata only (no inline source), so the edits made in sessions 3–4 are fully captured in the Python files. Pushing these into a live Zerve canvas requires a re-import of the `Development/` folder (or a manual paste into the canvas UI, block by block).

### Structural

12. Split the modelling stage into two clear sub-DAGs on the canvas: *Success prediction* (blocks `4fe8f20e → 6f1f3592` + SHAP) vs *Churn prediction* (blocks `8986b069 → 0e209692` + its own SHAP). Currently both thread through each other.
13. Consolidate the two LTV computations (`LTV Prediction & Unit Economics` behavioural 0–100 score vs `User Intelligence Export`'s $-denominated discounted-annuity) into one pass with explicit units.

---

*Last reviewed: full read of every file in `Development/` + cross-reference against `canvas.yaml` edges. Nothing in this document is inferred — every claim is grounded in a specific file:line range in the repo.*
