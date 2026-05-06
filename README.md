# ZerveChurn

> End-to-end user-behavior analytics & churn-prediction project built on the [Zerve](https://www.zerve.ai) canvas platform, exported to a plain Git repo for review, archival, and local inspection.

**Dataset:** 409,287 events · 5,410 users · 141 event types · Sep 1 – Dec 8, 2025 (98 days)
**Primary model:** Soft Voting ensemble (99.51% test accuracy / 0.9952 ROC-AUC on composite success label, retrained Apr 2026 with leaked boolean features removed — see `docs/pipeline_deep_dive.md §3.4` for the residual numeric-leakage caveat). **Leakage-free future-holdout sibling** (`scripts/future_holdout_retrain.py`, same ensemble on days 1–89 features → days 90–99 activity label): **ROC-AUC 0.9414** / accuracy 0.9044 — a ~5 pp AUC drop that quantifies how much of the composite-label headline was feature↔label leakage.
**Canvas:** 67 blocks · 78 edges · 1 layer (`Development`)

---

## Table of Contents

- [What's in this repo](#whats-in-this-repo)
- [Repository layout](#repository-layout)
- [How the Zerve canvas maps to files](#how-the-zerve-canvas-maps-to-files)
- [Pipeline stages](#pipeline-stages)
- [Data & model artifacts](#data--model-artifacts)
- [Generated reports](#generated-reports)
- [Frontend web application](#frontend-web-application)
- [Local tooling (`scripts/`)](#local-tooling-scripts)
- [Reproducing / running locally](#reproducing--running-locally)
- [Key findings](#key-findings)
- [Notes & caveats](#notes--caveats)
- [Further reading](#further-reading)

---

## What's in this repo

This is a **read-only export** of a Zerve project called *ZerveXHackerEarth (Clone)* plus the artifacts it produced when run:

1. **`canvas.yaml`** — declarative canvas graph (blocks + edges + layout). Source of truth for the visual DAG in Zerve.
2. **`Development/`** — one file per canvas block, exported as `.py` / `.md` / `.text`. This is where the actual analysis code lives.
3. **`Frontend/`** — Streamlit-based interactive web application for real-time churn prediction and exploratory data analysis.
4. **Data & model artifacts** — `user_retention.parquet` (raw events), `user_segments.csv`, `user_intelligence_export.csv`, `ensemble_models.pkl` (trained model bundle).
5. **Generated reports** — timestamped markdown files (`user_behavior_analytics_report_*.md`, `social_media_posts_*.md`, `weekly_insights_report.md`) produced by the "Export Report", "Social Media Post Drafts", and "Weekly Insights Executive Briefing" blocks.
6. **Local tooling** — `scripts/render_canvas_dag.py` (Mermaid DAG generator) + its output `canvas_dag.md`.

---

## Repository layout

```
ZerveChurn/
├── README.md                                         # you are here
├── .gitignore                                        # smart ignores; datasets are TRACKED
│
├── canvas.yaml                                       # Zerve canvas export (blocks + edges)
├── canvas_dag.md                                     # Mermaid DAG rendered from canvas.yaml
│
├── Development/                                      # one file per canvas block
│   ├── layer.yaml                                    # layer-level canvas metadata
│   ├── Example Dataset.py                            # entrypoint: loads user_retention.parquet
│   ├── Data Exploration.py
│   ├── 01_data_loading_and_overview.md               # (markdown block)
│   ├── 02_statistical_profiling.py
│   ├── …
│   ├── Churn Early Warning System.py
│   ├── Churn Early Warning — Ranked Action Table.py
│   ├── Hybrid GNN Churn Model & Community Analysis.py
│   ├── Weekly Insights Executive Briefing.text       # LLM prompt block
│   ├── Project README.md                             # Zerve-side project overview
│   └── Quality Assurance Checklist.md
│
├── Frontend/                                         # Streamlit web application
│   ├── streamlit_webapp.py                           # interactive churn predictor & EDA app
│   └── .streamlit/
│       └── config.toml                               # Streamlit configuration
│
├── scripts/                                          # local tooling — see "Local tooling" below
│   ├── render_canvas_dag.py
│   ├── run_canvas_locally.py
│   ├── future_holdout_retrain.py
│   ├── future_holdout_retrain_rich.py
│   ├── build_gnn_colab_notebook.py
│   └── _dump_cells.py
│
├── docs/                                             # deep-dive references
│   ├── pipeline_deep_dive.md
│   ├── repo_state_and_next_steps.md
│   ├── zerve_platform_report.md
│   └── notebook_mcp_issue_report.md                  # unrelated FastMCP bug report
│
├── notebooks/
│   └── gnn_colab_verification.ipynb                  # Colab notebook for GraphSAGE / Hybrid GNN
│
├── outputs/                                          # local-replay run artefacts
│   ├── _run_72f.log / _run_73.* / _run_74.* / _run_75.log    # canvas-replay run logs
│   ├── _future_holdout_89_9{,.rich}.json|log         # leakage-free holdout summaries
│   ├── _colab_gnn_cells_dump.txt                     # captured Colab cell trace
│   ├── _state/  (gitignored — ~16 GB pickled namespaces)
│   └── <Block Name>/fig_NN.png                       # matplotlib captures per block
│
├── user_retention.parquet                            # raw input: 409K event rows × 107 cols (~50 MB)
├── user_segments.csv                                 # master segmentation matrix (5,410 users × 16 cols)
├── user_intelligence_export.csv                      # downstream user-level intelligence export
├── ensemble_models.pkl                               # trained ensemble (6 models, ~6 MB)
│
├── user_behavior_analytics_report_20260424_114927.md # latest comprehensive analytics report
├── social_media_posts_20260424.md                    # latest marketing post drafts
└── weekly_insights_report.md                         # LLM-generated executive briefing
```

---

## How the Zerve canvas maps to files

Each **block** in `canvas.yaml` has a corresponding file in `Development/`, where:

| Block `type` in YAML | File extension | Role                                   |
| -------------------- | -------------- | -------------------------------------- |
| `1`                  | `.py`          | Code / compute block                   |
| `4`                  | `.md`          | Markdown / documentation block         |
| `9`                  | `.text`        | LLM prompt block (Bedrock Claude Haiku)|

**Edges** in `canvas.yaml` express data-flow / ordering dependencies between blocks. `scripts/render_canvas_dag.py` consumes both and emits `canvas_dag.md` as a Mermaid diagram grouped into seven logical stages.

Open `canvas_dag.md` in any Mermaid-aware Markdown viewer (VS Code preview with the *Markdown Preview Mermaid Support* extension, GitHub, Obsidian) to see the full DAG.

---

## Pipeline stages

The 67 blocks cluster into seven stages (full per-block descriptions live in `canvas.yaml`):

**1. Ingestion & EDA**
`Example Dataset` → `Data Exploration` / `load_and_inspect_dataset` → `02_statistical_profiling`, `Statistical Summaries`, `User Activity Analysis`, `Temporal Patterns`, `User Type & Anomaly Detection`, `data_quality_checks`, `date_range_and_temporal_coverage`, `Dataset Field-Level Description`, `Additional Exploratory Visualizations`.

**2. Event Semantics**
`Event Taxonomy & Categorization` (141 events → 22 categories) → `Workflow Stage Mapping` (→ 14 stages) → `Hierarchical Event Visualization`.

**3. Segmentation**
`Engagement Segmentation` (K-means), `Workflow Pattern Segmentation`, `Temporal & Monetization Segmentation`, `Feature Adoption Evolution`, `Lifecycle Stage Definition`, `Growth Trajectory Classification`, `Interactive Visualizations & Segment Export` → produces `user_segments.csv`.

**4. Behavioral Features**
`Session Pattern Analysis` (30-min gap sessionization), `Workflow Sequence Patterns` (n-grams), `Collaboration Signature & Final Matrix`, `Isolation Forest Anomaly Detection`, `Comprehensive Feature Engineering` (70+ features across 7 categories), `Engagement Momentum Tracking`, `Engagement Forecast per Segment`, `Cohort Behavioral DNA — Feature Engineering`.

**5. Modeling — Success / Churn / Survival / LTV**
This stage actually runs **two independent model families** that share feature names and output ranges, so naming can mislead:
- **Success ensemble** (`Primary Success Metrics` → `Composite Success Score & Labeling` → `Validation & Business Alignment` → `01_data_prep_train_val_test_split` → `02_base_models_ensemble`). Target = binary `alternative_label == 'High Value'`. Trains 4 base + 2 ensemble classifiers, persists the best to `ensemble_models.pkl`. This is what the 99.88% headline number refers to. `Churn Risk Scoring & Time-Based Predictions` then rebrands this model's probability as `churn_risk_score` on `active_users_survival`, which feeds `Integrated Dashboard`, `Advanced Synthesis`, `LTV Prediction`, and `Hybrid GNN Churn Model`.
- **Churn Early Warning ensemble** (`Churn Early Warning System` + `Ranked Action Table`). Target = `recency_days > 30`. Trains its own `GradientBoosting + HistGB + RandomForest` weighted 40/40/20 on 14 churn-specific engineered features, then composites with engagement-decay + recency penalty. Output = `churn_risk_scores` (separate DataFrame, same column range but different semantics), `top_20_at_risk`, `top_100_intervention`, `early_warning_active`. This is the credible churn output of the repo — see `docs/pipeline_deep_dive.md §3.5`.

Alongside these: `Survival Analysis Data Preparation` → `Kaplan-Meier Survival Curves by Segment` (hand-rolled KM, not `lifelines`), `LTV Prediction & Unit Economics` (behavioural 0–100 score — distinct from the $-denominated LTV in `User Intelligence Export`), and `Behavioral Economics Scoring` (`habit_score` / `loss_aversion_score` / optimal-day recommendations).

Note: `01_data_prep_train_val_test_split` uses plain `sklearn.utils.resample` upsampling, **not** SMOTE despite the README earlier calling it that.

**6. Graph / GNN**
`Collaboration Network & Centrality Analysis` (kNN cosine similarity on behavioural features, post-§7.4) → `Community Detection & Success Correlation` → `GNN Social Influence Graph Construction` → `GraphSAGE Training & Social Influence Embeddings` (PyTorch, trained via unsupervised link prediction, post-§7.5) → `Hybrid GNN Churn Model & Community Analysis`.

**7. Reporting & Export**
`SHAP Explainability Analysis`, `Causal Impact & Attribution Analysis`, `Integrated Dashboard Synthesis`, `Advanced Analysis Synthesis`, `Weekly Delta Metrics Computation`, `Weekly Insights Executive Briefing` (LLM — `anthropic.claude-3-haiku-20240307-v1:0` on Bedrock `eu-west-1`), `Executive Summary Report Generation`, `Save Report to File`, `Export Report as Markdown File`, `Social Media Post Drafts`, `User Intelligence Export` → produces the timestamped markdown reports and `user_intelligence_export.csv`.

---

## Data & model artifacts

| File                              | Rows         | Produced by                                                  | Notes                                    |
| --------------------------------- | ------------ | ------------------------------------------------------------ | ---------------------------------------- |
| `user_retention.parquet`          | 409,287 × 107 | (external, input)                                           | Raw event log — required entrypoint      |
| `user_segments.csv`               | 5,410 × 16   | `Interactive Visualizations & Segment Export.py`             | Master segmentation matrix               |
| `user_intelligence_export.csv`    | 5,410 × N    | `User Intelligence Export.py`                                | Downstream per-user intelligence table   |
| `ensemble_models.pkl`             | —            | `02_base_models_ensemble.py`                                 | 6-model ensemble, joblib-pickled         |

All four artefacts are **tracked in this repo** so audits and Colab runs work without external uploads. The `.gitignore` no longer blanket-ignores `*.parquet` / `*.csv` / `*.pkl`; if you fork the repo and want to drop the heavy binaries, uncomment the patterns at the bottom of `.gitignore`.

---

## Generated reports

All timestamped markdown files at the repo root are build outputs of reporting blocks:

- **`user_behavior_analytics_report_YYYYMMDD[_HHMMSS].md`** — comprehensive 8-section analytics report (`Export Report as Markdown File.py`).
- **`social_media_posts_YYYYMMDD[_HHMMSS].md`** — ready-to-publish post drafts (`Social Media Post Drafts.py`).
- **`weekly_insights_report.md`** — LLM-generated executive briefing from the Claude Haiku block (`Weekly Insights Executive Briefing.text` + `Save Report to File.py`).

Reports are regenerated on each block run; the timestamp suffix prevents overwrites. Only the **most recent** of each family is kept in this repo (older identical re-runs were pruned in the Apr 2026 cleanup — the reporting blocks emit hand-coded literals so the older files were byte-identical duplicates of the latest, see `docs/pipeline_deep_dive.md §6`).

---

## Frontend web application

A **Streamlit-based interactive web application** is available in `Frontend/streamlit_webapp.py` for real-time churn prediction and exploratory data analysis.

### Features

**Tab 1: Churn Predictor**
- Upload `ensemble_models.pkl` to load the trained model ensemble
- Select from 6 models: Random Forest, Gradient Boosting, AdaBoost, Logistic Regression, Voting Ensemble, Stacking Ensemble
- Enter user behavioral metrics (total events, tenure, days since first/last activity)
- Real-time churn risk scoring with:
  - Churn score (0-100) with visual gauge
  - Churn window prediction (30/60/90 days or beyond)
  - Risk tier classification (Low/Moderate/High)
  - Retention score
  - Model used display
- Actionable recommendations based on risk level

**Tab 2: EDA & Insights**
- Upload `user_retention.csv` to unlock full analysis
- Six comprehensive analysis sections:
  1. **Dataset Overview** - Row/column counts, unique users/events, memory usage, data types, missing values, top events, activity over time
  2. **Behavioral Metrics** - Browser/OS/country distributions, device & geography breakdowns
  3. **Success Scoring** - Composite success scores, labels, cohort heatmaps
  4. **Survival Curves** - Kaplan-Meier curves by risk segment, engagement level, and deployment status with median survival times
  5. **Cohort & Tenure Analysis** - User count & churn rate by tenure bucket, average lifetime events, days inactive vs events scatter
  6. **User Profile Explorer** - Browse full user-level summary table with all computed metrics

### Running the webapp

```powershell
# Install dependencies
pip install streamlit pandas numpy plotly

# Navigate to Frontend directory
cd Frontend

# Run the Streamlit app
streamlit run streamlit_webapp.py
```

The webapp uses a dark theme with the Zerve color palette and includes:
- Custom CSS styling for a modern, professional UI
- Interactive Plotly charts with hover tooltips
- Responsive layout that works on desktop and tablet
- Cached model loading and EDA pipeline for performance
- File upload handling for both model pickle and CSV datasets

### Configuration

Streamlit settings are in `Frontend/.streamlit/config.toml`:
- Primary color: `#7c6af7` (Zerve purple)
- Background: `#0f0f11` (dark)
- Max upload size: 400 MB
- Logger level: info

---

## Local tooling (`scripts/`)

### `scripts/render_canvas_dag.py`

Parses `canvas.yaml`, groups blocks into the seven stages above, and emits a Mermaid `flowchart LR` into `canvas_dag.md`. Regenerate whenever `canvas.yaml` changes:

```powershell
python scripts/render_canvas_dag.py canvas.yaml canvas_dag.md
```

Only dependency: `pyyaml`.

### `scripts/run_canvas_locally.py`

Replays the Zerve canvas **outside** Zerve. It topo-sorts `canvas.yaml`, executes each `Development/<name>.py` in a single shared Python namespace (mimicking Zerve's DAG scope), and captures figures into `outputs/<block>/`.

Key behaviors:

- **Shared namespace** — each block's top-level variables, functions and DataFrames carry over to downstream blocks, matching Zerve's runtime.
- **Skips non-code blocks** — markdown (`type=4`), GenAI prompt (`type=9`), and `(Copy)` duplicates.
- **Matplotlib capture** — forces `Agg` backend, redirects `plt.show()` into `outputs/<block>/fig_NN.png`.
- **Zerve helper stubs** — `spread()` / `gather()` / `attach_variable()` are stubbed so Fleet / deployment-style blocks do not crash.
- **Checkpointing** — `--checkpoint` pickles the namespace after every block; `--resume` picks up where the last successful block left off.
- **Surgical reruns** — `--from "Block Name"`, `--to "Block Name"`, repeatable `--only` / `--skip`.

Common invocations:

```powershell
# See the plan (what will run / what will be skipped)
python scripts/run_canvas_locally.py --list

# Full replay with figure capture + checkpoints
python scripts/run_canvas_locally.py --checkpoint

# Run just the EDA slice
python scripts/run_canvas_locally.py --from "Example Dataset" --to "Comprehensive Feature Engineering"

# Continue past block failures instead of stopping
python scripts/run_canvas_locally.py --continue-on-error
```

**Caveats.** Zerve blocks are written as cells, not standalone scripts, so some depend on variables that upstream blocks create under specific names. Expect failures on blocks whose upstream edges are missing in `canvas.yaml` (several reporting blocks in this canvas have no inbound edges at all — see `--list` output). Use `--from` / `--to` to run coherent slices rather than the whole DAG in one shot.

Dependencies: `pyyaml` + whatever the blocks themselves import (pandas, numpy, scikit-learn, lifelines, shap, networkx, joblib, matplotlib, seaborn, plotly, imbalanced-learn — see the *Reproducing / running locally* section above).

### `scripts/future_holdout_retrain.py`

Leakage-free retrain of the success ensemble using a **temporal** split — features from the first N days of the event log (default 89), target = activity in the remaining days. Same `{RF, GB, Ada, LR, Voting, Stacking}` ensemble as `Development/02_base_models_ensemble.py`, but with a label the model cannot observe through its features. Writes a structured JSON summary (via `--json-out`) and prints a comparison against the current composite-label baseline.

```powershell
# Default 89-day feature window / 9-day target window on the Sep 1 – Dec 8 dataset
python scripts/future_holdout_retrain.py --json-out outputs/_future_holdout_89_9.json

# Custom horizon
python scripts/future_holdout_retrain.py --feature-horizon-days 70 --json-out outputs/_future_holdout_70_28.json
```

Only dependencies beyond the main stack: none (uses `pandas`, `numpy`, `scikit-learn`).

### `scripts/future_holdout_retrain_rich.py`

Rich-features variant of the above. Keeps the same feature-window / target-window / ensemble shape, but computes 29 extra behavioural columns **strictly from the feature-window events** — 9 session-pattern (total sessions, avg/max session length, event density, deep-work ratio, inter-session gaps), 12 workflow (n-gram diversity, power-user / struggle scores, deployment / agent / create-run flags), 8 momentum (active days, last-7d vs. prior-7d event counts, accel / decel period counts, daily-events stdev). Evaluates every trained model on the test set and selects the best by val ROC-AUC.

```powershell
python scripts/future_holdout_retrain_rich.py --json-out outputs/_future_holdout_89_9_rich.json
```

On the default 89-day / 10-day split: best-by-val-AUC model is `voting`, test ROC-AUC 0.9168 — **−0.025 pp below the 4-feature baseline of 0.9414** (though accuracy and F1 both improve). With only 19 positive users in the test set, the extra features increase variance more than they add ranking signal. Full JSON + per-model test metrics: `outputs/_future_holdout_89_9_rich.json`.

### `scripts/build_gnn_colab_notebook.py`

Emits `notebooks/gnn_colab_verification.ipynb` — a Colab-ready notebook that runs only the **minimum upstream chain** required to verify the two heaviest GNN blocks (**block 65 GraphSAGE Training** and **block 67 Hybrid GNN Churn Model**). The generator parses `canvas.yaml`, walks backwards from those two targets, topo-sorts the 19 ancestor code blocks, and inlines each block body into a notebook cell. All cells share Colab's top-level namespace, matching Zerve's per-canvas variable scope.

```powershell
# (Re)generate the notebook from the current canvas.yaml + Development/*.py
python scripts/build_gnn_colab_notebook.py
```

Runtime requirements on Colab: default CPU runtime is sufficient (~5.4k graph nodes, ~40k edges; 30 GraphSAGE epochs finish in a few minutes — verified session 3). Upload `user_retention.parquet` (~50 MB, gitignored) via the Windsurf Colab extension's right-click → **Upload to Colab**, or mount Drive (Command Palette → "Mount Google Drive"), or use the in-notebook `files.upload()` widget fallback. Results land in `outputs/colab_gnn_results.json` inside the Colab runtime for download. Use this when the local laptop cannot finish the full `scripts/run_canvas_locally.py` replay because of RAM / disk pressure during GraphSAGE training.

**Verified Colab numbers (session 3, post-stabilization).** Block 17 (graph construction): 5,410 nodes / 40,097 edges. Block 18 (GraphSAGE training): the original `lr=5e-3` + plain `ReLU` config showed training collapse on Colab (pair-AUC 0.893 → 0.315 over 30 epochs). The source was patched to `lr=1e-3` + `LeakyReLU(0.1)` and the notebook regenerated; re-verification on Colab is pending. Block 19 (Hybrid GNN, captured pre-fix as a lower bound): hybrid dataset 4,593 × 33 (8 baseline + 19 GNN columns), 30.8% churn rate. Test-set ROC-AUC 0.8824 (hybrid) vs 0.8834 (baseline), F1 0.8005 vs 0.7964 — AUC lift −0.12 pp, F1 lift +0.52 pp on this dataset. See `docs/pipeline_deep_dive.md §7.5` for the full trace and follow-ups.

---

## Reproducing / running locally

This repo is primarily an **archive of a Zerve execution**. To re-run a block locally:

1. **Python environment** — Python 3.10+ recommended.
2. **Install the stack the blocks assume:**
   ```powershell
   pip install pandas numpy scipy scikit-learn imbalanced-learn matplotlib seaborn plotly lifelines shap networkx pyyaml joblib
   ```
   GNN blocks additionally need `torch` + `torch-geometric`.
   The LLM block needs AWS Bedrock credentials (`anthropic.claude-3-haiku-20240307-v1:0`, region `eu-west-1`).
3. **Ensure `user_retention.parquet` is at the repo root** — every downstream block assumes that path.
4. **Run blocks in dependency order** (follow edges in `canvas_dag.md` / `canvas.yaml`). A reasonable manual order starts with:
   ```
   Development/Example Dataset.py
   Development/Data Exploration.py
   Development/02_statistical_profiling.py
   …
   Development/Comprehensive Feature Engineering.py
   Development/01_data_prep_train_val_test_split.py
   Development/02_base_models_ensemble.py
   …
   ```
5. Blocks share state through module-level variables in the Zerve runtime. When running outside Zerve you may need to concatenate or adapt them — treat each `.py` as a cell, not a standalone script.

### Running the Streamlit webapp

For interactive churn prediction and EDA:

```powershell
# Install webapp dependencies
pip install streamlit pandas numpy plotly

# Navigate to Frontend directory
cd Frontend

# Run the Streamlit app
streamlit run streamlit_webapp.py
```

The webapp will open in your browser at `http://localhost:8501`. Upload `ensemble_models.pkl` from the repo root to enable predictions, or upload `user_retention.csv` (converted from `user_retention.parquet`) to unlock the full EDA & Insights tab.

---

## Key findings

From `Development/Project README.md` and `Development/key_findings_summary.md`:

- **5,410 unique users** across **409,287 events** over 98 days.
- **Power law**: top 5% of users (≈273) average 544 events vs. ~5 for bottom 62%.
- **`credits_used` dominates** at 39.1% of all events.
- **Peak activity** in November 2025 (55.7% of events); Thursday is the busiest weekday.
- **99.51% Soft-Voting ensemble test accuracy / 0.9952 ROC-AUC** on the composite success label (retrained Apr 2026 after `docs/pipeline_deep_dive.md §7.1` — the 5 boolean leakage features are no longer inputs; residual leakage from numeric features remains, see §3.4).
- **Churn Early Warning System** (Apr 2026 run): 0 Critical / 0 High / 8 Medium / 3,172 Low-risk among 3,180 active users. Top drivers from SHAP on the EWS RandomForest: recency score, 7d-vs-expected ratio, session frequency, tenure, historical daily rate.
- **Future-holdout sanity check** (`scripts/future_holdout_retrain.py`, post-§3.4 follow-up): same ensemble retrained on features from days 1–89 with label = activity in days 90–99 on the same 4 numeric features. AdaBoost wins val F1 0.935; test accuracy 0.9044, **ROC-AUC 0.9414** on 3,905 users (3.3% positive). Δ vs. composite-label baseline: −0.091 accuracy, −0.054 ROC-AUC — the expected "leakage-removal penalty". Residual 0.94 AUC is genuine past→future signal. Full JSON: `outputs/_future_holdout_89_9.json`.
- **Rich-features future-holdout** (`scripts/future_holdout_retrain_rich.py`, §7 item 15): same window / label / ensemble as above, but 33 features (4 base + 9 session-pattern + 12 workflow-n-gram + 8 momentum) computed strictly on the feature-window slice. Best-by-val-AUC is `voting`, test **ROC-AUC 0.9168** — _−0.025 pp below_ the 4-feature baseline. Accuracy 0.9317 and F1 0.9448 both improve, but no model configuration crossed the 0.95 AUC target the task set for itself. Interpretation: with only 19 positive users in the test split, 29 extra correlated features raise variance more than they add ranking signal; `total_sessions` (imp 0.27), `active_days` (0.14) and `days_since_last` (0.11) absorb ~52 % of importance between them. Full JSON + per-model test metrics: `outputs/_future_holdout_89_9_rich.json`.
- **31–60 day tenure** is the peak churn-vulnerability window (43% of observed churn).
- **271 anomalous users** surfaced by Isolation Forest.
- **12,641 sessions** identified (30-min gap threshold); 28% qualify as "deep work".

---

## Notes & caveats

Mechanical / reproducibility caveats:

- **Block code is not self-contained** — scripts in `Development/` expect the shared Zerve execution context (variables created by upstream blocks). Running a single `.py` in isolation will usually fail with `NameError`. Read `canvas.yaml` / `canvas_dag.md` to see upstream dependencies. Use `python scripts/run_canvas_locally.py` to topo-sort + replay the canvas in a shared namespace (`docs/repo_state_and_next_steps.md §3` lists the six small fixes needed for a fully clean local replay).
- **Large binaries are tracked, not ignored.** `user_retention.parquet` (~50 MB), `ensemble_models.pkl` (~6 MB), `user_segments.csv`, `user_intelligence_export.csv` are all committed. The `.gitignore` does ignore `outputs/_state/` (≈16 GB of pickled checkpoint namespaces produced by `run_canvas_locally.py --checkpoint`) and standard Python / IDE / OS noise.
- **Apr 2026 cleanup of repo root.** Earlier exports left 17 timestamped `user_behavior_analytics_report_*.md` and 5 `social_media_posts_*.md` files at the root, plus a leaked-f-string `user_behavior_analytics_report_{report_date.replace('-', '')}.md`. These were byte-identical duplicates from re-runs of hand-coded reporting blocks. Only the latest of each family is retained; the underlying source-block bug (template not interpolated) is tracked in `docs/repo_state_and_next_steps.md §7`.
- **LLM block requires AWS Bedrock access** — without credentials, `Weekly Insights Executive Briefing` / `weekly_insights_report.md` cannot be regenerated.

Data / model interpretation caveats (full analysis in `docs/pipeline_deep_dive.md`):

- **Three of five "primary success metrics" are structurally zero on this data.** `Primary Success Metrics.py` checks for event names like `api_deployed`, `credits_purchased`, `canvas_shared`, `comment_added` — none of which occur in `user_retention.parquet`. So `is_paid_user`, `has_deployment`, and `collaboration_success` are **False for every user**. Any chart or segmentation keyed on these is keying on constants. `Validation & Business Alignment.py` acknowledges this explicitly.
- **Residual label leakage even after Apr 2026 fix.** The original 99.88% success-ensemble accuracy came from feeding the same booleans (`long_term_retention`, `is_power_user`, `is_paid_user`, `has_deployment`, `collaboration_success`) into both the features and the target-generation function. Those 5 booleans were removed from the feature list in `01_data_prep_train_val_test_split.py` (Apr 2026) and the ensemble was retrained to 99.51% / 0.9952 ROC-AUC. The remaining 4 numeric features (`total_events`, `tenure_days`, `days_since_first`, `days_since_last`) are **still** direct inputs to the step-function thresholds in `Validation & Business Alignment.py:118-160` — the fact that accuracy only dropped ~0.4 pp is itself evidence the model is still mostly rediscovering the label-generation rule rather than generalizing. The Early Warning System ensemble (`Churn Early Warning System.py`) does not have this issue and is the credible churn signal.
- **[FIXED Apr 2026] Two different scores previously both called `churn_risk_score`.** `active_users_survival['churn_risk_score']` (the success-ensemble probability rebranded as churn risk) has been renamed `active_users_survival['success_score']`; its companion `predicted_risk_category` is now `success_tier` with values `Low` / `Medium` / `High`. The rename propagated into `Advanced Analysis Synthesis`, `Cohort Behavioral DNA`, `Integrated Dashboard`, `Behavioral Economics Scoring`, `Weekly Delta Metrics Computation`, `LTV Prediction & Unit Economics`, and `Hybrid GNN Churn Model`. The plural `churn_risk_scores` DataFrame from the Early Warning System still carries its own `churn_risk_score` column — untouched, because that one is a genuine churn score. **Residual issue:** most downstream call sites still treat `success_score` as if it were a churn probability (see `docs/pipeline_deep_dive.md §3.2`); the rename exposes the inversion but the math is unchanged.
- **[FIXED Apr 2026 — §7.4] The collaboration network is now behaviourally derived.** `Collaboration Network & Centrality Analysis.py` was rewritten to use `StandardScaler` + `cosine_similarity` kNN (`k=8`, `threshold=0.70`) on 10 behavioural features drawn from `behavioral_fingerprint` (sessions, deep-work ratio, power/struggle scores, sequence diversity, collaboration/sharing signals). The previous `np.random.choice` edge construction was removed. Empirical network on the 500-user high-activity sample: ~2,600 edges, density ~0.021. "Super connectors", modularity and t-tests against success scores are now statistics over a behaviourally-clustered graph.
- **[FIXED Apr 2026 — §7.5] GraphSAGE is now trained.** `GraphSAGE Training & Social Influence Embeddings.py` replaced the untrained numpy forward pass with a PyTorch `GraphSAGE2` module (d_in→32→16) + `torch.optim.Adam(lr=5e-3, weight_decay=1e-5)` + 30 epochs of unsupervised link-prediction (positive edges from `gnn_graph['edge_weight_map']`, uniform negative sampling at 1:1, loss `-logσ(pos) - logσ(-neg)`). Downstream `Hybrid GNN Churn Model & Community Analysis` consumes the learned embeddings unchanged.
- **Hand-rolled stand-ins for library algorithms.** The `shap` package is not used — `SHAP Explainability Analysis.py` implements Saabas tree-path decomposition in numpy. `lifelines` is not used — `Kaplan-Meier Survival Curves by Segment.py` implements KM in ~30 lines. `Community Detection & Success Correlation.py` implements greedy modularity by hand. Numbers are indicative; don't conflate with library-grade equivalents in external reports.
- **[FIXED Apr 2026] SHAP now targets the Early Warning System RandomForest** instead of the success-model RF. `SHAP Explainability Analysis.py` consumes `churn_ews_rf_model` + `churn_ews_feature_matrix` (exported publicly from `Churn Early Warning System.py`), so the "top churn drivers" in `Churn Early Warning — Ranked Action Table.py` now describe the actual churn ensemble's decisions, not the success ensemble's.
- **[FIXED Apr 2026] Wall-clock bug in recency features.** `Comprehensive Feature Engineering.py:102` and `Temporal & Monetization Segmentation.py:15` now use `user_retention['timestamp'].max()` as the reference date, so re-running the pipeline after Dec 2025 no longer pushes every user into the `'churned'` lifecycle bucket.
- **Reporting blocks have hand-authored numbers.** `Export Report as Markdown File.py` and `Social Media Post Drafts.py` write multi-hundred-line markdown files with statistics (e.g. "642 paid users", "workflow funnel drop-off 40%", "12% paid conversion") baked in as string literals. They do **not** re-compute from the pipeline's upstream variables and will not update if the data changes. The `weekly_insights_report.md` LLM briefing *is* data-driven (via `weekly_insights_metrics` JSON).
- **Primary-stage counts in the README key findings do reflect the data** (409K events, 5,410 users, 141 event types, 12,641 sessions, 271 isolation-forest anomalies). The 99.51% carries the residual-leakage caveat above; the "14 EWS critical alerts" figure is from an earlier run and no longer holds — the current EWS output is 0 Critical / 0 High / 8 Medium / 3,172 Low among active users.

---

## Further reading

- **`docs/pipeline_deep_dive.md`** — **start here** for a stage-by-stage reading of every block, cross-cutting findings, model-validity assessment, and the full list of data-integrity issues with file:line references. The material above is a digest of this document.
- **`docs/zerve_platform_report.md`** — everything about the Zerve platform itself (canvas/notebook duality, DAG execution semantics, S3-cache state machine, block types, Lambda/Fargate/GPU compute, Fleets, layer types, Git integration, AWS self-hosting). Read this if you need to understand *why* the codebase uses patterns like shared-namespace variables and defensive `del` prologues.
- **`docs/repo_state_and_next_steps.md`** — reviewer-grade audit of what currently exists, what runs end-to-end, what does not, and the smallest set of changes needed for clean local execution.
- **`docs/viva_presentation_guide.md`** — comprehensive viva/presentation guide with project overview, technical architecture, key models, findings, limitations, open issues, future work, presentation structure, and common Q&A.
- **`report.md`** — complete user behavior analytics & churn prediction report with detailed pipeline documentation, model validity assessment, and business insights.
- **`Final-Report.pdf`** — final project report document (PDF version).
- **`canvas_dag.md`** — Mermaid rendering of the 67-block / 78-edge DAG grouped by stage.
- **`Development/Project README.md`** and **`Development/Quality Assurance Checklist.md`** — the Zerve-side authored overview and QA notes (pre-date the issues documented above).

---

## License / attribution

Project exported from Zerve; retain the organization and project IDs recorded at the top of `canvas.yaml` for re-import.