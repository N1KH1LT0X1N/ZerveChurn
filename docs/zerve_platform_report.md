# Zerve.ai — Platform Deep-Dive & Pipeline Execution Report

> Research compiled from the official Zerve User Documentation (`docs.zerve.ai`) and the AWS Marketplace listing, cross-referenced against what we actually see in this repo's `canvas.yaml` export. Everything below is directly applicable to how **this** ZerveChurn project was built and ran.

---

## 1. What Zerve is (one paragraph)

Zerve markets itself as an **"Agentic Development Environment purpose-built for data science"**. It combines three things in one workspace:

1. A **visual canvas** (DAG of code blocks) *and* an equivalent **notebook view** of the same project.
2. A **serverless execution engine** that decouples compute from storage — every block runs in its own cloud container and its outputs are cached in S3, so downstream blocks can pick them up without re-running upstream work.
3. A **context-aware AI agent** that can read the canvas, your variables, and your pipeline, then author/edit/fix blocks on your behalf.

It is usable as **managed SaaS** (zerve.cloud) or **self-hosted inside your own AWS/GCP/Azure account** (VPC + S3 + ECS/Lambda + SageMaker integrations).

---

## 2. The two interchangeable views

The *same* project is accessible in two UIs, backed by the same data model:

| View          | Feel                                            | Best for                                  |
| ------------- | ----------------------------------------------- | ----------------------------------------- |
| **Canvas**    | Spatial DAG, drag-and-connect blocks            | Pipelines, DAG orchestration, deployments |
| **Notebook**  | Linear cells like Jupyter                       | Exploratory work, quick iteration         |

You can switch between them at will — both read/write `canvas.yaml` + `Development/` the same way.

---

## 3. Project anatomy (what gets stored in Git / on export)

When a canvas is committed via the built-in Git integration, Zerve writes a deterministic directory layout — **exactly what this repo contains**:

```
<repo>/
├── canvas.yaml                # canvas-level metadata: id, name, org, project_id,
│                              #   global_imports, requirements, linux_packages,
│                              #   environment_variables, layers[], layers[].blocks[],
│                              #   layers[].edges[]
└── Development/               # one folder per layer
    ├── layer.yaml             # layer-level copy of the subset of canvas.yaml
    ├── <Block Name>.py        # Python code block  (type 1, language=Python)
    ├── <Block Name>.R         # R block            (type 1, language=R)
    ├── <Block Name>.sql       # Query block        (type 2-ish, SQL / GraphQL)
    ├── <Block Name>.md        # Markdown / notes   (type 4)
    ├── <Block Name>.Rmd       # R-Markdown
    └── <Block Name>.text      # GenAI prompt block (type 9) — system/user prompt
```

Key YAML shapes we've verified in our `canvas.yaml`:

- **Block common fields:** `id`, `name`, `description`, `type` (numeric enum), `status`, `x/y/width/height` (canvas coordinates), `parent_id`, `properties{}`, `variables`, `compute_settings`, `canvas_id`, `layer_id`.
- **Block `type` enum** (inferred from our data):
  - `1` → code / compute block (Python, R, SQL)
  - `4` → markdown / note block
  - `9` → GenAI (LLM) prompt block; its `properties` object carries `provider`, `model`, `region`, `temperature`, `max_tokens`, `system_prompt`, `output_variable`, etc.
- **Edges:** `{id, source, target, layer_id, canvas_id}` — directed source→target references by block id.
- **Layer types:** `Development` (default IDE), `Deployment` (API), `Scheduled Jobs`, `SageMaker Deployments`.

> Our project only has a `Development` layer. That's why we don't see `deployment.yaml` / scheduled-job artifacts here.

---

## 4. Block types (the full menu)

From the Zerve docs' block-type catalog:

| Block            | Language/role                                | Notes                                                                |
| ---------------- | -------------------------------------------- | -------------------------------------------------------------------- |
| **Python**       | Full Python 3.11 with per-canvas venv         | Jupyter-like output per block; variables passed downstream natively  |
| **R**            | R + Rscript                                   | Full interop with Python via serialization                           |
| **Query**        | SQL over any pandas DF or native DB / GraphQL | No driver install needed — Zerve manages connectors                  |
| **Markdown**     | Documentation                                 | Pure notes, no execution                                             |
| **R Markdown**   | R + markdown                                  | Dynamic reports                                                      |
| **GenAI**        | LLM prompt (Bedrock + OpenAI)                 | Variables from upstream blocks can be substituted into prompts       |
| **UI Input/Output** | Streamlit-ish widgets                       | Text/number/dropdown/slider/file-upload → build mini apps on canvas  |
| **Aggregator**   | Fan-in from parallel Fleets                   | Uses `gather()` to collect results                                   |
| **Logic Gate**   | Branching / conditional                       | DAG control flow                                                     |
| **API Controller + API Route** | Deployment-layer HTTP endpoint  | Only valid inside a Deployment layer                                 |

### GenAI — supported models (Bedrock + OpenAI)

Amazon Titan (Text + Image), SDXL 0.8 / 1.0 (Image), AI21 Jurassic-2, Anthropic Claude Instant / v2 / v2:1 / 3 Sonnet / 3 Haiku, Cohere Command / Command Light, Meta Llama 2 & 3 (8B/13B/70B), Mistral 7B / Mixtral 8×7B / Mistral Large, plus **all OpenAI models via API**.

> Our `Weekly Insights Executive Briefing` block uses `anthropic.claude-3-haiku-20240307-v1:0` on Bedrock `eu-west-1` — this is the exact `properties` section on block `3aaed87c-…` in `canvas.yaml`.

### GenAI default system prompt

Zerve injects a stock system prompt into every Bedrock block unless you override it. It's context-aware: *"Your prompts originate from a code block in a directed acyclic graph of connected code blocks. You have access to the code used to generate variables you have access to the content and a description of each variable…"* That's why the LLM can reason about upstream variables *and* block code without you pasting them in.

---

## 5. How execution actually works

### 5.1 DAG semantics
- The canvas is a **Directed Acyclic Graph**. Values flow **left → right** along edges.
- Each block's code body sees the variables/functions/dataframes produced by its upstream blocks **as if they were in the same Python scope**.

### 5.2 Execution modes (canvas and block level)

| Scope  | Mode                | Behavior                                                                  |
| ------ | ------------------- | ------------------------------------------------------------------------- |
| Canvas | **Run All**         | Starts from the first unfinished block and runs left→right                |
| Canvas | **Force Run All**   | Reruns *every* block regardless of state; replaces cached data             |
| Block  | **Run This Block**  | Only active if block is unrun or code changed                              |
| Block  | **Run Up To Here**  | Runs all upstream blocks + this one                                        |
| Block  | **Force Run This**  | Re-executes ignoring state                                                  |
| Block  | **Force Run Up To** | Re-executes all upstream + this block, ignoring cached state                |

### 5.3 State machine — how cached data behaves

Zerve keeps a per-block "last-good" cache. Four scenarios:

1. **Code changed, run, errors out** → old data stays; downstream sees the last successful output.
2. **Code changed but not run** → old data stays; block is marked "stale".
3. **Code changed, currently running** → old data stays until the run completes.
4. **Code changed, run, succeeds** → cache is replaced by new outputs.

This is the key differentiator from Jupyter: a failed/partial re-run *never* corrupts the downstream pipeline.

### 5.4 Storage/compute separation

From the docs (verbatim intent):

> *"Once the code execution is done the data — all the variables, functions, dataframes, charts etc. — are cached, serialized and stored on disk (S3 in this case). This is then passed on to the next block as input for further code execution. This stored data is separated from the compute and won't be affected by code execution in downstream blocks."*

Concretely on self-hosted AWS:
- **State bucket:** `canvas-state-bucket-{org-uuid}` (S3, server-side encrypted).
- Each block's outputs (variables, DataFrames, matplotlib figures, plotly JSON, pickled models, DataFrame previews, stdout, stderr) are serialized there and keyed by block ID + run.
- A downstream block spins up a fresh compute container, pulls only the upstream cache entries it imports, executes, serializes its own outputs, and releases the compute. **Pure serverless.**

### 5.5 Compute types per block

Each block has a `compute_settings` object (we see `compute_environment_type: 1` on our `Example Dataset` block). The menu:

| Compute     | Limits                                                  | Best for                                         |
| ----------- | ------------------------------------------------------- | ------------------------------------------------ |
| **Lambda**  | ≤ 15 min execution, 10 GB memory                         | Light transforms, API responses, quick blocks     |
| **Fargate** | Longer-running, CPUs up to 120 GB memory                 | Heavy feature engineering, training on CPU        |
| **GPU**     | Selectable GPU tiers                                     | DL / LLM training, fine-tuning, inference         |

Compute type is *per block*, not per project, so cheap blocks stay on Lambda and only the modeling/GNN blocks burn Fargate/GPU.

### 5.6 Fleets — parallel execution

Zerve's built-in parallel primitive is `spread()` / `gather()`:

```python
# Upstream "spread" block
hyper_grid = [{'lr': lr, 'depth': d} for lr in [...] for d in [...]]
spread(hyper_grid)   # fans out 72+ concurrent compute blocks

# Downstream Aggregator block
results = gather()   # collects all 72 results into a list
```

Typical use cases listed in docs: hyperparameter sweeps (72 concurrent runs), category-level DataFrame processing (one fleet per airline / store / tenant), LLM evaluation (100 concurrent prompt runs).

---

## 6. Layers — the four flavors

The canvas is organized into **layers** (we only have one, `Development`):

1. **Development** — the IDE layer. Python / R / SQL / GraphQL / Markdown / RMarkdown / GenAI. Supports block **grouping** via `groupName.blockName` naming convention.
2. **Deployment (API)** — turns a Dev block into an HTTP endpoint via `API Controller` + `API Route` blocks. Config knobs: DNS, API key, Lambda vs Fargate, CPU/memory, payload/response validation, cURL/Python client codegen. Endpoint lives at `https://{domain}.zerve.cloud/{route}`.
3. **Scheduled Jobs** — cron-based recurrence (hourly/daily/weekly/custom cron). Once activated, the layer becomes **read-only** (edits require deactivation). Has an email-notification hook and a per-run status tracker with logs.
4. **SageMaker Deployments** — push a trained model as a SageMaker endpoint inside your AWS account (self-hosted path).

Separate from layers but deployed from the same org:
- **Hosted Apps** — upload a `.zip` of a Streamlit / R Shiny app and get a managed URL with build logs and custom DNS.

---

## 7. Everything about outputs (images, tables, logs, exports)

### 7.1 Per-block outputs rendered in the UI

When a Python/R block runs, Zerve captures and renders:

- **stdout / stderr** — shown under the block cell.
- **Cell return value** (Jupyter-style last expression).
- **Matplotlib figures** — rendered inline as PNGs (and cached in S3).
- **Plotly / Altair / Bokeh** — rendered as interactive HTML widgets.
- **Pandas DataFrames** — rendered as a paginated, sortable table preview with schema.
- **Markdown / R-Markdown** — rendered as formatted HTML.
- **Variables panel** — every top-level variable the block creates is listed with type, size, and a preview (this is what the AI agent reads when answering "what's in `df`?").
- **GenAI text output** — rendered as markdown, exposed as the `output_variable` (default: `output`).
- **GenAI image output** — rendered as inline image (Titan / SDXL). Saved to S3 and passable downstream.

### 7.2 What persists to S3 (the "state bucket")

- All of the above outputs, serialized (typically pickle for Python values, PNG/SVG for figures, JSON for plotly).
- Block metadata (run duration, timestamp, exit code).
- Canvas-level files (anything saved via the **Files** panel).
- For Deployment layer: compiled API bundles + Docker images + requirements.

### 7.3 What persists to the Git repo (what we see on disk)

The Git integration **only** commits source code + YAML, **not** execution outputs. That's why our repo has the `.py` / `.md` / `.text` block files but no per-block figure PNGs. The binary artifacts in our repo (`*.parquet`, `*.csv`, `*.pkl`, the timestamped reports) are there because the blocks themselves explicitly wrote them via Zerve's **Files** mechanism:

- **Files panel** — canvas-scoped file system (under the hood: an S3 prefix). Blocks read/write via normal `pd.read_parquet("user_retention.parquet")` / `open("report.md", "w")`.
- These files *do not* round-trip through the Git integration automatically — they are present in this repo only because someone explicitly downloaded them or the repo was zipped locally after execution.

### 7.4 Export / download paths

There are **three distinct export flavors**:

1. **Git commit of the canvas** — produces exactly the structure in §3. This is how `canvas.yaml` + `Development/` ended up in our repo.
2. **Download Deployment** (Deployment layer only) — produces a 4-part bundle:
   - `app/` — access files + API code.
   - `deployment.yaml` — env/config for Docker.
   - `Dockerfile` — build/run commands.
   - `requirements.txt` — pip deps.
   This is designed so you can `docker build && docker run` the deployed API standalone, outside Zerve.
3. **Files download** — any file in the canvas Files panel can be pulled to local disk. This is how `user_retention.parquet`, `user_segments.csv`, the timestamped reports, etc. end up on your machine.

### 7.5 Hosted endpoints produced by a full Zerve project

- **REST API** per deployed route (Lambda or Fargate-backed): `https://{org}.zerve.cloud/{route}`.
- **SageMaker endpoint** (if you use the SageMaker layer).
- **Hosted App URL** (Streamlit / R Shiny under `{app}.zerve.cloud`).
- **Scheduled-Job run history** (no URL, but tracked in the layer's status panel).

---

## 8. Assets, Requirements, Globals — project-wide scaffolding

### 8.1 Assets (reusable primitives, scoped to the canvas)

- **Functions & Classes** — importable across blocks without copy-paste.
- **Queries** — named SQL / GraphQL snippets.
- **Constants & Secrets** — env-var-like key-value store, encrypted for secrets; keeps API keys out of code.
- **Connections** — named DB credentials (Postgres, MySQL, MariaDB, Snowflake, Weaviate). A Query block references a Connection by name rather than embedding creds.

### 8.2 Requirements (per-canvas virtual environment)

- Each canvas ≈ its own venv.
- **Python 3.11.1** default, preloaded with common DS packages.
- Adds:
  - Python packages (with pinnable versions) → written to `canvas.yaml.requirements`.
  - Linux packages (apt) → `canvas.yaml.linux_packages`.
  - Environment variables → `canvas.yaml.environment_variables`.
- The **"Build Requirements"** action compiles a custom Docker executor for the canvas. Without a custom build, the canvas uses the org's default executor.

### 8.3 Global Imports

- Anything in **Global Imports** is auto-imported into every block (Python + R).
- Written to `canvas.yaml.python_global_imports` / `r_global_imports`.
- Our canvas has both empty — each block does its own imports.

---

## 9. Collaboration, versioning, AI agent

- **Real-time multi-user editing** (Google-Docs-style) on the canvas.
- **Comments** at block level (resolvable threads).
- **Sharing** — per-canvas / per-folder ACLs inside an Organization.
- **Source Control** — GitHub (Cloud + GHES), Bitbucket (OAuth + App), Azure DevOps. Branches = separate canvas clones with diff/PR flow; merging goes through the normal Git UI.
- **Include Repositories** — any Git repo can be mounted into a canvas; `import your_lib` works in block code.
- **AI Agent** — three routed actions: **Chat** (Q&A over canvas/data/variables), **Search** (live web), **Code** (mutate blocks, create new ones, debug, refactor). Voice input via mic and file uploads are supported. The agent uses the full canvas context (code + variable previews + descriptions) automatically.

---

## 10. Self-hosting architecture (AWS path, applicable to our project)

- One CloudFormation stack, ~10–15 min deploy, roughly opinionated:
  - **VPC** with 2 public subnets (load balancers, NAT) + 2 private subnets (compute).
  - **NAT + Internet Gateway**, 2 security groups (public / private).
  - **IAM role** scoped to resources prefixed `zerve-*`.
  - **S3 state bucket** `canvas-state-bucket-{orgUUID}`.
  - Control plane stays in `eu-west-1`; data plane deploys in your chosen region.
- Services potentially billed while using Zerve: EC2, ECS, ELB, VPC, ECR, SageMaker, RDS, Lambda, CodeBuild, S3, Data Transfer, Route 53, CloudWatch, Bedrock, API Gateway, KMS, CloudFormation.
- Zerve doesn't charge per-usage on top — you pay AWS directly for the compute you consume, plus your Zerve subscription.

---

## 11. How the ZerveChurn project actually ran (end-to-end)

Putting all of the above together for our specific repo:

1. **Canvas created** in the Zerve UI (or cloned from *ZerveXHackerEarth*), auto-assigned `canvas_id 6399a86d-…`, organization `d998b7aa-…`.
2. **Single `Development` layer** set up (`layer_id 9ed97209-…`).
3. **67 blocks** added (66 Python compute + 1 LLM Bedrock block), **78 edges** wiring them into a DAG (see `canvas_dag.md`).
4. `user_retention.parquet` uploaded to the canvas **Files** panel.
5. Blocks executed **left→right** following the DAG. Each block's compute ran serverless (most on default Lambda-style executor, with `Example Dataset` overridden to `compute_environment_type: 1`). Outputs (DataFrames, matplotlib/plotly charts, models, CSVs, markdown) were cached in the state S3 bucket.
6. Reporting blocks wrote artifacts back into the Files panel:
   - `user_segments.csv`, `user_intelligence_export.csv`, `ensemble_models.pkl`
   - `user_behavior_analytics_report_YYYYMMDD_HHMMSS.md`, `social_media_posts_YYYYMMDD_HHMMSS.md`
7. The **GenAI block** (`Weekly Insights Executive Briefing`, Claude 3 Haiku, eu-west-1) consumed the upstream `Weekly Delta Metrics Computation` variables via Zerve's automatic variable-injection system and produced `weekly_insights_report.md` (written to Files by the follow-on `Save Report to File` block).
8. The canvas was **committed via the Git integration**, writing `canvas.yaml` + `Development/` into this repo. The Files (parquet/csv/pkl/md reports) were downloaded separately and dropped next to the canvas locally — that's why they co-exist with the source files.

There is **no `Deployment` layer and no `Scheduled Jobs` layer** in this project. So no REST API, no cron — the pipeline exists purely as an interactive analytical canvas plus its cached outputs.

---

## 12. Plan — things we can do from here

Now that we understand the platform, the meaningful next moves on this local repo fall into four buckets:

### A. Re-importability back into Zerve
- Keep `canvas.yaml` + `Development/` intact and committed — Zerve can re-import this layout and reconstruct the canvas verbatim (that's the whole point of the Git integration).
- Document the **`canvas_id`, `layer_id`, `project_id`, `organization_id`** at the top of README so a future import knows which org/project to attach to. *(Already done in our README.)*

### B. Lightweight local replication of the pipeline
Because Zerve injects upstream variables into each block's scope automatically, running `Development/*.py` one-by-one outside Zerve will fail with `NameError`. To reproduce locally we'd need a thin **orchestrator** that:
1. Topo-sorts `canvas.yaml.edges`.
2. Executes each block's `.py` as a cell in a shared namespace (e.g. with `runpy.run_path(..., init_globals=ns)`).
3. Persists intermediate state to pickle between blocks (emulating Zerve's S3 cache).

This is ~100 lines of Python; if useful I can build it as `scripts/run_canvas_locally.py`.

### C. Output-archival script
Pull *every* block's most recent execution artifact (images, tables, logs) out of the Zerve state bucket into `outputs/` so the repo becomes a fully-inspectable archive of the run. Requires Zerve Developer API key + S3 access.

### D. Re-deployment options
- **As an API** → add a `Deployment` layer in Zerve, wrap the trained ensemble in an `API Route`, download the 4-part bundle, `docker build`, push to your own cluster.
- **As a scheduled weekly run** → add a `Scheduled Jobs` layer driven by the current DAG head; the Claude-Haiku `Weekly Insights Executive Briefing` block is already shaped for this.
- **As a Streamlit dashboard** → package the reporting blocks into a `Hosted App`.

### E. Houskeeping in this repo (already flagged in README)
- Remove `Development/Engagement Forecast per Segment (Copy).py` + its canvas node.
- Fix the broken filename `user_behavior_analytics_report_{report_date.replace('-', '')}.md`.
- Decide whether parquet/csv/pkl should be force-added to Git (currently ignored).

---

## 13. Primary sources

All URLs are under `https://docs.zerve.ai/guide/`:

- `welcome-to-zerve.md`
- `canvas-view/how-zerve-works.md`
- `canvas-view/layers-overview.md` + `…/development.md` + `…/deployment.md` + `…/deployment/create-deployment.md` + `…/deployment/download-deployment.md` + `…/scheduled-jobs.md`
- `canvas-view/blocks-and-connections/block-types.md` (+ `python.md`, `query.md`, `gen-ai.md`, `ui-blocks-input-and-output.md`)
- `canvas-view/blocks-and-connections/compute-settings/lambda-vs-fargate-vs-gpu.md`
- `canvas-view/assets.md`, `files.md`, `fleets.md`, `ai-agent.md`
- `canvas-view/installing-packages/requirements.md`
- `source-control-git/canvas-source-control-features.md`
- `hosted-apps.md`
- `integrations/cloud/aws-self-hosting.md`
- AWS Marketplace listing: *Zerve OS and AI Agent*, `aws.amazon.com/marketplace/pp/prodview-ktlnaltd6qqhc`
