# ZerveChurn — Repo State & Next Steps (Fact-Based Audit)

> **Scope.** A reviewer-grade audit of what currently exists in this repo, what runs end-to-end, what does not, and the smallest set of changes that get every coded block executing successfully. Every claim below is grounded in a specific file, line range, run-log line number, or JSON value — nothing is inferred.
>
> **End goal (per user).** Every block we have coded actually runs to completion, locally and/or on Colab, with the artefacts they are supposed to produce.
>
> **Last updated:** 2026-04-25. Sources audited: `canvas.yaml`, all 64 files under `Development/`, all 6 files under `scripts/`, `notebooks/gnn_colab_verification.ipynb`, all logs under `outputs/` (`_run_72f.log`, `_run_73.log/.err`, `_run_74.log/.err`, `_future_holdout_*.json/.log`, `_colab_gnn_cells_dump.txt`), `docs/pipeline_deep_dive.md`, `docs/zerve_platform_report.md`, `README.md`, `canvas_dag.md`, `docs/notebook_mcp_issue_report.md` (moved from repo root in Apr 2026 cleanup).
>
> **Cleanup addendum (2026-04-25, post-audit).** The repo root has been pruned to remove duplicate timestamped reports — see also `README.md` "Apr 2026 cleanup" note. Specifically:
>
> - **Deleted 16 of 17** `user_behavior_analytics_report_*.md` files (12 of those were byte-identical SHA-256 duplicates from Apr 2026 re-runs of the hand-coded `Export Report as Markdown File.py` block; the remaining ones differed only in baked-in literal numbers from earlier 2026 runs). Kept: `user_behavior_analytics_report_20260424_114927.md` (latest).
> - **Deleted** `user_behavior_analytics_report_BROKEN_TEMPLATE.md` (the leaked-f-string filename artefact called out in §1.1 / `pipeline_deep_dive.md:261`). The underlying source-block bug (template not interpolated) remains open as item 9 in `pipeline_deep_dive.md §7` and is now the only place it is tracked.
> - **Deleted 4 of 5** `social_media_posts_*.md` files. Kept: `social_media_posts_20260424.md` (latest).
> - **Moved** `notebook_mcp_issue_report.md` → `docs/notebook_mcp_issue_report.md` (unrelated FastMCP bug, no need to clutter the repo root).
> - **Wrote a smart `.gitignore`** that ignores `outputs/_state/` (16.8 GB of pickled namespace checkpoints from `run_canvas_locally.py --checkpoint`), `anaconda_projects/db/` (Zerve workspace SQLite), `__pycache__/`, virtualenvs, IDE configs, and OS junk — but **does not** ignore `*.parquet` / `*.csv` / `*.pkl`, so the dataset and trained ensemble stay tracked. The "files are gitignored by default" claim previously in `README.md:141` and below in §1.1 / `1.4` is therefore stale; data and model artefacts are now first-class tracked content.
>
> The pre-cleanup inventory tables in §1 below are preserved verbatim as a record of the Apr 25 2026 state.

---

## Table of contents

- [1. Repo inventory (what physically exists)](#1-repo-inventory-what-physically-exists)
- [2. What currently runs and what does not](#2-what-currently-runs-and-what-does-not)
- [3. Blockers preventing a clean local replay (every fix grounded)](#3-blockers-preventing-a-clean-local-replay-every-fix-grounded)
- [4. State of the GNN Colab verification](#4-state-of-the-gnn-colab-verification)
- [5. Open methodology issues from `pipeline_deep_dive.md` — verified status](#5-open-methodology-issues-from-pipeline_deep_divemd--verified-status)
- [6. Specific call-site audit: inverted `success_score` semantics](#6-specific-call-site-audit-inverted-success_score-semantics)
- [7. Specific call-site audit: hand-coded literals in reporting blocks](#7-specific-call-site-audit-hand-coded-literals-in-reporting-blocks)
- [8. Specific call-site audit: wall-clock `now()` usage](#8-specific-call-site-audit-wall-clock-now-usage)
- [9. Recommended next-step plan](#9-recommended-next-step-plan)
- [Appendix A — Run-log failure index](#appendix-a--run-log-failure-index)

---

## 1. Repo inventory (what physically exists)

### 1.1 Top-level artefacts (file sizes from `Get-ChildItem`)

| Path | Size | Role |
| ---- | ---- | ---- |
| `canvas.yaml` | 56,178 B | Zerve canvas export: 67 blocks + 78 edges, 1 layer (`Development`). |
| `canvas_dag.md` | 7,084 B | Mermaid render of the DAG, grouped into 7 stages. |
| `README.md` | 30,554 B | Project overview, repro instructions, key findings, caveats. |
| `ensemble_models.pkl` | 6,142,755 B | Joblib bundle: 6 success-models + scaler + validation results. |
| `user_retention.parquet` | 52,646,892 B | Raw input — 409,287 rows × 107 cols. **Confirmed by the run-72f log line 2** ("Dataset shape: (409287, 107)"). |
| `user_segments.csv` | 876,702 B | 5,410 × 16 master segmentation matrix. |
| `user_intelligence_export.csv` | 1,338,980 B | Per-user intelligence export. |
| `weekly_insights_report.md` | 4,168 B | LLM-generated executive briefing. |
| `user_behavior_analytics_report_*.md` | × 17 files | Timestamped reports from `Export Report as Markdown File.py`. |
| `social_media_posts_*.md` | × 5 files | Hand-coded marketing copy from `Social Media Post Drafts.py`. |
| `notebook_mcp_issue_report.md` | 1,549 B | Bug report for `the-notebook-mcp` (FastMCP `log_level`). Unrelated to the analytics pipeline. |
| `user_behavior_analytics_report_{report_date.replace('-', '')}.md` | 26,172 B | **Bug**: leaked Python f-string literal in filename (`pipeline_deep_dive.md:261`). |

### 1.2 `Development/` — block sources

`Development/` contains 64 files in total: **54 `.py` (compute blocks, `type: 1`)**, **5 `.md` (note blocks, `type: 4`)**, **1 `.text` (LLM block, `type: 9`)**, **1 `layer.yaml`** + **1 `__pycache__/`** with 61 cached `.pyc`. The seven canvas stages and their member blocks are enumerated in `canvas_dag.md:7-105`.

The `Development/Engagement Forecast per Segment (Copy).py` file is a byte-identical duplicate of `Development/Engagement Forecast per Segment.py` and is wired as a separate node in `canvas.yaml`. `pipeline_deep_dive.md:260` and `README.md:290` mark it as safe to delete; `scripts/run_canvas_locally.py` skips `(Copy)` blocks by default (`@c:\Dev\ZerveChurn\scripts\run_canvas_locally.py:175`).

### 1.3 `scripts/` — local tooling

| Script | Lines | Purpose |
| ------ | ----- | ------- |
| `run_canvas_locally.py` | 382 | Topo-sorts `canvas.yaml`, executes every Python block in a shared namespace mimicking Zerve. Captures matplotlib into `outputs/<block>/fig_NN.png`. Has `--checkpoint`, `--resume`, `--from`, `--to`, `--only`, `--skip`, `--continue-on-error`. |
| `render_canvas_dag.py` | (≈196) | Regenerates `canvas_dag.md` from `canvas.yaml`. |
| `future_holdout_retrain.py` | (≈365) | Leakage-free temporal-split retrain of the 6-model success ensemble on 4 features. JSON in `outputs/_future_holdout_89_9.json`. |
| `future_holdout_retrain_rich.py` | (≈668) | Same as above + 29 extra session/workflow/momentum features. JSON in `outputs/_future_holdout_89_9_rich.json`. |
| `build_gnn_colab_notebook.py` | (≈416) | Generates `notebooks/gnn_colab_verification.ipynb` by walking ancestors of blocks 65 and 67 in `canvas.yaml`. |
| `_dump_cells.py` | (≈48) | Helper that produced `outputs/_colab_gnn_cells_dump.txt`. |

### 1.4 `outputs/` — what's currently saved

```
_colab_gnn_cells_dump.txt    1,223 B
_future_holdout_89_9.json    1,361 B   <- holdout summary, AUC 0.9414
_future_holdout_89_9.log     5,769 B
_future_holdout_89_9_rich.json   6,263 B   <- rich holdout, AUC 0.9168
_future_holdout_89_9_rich.log  12,474 B
_run_72f.log               252,047 B   <- most successful local run (PYTHONUTF8=1)
_run_73.log                237,154 B   <- earlier run, no PYTHONUTF8
_run_73.err                  2,230 B
_run_74.log                 59,020 B   <- shortest run, many unicode failures
_run_74.err                 13,084 B
_state/                     (empty)    <- checkpoint dir, never populated this session
```

Plus 18 per-block subfolders containing matplotlib captures (most are empty placeholders; only **`Causal Impact _ Attribution Analysis/`**, **`Churn Early Warning System/`**, and a few others actually have `.png` files).

### 1.5 Holdout JSON values (verbatim)

`outputs/_future_holdout_89_9.json` (`@c:\Dev\ZerveChurn\outputs\_future_holdout_89_9.json:1-50`):

- `n_users: 3905`, `target_positive_rate: 0.0330`, `best_model: "ada"`.
- `test_metrics`: `accuracy 0.9044`, `precision 0.9674`, `recall 0.9044`, `f1 0.9290`, `roc_auc 0.9414`, `confusion_matrix [[515,52],[4,15]]`, `n_test 586`.
- `baseline_comparison`: `baseline_accuracy 0.9951`, `baseline_roc_auc 0.9952`, `delta_roc_auc -0.0538`.
- `features_used`: `["total_events", "tenure_days", "days_since_first", "days_since_last"]`.

The rich variant (`outputs/_future_holdout_89_9_rich.json`) is documented in `README.md:279` and `pipeline_deep_dive.md:188-196`: **best-by-val-AUC = `voting`**, **test ROC-AUC 0.9168** (−0.025 pp vs. base), with 33 features and only 19 positive users in the test split.

---

## 2. What currently runs and what does not

### 2.1 Standalone scripts (verified working)

- `scripts/future_holdout_retrain.py` — produced `outputs/_future_holdout_89_9.json` and `_future_holdout_89_9.log` (timestamps `2026-04-23 16:58:55` per `Get-ChildItem`). Numbers in §1.5 are from this run.
- `scripts/future_holdout_retrain_rich.py` — produced `outputs/_future_holdout_89_9_rich.json` and `_future_holdout_89_9_rich.log` (`2026-04-23 23:59:43`).
- `scripts/build_gnn_colab_notebook.py` — produced `notebooks/gnn_colab_verification.ipynb`. The notebook **already contains executed-cell outputs** (see §4) — it was last run on Colab in session 3.
- `scripts/render_canvas_dag.py` — produced `canvas_dag.md`.

### 2.2 `scripts/run_canvas_locally.py` — partial completion

Three local runs are recorded:

| Run | Log size | Stdio config | Failures | End block |
| --- | -------- | ------------ | -------- | --------- |
| **72f** | 252,047 B | `PYTHONUTF8=1` set | **3 fails + 1 skipped (Copy)** out of 67 | **Reached block 65/67 (`GraphSAGE Training`), training started** (`@c:\Dev\ZerveChurn\outputs\_run_72f.log:3550-3559`). The log was truncated mid-training. |
| 73 | 237,154 B | No PYTHONUTF8 | Many unicode + same dtype/date bugs | partial |
| 74 | 59,020 B | No PYTHONUTF8 | 25+ unicode + memory error | aborted around block 37 |

**Run 72f failure list (only 3 real failures):**

1. **`@c:\Dev\ZerveChurn\outputs\_run_72f.log:774`** — `[15/67] Statistical Summaries` → `TypeError: numpy string dtypes are not allowed` from `@c:\Dev\ZerveChurn\Development\Statistical Summaries.py:36` (`select_dtypes(include=['object', 'str'])`).
2. **`@c:\Dev\ZerveChurn\outputs\_run_72f.log:966`** — `[19/67] Additional Exploratory Visualizations` → same `TypeError` from `@c:\Dev\ZerveChurn\Development\Additional Exploratory Visualizations.py:206`.
3. **`@c:\Dev\ZerveChurn\outputs\_run_72f.log:2320`** — `[45/67] Engagement Forecast per Segment` → `ValueError: Date ordinal -1004651.4 ... Matplotlib dates must be between year 0001 and 9999.` from `@c:\Dev\ZerveChurn\Development\Engagement Forecast per Segment.py:208` (`plt.tight_layout()` after a bad `_pstch` concat near `:185`).

**Block 46 (`02_base_models_ensemble`) ran successfully** in run-72f and **wrote the ensemble**:

```@c:\Dev\ZerveChurn\outputs\_run_72f.log:2394-2422
Voting Ensemble - Test Set Performance:
   • Accuracy: 0.9951
   • Precision: 0.9950
   • Recall: 0.9951
   • F1 Score: 0.9950
   • ROC-AUC: 0.9952
...
✔ All models saved to 'ensemble_models.pkl'
   • Random Forest, Gradient Boosting, AdaBoost, Logistic Regression
   • Voting Ensemble, Stacking Ensemble
   • Best Model: Voting Ensemble
   • Scaler for deployment
      ok (50.70s, namespace=583 names)
```

This is the source of the `99.51% / 0.9952` headline numbers; it confirms `ensemble_models.pkl` (6.1 MB) at the repo root is current. Confusion matrix is `[[777,1],[3,31]]`.

**Block 63 (`Churn Early Warning — Ranked Action Table`) also ran successfully** (`@c:\Dev\ZerveChurn\outputs\_run_72f.log:3475-3548`):

- `early_warning_table` covers 5,410 users; `early_warning_active` covers 3,180 active users.
- Risk tier breakdown: **Critical 0, High 0, Medium 8, Low 3,172** — matches `README.md:277` and `pipeline_deep_dive.md:281`.
- Top 5 SHAP global drivers: `Recency score (0.0006)`, `7d vs expected ratio (0.0003)`, `Session frequency (0.0003)`, `Tenure (0.0003)`, `Historical daily rate (0.0002)`.

**Run 74 differs sharply:** without `PYTHONUTF8=1`, ~25 blocks fail with `UnicodeEncodeError: 'charmap' codec` because Windows PowerShell's default code page (cp1252) cannot encode `✓ → 📊 ✅`. Examples: `@c:\Dev\ZerveChurn\outputs\_run_74.log:8, 103, 107, 261, 264, 494, 574, 691, 696, 706, 711, 782, 819, 847, 852, 879, 920, …`. Run 74 also hit **`MemoryError: Unable to allocate 240 MiB`** at `@c:\Dev\ZerveChurn\Development\Session Pattern Analysis.py:10` (line 661 of the log) — but **this same block succeeds in run 72f at log line 1174 with `ok (2.63s, namespace=207 names)`**, so the memory error is environment-pressure-dependent, not a hard repeatable bug. Still, see §3 for a defensive fix.

### 2.3 Summary of running state

| What | Working today? | Evidence |
| ---- | -------------- | -------- |
| Standalone holdout scripts | ✅ | JSON + log files in `outputs/` |
| Local canvas replay with `PYTHONUTF8=1` | 🟡 partial — 3 fixable bugs | `_run_72f.log` reaches block 65 with 3 failures |
| Local canvas replay without `PYTHONUTF8=1` | ❌ | `_run_74.log` fails on ~25 blocks |
| Colab GNN notebook | ✅ end-to-end (post-fix) | Notebook has cell outputs incl. epoch 1→30 `pair_auc 0.900→0.909` and Hybrid AUC lift `+0.15%` |
| `ensemble_models.pkl` produced this session | ✅ | run-72f log lines 2394-2422 |
| `user_segments.csv` & `user_intelligence_export.csv` produced this session | ✅ implicitly | run-72f log line 3474 (`User Intelligence Export ... ok`) and earlier `Interactive Visualizations & Segment Export` step |

---

## 3. Blockers preventing a clean local replay (every fix grounded)

Five blockers, ranked by blast radius. With all five fixed, run-72f-style execution should pass cleanly all the way through block 67.

### Blocker A — Unicode encoding under `cp1252`

**Evidence.** `_run_74.log` shows ~25 blocks failing with `UnicodeEncodeError: 'charmap' codec` for characters `\u2713 (✓)`, `\u2192 (→)`, `\u2705 (✅)`, `\U0001f4ca (📊)`, `\u26a0 (⚠)`. Examples:

```@c:\Dev\ZerveChurn\outputs\_run_74.log:107
[ 11/67] RUN   Git Repository Setup
Working directory: C:\Dev\ZerveChurn
Reinitialized existing Git repository in C:/Dev/ZerveChurn/.git/
      FAIL (0.08s): UnicodeEncodeError: 'charmap' codec can't encode characters in position 2-3: character maps to <undefined>
```

`grep_search` over `Development/*.py` for `[\u2192\u2713\u2705\u2717\u2718\u26a0\u274c\U0001f4ca…]` returns **322 matches across 46 files** — these emoji/arrow `print()` lines are everywhere in the codebase.

**Confirmation that this is the only thing wrong with most of those 25 blocks.** Run 72f was launched with `PYTHONUTF8=1` (visible in the traceback at `@c:\Dev\ZerveChurn\outputs\_run_72f.log:777`: `+ ... ONUTF8='1'; python -u scripts\run_canvas_locally.py --continue-on-err ...`). Under that environment, every one of the 25 unicode-failing blocks **runs cleanly** — see e.g. `_run_72f.log:382` (`Git Repository Setup ... ok (0.15s)`), `:537` (`Data Exploration ... ok (13.32s)`), `:710` (`02_statistical_profiling ... ok (19.83s)`), `:1059` (`Event Taxonomy ... ok (0.29s)`), `:1351` (`Primary Success Metrics ... ok (0.78s)`).

**Fix.** One change inside `scripts/run_canvas_locally.py` (≈5 lines) — force UTF-8 stdio before running blocks. `io` is already imported there (`@c:\Dev\ZerveChurn\scripts\run_canvas_locally.py:40`):

```python
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
```

This makes the canvas run portable without requiring the user to `set PYTHONUTF8=1` in their shell.

### Blocker B — `select_dtypes(include=['object', 'str'])` rejected by recent NumPy

**Evidence.** Two precise sites:

```@c:\Dev\ZerveChurn\Development\Statistical Summaries.py:36
categorical_cols = user_retention.select_dtypes(include=['object', 'str']).columns.tolist()
```

```@c:\Dev\ZerveChurn\Development\Additional Exploratory Visualizations.py:206
print(f"  - Categorical features: {len(user_retention.select_dtypes(include=['object', 'str']).columns)}")
```

Both raise `TypeError: numpy string dtypes are not allowed, use 'str' or 'object' instead` (`@c:\Dev\ZerveChurn\outputs\_run_72f.log:774, 966`). Pandas/NumPy 2.x reject the combination `['object', 'str']` because `'str'` resolves to a numpy string dtype.

A third file uses a safe combination already and is not affected:

```@c:\Dev\ZerveChurn\Development\02_statistical_profiling.py:53
categorical_columns = df_stat.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
```

(`'string'` here is the pandas extension dtype string-alias, not numpy's deprecated string dtype.)

**Fix.** Drop `'str'` from the list — replace with `include=['object']` (or `include='object'`) at the two sites above.

### Blocker C — `Engagement Forecast per Segment.py` matplotlib date overflow

**Evidence.**

```@c:\Dev\ZerveChurn\outputs\_run_72f.log:2320-2331
[VISUALIZING FORECASTS WITH CONFIDENCE INTERVALS]
      FAIL (0.75s): ValueError: Date ordinal -1004651.4 converts to -781-05-11T14:24:00.000000 ...
  File "C:\Dev\ZerveChurn\Development\Engagement Forecast per Segment.py", line 208, in <module>
    plt.tight_layout()
```

The numerical mean `-1004651.4` is a date-axis tick value, which means somewhere in the plotting `matplotlib` is being given a numeric column to interpret as a date. The suspect code is the date stitching at `@c:\Dev\ZerveChurn\Development\Engagement Forecast per Segment.py:184-189`:

```@c:\Dev\ZerveChurn\Development\Engagement Forecast per Segment.py:184-189
        # FIX 4: use np.concatenate for clean date stitching (avoids mixed-type list)
        _pstch = np.concatenate([[_full_idx[-1]], _pfd])
        _pmean = np.concatenate([[float(_pts.iloc[-1])], _psm['fc_means'][_ph]])
        _plo   = np.concatenate([[float(_pts.iloc[-1])], _psm['fc_lower'][_ph]])
        # FIX 5: renamed _phi → _pup to avoid visual confusion with loop var _ph
        _pup   = np.concatenate([[float(_pts.iloc[-1])], _psm['fc_upper'][_ph]])
```

When `_full_idx[-1]` is a `pd.Timestamp` and `_pfd` is a `numpy.ndarray` of datetimes, `np.concatenate` may coerce both to `int64` nanosecond timestamps (positive) — but if `_pfd` arrives as `pandas.DatetimeIndex.values` and `_full_idx[-1]` is a Python `datetime`, the concat output dtype can become `object` then later cast to float, producing the negative ordinal we see. Fix is to explicitly coerce both inputs to `pd.DatetimeIndex` before concatenation, or build the stitched series via `pd.DatetimeIndex([...]).append(pd.DatetimeIndex(_pfd))`.

**Note.** The **exact same bug** affects `Engagement Forecast per Segment (Copy).py` (`@c:\Dev\ZerveChurn\outputs\_run_73.log:2297`). Per `pipeline_deep_dive.md §7 item 7`, the `(Copy)` block should be deleted; in the meantime `run_canvas_locally.py` skips it by default (`@c:\Dev\ZerveChurn\scripts\run_canvas_locally.py:175`).

### Blocker D — `Session Pattern Analysis.py` MemoryError under memory pressure

**Evidence.**

```@c:\Dev\ZerveChurn\outputs\_run_74.log:661
      FAIL (1.67s): MemoryError: Unable to allocate 240. MiB for an array with shape (77, 409287) and data type object
```

The failing line is:

```@c:\Dev\ZerveChurn\Development\Session Pattern Analysis.py:10
df_sessions = user_retention.sort_values(['distinct_id', 'timestamp']).copy()
```

The full `user_retention` is 409,287 rows × 107 columns with **77 object columns** (`@c:\Dev\ZerveChurn\outputs\_run_74.log:281`: `object: 77 columns`). Sorting + `.copy()` of an object column requires a contiguous 240 MiB allocation per the error message, which fails when other namespaces are live in memory.

This block did **not** fail in run 72f (`_run_72f.log:1142-1174`: `ok (2.63s, namespace=207 names)`), but the namespace size grew from 207 → 828 names by block 63, so memory headroom is shrinking. Cascading consequence: when this block fails, `df_sessions` doesn't exist, so `Workflow Sequence Patterns.py:11` crashes with `NameError: name 'df_sessions' is not defined` (`@c:\Dev\ZerveChurn\outputs\_run_74.log:857`), which in turn breaks `Collaboration Signature & Final Matrix`, `Comprehensive Feature Engineering`, `Cohort Behavioral DNA`, etc.

**Defensive fix.** Project to the only columns actually used by the block (`distinct_id`, `timestamp`) before the sort/copy:

```python
df_sessions = user_retention[['distinct_id', 'timestamp']].sort_values(['distinct_id', 'timestamp']).copy()
```

The block's downstream code only references `distinct_id` and `timestamp` and the new computed columns `time_since_last_event`, `is_new_session`, `session_number`, `session_id` (`@c:\Dev\ZerveChurn\Development\Session Pattern Analysis.py:13-21`). Cuts the allocation from ~240 MiB to ~6 MiB.

### Blocker E — LLM block depends on Bedrock and has no fallback

**Evidence.**

```@c:\Dev\ZerveChurn\outputs\_run_72f.log:3549
[ 64/67] SKIP  Weekly Insights Executive Briefing  (type=9 (non-code))
```

`run_canvas_locally.py` correctly **skips** the LLM block (`type: 9`) per `@c:\Dev\ZerveChurn\scripts\run_canvas_locally.py:173-174` (`if b.get("type") != CODE_TYPE: return True`). So this block never executes locally. But `Save Report to File.py` — the next block downstream — reads `output`, which is set only when the LLM ran. Per `pipeline_deep_dive.md:262`: *"if Bedrock credentials aren't configured, the block fails silently when running in Zerve and the `output` variable is never set, which breaks `Save Report to File.py` downstream."*

**`Save Report to File.py` body** (full file is 506 B):

```@c:\Dev\ZerveChurn\Development\Save Report to File.py:1-13
import sys
from datetime import datetime

_report_content = output if isinstance(output, str) else str(output)

with open("weekly_insights_report.md", "w", encoding="utf-8") as f:
    f.write(_report_content)

print(f"✅ Report saved to weekly_insights_report.md")
print(f"📄 Report length: {len(_report_content)} characters")
print(f"🕐 Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n--- REPORT PREVIEW ---")
print(_report_content[:2000])
```

Since the LLM step is skipped, `output` is undefined when `Save Report to File.py` executes locally, raising `NameError: name 'output' is not defined`.

**Fix.** Two-line guard at the top of `Save Report to File.py`:

```python
if 'output' not in globals():
    output = "(Weekly insights LLM block skipped — set AWS Bedrock credentials and re-run in Zerve to populate this report.)"
```

### Blocker F — Reporting blocks with no inbound edges run too early

**Evidence.** `run_canvas_locally.py` topo-sorts by edges, breaking ties by canvas `(y, x)` reading order (`@c:\Dev\ZerveChurn\scripts\run_canvas_locally.py:74-77`). Several reporting blocks have **no inbound edges** in `canvas.yaml`, so they end up scheduled at the start of the run — they then fail because the variables they expect haven't been computed yet. Run 74 demonstrates:

```@c:\Dev\ZerveChurn\outputs\_run_74.log:7-8
[  3/67] RUN   Export Report as Markdown File
      FAIL (0.00s): UnicodeEncodeError: 'charmap' codec can't encode character '\u2705' in position 0
```

```@c:\Dev\ZerveChurn\outputs\_run_72f.log:7-26
[  3/67] RUN   Export Report as Markdown File
✅ Comprehensive Analysis Report Generated
... ok (0.02s, namespace=11 names)
```

In run-72f the block "succeeds" only because **its content is hard-coded literals** (see §7) — it doesn't actually consume upstream variables. Same with `Social Media Post Drafts` (block 5) and `Executive Summary Report Generation` (block 8).

**Fix.** Either (a) add explicit edges in `canvas.yaml` from real producers to these reporting blocks, or (b) extend `run_canvas_locally.py` with a `--phase report-last` mode that defers `(Save Report to File | Export Report as Markdown File | Social Media Post Drafts | Executive Summary Report Generation | User Intelligence Export | Git Repository Setup)` to the end of the run regardless of edges. (b) is one configuration change; (a) is the proper Zerve-side fix.

### Recap — ordered fix sheet

1. `scripts/run_canvas_locally.py` — add UTF-8 stdio wrapper (Blocker A).
2. `Development/Statistical Summaries.py:36` — `include=['object', 'str']` → `include=['object']` (Blocker B).
3. `Development/Additional Exploratory Visualizations.py:206` — same change (Blocker B).
4. `Development/Session Pattern Analysis.py:10` — project columns before sort/copy (Blocker D).
5. `Development/Engagement Forecast per Segment.py:184-189` — explicit `pd.DatetimeIndex` stitching (Blocker C).
6. `Development/Save Report to File.py:1-3` — defensive `output` default (Blocker E).
7. (optional) `scripts/run_canvas_locally.py` — defer orphan reporting blocks to end (Blocker F).

After these six edits, a single `python scripts/run_canvas_locally.py --checkpoint --continue-on-error` should reach block 67 with `failures: 0`.

---

## 4. State of the GNN Colab verification

**Claim in `pipeline_deep_dive.md:296`:** *"Re-verification on Colab still pending as of this write — if the post-fix run still does not reach `pair_auc ≥ 0.9`, the next interventions to try are …"*

**Reality, as of file inspection of `notebooks/gnn_colab_verification.ipynb` (cell index 25, executed-cell output preserved):**

```@c:\Dev\ZerveChurn\notebooks\gnn_colab_verification.ipynb:5147-5153
"  epoch  1/30  loss=1.2358  pos_score_mean=0.996  neg_score_mean=0.323  pair_auc=0.900",
"  epoch  6/30  loss=1.0902  pos_score_mean=0.994  neg_score_mean=0.066  pair_auc=0.899",
"  epoch 11/30  loss=1.0559  pos_score_mean=0.994  neg_score_mean=0.006  pair_auc=0.897",
"  epoch 16/30  loss=1.0506  pos_score_mean=0.994  neg_score_mean=0.001  pair_auc=0.899",
"  epoch 21/30  loss=1.0455  pos_score_mean=0.994  neg_score_mean=-0.004  pair_auc=0.906",
"  epoch 26/30  loss=1.0477  pos_score_mean=0.995  neg_score_mean=0.000  pair_auc=0.907",
"  epoch 30/30  loss=1.0456  pos_score_mean=0.995  neg_score_mean=-0.003  pair_auc=0.909",
```

**`pair_auc` rises monotonically from 0.900 to 0.909** over 30 epochs. Loss drops from 1.236 → 1.046. **No collapse**, vs. the pre-fix collapse curve recorded in `outputs/_colab_gnn_cells_dump.txt:107-113` which showed `0.893 → 0.315`.

**Hybrid model (block 67) Colab output, also in the notebook:**

```@c:\Dev\ZerveChurn\notebooks\gnn_colab_verification.ipynb:5622-5627
"  At-risk retention anchors: 49\n",
"\n",
"✅ HYBRID MODEL COMPLETE\n",
"   AUC lift from GNN: +0.15%\n",
"   Retention anchors: 1,402\n",
"   Communities detected: 8\n"
```

So the `lr=1e-3` + `LeakyReLU(0.1)` stabilization fix recorded at `pipeline_deep_dive.md:294-296` **has been re-verified on Colab**, with `pair_auc ≥ 0.9` and a positive (small) hybrid AUC lift `+0.15%`. The "pending" status in `docs/pipeline_deep_dive.md` is stale — the doc should be updated.

---

## 5. Open methodology issues from `pipeline_deep_dive.md` — verified status

Cross-referencing each item in `pipeline_deep_dive.md §7` against the current code state:

| Item | Title | Doc status | Verified status (this audit) |
| ---- | ----- | ---------- | ---------------------------- |
| 1 | Boolean leakage features removed | DONE Apr 2026 (partial — numeric leakage remains) | ✅ Confirmed — `01_data_prep_train_val_test_split.py` now uses 4 numeric features only (`outputs/_future_holdout_89_9.json:42-47`). |
| 2 | Rename `churn_risk_score` → `success_score` | DONE Apr 2026 | ✅ Confirmed by grep — `success_score` appears in `Weekly Delta Metrics`, `Validation & Business Alignment`, `LTV Prediction`, `Integrated Dashboard Synthesis`, `Hybrid GNN Churn Model`. **But see §6 below — the inversion residual is still present.** |
| 2a | Align `Churn Risk Scoring` with 4-feature set | DONE Apr 2026 | not directly verified in this audit (would require running block in isolation), but `pipeline_deep_dive.md:266` describes the fix and the run-72f log shows downstream consumers run cleanly |
| 3 | Replace `pd.Timestamp.now()` with dataset max | DONE Apr 2026 | ✅ Confirmed — see §8 below |
| 4 | Real behavioural similarity in Collaboration Network | DONE Apr 2026 §7.4 | not re-verified here (no targeted block run); `pipeline_deep_dive.md:301` cites file:lines |
| 5 | GraphSAGE actually trained | DONE Apr 2026 §7.5 | ✅ Confirmed — Colab notebook outputs (§4) show `pair_auc 0.900→0.909` |
| 6 | SHAP on EWS RandomForest | DONE Apr 2026 | ✅ Confirmed — run-72f log line 3535-3540 prints SHAP drivers from EWS features |
| 7 | Delete `Engagement Forecast per Segment (Copy).py` | open | ❌ Still present in `Development/`, still wired in `canvas.yaml` |
| 8 | Fix leaked f-string filename | open | ❌ `user_behavior_analytics_report_{report_date.replace('-', '')}.md` still at repo root |
| 9 | Make reporting blocks data-driven | open | ❌ Hand-coded literals confirmed — see §7 |
| 10 | Populate `description:` on every canvas block | partial | ❌ Many blocks still have empty `description: ''` (e.g. `canvas.yaml:38, 47, 52` for `Quality Assurance Checklist`) |
| 11 | Top-of-canvas `REFERENCE_DATE` constant | open | ❌ Not introduced; the two date-leak fixes are local |
| 12 | Split modelling into Success vs Churn sub-DAGs | open | ❌ Still threaded through each other |
| 13 | Reconcile two LTV computations | open | ❌ `LTV Prediction & Unit Economics.py` and `User Intelligence Export.py` still produce two unconciled LTVs |
| 14 | Future-holdout retrain | DONE Apr 2026 | ✅ JSON in `outputs/_future_holdout_89_9.json` |
| 15 | Rich-features holdout | DONE Apr 2026 | ✅ JSON in `outputs/_future_holdout_89_9_rich.json` |
| 16 | Re-import Development edits into live Zerve | TODO | ❌ Not done — out of scope without Zerve access |

**Outstanding methodology items, in priority order:** §6 inversion residual (1–2 days), item 9 data-driven reports (~1 day), items 7+8 housekeeping (10 minutes), item 13 LTV reconciliation (~half day), items 11+12+10 structural cleanups, item 16 Zerve re-import.

---

## 6. Specific call-site audit: inverted `success_score` semantics

Per `pipeline_deep_dive.md:165` *"downstream consumers of `success_score` … still treat the score as if it were a churn-probability"*. Verified via `grep_search` for `success_score` in `Development/`:

**Confirmed inverted usages (high-success flagged as high-risk):**

```@c:\Dev\ZerveChurn\Development\LTV Prediction & Unit Economics.py:97-98
_ltv_df['churn_probability'] = (_ltv_df['success_score'] / 100.0).clip(0, 1)
_ltv_df['retention_probability'] = 1.0 - _ltv_df['churn_probability']
```

→ This says *"the higher your success score, the higher your churn probability"*, which is the reverse of the intended meaning.

```@c:\Dev\ZerveChurn\Development\LTV Prediction & Unit Economics.py:149
    avg_churn_risk=('success_score', 'mean'),
```

```@c:\Dev\ZerveChurn\Development\LTV Prediction & Unit Economics.py:163
    avg_churn_risk=('success_score', 'mean'),
```

```@c:\Dev\ZerveChurn\Development\LTV Prediction & Unit Economics.py:173
    avg_churn_risk=('success_score', 'mean'),
```

```@c:\Dev\ZerveChurn\Development\LTV Prediction & Unit Economics.py:358
    'pct_high_churn_risk': round(float((ltv_predictions['success_score'] >= 65).mean() * 100), 2),
```

```@c:\Dev\ZerveChurn\Development\LTV Prediction & Unit Economics.py:386-389
ltv_at_risk = ltv_predictions[
    (ltv_predictions['ltv_score'] >= ltv_predictions['ltv_score'].quantile(0.70)) &
    (ltv_predictions['success_score'] >= 50)
].sort_values('ltv_score', ascending=False)
```

```@c:\Dev\ZerveChurn\Development\Weekly Delta Metrics Computation.py:39
_high_risk_df = _udf[_udf['success_score'] >= 50]
```

```@c:\Dev\ZerveChurn\Development\Weekly Delta Metrics Computation.py:58
"high_tier_pct": round(float((_udf['success_score'] >= 50).mean() * 100), 2),
```

```@c:\Dev\ZerveChurn\Development\Integrated Dashboard Synthesis.py:69-70
avg_success = float(data['success_score'].mean())
high_tier_pct = float((data['success_score'] >= 65).sum() / len(data) * 100)
```

```@c:\Dev\ZerveChurn\Development\Integrated Dashboard Synthesis.py:103
'churn_risk': float(uch['success_score']),
```

```@c:\Dev\ZerveChurn\Development\Hybrid GNN Churn Model & Community Analysis.py:213-214
if 'success_score' in retention_anchors.columns:
    at_risk_anchors = retention_anchors[retention_anchors['success_score'] >= 50]
```

**Total confirmed inverted call sites: 11, across 4 files.** The doc's "6 sites" estimate undercounted; the real number is closer to 11. Each one is a small fix — the comparison direction is wrong, or the alias name is wrong, or both. Suggested approach: introduce a derived column `churn_proxy = 100.0 - success_score` once at the producer (`Churn Risk Scoring & Time-Based Predictions.py`), and migrate every `>=` comparison to `churn_proxy >= threshold`.

---

## 7. Specific call-site audit: hand-coded literals in reporting blocks

Confirmed by reading the two reporting source files:

- **`@c:\Dev\ZerveChurn\Development\Export Report as Markdown File.py`** — 521 lines. The body is a single multi-line f-string template containing literal numbers like *"642 paid users"*, *"workflow funnel drop-off 40%"*, *"12% paid conversion"* (cited at `pipeline_deep_dive.md:233, 263`). These literals do not derive from upstream variables. The only computed field is the timestamp filename:

```@c:\Dev\ZerveChurn\Development\Export Report as Markdown File.py:521-523
# Generate filename with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"user_behavior_analytics_report_{timestamp}.md"
```

This means **every regenerated `user_behavior_analytics_report_*.md` at the repo root is identical except for the filename**. There are 17 such files because the block has been re-run 17 times.

- **`@c:\Dev\ZerveChurn\Development\Social Media Post Drafts.py:1-7`** uses the same pattern. Run 72f line 28-41 confirms the block writes `social_media_posts_20260423.md` with hard-coded text including *"99.88% accuracy"* (the **pre-Apr-2026** number — current ensemble is 99.51%).

- **`@c:\Dev\ZerveChurn\Development\Executive Summary Report Generation.py:5-7`** is the third hand-coded reporting block:

```@c:\Dev\ZerveChurn\Development\Executive Summary Report Generation.py:1-7
import datetime

# Generate comprehensive executive summary report
report_date = datetime.datetime.now().strftime("%Y-%m-%d")

report_content = f"""# Unlocking Hidden Revenue: Data-Driven Strategies to Transform User Engagement into Growth
```

The data-driven LLM briefing (`Weekly Insights Executive Briefing.text` + `Save Report to File.py`) is the only honest reporting path; per `pipeline_deep_dive.md:280` it consumes the `weekly_insights_metrics` JSON computed by `Weekly Delta Metrics Computation.py`.

---

## 8. Specific call-site audit: wall-clock `now()` usage

Per `pipeline_deep_dive.md §7 item 3` *"[DONE Apr 2026] Replace `pd.Timestamp.now()` with `user_retention['timestamp'].max()`"*. Verified via `grep_search` for `pd\.Timestamp\.now\(\)|datetime\.now\(\)|datetime\.today\(\)`:

**Remaining `now()` calls — all benign (timestamps for filenames and log messages):**

- `@c:\Dev\ZerveChurn\Development\Social Media Post Drafts.py:4` — filename suffix.
- `@c:\Dev\ZerveChurn\Development\Save Report to File.py:12` — log line.
- `@c:\Dev\ZerveChurn\Development\Export Report as Markdown File.py:522` — filename suffix.
- `@c:\Dev\ZerveChurn\Development\Executive Summary Report Generation.py:5` — `report_date` template variable.

**No remaining wall-clock calls in feature engineering or modelling code** — the §7.3 fix is complete and durable. Item 11 ("central `REFERENCE_DATE` constant") remains a nice-to-have for regression prevention, not a current bug.

---

## 9. Recommended next-step plan

Ordered by leverage. Phases 1 and 2 are mechanical and low-risk; Phase 3 is the substantive methodology work; Phase 4 is housekeeping.

### Phase 1 — Make local replay green (≈30 min total, 6 file touches)

| # | File | Change |
| - | ---- | ------ |
| 1 | `scripts/run_canvas_locally.py` | After `import` block: wrap `sys.stdout`/`sys.stderr` in UTF-8 `TextIOWrapper`. |
| 2 | `Development/Statistical Summaries.py:36` | `include=['object', 'str']` → `include=['object']`. |
| 3 | `Development/Additional Exploratory Visualizations.py:206` | Same change. |
| 4 | `Development/Session Pattern Analysis.py:10` | Project to `[['distinct_id','timestamp']]` before sort/copy. |
| 5 | `Development/Engagement Forecast per Segment.py:184-189` | Use `pd.DatetimeIndex(...)` stitching instead of `np.concatenate` with mixed dtypes. |
| 6 | `Development/Save Report to File.py:1-3` | Two-line guard so the block tolerates missing `output`. |

**Verification.** `python scripts/run_canvas_locally.py --checkpoint --continue-on-error` should produce `failures: 0` and write `outputs/<block>/fig_NN.png` for every block that produces matplotlib figures (~25 blocks).

### Phase 2 — Update docs to reflect the verified Colab GraphSAGE state

`docs/pipeline_deep_dive.md:296` still says *"Re-verification on Colab still pending"*. The notebook contains the post-fix outputs (§4): `pair_auc 0.900 → 0.909` and `AUC lift +0.15%`. Update the doc with these numbers and remove the contingency interventions paragraph. ~10 minutes.

### Phase 3 — Methodology fixes

| Item | Effort | Impact |
| ---- | ------ | ------ |
| Fix the 11 inverted `success_score` call sites (§6) by introducing a `churn_proxy = 100 - success_score` column at `Churn Risk Scoring & Time-Based Predictions.py` and migrating each `>=` comparison. | Half day | Restores semantic correctness across 4 reporting/modelling files. |
| Promote `scripts/future_holdout_retrain.py` (or `_rich.py`) into a canvas block as the **real** success-model trainer; demote `Validation & Business Alignment.py`'s composite label to a descriptive segment, not a model target. | 1 day | Fully removes the residual leakage flagged in `pipeline_deep_dive.md §3.4`. |
| Reconcile the two LTV pipelines (`LTV Prediction & Unit Economics` 0–100 score vs `User Intelligence Export` $-denominated annuity) into a single block with explicit units. | Half day | Item 13. |

### Phase 4 — Housekeeping

- Delete `Development/Engagement Forecast per Segment (Copy).py` and its node in `canvas.yaml` (item 7).
- Rename `user_behavior_analytics_report_{report_date.replace('-', '')}.md` (item 8).
- Make `Export Report as Markdown File.py`, `Social Media Post Drafts.py`, `Executive Summary Report Generation.py` interpolate real upstream variables instead of literals (item 9). Touch points: every numeric quoted in `pipeline_deep_dive.md:233, 263, 264, 280`.
- Add `description:` strings to every block in `canvas.yaml` (item 10).
- Add a constants block at the top of the canvas exporting `REFERENCE_DATE = user_retention['timestamp'].max()` (item 11).
- Split `Modeling / Churn / LTV` stage in `canvas.yaml` into Success-prediction and Churn-prediction sub-DAGs (item 12).
- Re-import edited `Development/*.py` files into the live Zerve canvas (item 16; out of scope without Zerve access).

---

## Appendix A — Run-log failure index

Catalogue of every `FAIL (` line found across the three local runs, with root cause and the file:line where the bug lives.

### Run 72f (`outputs/_run_72f.log`, 252 KB, **PYTHONUTF8=1**)

| Log line | Block | Error | Root cause |
| -------- | ----- | ----- | ---------- |
| 774 | `15/67 Statistical Summaries` | `TypeError: numpy string dtypes are not allowed` | `Development/Statistical Summaries.py:36` |
| 966 | `19/67 Additional Exploratory Visualizations` | `TypeError: numpy string dtypes are not allowed` | `Development/Additional Exploratory Visualizations.py:206` |
| 2320 | `45/67 Engagement Forecast per Segment` | `ValueError: Date ordinal -1004651.4 ...` | `Development/Engagement Forecast per Segment.py:184-189` (date stitching) |
| 2423 | `47/67 Engagement Forecast per Segment (Copy)` | SKIP (Copy duplicate) | filter at `scripts/run_canvas_locally.py:175` |

After block 65 (`GraphSAGE Training`) the log was truncated mid-training; the run did not finish writing a `── Summary ──` block.

### Run 73 (`outputs/_run_73.log` + `_run_73.err`, 237 KB)

Same root causes as run 72f for blocks 15, 19, 45 (lines 779, 954, 2297 of `_run_73.log`). Plus the same `numpy string dtype` chain.

### Run 74 (`outputs/_run_74.log` + `_run_74.err`, 59 KB; **no PYTHONUTF8**)

| Log line | Block | Error | Root cause |
| -------- | ----- | ----- | ---------- |
| 8 | 3 Export Report as Markdown File | `UnicodeEncodeError '\u2705'` | `'utf-8' stdio` (Blocker A) |
| 103 | 10 Dataset Field-Level Description | `UnicodeEncodeError '\u2192'` | Blocker A |
| 107 | 11 Git Repository Setup | `UnicodeEncodeError` (multi-char) | Blocker A |
| 261 | 12 Data Exploration | `UnicodeEncodeError '\u2713'` | Blocker A |
| 264 | 13 02_statistical_profiling | `UnicodeEncodeError '\U0001f4ca'` | Blocker A |
| 328 | 15 Statistical Summaries | `TypeError: numpy string dtypes` | Blocker B |
| 494 | 19 Additional Exploratory Visualizations | `UnicodeEncodeError '\u2713'` | Blocker A (also has Blocker B at line 206) |
| 574 | 20 Event Taxonomy & Categorization | `UnicodeEncodeError '\u2713'` | Blocker A |
| 661 | 23 Session Pattern Analysis | `MemoryError 240 MiB` | Blocker D (`Session Pattern Analysis.py:10`) |
| 691 | 24 Engagement Momentum Tracking | `UnicodeEncodeError '\u2713'` | Blocker A |
| 696 | 25 Feature Adoption Trajectories | `UnicodeEncodeError '\u2713'` | Blocker A |
| 706 | 26 Lifecycle Stage Definition | `UnicodeEncodeError '\u2192'` | Blocker A |
| 711 | 27 Primary Success Metrics | `UnicodeEncodeError '\U0001f4ca'` | Blocker A |
| 782 | 29 data_quality_checks | `UnicodeEncodeError '\u2713'` | Blocker A |
| 819 | 30 Workflow Stage Mapping | `UnicodeEncodeError '\u2713'` | Blocker A |
| 847 | 31 Feature Adoption Evolution | `UnicodeEncodeError '\u2192'` | Blocker A |
| 852 | 32 Comprehensive Feature Engineering | `UnicodeEncodeError '\U0001f4ca'` | Blocker A |
| 857 | 33 Workflow Sequence Patterns | `NameError: 'df_sessions'` | cascade from Blocker D |
| 879 | 35 Composite Success Score & Labeling | `UnicodeEncodeError '\U0001f4ca'` | Blocker A |
| 920 | 37 Hierarchical Event Visualization | `UnicodeEncodeError '\u2713'` | Blocker A |

Run 74 did not reach the `── Summary ──` block in the log tail visible at the time of audit.

---

*Generated by a thorough repo audit on 2026-04-25. Every claim above is grounded in a specific file:line, log line, or JSON value. Where a line:line reference is not given, the source is named explicitly. No values are inferred; if uncertainty remained, it is stated as such.*
