"""Generate a self-contained Google Colab notebook that runs only the
minimum ancestor chain of the GraphSAGE (block 65) and Hybrid GNN (block 67)
blocks, so the heavy GNN training can be verified on Colab instead of the
local laptop.

Run:
    python scripts/build_gnn_colab_notebook.py

Output:
    notebooks/gnn_colab_verification.ipynb

The script parses canvas.yaml, topo-sorts the full DAG, walks backwards from
the two GNN targets to collect their transitive ancestors, then emits one
Colab code cell per ancestor block (plus setup / data-upload / stubs /
results-export cells around them). The generated notebook keeps the Zerve
"single shared global namespace" semantics by running every cell in the
Colab runtime's top-level scope.
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CANVAS = ROOT / "canvas.yaml"
DEV = ROOT / "Development"
OUT = ROOT / "notebooks" / "gnn_colab_verification.ipynb"

TARGETS = [
    "GraphSAGE Training & Social Influence Embeddings",
    "Hybrid GNN Churn Model & Community Analysis",
]

# ───────────────────────── Canvas → ancestor plan ──────────────────────────

def load_ordered_ancestors() -> list[dict]:
    canvas = yaml.safe_load(CANVAS.read_text(encoding="utf-8"))
    layer = canvas["layers"][0]
    blocks = layer["blocks"]
    edges = layer.get("edges", [])
    by_id = {b["id"]: b for b in blocks}

    fwd: dict[str, list[str]] = defaultdict(list)
    rev: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        s, t = e["source"], e["target"]
        if s in by_id and t in by_id:
            fwd[s].append(t)
            rev[t].append(s)

    name_to_id = {b["name"]: b["id"] for b in blocks}
    missing = [t for t in TARGETS if t not in name_to_id]
    if missing:
        raise SystemExit(f"Missing target blocks in canvas.yaml: {missing}")

    ancestors: set[str] = set()
    stack = [name_to_id[t] for t in TARGETS]
    while stack:
        n = stack.pop()
        if n in ancestors:
            continue
        ancestors.add(n)
        for p in rev.get(n, []):
            if p not in ancestors:
                stack.append(p)

    # Topo-sort restricted to the ancestor subgraph, tie-break by (y, x, name)
    indeg: dict[str, int] = {n: 0 for n in ancestors}
    for n in ancestors:
        for p in rev.get(n, []):
            if p in ancestors:
                indeg[n] += 1

    def key(bid: str) -> tuple[float, float, str]:
        b = by_id[bid]
        return (b.get("y", 0), b.get("x", 0), b["name"])

    ready: deque[str] = deque(
        sorted([n for n, d in indeg.items() if d == 0], key=key)
    )
    ordered: list[dict] = []
    while ready:
        n = ready.popleft()
        ordered.append(by_id[n])
        for ch in sorted(fwd.get(n, []), key=key):
            if ch in ancestors:
                indeg[ch] -= 1
                if indeg[ch] == 0:
                    ready.append(ch)

    if len(ordered) != len(ancestors):
        raise SystemExit("Cycle detected in ancestor subgraph")
    return ordered


# ───────────────────────── Notebook cell helpers ───────────────────────────

def _src(text: str) -> list[str]:
    """nbformat expects cell source as a list of lines, each (except the
    last) terminated by \\n."""
    lines = text.splitlines(keepends=True)
    return lines if lines else [""]


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _src(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _src(text),
    }


# ───────────────────────── Static cells ────────────────────────────────────

HEADER_MD = """# ZerveChurn — GNN Verification (Colab)

This notebook reproduces **blocks 65 (GraphSAGE Training)** and **67 (Hybrid
GNN Churn Model)** from the Zerve canvas by running only the minimum
upstream chain required to build their inputs. Generated automatically from
`canvas.yaml` by `scripts/build_gnn_colab_notebook.py`.

The notebook is **self-contained** — all 19 ancestor blocks are inlined
below, so no `git clone` is needed.

## How to run (Windsurf / VS Code with the Google Colab extension)

1. Open this `.ipynb` in Windsurf. Click **Select Kernel → Colab → Auto
   Connect** (top-right of the notebook) so cells execute on a hosted Colab
   runtime. CPU runtime is fine; the graph is ~5.4k nodes / ~40k edges and
   30 GraphSAGE epochs finish in a few minutes.
2. Run the **Setup** cell (installs `pyarrow` / `pyyaml`, `chdir`s to
   `/content`).
3. Get `user_retention.parquet` onto the runtime. Pick one:
   - **Recommended (extension-native)**: right-click
     `user_retention.parquet` in the Windsurf Explorer panel and choose
     **Upload to Colab**. It lands at `/content/user_retention.parquet`.
   - **Google Drive**: Command Palette (`Ctrl/Cmd+Shift+P`) →
     `Colab: Mount Google Drive to Server...` — then put the parquet at
     `MyDrive/ZerveChurn/user_retention.parquet`.
   - **Browser-Colab fallback**: the **Data check** cell below will offer
     `google.colab.files.upload()` if neither of the above is detected.
4. Run the **Data check** cell. It auto-locates the parquet at `/content/`,
   in the cwd, or under common Drive paths, and bails out with clear
   instructions if it can't find it.
5. `Run All` from there. The two GNN target blocks print their metrics and
   `outputs/colab_gnn_results.json` is written for download via the
   extension's file panel.

## How to run (plain browser Colab)

Same as above, except in step 3 use `Files → Upload` in the left sidebar
(or let the fallback `files.upload()` widget run in the Data check cell).
"""

SETUP_CODE = """# === Setup: working dir + deps (self-contained, no git clone) ===
import os, sys, subprocess, pathlib

# /content is where the Colab-VSCode extension's "Upload to Colab" drops
# files and where google.colab.files.upload() also lands them by default.
# Using it as cwd means every inlined block's `pd.read_parquet(...)`
# resolves without path surgery.
WORK_DIR = "/content"
pathlib.Path(WORK_DIR).mkdir(parents=True, exist_ok=True)
os.chdir(WORK_DIR)
pathlib.Path("outputs").mkdir(exist_ok=True)
print("cwd:", os.getcwd())

# Colab ships torch, sklearn, pandas, numpy, scipy, matplotlib, pyarrow by
# default. Install only what might be missing.
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "pyarrow", "pyyaml"], check=True)

import torch, sklearn, pandas as pd, numpy as np
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
print("sklearn:", sklearn.__version__, "| pandas:", pd.__version__, "| numpy:", np.__version__)
"""

DATA_CODE = r"""# === Data check: locate user_retention.parquet (~50 MB) ===
# Supports three upload paths so the same cell works in Windsurf/VSCode
# (Colab extension), browser Colab, and Drive-mount setups.

import os, pathlib, shutil

PARQUET = "user_retention.parquet"
CWD_PATH = pathlib.Path(PARQUET)

CANDIDATE_PATHS = [
    CWD_PATH,
    pathlib.Path("/content") / PARQUET,
    pathlib.Path("/content/drive/MyDrive/ZerveChurn") / PARQUET,
    pathlib.Path("/content/drive/MyDrive") / PARQUET,
]

found = next((p for p in CANDIDATE_PATHS if p.exists()), None)

if found is None:
    print("user_retention.parquet NOT FOUND on the runtime.")
    print()
    print("Upload it using ONE of these (no need to re-run the Setup cell):")
    print()
    print("  [A] Windsurf / VS Code with Colab extension (recommended)")
    print("      \u2192 Right-click 'user_retention.parquet' in the Windsurf Explorer")
    print("      \u2192 Choose 'Upload to Colab'")
    print("      \u2192 File will land at /content/user_retention.parquet")
    print()
    print("  [B] Google Drive mount")
    print("      \u2192 Command Palette (Ctrl/Cmd+Shift+P)")
    print("      \u2192 'Colab: Mount Google Drive to Server...'")
    print("      \u2192 Put the parquet at MyDrive/ZerveChurn/user_retention.parquet")
    print()
    print("  [C] Browser-Colab widget fallback (attempting now)")
    try:
        from google.colab import files
        up = files.upload()
        for fname in up:
            target = pathlib.Path("/content") / PARQUET
            if fname != PARQUET or not target.exists():
                shutil.move(fname, str(target))
        found = pathlib.Path("/content") / PARQUET
        if not found.exists():
            raise RuntimeError("Upload did not produce user_retention.parquet")
    except Exception as exc:
        raise SystemExit(
            f"Could not load user_retention.parquet: {exc}. "
            "Use option [A] or [B] above, then re-run this cell."
        )

# Ensure the file is at cwd so all downstream blocks' relative reads work.
if found.resolve() != CWD_PATH.resolve():
    shutil.copy(str(found), str(CWD_PATH))

size_mb = CWD_PATH.stat().st_size / 1e6
print(f"\u2713 user_retention.parquet ready at {CWD_PATH.resolve()} ({size_mb:.1f} MB)")
"""

STUBS_CODE = """# === Zerve runtime stubs ===
# Mirror the minimal helpers scripts/run_canvas_locally.py injects so any
# fleet-style helpers don't blow up. None of the 19 ancestor blocks actually
# use these, but keeping them defined is cheap insurance.

_fleet_state = {"last_spread": None}

def spread(items):
    _fleet_state["last_spread"] = list(items)
    return list(items)

def gather():
    return list(_fleet_state.get("last_spread") or [])

def attach_variable(*_a, **_kw):
    return None

# Silence sklearn FutureWarnings so the important prints stay visible
import warnings
warnings.filterwarnings("ignore")
"""

RESULTS_CODE = """# === Export GNN verification results ===
# Persists the numbers from blocks 65 + 67 so you can download them with
# Colab's file browser (left sidebar → outputs/).

import json, os, pathlib, datetime

pathlib.Path("outputs").mkdir(exist_ok=True)

results = {
    "generated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "block_65_graphsage": {},
    "block_67_hybrid_gnn": {},
}

# GraphSAGE (block 65) typically exposes: gnn_embedding_df, influence_scores,
# kmeans model, training loss trace. We capture what is in scope.
for name in ("gnn_embedding_df", "influence_df", "community_summary"):
    if name in globals():
        obj = globals()[name]
        try:
            path = f"outputs/{name}.csv"
            obj.to_csv(path, index=False)
            results["block_65_graphsage"][name] = {
                "rows": len(obj),
                "cols": list(obj.columns)[:20],
                "saved": path,
            }
        except Exception as exc:
            results["block_65_graphsage"][name] = {"error": str(exc)}

# Hybrid GNN (block 67) exposes hybrid_df, plus the AUC/F1 comparisons.
for name in ("hybrid_df", "hybrid_df_clean", "model_comparison_df",
             "gnn_model_results", "baseline_results", "hybrid_results"):
    if name in globals():
        obj = globals()[name]
        try:
            if hasattr(obj, "to_csv"):
                path = f"outputs/{name}.csv"
                obj.to_csv(path, index=False)
                results["block_67_hybrid_gnn"][name] = {
                    "type": "dataframe",
                    "rows": len(obj),
                    "saved": path,
                }
            else:
                results["block_67_hybrid_gnn"][name] = {
                    "type": type(obj).__name__,
                    "repr": repr(obj)[:500],
                }
        except Exception as exc:
            results["block_67_hybrid_gnn"][name] = {"error": str(exc)}

out_path = "outputs/colab_gnn_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"Wrote {out_path}")
print(json.dumps(results, indent=2, default=str)[:2000])

# One-click download helper
try:
    from google.colab import files as _f
    print("\\nTo download results locally, run:  files.download('outputs/colab_gnn_results.json')")
except ImportError:
    pass
"""

FOOTER_MD = """## Interpreting the output

- **Block 65 (GraphSAGE Training)**: you should see 30 training epochs with
  decreasing loss, converging `pair_auc` ≥ 0.95 on link prediction. The cell
  produces `gnn_embedding_df` — one 16-dim embedding per user — plus
  `influence_score` and KMeans communities.
- **Block 67 (Hybrid GNN Churn Model)**: compares baseline behavioural-only
  vs. hybrid (baseline + GNN) churn classifiers. Expected: hybrid AUC
  improves over baseline by a small but real margin on this dataset.

Both artefacts end up in `outputs/`. The JSON summary lives at
`outputs/colab_gnn_results.json` — download via the left sidebar or the
`files.download()` call printed by the export cell.

If **block 65 hangs** or OOMs on a free Colab runtime, switch to a T4 GPU
runtime (`Runtime → Change runtime type → T4 GPU`) and the training loop
will auto-use it since block 65 was written without a device pin — torch
will default to CPU tensors, so performance is dominated by the sparse
matmul which is already fast on CPU for 5.4k nodes. No changes required.
"""


# ───────────────────────── Assembly ────────────────────────────────────────

def build_block_cell(idx: int, total: int, block: dict) -> dict:
    name = block["name"]
    src_path = DEV / f"{name}.py"
    if not src_path.exists():
        raise SystemExit(f"Missing source file for block {idx}: {src_path}")
    body = src_path.read_text(encoding="utf-8")

    header = (
        f"# ══════════════════════════════════════════════════════════════\n"
        f"# Block {idx}/{total}: {name}\n"
        f"# Source: Development/{name}.py\n"
        f"# ══════════════════════════════════════════════════════════════\n\n"
    )
    return code_cell(header + body)


def build_notebook() -> dict:
    ancestors = load_ordered_ancestors()
    total = len(ancestors)

    cells: list[dict] = [
        md_cell(HEADER_MD),
        md_cell("## 1. Setup"),
        code_cell(SETUP_CODE),
        md_cell("## 2. Upload dataset"),
        code_cell(DATA_CODE),
        md_cell("## 3. Zerve runtime stubs"),
        code_cell(STUBS_CODE),
        md_cell(f"## 4. Run the {total} ancestor blocks in topological order"),
    ]

    for i, block in enumerate(ancestors, 1):
        cells.append(md_cell(f"### 4.{i}  {block['name']}"))
        cells.append(build_block_cell(i, total, block))

    cells.append(md_cell("## 5. Export results"))
    cells.append(code_cell(RESULTS_CODE))
    cells.append(md_cell(FOOTER_MD))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    nb_dict = build_notebook()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Normalize via nbformat so cell IDs are present (required by
    # nbformat_minor >= 5) and the document validates cleanly.
    try:
        import warnings as _w
        import nbformat
        from nbformat import MissingIDFieldWarning
        nb = nbformat.from_dict(nb_dict)
        with _w.catch_warnings():
            _w.simplefilter("ignore", MissingIDFieldWarning)
            _, nb = nbformat.validator.normalize(nb)
            nbformat.validate(nb)
        with OUT.open("w", encoding="utf-8") as f:
            nbformat.write(nb, f)
    except ImportError:
        OUT.write_text(
            json.dumps(nb_dict, indent=1, ensure_ascii=False), encoding="utf-8"
        )

    code_cells = sum(1 for c in nb_dict["cells"] if c["cell_type"] == "code")
    md_cells = sum(1 for c in nb_dict["cells"] if c["cell_type"] == "markdown")
    size_kb = OUT.stat().st_size / 1024
    print(f"Wrote {OUT.relative_to(ROOT)}  ({code_cells} code / {md_cells} md cells, {size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
