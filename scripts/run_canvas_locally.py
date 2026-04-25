"""Locally execute a Zerve canvas export.

Parses ``canvas.yaml``, topo-sorts blocks by edges, and executes each
``Development/<name>.py`` in a single shared globals namespace — mimicking
Zerve's per-canvas variable scope, where every block sees the variables,
functions and dataframes produced by its upstream blocks.

What it does for you:

- Skips non-code blocks (markdown ``type=4``, GenAI prompt ``type=9``).
- Skips duplicate ``(Copy)`` blocks by default.
- Forces matplotlib into a non-GUI ``Agg`` backend and redirects every
  ``plt.show()`` / figure creation into ``outputs/<block_name>/fig_<n>.png``.
- Stubs Zerve-only helpers (``spread``/``gather``) so Fleet blocks do not
  explode when run outside the platform.
- Changes CWD to the project root so blocks reading ``user_retention.parquet``
  et al. work as-is.
- Optional ``--checkpoint`` pickles the shared namespace after every block
  and can resume from the last successful block.
- ``--from`` / ``--to`` / ``--only`` / ``--skip`` for surgical re-runs.

Usage examples::

    # Dry-run the plan
    python scripts/run_canvas_locally.py --list

    # Full run with figure capture + checkpoints
    python scripts/run_canvas_locally.py --checkpoint

    # Resume after a failure
    python scripts/run_canvas_locally.py --checkpoint --resume

    # Run only EDA blocks
    python scripts/run_canvas_locally.py --to "Comprehensive Feature Engineering"
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import pickle
import sys
import time
import traceback
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import yaml

# ── Force UTF-8 stdio so blocks that print "✓ → 📊 ✅ ⚠" don't crash on
# Windows consoles whose default code page is cp1252. Equivalent to
# launching with PYTHONUTF8=1 but baked into the runner so users don't
# need to remember the env-var (Blocker A in docs/repo_state_and_next_steps.md).
try:
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
except Exception:  # pragma: no cover - non-fatal; fall back to original stdio
    pass

ROOT = Path(__file__).resolve().parent.parent

# Reporting blocks that have no inbound edges in canvas.yaml but consume
# variables produced upstream (or whose only role is to dump artifacts to
# disk). They get scheduled at the *start* of a topo-sort because they have
# zero indegree, then fail because their dependencies haven't run. Defer
# them to the end of the plan so they run after every real producer.
# (Blocker F in docs/repo_state_and_next_steps.md)
LATE_REPORTING_BLOCKS = {
    "Save Report to File",
    "Export Report as Markdown File",
    "Social Media Post Drafts",
    "Executive Summary Report Generation",
    "User Intelligence Export",
    "Git Repository Setup",
}


# ─────────────────────────── Canvas parsing ────────────────────────────────

def load_canvas(canvas_path: Path) -> dict:
    with canvas_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def topo_sort(blocks: list[dict], edges: list[dict]) -> list[dict]:
    """Kahn's algorithm over block ids; stable by (y, x) to match canvas
    reading order when multiple blocks are ready."""
    by_id = {b["id"]: b for b in blocks}
    indeg: dict[str, int] = {b["id"]: 0 for b in blocks}
    adj: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        s, t = e["source"], e["target"]
        if s in by_id and t in by_id:
            adj[s].append(t)
            indeg[t] += 1

    def sort_key(bid: str) -> tuple[float, float, str]:
        b = by_id[bid]
        return (b.get("y", 0), b.get("x", 0), b["name"])

    ready: deque[str] = deque(sorted([bid for bid, d in indeg.items() if d == 0], key=sort_key))
    out: list[dict] = []
    while ready:
        bid = ready.popleft()
        out.append(by_id[bid])
        for nxt in sorted(adj[bid], key=sort_key):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
    if len(out) != len(blocks):
        missing = set(by_id) - {b["id"] for b in out}
        raise RuntimeError(f"Cycle or unreachable blocks detected: {missing}")
    return out


# ─────────────────────────── Matplotlib plumbing ───────────────────────────

def install_mpl_capture(outputs_dir: Path):
    """Route every figure into outputs/<block>/fig_N.png instead of popping
    a GUI window. ``set_current_block()`` swaps the target subdirectory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    state = {"block": "__init__", "count": 0}

    def _dest() -> Path:
        d = outputs_dir / state["block"]
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save_all():
        for num in plt.get_fignums():
            fig = plt.figure(num)
            state["count"] += 1
            path = _dest() / f"fig_{state['count']:02d}.png"
            try:
                fig.savefig(path, dpi=120, bbox_inches="tight")
            except Exception as exc:  # pragma: no cover - best effort
                print(f"    [mpl] could not save {path}: {exc}", file=sys.stderr)
        plt.close("all")

    _orig_show = plt.show
    plt.show = lambda *a, **k: _save_all()  # type: ignore[assignment]

    def set_current_block(name: str):
        # Save anything that leaked out of the previous block first.
        _save_all()
        state["block"] = _safe_name(name)
        state["count"] = 0

    def flush():
        _save_all()
        plt.show = _orig_show  # type: ignore[assignment]

    return set_current_block, flush


def _safe_name(name: str) -> str:
    keep = "-_.() "
    return "".join(c if c.isalnum() or c in keep else "_" for c in name).strip()


# ─────────────────────────── Zerve helper stubs ────────────────────────────

def zerve_stubs() -> dict[str, Any]:
    """Minimal stand-ins for Zerve's canvas-runtime helpers."""
    _fleet_state: dict[str, Any] = {"last_spread": None, "results": []}

    def spread(items):
        # In Zerve this fans out N parallel blocks. Locally we just keep the
        # list; the downstream Aggregator's `gather()` returns it unchanged.
        _fleet_state["last_spread"] = list(items)
        print(f"    [stub] spread() captured {len(items)} items (not parallelised)")
        return list(items)

    def gather():
        items = _fleet_state.get("last_spread") or []
        print(f"    [stub] gather() -> {len(items)} items")
        return items

    def attach_variable(*_a, **_kw):
        return None

    return {"spread": spread, "gather": gather, "attach_variable": attach_variable}


# ─────────────────────────── Block execution ───────────────────────────────

CODE_TYPE = 1        # Python / R / SQL compute block
MARKDOWN_TYPE = 4    # Notes
LLM_TYPE = 9         # GenAI prompt (Bedrock)


def should_skip_block(b: dict, skip_copies: bool) -> tuple[bool, str]:
    if b.get("type") != CODE_TYPE:
        return True, f"type={b.get('type')} (non-code)"
    if skip_copies and "(Copy)" in b["name"]:
        return True, "duplicate (Copy)"
    return False, ""


def find_block_file(dev_dir: Path, block_name: str) -> Path | None:
    # Exported filename matches block name + .py for Python blocks.
    candidate = dev_dir / f"{block_name}.py"
    if candidate.exists():
        return candidate
    # Some names contain characters the filesystem rewrote — best-effort
    # fallback.
    for p in dev_dir.glob("*.py"):
        if p.stem == block_name:
            return p
    return None


def run_block(block: dict, code_path: Path, ns: dict[str, Any]) -> None:
    code = code_path.read_text(encoding="utf-8")
    compiled = compile(code, str(code_path), "exec")
    ns["__file__"] = str(code_path)
    ns["__name__"] = "__zerve_block__"
    exec(compiled, ns)


# ─────────────────────────── Orchestration ─────────────────────────────────

def build_plan(
    canvas: dict,
    only: set[str],
    skip: set[str],
    from_name: str | None,
    to_name: str | None,
    skip_copies: bool,
) -> list[dict]:
    blocks = canvas["layers"][0]["blocks"]
    edges = canvas["layers"][0].get("edges", [])
    ordered = topo_sort(blocks, edges)

    names_in_order = [b["name"] for b in ordered]
    start_idx = names_in_order.index(from_name) if from_name else 0
    end_idx = names_in_order.index(to_name) + 1 if to_name else len(ordered)
    window = ordered[start_idx:end_idx]

    plan = []
    deferred = []
    for b in window:
        if only and b["name"] not in only:
            continue
        if b["name"] in skip:
            continue
        skip_it, reason = should_skip_block(b, skip_copies)
        entry = {"block": b, "skip_reason": reason if skip_it else None}
        # Defer orphan reporting blocks to the end so they execute after their
        # logical (un-edged) producers. See LATE_REPORTING_BLOCKS comment.
        if not skip_it and b["name"] in LATE_REPORTING_BLOCKS:
            deferred.append(entry)
        else:
            plan.append(entry)
    plan.extend(deferred)
    return plan


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a Zerve canvas export locally.")
    ap.add_argument("--canvas", default=str(ROOT / "canvas.yaml"))
    ap.add_argument("--dev-dir", default=str(ROOT / "Development"))
    ap.add_argument("--outputs", default=str(ROOT / "outputs"))
    ap.add_argument("--checkpoint-dir", default=str(ROOT / "outputs" / "_state"))
    ap.add_argument("--from", dest="from_name")
    ap.add_argument("--to", dest="to_name")
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--skip", action="append", default=[])
    ap.add_argument("--include-copies", action="store_true", help="Do not skip '(Copy)' blocks")
    ap.add_argument("--list", action="store_true", help="Print execution plan and exit")
    ap.add_argument("--dry-run", action="store_true", help="Skip execution, keep everything else")
    ap.add_argument("--checkpoint", action="store_true", help="Pickle namespace after each block")
    ap.add_argument("--resume", action="store_true", help="Load last checkpoint before running")
    ap.add_argument("--stop-on-error", action="store_true", default=True)
    ap.add_argument("--continue-on-error", action="store_true",
                    help="Log errors and keep going instead of stopping")
    args = ap.parse_args()

    if args.continue_on_error:
        args.stop_on_error = False

    canvas_path = Path(args.canvas)
    dev_dir = Path(args.dev_dir)
    outputs_dir = Path(args.outputs)
    ckpt_dir = Path(args.checkpoint_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    canvas = load_canvas(canvas_path)
    plan = build_plan(
        canvas,
        only=set(args.only),
        skip=set(args.skip),
        from_name=args.from_name,
        to_name=args.to_name,
        skip_copies=not args.include_copies,
    )

    # ── Listing mode ──
    if args.list:
        print(f"# Execution plan ({len(plan)} entries)\n")
        for i, item in enumerate(plan, 1):
            b = item["block"]
            status = f"SKIP ({item['skip_reason']})" if item["skip_reason"] else "RUN"
            print(f"{i:3d}. [{status:<30}] {b['name']}")
        return 0

    # ── Namespace + stubs ──
    ns: dict[str, Any] = {}
    ns.update(zerve_stubs())

    if args.resume and args.checkpoint:
        last = ckpt_dir / "_latest.pkl"
        if last.exists():
            try:
                with last.open("rb") as f:
                    ns.update(pickle.load(f))
                print(f"[resume] loaded {len(ns)} names from {last}")
            except Exception as exc:
                print(f"[resume] failed to load checkpoint: {exc}", file=sys.stderr)

    set_block, flush_mpl = install_mpl_capture(outputs_dir)

    # Zerve blocks assume cwd == project root.
    prev_cwd = Path.cwd()
    os.chdir(ROOT)

    failures: list[tuple[str, str]] = []
    started = time.time()
    try:
        for i, item in enumerate(plan, 1):
            b = item["block"]
            name = b["name"]
            if item["skip_reason"]:
                print(f"[{i:3d}/{len(plan)}] SKIP  {name}  ({item['skip_reason']})")
                continue

            code_path = find_block_file(dev_dir, name)
            if code_path is None:
                print(f"[{i:3d}/{len(plan)}] MISS  {name}  (no .py in Development/)")
                failures.append((name, "no .py file"))
                if args.stop_on_error:
                    break
                continue

            set_block(name)
            print(f"[{i:3d}/{len(plan)}] RUN   {name}")
            t0 = time.time()
            if args.dry_run:
                print("      (dry-run)")
                continue

            try:
                run_block(b, code_path, ns)
                dt = time.time() - t0
                new_names = [k for k in ns if not k.startswith("_")]
                print(f"      ok ({dt:.2f}s, namespace={len(new_names)} names)")
            except Exception as exc:  # noqa: BLE001 - we want to see anything
                dt = time.time() - t0
                print(f"      FAIL ({dt:.2f}s): {exc.__class__.__name__}: {exc}")
                traceback.print_exc(limit=3)
                failures.append((name, f"{exc.__class__.__name__}: {exc}"))
                if args.stop_on_error:
                    break

            if args.checkpoint:
                _write_checkpoint(ckpt_dir, name, ns)
    finally:
        flush_mpl()
        os.chdir(prev_cwd)

    total = time.time() - started
    print("\n── Summary ──")
    print(f"  total time: {total:.1f}s")
    print(f"  failures  : {len(failures)}")
    for name, reason in failures:
        print(f"    - {name}: {reason}")
    return 0 if not failures else 1


def _write_checkpoint(ckpt_dir: Path, block_name: str, ns: dict[str, Any]) -> None:
    """Pickle only the picklable names. Silent-skip anything that resists."""
    picklable: dict[str, Any] = {}
    for k, v in ns.items():
        if k.startswith("__") or callable(v) and getattr(v, "__module__", "") == "builtins":
            continue
        try:
            pickle.dumps(v)
        except Exception:
            continue
        picklable[k] = v
    safe = _safe_name(block_name)
    path = ckpt_dir / f"{safe}.pkl"
    try:
        with path.open("wb") as f:
            pickle.dump(picklable, f)
        latest = ckpt_dir / "_latest.pkl"
        with latest.open("wb") as f:
            pickle.dump(picklable, f)
    except Exception as exc:
        print(f"      [checkpoint] skipped: {exc}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
