"""Render a Mermaid DAG from a Zerve canvas.yaml.

Usage:
    python scripts/render_canvas_dag.py [canvas.yaml] [output.md]

Defaults to ./canvas.yaml -> ./canvas_dag.md.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml


# Block type -> shape + label prefix
TYPE_STYLE = {
    1: ("[", "]"),       # code/compute -> rectangle
    4: ("([", "])"),     # markdown/note -> stadium
    9: ("{{", "}}"),     # LLM/agent -> hexagon
}

STAGE_RULES = [
    ("ingest",     ["load", "example dataset", "data exploration", "statistical_profiling",
                      "statistical summaries", "data_quality", "data exploration",
                      "temporal patterns", "additional exploratory", "user activity",
                      "user type", "date_range", "field-level"]),
    ("semantics",  ["taxonomy", "workflow stage mapping", "hierarchical event"]),
    ("segment",    ["segmentation", "feature adoption", "interactive visualizations",
                      "lifecycle stage", "growth trajectory"]),
    ("behavior",   ["session pattern", "workflow sequence", "collaboration signature",
                      "isolation forest", "feature engineering", "cohort behavioral",
                      "engagement momentum", "engagement forecast"]),
    ("modeling",   ["primary success", "composite success", "validation & business",
                      "data_prep_train", "base_models", "survival", "kaplan",
                      "churn risk", "ltv", "early warning", "behavioral economics"]),
    ("graph",      ["network & centrality", "community detection", "gnn ",
                      "graphsage", "hybrid gnn"]),
    ("report",     ["shap", "causal", "dashboard", "delta metrics", "executive",
                      "save report", "export report", "summary report",
                      "social media", "readme", "quality assurance", "key_findings",
                      "comprehensive analysis report", "user intelligence export",
                      "advanced analysis synthesis", "git repository"]),
]

STAGE_ORDER = ["ingest", "semantics", "segment", "behavior", "modeling", "graph", "report", "other"]
STAGE_TITLE = {
    "ingest":    "Ingestion & EDA",
    "semantics": "Event Semantics",
    "segment":   "Segmentation",
    "behavior":  "Behavioral Features",
    "modeling":  "Modeling / Churn / LTV",
    "graph":     "Graph / GNN",
    "report":    "Reporting & Export",
    "other":     "Other",
}


def classify(name: str) -> str:
    low = name.lower()
    for stage, kws in STAGE_RULES:
        for kw in kws:
            if kw in low:
                return stage
    return "other"


def short_id(uid: str) -> str:
    # Mermaid node ids must be safe; take first 8 hex chars prefixed with n.
    return "n" + uid.replace("-", "")[:8]


def escape_label(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    # Mermaid-safe: avoid quotes/brackets/pipes inside label.
    text = text.replace('"', "'").replace("|", "/").replace("[", "(").replace("]", ")")
    if len(text) > 60:
        text = text[:57] + "..."
    return text


def render(canvas_path: Path, out_path: Path) -> None:
    data = yaml.safe_load(canvas_path.read_text(encoding="utf-8"))
    layer = data["layers"][0]
    blocks = layer["blocks"]
    edges = layer.get("edges", [])

    # Group blocks by stage
    grouped: dict[str, list[dict]] = {s: [] for s in STAGE_ORDER}
    for b in blocks:
        grouped[classify(b["name"])].append(b)

    lines: list[str] = []
    lines.append(f"# {data['name']} - Canvas DAG\n")
    lines.append(f"- Blocks: **{len(blocks)}**")
    lines.append(f"- Edges: **{len(edges)}**")
    lines.append(f"- Source: `{canvas_path.name}`\n")
    lines.append("```mermaid")
    lines.append("flowchart LR")
    lines.append("  classDef code fill:#e6f2ff,stroke:#3b82f6,color:#0b2545;")
    lines.append("  classDef note fill:#fff7d6,stroke:#d4a017,color:#4a3a00;")
    lines.append("  classDef llm  fill:#f3e8ff,stroke:#8b5cf6,color:#2e1065;")

    # Subgraphs per stage
    for stage in STAGE_ORDER:
        bs = grouped[stage]
        if not bs:
            continue
        lines.append(f'  subgraph {stage}["{STAGE_TITLE[stage]}"]')
        lines.append("    direction TB")
        for b in sorted(bs, key=lambda x: (x.get("y", 0), x.get("x", 0))):
            nid = short_id(b["id"])
            open_br, close_br = TYPE_STYLE.get(b.get("type", 1), ("[", "]"))
            label = escape_label(b["name"])
            lines.append(f'    {nid}{open_br}"{label}"{close_br}')
        lines.append("  end")

    # Class assignments
    code_ids, note_ids, llm_ids = [], [], []
    for b in blocks:
        nid = short_id(b["id"])
        t = b.get("type", 1)
        (llm_ids if t == 9 else note_ids if t == 4 else code_ids).append(nid)
    if code_ids:
        lines.append("  class " + ",".join(code_ids) + " code;")
    if note_ids:
        lines.append("  class " + ",".join(note_ids) + " note;")
    if llm_ids:
        lines.append("  class " + ",".join(llm_ids) + " llm;")

    # Edges
    known = {b["id"] for b in blocks}
    for e in edges:
        s, t = e["source"], e["target"]
        if s in known and t in known:
            lines.append(f"  {short_id(s)} --> {short_id(t)}")

    lines.append("```\n")

    # Legend
    lines.append("## Legend\n")
    lines.append("- Rectangle = code/compute block (`type: 1`)")
    lines.append("- Stadium = markdown/note block (`type: 4`)")
    lines.append("- Hexagon = LLM/agent block (`type: 9`)\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path} ({len(blocks)} blocks, {len(edges)} edges)")


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("canvas.yaml")
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("canvas_dag.md")
    render(src, dst)
