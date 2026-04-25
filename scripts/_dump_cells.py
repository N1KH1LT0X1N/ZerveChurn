"""Throwaway: dump outputs of target GNN cells. Delete after use."""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "gnn_colab_verification.ipynb"
OUT = Path(__file__).resolve().parents[1] / "outputs" / "_colab_gnn_cells_dump.txt"

nb = json.loads(NB.read_text(encoding="utf-8"))
targets = ["Block 17/19", "Block 18/19", "Block 19/19", "Export GNN verification"]

lines = []
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if not any(t in src for t in targets):
        continue
    head = src.splitlines()[0] if src else ""
    lines.append("=" * 100)
    lines.append(f"CELL {i}  type={c['cell_type']}  ec={c.get('execution_count')}  head={head[:120]}")
    lines.append("-- OUTPUTS --")
    for o in c.get("outputs", []):
        ot = o.get("output_type")
        if ot == "stream":
            lines.append(f"[stream:{o.get('name')}]")
            lines.append("".join(o.get("text", [])))
        elif ot in ("execute_result", "display_data"):
            data = o.get("data", {})
            if "text/plain" in data:
                lines.append(f"[{ot}:text/plain]")
                lines.append("".join(data["text/plain"]))
            other = [k for k in data.keys() if k != "text/plain"]
            if other:
                lines.append(f"[{ot} other mimetypes: {other}]")
        elif ot == "error":
            lines.append(f"[error] {o.get('ename')}: {o.get('evalue')}")
            lines.append("\n".join(o.get("traceback", [])))
        else:
            lines.append(f"[unknown output_type={ot}]")
    lines.append("")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")
