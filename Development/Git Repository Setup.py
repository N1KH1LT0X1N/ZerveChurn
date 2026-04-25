import subprocess
import os

# ── 1. Initialise Git repo in the current working directory ──────────────────
cwd = os.getcwd()
print(f"Working directory: {cwd}")

init_result = subprocess.run(
    ["git", "init"],
    capture_output=True, text=True, cwd=cwd
)
print(init_result.stdout.strip())
if init_result.returncode != 0:
    print("STDERR:", init_result.stderr.strip())

# ── 2. Write a comprehensive Python / data-science .gitignore ────────────────
gitignore_content = """# ── Byte-compiled / cache ────────────────────────────────────────────────────
__pycache__/
*.py[cod]
*$py.class
*.pyo

# ── Distribution / packaging ─────────────────────────────────────────────────
.Python
build/
dist/
*.egg-info/
.eggs/
*.egg

# ── Virtual environments ─────────────────────────────────────────────────────
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# ── Secrets & credentials ────────────────────────────────────────────────────
.env
.env.*
secrets.yaml
secrets.json
credentials.json
*.pem
*.key

# ── Jupyter Notebooks ────────────────────────────────────────────────────────
.ipynb_checkpoints/
*.ipynb

# ── Data files (large / generated) ───────────────────────────────────────────
*.csv
*.parquet
*.pkl
*.pickle
*.h5
*.hdf5
*.feather
*.arrow
*.db
*.sqlite
*.sqlite3
*.json
*.ndjson
*.tsv

# ── ML / model artefacts ─────────────────────────────────────────────────────
*.model
*.bin
*.pt
*.pth
*.onnx
*.pb
*.tflite
saved_model/
checkpoints/
mlruns/
wandb/

# ── Reports & exports ────────────────────────────────────────────────────────
*.html
*.pdf
reports/
outputs/

# ── OS & editor noise ────────────────────────────────────────────────────────
.DS_Store
Thumbs.db
.idea/
.vscode/
*.swp
*.swo
*~

# ── Logs ─────────────────────────────────────────────────────────────────────
*.log
logs/
"""

gitignore_path = os.path.join(cwd, ".gitignore")
with open(gitignore_path, "w") as f:
    f.write(gitignore_content.strip() + "\n")

print(f"\n.gitignore written to: {gitignore_path}")
print(f"  → {len([l for l in gitignore_content.strip().splitlines() if l and not l.startswith('#')])} active ignore patterns")

# ── 3. Display git status to confirm setup ───────────────────────────────────
status_result = subprocess.run(
    ["git", "status"],
    capture_output=True, text=True, cwd=cwd
)
print("\n── git status ───────────────────────────────────────────────────────────")
print(status_result.stdout.strip())
if status_result.returncode != 0:
    print("STDERR:", status_result.stderr.strip())

print("\n✅  Git repository initialised with Python/data-science .gitignore.")
