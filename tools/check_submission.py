"""Pre-submission checker for the Group 11 notebook.

Usage:  python tools/check_submission.py path/to/demand_forecast.ipynb [path/to/project_dir]

Checks (standard library only):
  * notebook was executed: every code cell has an execution count, in order
  * no error outputs
  * the rubric section headings are present
  * nothing that looks like a password / private key is in the notebook or project files
  * outputs/results.json exists and contains the key fields
  * requirements.txt exists
"""
import json
import os
import re
import sys
from pathlib import Path

REQUIRED_HEADINGS = [
    "configuration", "clean", "data parallelism", "split", "feature",
    "hyperparameter", "evaluation", "benchmark", "amdahl", "non-determinism",
    "reproducib",
]
SECRET_PATTERNS = [
    re.compile(r"password\s*[:=]", re.I),
    re.compile(r"BEGIN (OPENSSH|RSA|EC) PRIVATE KEY"),
]
# Optionally set env var GROUP_SECRET to the real password before running, so it is searched for too
# (it is never printed or stored).
if os.environ.get("GROUP_SECRET"):
    SECRET_PATTERNS.append(re.compile(re.escape(os.environ["GROUP_SECRET"])))
RESULT_KEYS = ["rows_raw", "rows_kept", "test_metrics", "speedup"]
SKIP_DIRS = {"venv", ".venv", ".git", "__pycache__", ".ipynb_checkpoints"}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    nb_path = Path(sys.argv[1])
    project = Path(sys.argv[2]) if len(sys.argv) > 2 else nb_path.parent
    problems, warnings = [], []

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    code = [c for c in cells if c.get("cell_type") == "code"]
    counts = [c.get("execution_count") for c in code]
    if any(c is None for c in counts):
        problems.append(f"{sum(c is None for c in counts)} code cell(s) never executed")
    elif counts != sorted(counts) or (counts and counts[0] != 1):
        warnings.append("execution counts not 1..N in order - was it run with Restart & Run All?")

    for i, c in enumerate(code):
        for out in c.get("outputs", []):
            if out.get("output_type") == "error":
                problems.append(f"code cell {i} has an error output: {out.get('ename')}")

    md = "\n".join("".join(c.get("source", "")) for c in cells if c.get("cell_type") == "markdown").lower()
    headings = "\n".join(l for l in md.splitlines() if l.lstrip().startswith("#"))
    for h in REQUIRED_HEADINGS:
        if h not in headings:
            warnings.append(f"no markdown heading containing '{h}'")

    files = [nb_path] + [p for p in project.rglob("*")
                         if p.is_file() and not SKIP_DIRS.intersection(p.parts)
                         and p.suffix in {".py", ".ipynb", ".md", ".txt", ".json", ".slurm", ".cfg", ".toml", ".yml", ".yaml"}]
    for f in set(files):
        if f.name == "check_submission.py":
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        for pat in SECRET_PATTERNS:
            if pat.search(text):
                problems.append(f"possible secret in {f} ")

    big = [p for p in project.rglob("*.parquet") if not SKIP_DIRS.intersection(p.parts)]
    if big:
        problems.append(f"data files found in project (don't submit/commit them): {big[:3]}")

    res = project / "outputs" / "results.json"
    if not res.exists():
        problems.append("outputs/results.json missing (poster numbers must come from it)")
    else:
        data = json.loads(res.read_text(encoding="utf-8"))
        flat = json.dumps(data).lower()
        for k in RESULT_KEYS:
            if k not in flat:
                warnings.append(f"results.json has no field like '{k}'")

    if not (project / "requirements.txt").exists():
        problems.append("requirements.txt missing")

    for w in warnings:
        print("WARNING:", w)
    for p in problems:
        print("PROBLEM:", p)
    print("\nRESULT:", "FAIL" if problems else "PASS", f"({len(problems)} problems, {len(warnings)} warnings)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
