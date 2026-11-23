"""Compare two results.json files, ignoring fields that are SUPPOSED to change between runs.

Usage: python scripts/compare_results.py outputs/results_run1.json outputs/results.json
Ignored: timings of this run, the deliberately broken non-determinism demos, and the Slurm job id.
"""
import json
import sys

IGNORE = {"timing_this_run", "broken_evidence_this_run", "slurm_job_id", "task_seconds_by_model",
          "shard_seconds_min", "shard_seconds_max", "peak_rss_mb_max"}


def strip(obj):
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if k not in IGNORE}
    if isinstance(obj, list):
        return [strip(v) for v in obj]
    return obj


def diff(a, b, path=""):
    if type(a) is not type(b):
        yield f"{path}: {a!r} != {b!r}"
    elif isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                yield f"{path}/{k}: only in one file"
            else:
                yield from diff(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            yield from diff(x, y, f"{path}[{i}]")
    elif a != b and not (isinstance(a, float) and a != a and b != b):   # NaN == NaN here
        yield f"{path}: {a!r} != {b!r}"


if __name__ == "__main__":
    a, b = (strip(json.load(open(p))) for p in sys.argv[1:3])
    problems = list(diff(a, b))
    print("\n".join(problems) if problems else "IDENTICAL (apart from timings and the deliberately broken demos)")
    sys.exit(1 if problems else 0)
