"""Sequential vs parallel timing for the three workloads (run with: sbatch run_benchmark.slurm).

For every workload and worker count: one discarded warm-up run, then N_REPEATS timed runs
with time.perf_counter(). Every run's output is asserted equal to the sequential output -
if a parallel run ever differed, the benchmark would stop with an error.

Writes outputs/benchmarks.csv (every timed run), outputs/benchmark_summary.csv
(mean, std, speedup, efficiency, Amdahl p) and outputs/benchmark_meta.json (hardware, load).
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demand.parallel import SINGLE_THREAD_ENV  # noqa: E402

os.environ.update(SINGLE_THREAD_ENV)

import pandas as pd  # noqa: E402

from demand import cleaning, config, features, models, parallel  # noqa: E402


def shell(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"unavailable: {e}"


def main() -> None:
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    cpus = config.available_cpus()
    workers = [k for k in config.WORKER_COUNTS if k <= cpus]
    files = config.month_files()
    meta = {
        "hostname": platform.node(), "cpu_model": shell("lscpu | grep 'Model name' | cut -d: -f2 | xargs"),
        "cpus_allocated": cpus, "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "cpus_visible": len(os.sched_getaffinity(0)), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "load_average_start": os.getloadavg(), "squeue_jobs_start": shell("squeue -h | wc -l"),
        "python": platform.python_version(), "repeats": config.N_REPEATS, "worker_counts": workers,
        "start": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(json.dumps(meta, indent=1), flush=True)
    meta["pool_startup_s"] = {k: parallel.pool_startup_seconds(k) for k in workers}

    # Reference (sequential) outputs, computed once and used to verify every run.
    ref_y, ref_stats = cleaning.combine_months(parallel.run_sequential(cleaning.clean_month, files))
    ref_X = features.join_feature_blocks(parallel.run_sequential(features.run_feature_task, features.feature_tasks(ref_y)))
    train = features.split(features.model_table(ref_X, ref_y))["train"]
    Xtr, ytr = train[config.FEATURE_COLUMNS], train["y"]
    tasks = models.search_tasks()
    ref_search = models.run_search(Xtr, ytr, tasks, None).drop(columns="seconds")

    def check_cleaning(res):
        y, st = cleaning.combine_months(res)
        assert y.equals(ref_y) and st.equals(ref_stats), "cleaning output differs!"

    def check_features(res):
        assert features.join_feature_blocks(res).equals(ref_X), "features differ!"

    def check_search(res):
        assert res.drop(columns="seconds").equals(ref_search), "search output differs!"

    workloads = {
        "cleaning": (lambda: parallel.run_sequential(cleaning.clean_month, files),
                     lambda k: parallel.run_parallel(cleaning.clean_month, files, k), check_cleaning),
        "features": (lambda: parallel.run_sequential(features.run_feature_task, features.feature_tasks(ref_y)),
                     lambda k: parallel.run_parallel(features.run_feature_task, features.feature_tasks(ref_y), k),
                     check_features),
        "search": (lambda: models.run_search(Xtr, ytr, tasks, None),
                   lambda k: models.run_search(Xtr, ytr, tasks, k), check_search),
    }

    rows = []
    for name, (seq_fn, par_fn, check) in workloads.items():
        runs = [("sequential", 1, seq_fn)] + [("parallel", k, (lambda k=k: par_fn(k))) for k in workers]
        for mode, k, fn in runs:
            times, result = parallel.time_runs(fn, config.N_REPEATS, warmup=True)
            check(result)   # the benchmark is only valid if the output is identical
            for i, t in enumerate(times):
                rows.append({"workload": name, "mode": mode, "workers": k, "repeat": i + 1, "seconds": t})
            print(f"{name:9s} {mode:10s} k={k}: mean {sum(times)/len(times):7.2f}s  verified equal", flush=True)
            pd.DataFrame(rows).to_csv(config.OUTPUT_DIR / "benchmarks.csv", index=False)

    bench = pd.DataFrame(rows)
    summary = parallel.speedup_table(bench)
    summary.to_csv(config.OUTPUT_DIR / "benchmark_summary.csv", index=False)
    meta.update({"end": time.strftime("%Y-%m-%d %H:%M:%S"), "load_average_end": os.getloadavg(),
                 "squeue_jobs_end": shell("squeue -h | wc -l")})
    (config.OUTPUT_DIR / "benchmark_meta.json").write_text(json.dumps(meta, indent=1))
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
