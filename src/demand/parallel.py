"""Sequential and parallel runners plus the timing harness.

We use `concurrent.futures.ProcessPoolExecutor` (as in the week 9 lab). Processes, not
threads, because pandas/sklearn work is CPU-bound Python and threads would share one GIL.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Iterable

# Single-threaded maths in every process: otherwise each worker starts its own BLAS/OpenMP
# thread pool, 8 workers x 64 threads oversubscribe the CPUs, and results can depend on the
# number of threads. Set before numpy/sklearn are imported in the worker.
SINGLE_THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def pin_single_thread() -> None:
    """Pool initializer: force single-threaded maths libraries inside each worker."""
    os.environ.update(SINGLE_THREAD_ENV)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass


def run_sequential(func: Callable, items: Iterable) -> list:
    """Plain loop in the current process - the reference result and the T_seq timing."""
    return [func(x) for x in items]


def run_parallel(func: Callable, items: Iterable, workers: int) -> list:
    """Same work spread over `workers` processes.

    `executor.map` returns results in INPUT order, not completion order, so the combined
    result is deterministic (see the non-determinism section for what happens otherwise).
    """
    items = list(items)
    with ProcessPoolExecutor(max_workers=workers, initializer=pin_single_thread) as ex:
        return list(ex.map(func, items, chunksize=1))


def _noop(x: Any) -> Any:
    return x


def pool_startup_seconds(workers: int, repeats: int = 3) -> float:
    """Overhead of creating a pool and running empty tasks (measured part of Amdahl's 'serial' cost)."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run_parallel(_noop, range(workers), workers)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def time_runs(fn: Callable[[], Any], repeats: int, warmup: bool = True) -> tuple[list[float], Any]:
    """Call `fn` once as a discarded warm-up, then `repeats` timed calls (time.perf_counter).

    Returns the list of wall-clock times and the result of the last call (for verification).
    """
    if warmup:
        fn()
    times, result = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return times, result


def speedup_table(bench) -> "pd.DataFrame":
    """Mean/std time per (workload, mode, workers) plus speedup, efficiency and Amdahl's p.

    speedup S_k = T_seq / T_k, efficiency E_k = S_k / k,
    Amdahl: S = 1 / ((1-p) + p/k)  =>  p = (1 - 1/S) / (1 - 1/k)   (k > 1).
    """
    import pandas as pd
    g = (bench.groupby(["workload", "mode", "workers"])["seconds"]
         .agg(["mean", "std", "count"]).reset_index())
    rows = []
    for wl, sub in g.groupby("workload"):
        t_seq = float(sub.loc[sub["mode"] == "sequential", "mean"].iloc[0])
        for _, r in sub.iterrows():
            k, s = int(r["workers"]), t_seq / r["mean"]
            p = (1 - 1 / s) / (1 - 1 / k) if k > 1 else float("nan")
            rows.append({"workload": wl, "mode": r["mode"], "workers": k,
                         "mean_s": r["mean"], "std_s": r["std"], "repeats": int(r["count"]),
                         "speedup": s, "efficiency": s / k, "amdahl_p": p,
                         "max_speedup": 1 / (1 - p) if k > 1 and 0 <= p < 1 else float("nan")})
    return pd.DataFrame(rows)
