"""Non-determinism demos: each source has a `broken` version and a `fixed` version.

All demos use the real pipeline (the real `clean_month` on the real month files) except the
random-search demo, which uses the real training data with a deliberately unseeded RNG.
"""
from __future__ import annotations

import hashlib
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from . import config
from .cleaning import clean_month
from .models import cv_folds, fit_predict, metrics
from .parallel import pin_single_thread


def series_hash(s: pd.Series) -> str:
    """Short fingerprint of a series' values AND index order (any reordering changes it)."""
    h = hashlib.sha256(s.to_numpy().tobytes() + s.index.asi8.tobytes())
    return h.hexdigest()[:12]


def clean_month_jittered(path) -> tuple[pd.Series, dict]:
    """The real clean_month plus a random 0-2 s sleep.

    The sleep only exaggerates what happens on a busy shared server anyway (workers finish
    at slightly different times); it makes the completion-order effect visible every run.
    """
    time.sleep(random.Random().uniform(0, 2))   # unseeded on purpose: models server noise
    return clean_month(path)


# ---------------------------------------------------------------- source 1 + 2: completion order
def combine_broken(files, workers: int) -> dict:
    """BROKEN: take results as they complete (as_completed) and combine in that order.

    - series rows are concatenated in completion order       -> source 1 (completion order)
    - fare partial sums are added in completion order (float) -> source 2 (reduction order)
    """
    parts, fare_total, order = [], 0.0, []
    with ProcessPoolExecutor(max_workers=workers, initializer=pin_single_thread) as ex:
        futures = [ex.submit(clean_month_jittered, f) for f in files]
        for fut in as_completed(futures):
            s, st = fut.result()
            parts.append(s)
            fare_total += st["fare_sum"]
            order.append(st["month"])
    y = pd.concat(parts)
    return {"series_hash": series_hash(y), "first_hour": str(y.index[0]),
            "fare_total": fare_total, "fare_hex": fare_total.hex(), "order": order}


def combine_fixed(files, workers: int) -> dict:
    """FIXED: same jittered workers, but results are sorted by shard id (month) before combining,
    and the floating-point reduction is done in that fixed order (plus math.fsum, which is
    exactly rounded and therefore order-independent)."""
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=pin_single_thread) as ex:
        futures = [ex.submit(clean_month_jittered, f) for f in files]
        for fut in as_completed(futures):          # still arrives in random order ...
            results.append(fut.result())
    results.sort(key=lambda r: r[1]["month"])     # ... but the order is fixed before combining
    y = pd.concat([s for s, _ in results])
    fare_total = 0.0
    for _, st in results:
        fare_total += st["fare_sum"]
    fare_fsum = math.fsum(st["fare_sum"] for _, st in results)
    return {"series_hash": series_hash(y), "first_hour": str(y.index[0]),
            "fare_total": fare_total, "fare_hex": fare_total.hex(),
            "fare_fsum_hex": fare_fsum.hex(), "order": [st["month"] for _, st in results]}


def tiny_float_example() -> dict:
    """(a + b) + c != a + (b + c) in floating point."""
    a, b, c = 1e16, -1e16, 1.0
    return {"(a+b)+c": (a + b) + c, "a+(b+c)": a + (b + c)}


# ---------------------------------------------------------------- source 3: unseeded randomness
def _random_search(X: pd.DataFrame, y: pd.Series, rng: np.random.Generator, n_trials: int = 6,
                   seeds: list[int | None] | None = None) -> dict:
    """Random search over HGB settings scored on the last CV fold.

    Each trial draws its hyperparameters from `rng` and uses early stopping, whose internal
    validation split is random unless `random_state` is fixed.
    """
    tr, va = cv_folds(len(X))[-1]
    best = None
    for i in range(n_trials):
        lr = float(10 ** rng.uniform(-2, -0.7))
        leaves = int(rng.integers(8, 64))
        from sklearn.ensemble import HistGradientBoostingRegressor
        est = HistGradientBoostingRegressor(learning_rate=lr, max_leaf_nodes=leaves, max_iter=200,
                                            early_stopping=True, random_state=seeds[i] if seeds else None)
        est.fit(X.iloc[tr], y.iloc[tr])
        rmse = metrics(y.iloc[va], est.predict(X.iloc[va]))["rmse"]
        if best is None or rmse < best["rmse"]:
            best = {"rmse": rmse, "learning_rate": lr, "max_leaf_nodes": leaves}
    return best


def random_search_broken(X, y) -> dict:
    """BROKEN: unseeded generator and unseeded model -> a different 'best' setting every run."""
    return _random_search(X, y, np.random.default_rng())


def random_search_fixed(X, y) -> dict:
    """FIXED: one base seed for the generator and a per-trial seed SEED + trial id for the model."""
    n = 6
    return _random_search(X, y, np.random.default_rng(config.SEED), n_trials=n,
                          seeds=[config.SEED + i for i in range(n)])
