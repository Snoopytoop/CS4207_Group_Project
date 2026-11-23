"""Baselines, models, and the hyperparameter search as task parallelism (#2).

One search TASK = one (model, hyperparameters, target, fold) combination. Each task fits its
own model on its own training block and scores its own validation block: it shares nothing
with any other task, so all tasks can run at the same time in any order.
"""
from __future__ import annotations

import itertools
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .parallel import pin_single_thread


# ---------------------------------------------------------------- metrics
def metrics(y_true, y_pred) -> dict:
    """MAE and RMSE in pickups/hour, WAPE = total absolute error as % of total demand."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    nz = y_true > 0
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "wape_pct": float(100 * np.abs(err).sum() / y_true.sum()),
        "mape_pct": float(100 * np.mean(np.abs(err[nz]) / y_true[nz])),
    }


# ---------------------------------------------------------------- models
def make_model(model: str, params: dict):
    """Ridge needs scaled inputs (penalty treats all coefficients alike); the scaler sits in a
    Pipeline so it is re-fitted on each fold's training rows only - no leakage."""
    if model == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=params["alpha"]))
    if model == "hgb":
        return HistGradientBoostingRegressor(
            learning_rate=params["learning_rate"], max_leaf_nodes=params["max_leaf_nodes"],
            max_iter=params["max_iter"], early_stopping=False, random_state=config.SEED)
    raise ValueError(model)


def fit_predict(model: str, params: dict, target: str,
                X_tr: pd.DataFrame, y_tr: pd.Series, X_te: pd.DataFrame) -> np.ndarray:
    """Fit on (X_tr, y_tr) and predict X_te.

    target='diff168' -> the model learns the CORRECTION to 'same hour last week'
    (y - lag_168), and we add lag_168 back afterwards.
    """
    est = make_model(model, params)
    if target == "level":
        est.fit(X_tr, y_tr)
        return est.predict(X_te)
    est.fit(X_tr, y_tr - X_tr["lag_168"])
    return est.predict(X_te) + X_te["lag_168"].to_numpy()


# ---------------------------------------------------------------- search tasks
def search_settings() -> list[dict]:
    """Every (model, params, target) setting in a fixed order."""
    out = []
    for target in config.TARGETS:
        for a in config.RIDGE_ALPHAS:
            out.append({"model": "ridge", "params": {"alpha": a}, "target": target})
        for lr, leaves, it in itertools.product(config.HGB_LEARNING_RATES,
                                                config.HGB_MAX_LEAF_NODES, config.HGB_MAX_ITER):
            out.append({"model": "hgb", "target": target,
                        "params": {"learning_rate": lr, "max_leaf_nodes": leaves, "max_iter": it}})
    for i, s in enumerate(out):
        s["setting_id"] = i
        s["label"] = f"{s['model']}[{s['target']}] " + ", ".join(f"{k}={v}" for k, v in s["params"].items())
    return out


def search_tasks() -> list[dict]:
    """Settings x folds, each with a unique task_id. The list order is fixed."""
    tasks = []
    for s in search_settings():
        for fold in range(config.N_CV_FOLDS):
            tasks.append({**s, "fold": fold, "task_id": len(tasks)})
    return tasks


def cv_folds(n_rows: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds: each validates on the 4 weeks after its training block."""
    tss = TimeSeriesSplit(n_splits=config.N_CV_FOLDS, test_size=config.CV_TEST_SIZE)
    return list(tss.split(np.arange(n_rows)))


# Training data is sent to each worker ONCE (pool initializer) instead of with every task.
_DATA: dict = {}


def init_search_worker(X: pd.DataFrame, y: pd.Series) -> None:
    pin_single_thread()
    _DATA["X"], _DATA["y"], _DATA["folds"] = X, y, cv_folds(len(X))


def fit_and_score(task: dict) -> dict:
    """One independent search task -> (task_id, rmse, mae, seconds)."""
    t0 = time.perf_counter()
    X, y = _DATA["X"], _DATA["y"]
    tr, va = _DATA["folds"][task["fold"]]
    pred = fit_predict(task["model"], task["params"], task["target"],
                       X.iloc[tr], y.iloc[tr], X.iloc[va])
    m = metrics(y.iloc[va], pred)
    return {"task_id": task["task_id"], "setting_id": task["setting_id"], "fold": task["fold"],
            "rmse": m["rmse"], "mae": m["mae"], "seconds": time.perf_counter() - t0}


def run_search(X: pd.DataFrame, y: pd.Series, tasks: list[dict], workers: int | None) -> pd.DataFrame:
    """workers=None -> sequential loop in this process; otherwise a process pool.

    Results are sorted by task_id, so the table is identical whatever order tasks finished in.
    """
    if workers is None:
        init_search_worker(X, y)
        rows = [fit_and_score(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_search_worker,
                                 initargs=(X, y)) as ex:
            rows = list(ex.map(fit_and_score, tasks, chunksize=1))
    return pd.DataFrame(rows).sort_values("task_id").reset_index(drop=True)


def summarise_search(results: pd.DataFrame) -> pd.DataFrame:
    """Mean validation RMSE/MAE per setting over the folds, best first."""
    settings = pd.DataFrame(search_settings())[["setting_id", "model", "target", "label"]]
    agg = (results.groupby("setting_id")
           .agg(cv_rmse=("rmse", "mean"), cv_mae=("mae", "mean"), task_seconds=("seconds", "sum"))
           .reset_index())
    return settings.merge(agg, on="setting_id").sort_values(["cv_rmse", "setting_id"]).reset_index(drop=True)
