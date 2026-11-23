"""Feature extraction as three INDEPENDENT tasks (task parallelism #1).

Each task reads only the hourly series (or its index) and returns a DataFrame on the same
index. No task needs another task's output, so they can run at the same time, and we rejoin
them in the fixed order `config.FEATURE_TASKS`.

Leakage rule: a feature for hour t may only use hours <= t-1. Lags are shift(k>=1) and every
rolling window is computed on y.shift(1) - shift FIRST, then roll.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def lag_features(y: pd.Series) -> pd.DataFrame:
    """Pickups 1, 2, 3, 24, 48 and 168 hours ago."""
    return pd.DataFrame({f"lag_{k}": y.shift(k) for k in config.LAGS}, index=y.index)


def rolling_features(y: pd.Series) -> pd.DataFrame:
    """Mean / std / max of the last 3, 24 and 168 hours, ending at t-1."""
    past = y.shift(1)
    cols = {}
    for w in config.ROLL_WINDOWS:
        roll = past.rolling(w)
        for s in config.ROLL_STATS:
            cols[f"roll_{s}_{w}"] = getattr(roll, s)()
    return pd.DataFrame(cols, index=y.index)


def us_holidays(years: list[int]) -> set:
    """US federal holidays (observed dates) for the given years."""
    try:
        import holidays
        return set(holidays.US(years=years, observed=True).keys())
    except ImportError:   # fallback: the 2026 dates only
        return {pd.Timestamp(d).date() for d in
                ["2026-01-01", "2026-01-19", "2026-02-16", "2026-05-25", "2026-06-19",
                 "2026-07-03", "2026-09-07"]}


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hour and weekday as sin/cos pairs (23:00 is next to 00:00), weekend, month, holiday.

    These are known in advance for any future hour, so they cannot leak.
    """
    hour, dow = index.hour.to_numpy(), index.dayofweek.to_numpy()
    hol = us_holidays(sorted(set(index.year)))
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7), "dow_cos": np.cos(2 * np.pi * dow / 7),
        "is_weekend": (dow >= 5).astype(int),
        "month": index.month.to_numpy(),
        "is_holiday": np.array([d in hol for d in index.date]).astype(int),
    }, index=index)


def run_feature_task(task: tuple[str, pd.Series]) -> tuple[str, pd.DataFrame]:
    """Worker entry point: (task name, series) -> (task name, feature block)."""
    name, y = task
    if name == "lag":
        return name, lag_features(y)
    if name == "rolling":
        return name, rolling_features(y)
    if name == "calendar":
        return name, calendar_features(y.index)
    raise ValueError(name)


def join_feature_blocks(blocks: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Rejoin the task outputs in the FIXED order of config.FEATURE_TASKS / FEATURE_COLUMNS."""
    by_name = dict(blocks)
    out = pd.concat([by_name[n] for n in config.FEATURE_TASKS], axis=1)
    return out[config.FEATURE_COLUMNS]


def feature_tasks(y: pd.Series) -> list[tuple[str, pd.Series]]:
    """The three independent feature tasks, in the fixed order."""
    return [(name, y) for name in config.FEATURE_TASKS]


def model_table(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Features + target, dropping the first hours whose 168-hour history is incomplete.

    We drop rather than impute: inventing a value for 'pickups a week ago' would be a guess.
    """
    return X.assign(y=y).dropna()


def split(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Time-based split - slices by date, never shuffled."""
    return {
        "train": table.loc[:config.TRAIN_END],
        "val": table.loc[config.VAL_START:config.VAL_END],
        "test": table.loc[config.TEST_START:config.TEST_END],
    }
