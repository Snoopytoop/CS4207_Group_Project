"""Cleaning one month of trips and turning it into an hourly pickup-count series.

`clean_month` is THE function that data parallelism applies to every shard (= month file).
It is a pure function of its input file: same file in -> same output out, whichever
process runs it. That is what makes the sequential and parallel results identical.
"""
from __future__ import annotations

import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .io import read_month

# Rule names in the order they are applied (also the row order of the cleaning table).
RULES = [
    "pickup_outside_month",
    "duration_out_of_range",
    "distance_out_of_range",
    "fare_not_positive",
    "duplicate_rows",
]


def apply_rules(df: pd.DataFrame, month: str) -> tuple[pd.DataFrame, dict]:
    """Apply the cleaning rules in a fixed order, counting how many rows each rule removes."""
    removed: dict[str, int] = {}
    start = pd.Timestamp(f"{month}-01")
    end = start + pd.offsets.MonthBegin(1)

    def drop(mask: pd.Series, rule: str) -> pd.DataFrame:
        removed[rule] = int(mask.sum())
        return df.loc[~mask]

    # 1. Spill-over from the previous/next month and bad clocks (e.g. a pickup dated 2001).
    pick = df["tpep_pickup_datetime"]
    df = drop((pick < start) | (pick >= end), "pickup_outside_month")

    # 2. Dropoff before pickup, under 1 minute or over 3 hours.
    minutes = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
    df = drop((minutes < config.MIN_DURATION_MIN) | (minutes > config.MAX_DURATION_MIN),
              "duration_out_of_range")

    # 3. Zero distance or meter errors (hundreds of thousands of miles).
    dist = df["trip_distance"]
    df = drop((dist <= 0) | (dist > config.MAX_DISTANCE_MILES), "distance_out_of_range")

    # 4. Refunds, voids and disputes.
    df = drop(df["fare_amount"] <= config.MIN_FARE, "fare_not_positive")

    # 5. Exact duplicates on every column we read.
    df = drop(df.duplicated(), "duplicate_rows")
    return df, removed


def encode_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Zone and rate codes are labels, not quantities, so store them as categories.

    Missing rate codes (Flex Fare trips) and code 99 both become an explicit 'unknown' level.
    """
    rate = df["RatecodeID"].astype("Int64").astype("string")
    rate = rate.where(df["RatecodeID"].notna() & (df["RatecodeID"] != config.UNKNOWN_RATECODE),
                      "unknown")
    return df.assign(
        PULocationID=df["PULocationID"].astype("category"),
        DOLocationID=df["DOLocationID"].astype("category"),
        RatecodeID=rate.astype("category"),
    )


def hourly_counts(pickups: pd.Series, month: str) -> tuple[pd.Series, int]:
    """Count pickups per clock hour and re-index to EVERY hour of the month.

    An hour with no recorded pickups is genuinely zero demand, so missing hours are filled
    with 0 (and counted). TLC timestamps are naive local time, so the re-index is on a naive
    hourly grid: 02:00 on the spring-forward day does not exist on the clock and shows as 0.
    """
    start = pd.Timestamp(f"{month}-01")
    full = pd.date_range(start, start + pd.offsets.MonthBegin(1), freq="h", inclusive="left")
    counts = pickups.dt.floor("h").value_counts().sort_index()
    missing = int(len(full.difference(counts.index)))
    series = counts.reindex(full, fill_value=0).astype("int64")
    series.index.name, series.name = "hour", "pickups"
    return series, missing


def clean_month(path: Path | str) -> tuple[pd.Series, dict]:
    """Read, clean and aggregate ONE month file. Returns (hourly pickup counts, statistics).

    The statistics dict contains only deterministic numbers (no timings), so the sequential
    and parallel runs can be compared with `==`.
    """
    month = config.month_of(path)
    raw = read_month(Path(path))
    df, removed = apply_rules(raw, month)
    df = encode_categories(df)
    series, missing = hourly_counts(df["tpep_pickup_datetime"], month)
    stats = {
        "month": month,
        "raw_rows": len(raw),
        **{f"removed_{r}": removed[r] for r in RULES},
        "kept_rows": len(df),
        "flex_fare_kept": int((df["payment_type"] == config.FLEX_FARE_PAYMENT_TYPE).sum()),
        "ratecode_unknown_kept": int((df["RatecodeID"] == "unknown").sum()),
        "missing_hours": missing,
        "fare_sum": float(df["fare_amount"].sum()),   # partial sum, used by the reduction demo
    }
    return series, stats


def clean_month_profiled(path: Path | str) -> dict:
    """Run `clean_month` and report its wall time and the worker's peak memory (for Criterion 3)."""
    t0 = time.perf_counter()
    _, stats = clean_month(path)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024   # Linux reports KiB
    return {"month": stats["month"], "seconds": time.perf_counter() - t0, "peak_rss_mb": peak_mb}


def hourly_counts_streaming(path: Path | str) -> pd.Series:
    """Same hourly counts as `clean_month`, but streamed with `iter_batches` (memory-lean).

    Rules 1-4 are row-local, so they can be applied batch by batch and the counts added up.
    The duplicate rule needs the whole month, so it is skipped here; the notebook checks
    that the result equals `clean_month` whenever a month has no duplicates.
    """
    from .io import iter_month_batches
    month = config.month_of(path)
    total = None
    for batch in iter_month_batches(Path(path)):
        df, _ = apply_rules(batch, month)
        part, _ = hourly_counts(df["tpep_pickup_datetime"], month)
        total = part if total is None else total + part
    return total


def combine_months(results: list[tuple[pd.Series, dict]]) -> tuple[pd.Series, pd.DataFrame]:
    """Reduce step: join per-month series in month order and stack the per-month stats.

    Months never overlap in time, so joining is a plain concatenation - no partial counts
    for the same hour need adding together.
    """
    results = sorted(results, key=lambda r: r[1]["month"])   # fixed order, whatever arrived first
    series = pd.concat([s for s, _ in results])
    stats = pd.DataFrame([st for _, st in results]).set_index("month")
    return series, stats


def cleaning_table(stats: pd.DataFrame) -> pd.DataFrame:
    """Rule -> rows removed and % of raw rows, over all months (the poster's cleaning table)."""
    raw = int(stats["raw_rows"].sum())
    rows = [{"rule": "raw rows", "rows": raw, "pct_of_raw": 100.0}]
    for r in RULES:
        n = int(stats[f"removed_{r}"].sum())
        rows.append({"rule": r, "rows": n, "pct_of_raw": 100 * n / raw})
    kept = int(stats["kept_rows"].sum())
    rows.append({"rule": "kept rows", "rows": kept, "pct_of_raw": 100 * kept / raw})
    out = pd.DataFrame(rows)
    out["pct_of_raw"] = out["pct_of_raw"].round(3)
    return out


def peak_profile(series: pd.Series) -> pd.DataFrame:
    """Descriptive peak-period demand: mean pickups/hour in each window, weekday vs weekend."""
    df = series.to_frame("pickups")
    df["weekend"] = np.where(df.index.day_of_week >= 5, "weekend", "weekday")
    rows = []
    for name, (h0, h1) in config.PEAK_WINDOWS.items():
        in_win = (df.index.hour >= h0) & (df.index.hour <= h1)
        for wk, grp in df[in_win].groupby("weekend"):
            rows.append({"window": f"{name} {h0:02d}:00-{h1:02d}:59", "day_type": wk,
                         "mean_pickups_per_hour": round(grp["pickups"].mean(), 1)})
    rows.append({"window": "all hours", "day_type": "all",
                 "mean_pickups_per_hour": round(df["pickups"].mean(), 1)})
    return pd.DataFrame(rows)
