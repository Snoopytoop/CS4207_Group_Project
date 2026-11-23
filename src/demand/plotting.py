"""Every figure in the project, one consistent style, saved at 300 dpi.

Colour roles (fixed across all figures and the poster):
  blue   = data parallelism / the model        orange = task parallelism / best baseline
  aqua   = third workload                       grey   = actual values, ideal lines, context
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SHADE = "#0b0b0b", "#52514e", "#e4e3df", "#f1f0ec"
WORKLOAD_COLOURS = {"cleaning": BLUE, "search": ORANGE, "features": AQUA}
WORKLOAD_LABELS = {"cleaning": "Data parallel: clean 33 months",
                   "search": "Task parallel: hyperparameter search",
                   "features": "Task parallel: 3 feature tasks"}

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 15, "axes.titlesize": 16, "axes.titleweight": "bold", "axes.labelsize": 15,
    "legend.fontsize": 13,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 2,
})


def _save(fig, name: str) -> Path:
    config.FIGURE_DIR.mkdir(exist_ok=True)
    path = config.FIGURE_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path


def _shade_weekends(ax, index: pd.DatetimeIndex) -> None:
    days = pd.Series(index.normalize().unique())
    for d in days[days.dt.day_of_week == 5]:
        ax.axvspan(d, d + pd.Timedelta(days=2), color=SHADE, lw=0, zorder=0)


def hourly_series(y: pd.Series, zoom_month: str = "2026-04") -> Path:
    """Top: the whole hourly series (daily mean on top) with train/val/test periods.
    Bottom: one month in full hourly detail with weekends shaded."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [1, 1]})
    a1.plot(y.index, y.values, color="#b9b8b2", lw=0.4, label="Hourly pickups")
    a1.plot(y.resample("D").mean(), color=INK, lw=1.2, label="Daily mean")
    spans = [(y.index[0], pd.Timestamp(config.TRAIN_END), "Train", "#dbe8f8"),
             (pd.Timestamp(config.VAL_START), pd.Timestamp(config.VAL_END), "Val", "#fde3d6"),
             (pd.Timestamp(config.TEST_START), pd.Timestamp(config.TEST_END), "Test", "#d3efe4")]
    top = y.max() * 1.08
    for s, e, lab, c in spans:
        a1.axvspan(s, e, color=c, lw=0, zorder=0)
        a1.text(s + (e - s) / 2, top, lab, ha="center", va="bottom", color=MUTED, fontsize=11)
    a1.set_ylim(0, top * 1.08)
    a1.set_ylabel("Pickups per hour")
    a1.set_title("NYC yellow-taxi pickups per hour, Jan 2024 - Sep 2026")
    a1.legend(loc="lower left", ncol=2)

    m = y.loc[zoom_month]
    _shade_weekends(a2, m.index)
    a2.plot(m.index, m.values, color=BLUE, lw=1.4)
    a2.set_ylim(0, None)
    a2.set_ylabel("Pickups per hour")
    a2.set_title(f"Zoom: {pd.Timestamp(zoom_month + '-01'):%B %Y} (weekends shaded)")
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
    fig.tight_layout()
    return _save(fig, "hourly_series.png")


def daily_weekly_profile(y: pd.Series) -> Path:
    """Average pickups by hour of day, weekdays vs weekends."""
    df = y.to_frame("p")
    df["hour_of_day"], df["weekend"] = df.index.hour, df.index.day_of_week >= 5
    prof = df.groupby(["weekend", "hour_of_day"])["p"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(prof.index, prof[False], color=BLUE, marker="o", ms=4, label="Weekday")
    ax.plot(prof.index, prof[True], color=ORANGE, marker="o", ms=4, label="Weekend")
    for name, (h0, h1) in config.PEAK_WINDOWS.items():
        ax.axvspan(h0 - 0.5, h1 + 0.5, color=SHADE, lw=0, zorder=0)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean pickups per hour")
    ax.set_ylim(0, None)
    ax.set_title("Daily profile (shaded: 07-10 and 16-20 peak windows)")
    ax.legend()
    return _save(fig, "daily_weekly_profile.png")


def time_split(index: pd.DatetimeIndex, history_end: pd.Timestamp, fold_ranges: list) -> Path:
    """Timeline of the split, zoomed on 2026 so the folds, validation and test months are visible.

    Training actually starts in Jan 2024 (after a 168-hour lag warm-up); the arrow at the left says so.
    """
    num = mdates.date2num
    left = pd.Timestamp("2026-01-01")
    right = pd.Timestamp(config.TEST_END) + pd.Timedelta(days=3)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    bars = [  # (lane, start, end, colour, label inside bar, text colour)
        (2, left, pd.Timestamp(config.TRAIN_END), "#9ec5f4", f"Train  (from {history_end:%d %b %Y})", INK),
        (2, pd.Timestamp(config.VAL_START), pd.Timestamp(config.VAL_END), ORANGE, "Val", "white"),
        (2, pd.Timestamp(config.TEST_START), pd.Timestamp(config.TEST_END), AQUA, "Test", "white"),
    ]
    for i, (s, e) in enumerate(fold_ranges):
        bars.append((1, s, e, "#fbc3a8", f"F{i + 1}", INK))
    for lane, s, e, c, lab, tc in bars:
        ax.broken_barh([(num(s), num(e) - num(s))], (lane - 0.32, 0.64), color=c, edgecolor="white", lw=2)
        ax.text(num(s) + (num(e) - num(s)) / 2, lane, lab, ha="center", va="center", color=tc,
                fontsize=14, fontweight="bold")
    ax.annotate("", xy=(num(left), 2), xytext=(num(left) + 22, 2),
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.5})
    ax.set_yticks([1, 2]); ax.set_yticklabels(["CV folds\n(inside train)", "Split"])
    ax.set_ylim(0.4, 2.6)
    ax.set_xlim(num(left), num(right))
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_title("Each fold validates on the 4 weeks after its training block")
    return _save(fig, "time_split.png")


def speedup(summary: pd.DataFrame) -> Path:
    """Speedup vs workers per workload, with the ideal y = x line."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    par = summary[summary["mode"] == "parallel"]
    kmax = int(par["workers"].max())
    ax.plot([1, kmax], [1, kmax], ls="--", color=MUTED, lw=1.2, label="Ideal (speedup = workers)")
    for wl, sub in par.groupby("workload"):
        sub = sub.sort_values("workers")
        ax.plot(sub["workers"], sub["speedup"], marker="o", ms=8, color=WORKLOAD_COLOURS[wl],
                label=WORKLOAD_LABELS[wl])
        last = sub.iloc[-1]
        ax.annotate(f"{last['speedup']:.1f}x", (last["workers"], last["speedup"]),
                    xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=15, fontweight="bold")
    ax.axhline(1, color=GRID, lw=1)
    ax.set_xticks(sorted(par["workers"].unique()))
    ax.set_xlabel("Worker processes")
    ax.set_ylabel("Speedup  (T_sequential / T_parallel)")
    ax.set_xlim(0.5, kmax + 1.5)
    ax.set_ylim(0, kmax + 0.5)
    ax.set_title("Speedup vs workers (mean of 3 timed runs)")
    ax.legend(loc="upper left")
    return _save(fig, "speedup.png")


def task_durations(search: pd.DataFrame) -> Path:
    """Load imbalance: total seconds of each search setting (summed over folds)."""
    s = search.sort_values("task_seconds")
    fig, ax = plt.subplots(figsize=(8, 6))
    colours = [ORANGE if m == "hgb" else BLUE for m in s["model"]]
    ax.barh(range(len(s)), s["task_seconds"], color=colours, height=0.7)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s["label"], fontsize=7)
    ax.set_xlabel("Seconds (sum over 4 folds, one worker)")
    ax.set_title("Search task durations: HGB tasks are 10-200x longer than Ridge")
    ax.grid(axis="y", visible=False)
    return _save(fig, "task_durations.png")


def forecast_vs_actual(y_true: pd.Series, y_model: pd.Series, y_base: pd.Series,
                       model_name: str, base_name: str, start: str, days: int = 7) -> Path:
    """One test week: actual vs model vs best baseline."""
    sl = slice(pd.Timestamp(start), pd.Timestamp(start) + pd.Timedelta(days=days) - pd.Timedelta(hours=1))
    fig, ax = plt.subplots(figsize=(12, 4))
    _shade_weekends(ax, y_true.loc[sl].index)
    ax.plot(y_true.loc[sl], color=INK, lw=2.2, label="Actual pickups")
    ax.plot(y_model.loc[sl], color=BLUE, lw=1.8, label=f"Forecast: {model_name}")
    ax.plot(y_base.loc[sl], color=ORANGE, lw=1.4, ls="--", label=f"Baseline: {base_name}")
    ax.set_ylabel("Pickups per hour")
    ax.set_ylim(0, None)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
    ax.set_title("One-hour-ahead forecast on a held-out test week")
    ax.legend(loc="upper left", ncol=3, fontsize=13)
    return _save(fig, "forecast_vs_actual.png")


def error_vs_baseline(metrics_df: pd.DataFrame) -> Path:
    """Test error (WAPE %) per method, lower is better. Row 0 of metrics_df = the chosen model."""
    chosen = metrics_df["method"].iloc[0]
    m = metrics_df.sort_values("wape_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    colours = [BLUE if r == chosen else ("#9ec5f4" if r.startswith("Model") else "#b9b8b2") for r in m["method"]]
    labels = [r.replace("Model: ", "").replace("Baseline: ", "No model: ").replace(" (", "\n(") for r in m["method"]]
    bars = ax.barh(labels, m["wape_pct"], color=colours, height=0.7)
    for b, v in zip(bars, m["wape_pct"]):
        ax.text(b.get_width() + 0.3, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", color=INK,
                fontsize=15, fontweight="bold")
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=13)
    ax.set_xlabel("Error as % of actual pickups (WAPE) - lower is better")
    ax.set_xlim(0, m["wape_pct"].max() * 1.2)
    ax.grid(axis="y", visible=False)
    return _save(fig, "error_vs_baseline.png")
