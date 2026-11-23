"""Every setting for the project lives here, so the notebook and scripts never hard-code numbers.

Change a value here and re-run the notebook; nothing else needs editing.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
DATA_DIR = Path("/home/cs4207/data")          # read-only course data, never copied or modified
FILE_PATTERN = "yellow_tripdata_*.parquet"
PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURE_DIR = PROJECT_DIR / "figures"


def month_files() -> list[Path]:
    """All monthly Parquet files on disk, sorted oldest first (one file = one shard)."""
    return [Path(p) for p in sorted(glob.glob(str(DATA_DIR / FILE_PATTERN)))]


def month_of(path: Path | str) -> str:
    """'yellow_tripdata_2026-04.parquet' -> '2026-04'."""
    return Path(path).stem.split("_")[-1]


# ---------------------------------------------------------------- reading
# Column pruning: we read 10 of the 19-21 columns. Everything else (tips, tolls, surcharges ...)
# is either irrelevant to pickup counts or would leak information about the hour being predicted.
READ_COLUMNS = [
    "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID",
    "trip_distance", "fare_amount", "total_amount", "RatecodeID", "payment_type",
]

# ---------------------------------------------------------------- cleaning thresholds
MIN_DURATION_MIN = 1.0       # shorter than 1 minute: meter started and stopped by mistake / test record
MAX_DURATION_MIN = 180.0     # longer than 3 hours: meter left running
MAX_DISTANCE_MILES = 100.0   # NYC taxi trips over 100 miles are meter errors (max seen: 281,576 miles)
MIN_FARE = 0.0               # fare <= 0: refunds, voids, disputes (rows are kept only if fare > 0)
FLEX_FARE_PAYMENT_TYPE = 0   # TLC data dictionary: payment_type 0 = "Flex Fare trip"
UNKNOWN_RATECODE = 99        # TLC data dictionary: RatecodeID 99 = "Null/unknown"

# ---------------------------------------------------------------- time split (time-based, never random)
TRAIN_END = "2026-07-31 23:00"   # train: first month .. Jul 2026
VAL_START, VAL_END = "2026-08-01 00:00", "2026-08-31 23:00"     # model selection only
TEST_START, TEST_END = "2026-09-01 00:00", "2026-09-30 23:00"   # scored once at the very end
N_CV_FOLDS = 4               # TimeSeriesSplit folds inside the training period
CV_TEST_SIZE = 24 * 28       # each fold validates on the 4 weeks after its training block

# ---------------------------------------------------------------- features (fixed column order)
LAGS = [1, 2, 3, 24, 48, 168]
ROLL_WINDOWS = [3, 24, 168]
ROLL_STATS = ["mean", "std", "max"]
FEATURE_TASKS = ["lag", "rolling", "calendar"]   # tasks are re-joined in exactly this order
FEATURE_COLUMNS = (
    [f"lag_{k}" for k in LAGS]
    + [f"roll_{s}_{w}" for w in ROLL_WINDOWS for s in ROLL_STATS]
    + ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "month", "is_holiday"]
)

# ---------------------------------------------------------------- hyperparameter grid
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]
HGB_LEARNING_RATES = [0.05, 0.1]
HGB_MAX_LEAF_NODES = [15, 31]
HGB_MAX_ITER = [200, 500]
TARGETS = ["level", "diff168"]   # predict y directly, or the correction to "same hour last week"

# ---------------------------------------------------------------- parallelism and reproducibility
SEED = 42
WORKER_COUNTS = [1, 2, 4, 8]
N_REPEATS = 3                # timed repeats after one discarded warm-up
PEAK_WINDOWS = {"morning_peak": (7, 9), "evening_peak": (16, 19)}   # inclusive hour ranges


def available_cpus() -> int:
    """CPUs Slurm allocated to this job.

    Rockhill's Slurm does not pin jobs to cores (a job 'sees' all 64), so we trust
    SLURM_CPUS_PER_TASK first and only fall back to the CPU affinity / machine count.
    """
    try:
        visible = len(os.sched_getaffinity(0))
    except AttributeError:   # not Linux
        visible = os.cpu_count() or 1
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    return min(int(slurm), visible) if slurm else visible
