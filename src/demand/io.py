"""Efficient Parquet reading.

Three techniques from the course:
1. Footer / metadata inspection: row counts, row groups and schema WITHOUT reading any data.
2. Column pruning: only the columns we need are decoded (`columns=`).
3. Streaming with `iter_batches`: process a file one batch at a time to cap memory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config


def file_inventory(files: list[Path]) -> pd.DataFrame:
    """Read only the Parquet footer of each file: rows, row groups, columns, size on disk."""
    rows = []
    for f in files:
        md = pq.ParquetFile(f).metadata          # footer only - no data pages are read
        rows.append({
            "month": config.month_of(f),
            "file": Path(f).name,
            "rows": md.num_rows,
            "row_groups": md.num_row_groups,
            "columns": md.num_columns,
            "size_mb": round(Path(f).stat().st_size / 1e6, 1),
        })
    return pd.DataFrame(rows)


def schema_differences(files: list[Path]) -> pd.DataFrame:
    """List the columns that appear in some files but not others (schema drift over time)."""
    names = {config.month_of(f): set(pq.ParquetFile(f).schema_arrow.names) for f in files}
    common = set.intersection(*names.values())
    rows = []
    for col in sorted(set.union(*names.values()) - common):
        months = [m for m, cols in names.items() if col in cols]
        rows.append({"column": col, "first_month": months[0], "n_months": len(months)})
    return pd.DataFrame(rows)


def read_month(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read one month with column pruning (only `columns` are decoded)."""
    columns = columns or config.READ_COLUMNS
    return pq.read_table(path, columns=columns).to_pandas()


def iter_month_batches(path: Path, columns: list[str] | None = None,
                       batch_size: int = 500_000) -> Iterator[pd.DataFrame]:
    """Stream one month in batches - peak memory is one batch instead of the whole file."""
    columns = columns or config.READ_COLUMNS
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
        yield pa.Table.from_batches([batch]).to_pandas()
