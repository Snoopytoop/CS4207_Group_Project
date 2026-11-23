"""Phase 1 exploration: footer inspection of every file + a column-pruned look at one month."""
import glob, sys
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd

files = sorted(glob.glob("/home/cs4207/data/yellow_tripdata_*.parquet"))
rows = []
schemas = {}
for f in files:
    pf = pq.ParquetFile(f)
    md = pf.metadata
    rows.append({"file": f.split("/")[-1], "rows": md.num_rows, "row_groups": md.num_row_groups,
                 "columns": md.num_columns})
    schemas[f.split("/")[-1]] = [(fld.name, str(fld.type)) for fld in pf.schema_arrow]
print(pd.DataFrame(rows).to_string())

# schema differences between files
base = schemas[rows[0]["file"]]
print("\nSchema of first file:", base)
for name, s in schemas.items():
    if s != base:
        print("DIFF", name, sorted(set(s) ^ set(base)))

month = sys.argv[1] if len(sys.argv) > 1 else "2026-04"
f = f"/home/cs4207/data/yellow_tripdata_{month}.parquet"
t = pq.read_table(f)
df = t.to_pandas()
print("\n=== one month:", month, df.shape)
print(df.isna().sum().to_string())
print(df.describe(include="all").T.to_string())
for c in df.columns:
    if df[c].nunique() < 30:
        print("\nvalue counts", c); print(df[c].value_counts(dropna=False).to_string())
dur = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
print("\nduration min:", dur.describe().to_string())
print("pickup range", df.tpep_pickup_datetime.min(), df.tpep_pickup_datetime.max())
print("out-of-month pickups", (df.tpep_pickup_datetime.dt.strftime('%Y-%m') != month).sum())
print("dup rows", df.duplicated().sum())
print("max distance", df.trip_distance.max(), " >100:", (df.trip_distance > 100).sum())
