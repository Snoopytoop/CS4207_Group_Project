"""Experiment behind the float-reduction demo: chance that two random shard orders give the same total.

Run: srun --partition=cs4207 --cpus-per-task=8 --mem=32G python scripts/float_order_experiment.py
"""
import sys, random
from collections import Counter
sys.path.insert(0, "src")
import numpy as np
from demand import config, parallel
from demand.cleaning import apply_rules
from demand.io import read_month


def parts(path):
    m = config.month_of(path)
    df, _ = apply_rules(read_month(path), m)
    h = df["tpep_pickup_datetime"].dt.floor("h")
    f = df["fare_amount"]
    return {"daily_sum": f.groupby(df["tpep_pickup_datetime"].dt.floor("D")).sum().to_numpy(),
            "hourly_mean_fare": f.groupby(h).mean().to_numpy(),
            "hourly_sum_sq": (f * f).groupby(h).sum().to_numpy(),
            "hourly_mean_speed": (df["trip_distance"] / ((df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 3600)).groupby(h).mean().to_numpy()}


res = parallel.run_parallel(parts, config.month_files(), 8)
rng = random.Random(0)
for name in res[0]:
    c = Counter()
    for _ in range(3000):
        order = res[:]; rng.shuffle(order)
        tot = 0.0
        for r in order:
            for v in r[name]:
                tot += float(v)
        c[tot] += 1
    p_same = sum((n / 3000) ** 2 for n in c.values())
    print(f"{name:18s} distinct {len(c):5d}   P(two runs identical) = {p_same:.3f}")
