# EXPLAINER – what we did, why, and how to defend it

Plain-English guide to every section of `demand_forecast.ipynb`. Numbers come from `outputs/results.json`
(the notebook writes it; the poster reads it). Read this with the notebook open.

---

## The one-paragraph story (learn this by heart)

We forecast how many yellow-taxi pickups New York will have **in the next hour**, so fleets can send drivers
where demand is about to appear. We used **all 33 monthly files** (Jan 2024 – Sep 2026, ~123.5 million trips),
cleaned them with five ordered rules, counted pickups per hour (24,096 hours), built lag / rolling / calendar
features that only look backwards, and compared Ridge regression and gradient boosting against two no-model
baselines. The pipeline uses **data parallelism** (one process per month file for cleaning) and **task
parallelism** (feature blocks; 96 independent hyperparameter-search fits). Every parallel output is checked to
be **identical** to the sequential one; we timed both, computed speedup, efficiency and Amdahl's *p*, and showed
three sources of **non-determinism** with their fixes.

---

## Section 0–1: configuration and environment

- **What:** prints versions, hardware and every setting.
- **Why one config file (`src/demand/config.py`):** the rubric asks for seeds, worker counts, paths and
  thresholds in one place. Changing a threshold there changes it everywhere.
- **Why code in `src/demand/` and not all in the notebook:** `ProcessPoolExecutor` sends a *function name* to
  each worker process, and the worker must be able to import it. Functions defined inside a notebook cannot
  always be imported by workers (same reason the week 9 lab used `workers.py`).
- **Single-threaded maths (`OMP_NUM_THREADS=1` etc.):** numpy/scikit-learn can use many threads internally. With
  8 worker processes that could mean 8 × 64 threads fighting for 8 cores (oversubscription), and the result of a
  multi-threaded sum can depend on how many threads split it. Pinning to 1 thread makes timings fair and results
  reproducible.
- **Partition:** we used Slurm partition `cs4207` (the course partition). The `student` partition existed but
  would not start jobs until the evening of 23 Nov. Rockhill's Slurm does not pin a job to its cores, so a job
  "sees" all 64 CPUs; we use the Slurm allocation (`SLURM_CPUS_PER_TASK = 8`) to decide worker counts.

## Section 2: data inventory (footer inspection)

- A Parquet file stores **metadata in its footer**: number of rows, row groups and the schema. Reading it costs
  almost nothing and tells us the size of every shard before we read any data.
- Findings: 33 files, 2.96 M – 4.59 M rows each (largest/smallest ≈ 1.55), 3–5 row groups each. Schema drift:
  `cbd_congestion_fee` appears from Jan 2025 (Manhattan congestion pricing) and `request_source` from Jun 2026.
  We do not use either.

## Section 3: cleaning and preprocessing

### What's wrong with the raw data (April 2026, the month the course example used)
- 3,831,240 rows. Pickup times run from **2001-01-01** to 2026-05-01: 12 rows are outside April.
- Durations from 0 seconds to **9,984 minutes** (almost 7 days).
- Distances from 0 to **281,576 miles** (a meter error); 136 trips over 100 miles.
- Fares from **−$1,274.90** (refunds) to $1,442.20.
- 799,786 rows (20.9%) have nulls in `RatecodeID`, `passenger_count`, `store_and_fwd_flag`,
  `congestion_surcharge` and `Airport_fee`.

### Where is "Flex Fare"?
There is no column called Flex Fare. The TLC data dictionary says **`payment_type = 0` means a Flex Fare trip**
(a fixed, up-front price booked through an app) and `RatecodeID = 99` means unknown. The cross-tab in the
notebook shows the null-`RatecodeID` rows are exactly the `payment_type = 0` rows. So the "nulls" are not broken
records: they are Flex Fare trips, for which the meter-related fields are simply not filled in.

**Decision: keep them.** Our target is *how many pickups happen*. A Flex Fare trip is a real pickup. Dropping
them would delete 20–30% of demand in recent months and create a fake drop in demand when Flex Fare became
common in 2025. We turn `RatecodeID` into a category with an explicit **`unknown`** level (null and 99). If a
Flex Fare row is junk (zero distance or zero fare) the general rules below remove it anyway.

### The rules, in order (thresholds in `config.py`)
| # | Rule | Why this threshold |
|---|---|---|
| 1 | Pickup outside the file's calendar month | Spill-over records from the previous/next month and broken clocks (2001). They would double-count hours that belong to another file. |
| 2 | Duration < 1 minute, > 3 hours, or dropoff before pickup | < 1 min = meter started/stopped by mistake or a test record; > 3 h = meter left running. A real NYC trip over 3 hours is extremely rare. |
| 3 | Distance ≤ 0 or > 100 miles | 0 miles = no trip happened; > 100 miles is outside any normal taxi trip and includes meter errors of 281,576 miles. |
| 4 | Fare ≤ $0 | Refunds, voids and disputes (negative fares mirror an earlier charge). |
| 5 | Exact duplicate rows | The same trip recorded twice. |

**Order matters** because a row can break several rules; it is counted under the first rule it breaks. That is
why the table must say "applied in this order".

**All months:** 123,471,721 raw → 115,358,059 kept (93.4%). Removed: 802 out-of-month, 2,413,007 duration,
2,306,367 distance, 3,393,484 fare, 2 duplicates. **April 2026 reproduces the course worked example exactly**
(12 / 89,425 / 66,135 / 14,040 removed, 3,661,628 kept) – a good sign our rules match the course's.

### Categories
`PULocationID`, `DOLocationID` (taxi zones 1–265) and `RatecodeID` are **labels**, not quantities: zone 200 is
not "twice" zone 100. Storing them as `category` makes that explicit and saves memory. Our citywide forecast
counts all zones together, so the zone category is not a model input – it would be for a per-zone extension.

### Hourly aggregation and missing hours
- Each kept trip is assigned to the clock hour of its pickup (`floor("h")`) and counted.
- We **re-index to every hour** of the month and fill gaps with **0**: an hour with no recorded pickups means
  zero demand, not unknown demand.
- Only **3 hours** were missing: 02:00 on 10 Mar 2024, 9 Mar 2025 and 8 Mar 2026 – the US **daylight-saving**
  spring-forward hour, which does not exist on the clock (TLC timestamps are local time without a time zone).
  We leave them at 0 and say so. The autumn fall-back hour (1 Nov) happens twice and is counted together.
- Result: **24,096 hourly observations**, mean ≈ 4,787 pickups/hour.

### Efficient loading and memory
- **Column pruning:** we read 10 of the 19–21 columns (`columns=[...]`), so tips, tolls and surcharges are
  never decoded.
- **Footer inspection** (above) without reading data.
- **`iter_batches`:** `hourly_counts_streaming` reads a month 500,000 rows at a time. Rules 1–4 only look at one
  row, so per-batch hourly counts can be added. The notebook shows it gives exactly the same series as reading
  the whole month. We kept whole-month reading as the main path because the duplicate rule needs the whole month
  and one month only takes ~2 GB per process.

## Section 4: data parallelism

- **Data parallelism** = the *same* function on *different pieces* of the data at the same time.
- **Shard = one month file.** Why: it matches how the data is stored (no worker reads across files); each
  shard keeps its own time order; months cover non-overlapping hours, so combining is just concatenating in
  month order (no partial counts for the same hour to add up); and the shards are similar in size.
- **Map:** `ProcessPoolExecutor.map(clean_month, files)`. **Reduce:** `combine_months` sorts the results by
  month and concatenates the series; per-month cleaning statistics are stacked and summed.
- **Why processes, not threads:** pandas work here is CPU-bound Python; threads in one process share the GIL.
- **Verification:** `parallel.equals(sequential)` is printed as `True` – both the hourly series and the cleaning
  statistics table. The benchmark script asserts the same on every timed run.
- **Balance:** 33 shards on 8 workers = 5 "rounds": after 32 shards one worker cleans the 33rd while 7 wait.
  Shard sizes also differ by up to 1.55×. Both are **load imbalance**. A finer split (month × row group, ~140
  shards) would balance better but adds more scheduling overhead; we kept the simple, explainable version.
- **Memory:** each worker peaks at about 2 GB (one month of 10 columns plus intermediate copies). 8 workers ≈
  16 GB, within the 32 GB we requested. With bigger files we would switch to `iter_batches`.

## Section 5: time-based split and leakage

- **Why not a random split:** neighbouring hours are almost identical. With a random split the model would be
  trained on 14:00 and tested on 13:00 of the same day – it would look great and fail in real use.
- Split: **history** (first 169 hours, only needed to compute the 168-hour lag; dropped, not imputed) →
  **train** to 31 Jul 2026 → **validation** Aug 2026 → **test** Sep 2026.
- **Hyperparameter search** uses 4 **expanding-window** folds (`TimeSeriesSplit`) inside train; each fold trains on
  everything before a 4-week block and validates on that block.
- **Leakage controls:** lags are `shift(k≥1)`; rolling windows are computed on `y.shift(1)` (shift *first*,
  then roll – otherwise the window would include the hour being predicted); no fares/tips/distances of the
  predicted hour; the scaler is inside a `Pipeline`, so it is re-fitted on each fold's training rows; the test
  month is used once. The notebook **asserts** that `lag_1` equals the previous hour and that rolling windows end
  at t−1.

## Section 6: task parallelism 1 – feature extraction

- **Task parallelism** = *different* jobs at the same time (here: three different feature builders).
- `lag_features` (1, 2, 3, 24, 48, 168 h ago), `rolling_features` (mean/std/max of the last 3, 24, 168 h),
  `calendar_features` (hour and weekday as sin/cos so 23:00 is next to 00:00, weekend, month, US holiday).
- **Independent** because each reads only the hourly series and writes its own columns; none needs another's
  output. They are rejoined in a **fixed column order** (`config.FEATURE_COLUMNS`), so the table is identical
  whichever finishes first – verified with `equals`.
- **Honest result:** the series has only ~24,000 rows, each task takes milliseconds, and starting a process
  and pickling the series costs more than that. Parallel is **slower**. This is Amdahl + overheads: when the
  parallel work is tiny, the fixed overhead dominates. It would pay off with per-zone features (265 zones).

## Section 7: task parallelism 2 – hyperparameter search

- **Baselines:** *same hour yesterday* (`lag_24`) and *same hour last week* (`lag_168`). The course example
  found "same hour last week" beat tuned Ridge, so a model must be compared to it.
- **Models:** Ridge (linear, needs scaling) and HistGradientBoosting (trees, can learn interactions such as
  "Friday 23:00 is busy but Monday 23:00 is not"). Each can predict the **level** or the **correction to last
  week** (`y − lag_168`).
- **Grid:** Ridge alpha ∈ {0.1, 1, 10, 100}; HGB learning rate ∈ {0.05, 0.1} × max leaf nodes ∈ {15, 31} ×
  iterations ∈ {200, 500}; × 2 targets = 24 settings; × 4 folds = **96 tasks**.
- **Why independent:** each task fits its own new model on its own training block and scores its own
  validation block. No task reads another task's result.
- **Implementation:** `fit_and_score(task)` returns `(task_id, rmse, mae, seconds)`. The training data is sent to
  each worker once (pool `initializer`), not with each of the 96 tasks. Results are **sorted by `task_id`**.
- **Selection:** mean CV RMSE per setting → the best Ridge and the best HGB are trained on all of train and scored
  on **August** (validation) → lowest validation RMSE is chosen. The test month plays no part.
- **Load imbalance:** a Ridge fit takes ~0.01 s, a 500-iteration HGB fit several seconds. `map(chunksize=1)`
  hands out tasks one at a time so a free worker always takes the next one, but at the end a few long HGB tasks
  run while the other workers are idle (see `figures/task_durations.png`).

## Section 8: final test

- The chosen model is **retrained on train + validation** (Jan 2024 – Aug 2026), then predicts each hour of
  September 2026 one hour ahead. Each prediction uses the *actual* pickups up to t−1 – exactly what a live
  system would know.
- **Metrics:** MAE (average miss in pickups/hour), RMSE (penalises big misses more), **WAPE** (total absolute
  error ÷ total pickups – the "% of actual volume" in the course example), MAPE.
- *Results paragraph – filled in after the final run (see below).*

## Section 9: benchmarks and Amdahl

- **Speedup** S_k = T_sequential / T_k. **Efficiency** E_k = S_k / k (1.0 = perfect use of k workers).
- **Amdahl's law:** if a fraction *p* of the work can run in parallel, S_k = 1 / ((1 − p) + p/k). Solving for
  p from a measured S_k: p = (1 − 1/S_k) / (1 − 1/k). Max speedup with infinite workers = 1 / (1 − p).
  Here *p* is an **effective** value: it lumps true serial code and overheads together.
- **Method:** true sequential loop; a 1-worker pool (shows pool overhead with no parallelism); 2, 4, 8 workers.
  One **warm-up** run discarded (first run pays for reading files from disk into the cache and importing
  libraries in new processes), then **3 timed runs**, mean ± std. Each run is asserted equal to sequential.
- **Overheads:** process start-up (measured: empty pool), pickling results/data between processes, all workers
  reading Parquet from the same disk, load imbalance, and – important on Rockhill – **other users**: the load
  average was ~124 on 64 hardware threads during our runs, from long-running jobs that do not go through Slurm.
- *Numbers – filled in after the final run.*

## Section 10: non-determinism

| Source | Broken version | Fix | Limits of the fix |
|---|---|---|---|
| **Completion order** | `submit` + `as_completed`, concatenating months as they finish → rows in a different order each run (different hash / first hour) | sort by month before combining (or use `map`, which returns input order) | must wait for all shards before combining; costs a sort |
| **Float reduction order** | total fare = sum of 33 float partial sums in completion order → last digits change | add in a fixed (month) order; or `math.fsum`, which is exactly rounded so order doesn't matter | fixed order = repeatable, not more accurate; `fsum` is slower and only for sums |
| **Unseeded randomness** | random search with an unseeded generator + HGB early stopping with a random validation split → different "best" setting each run | base seed 42 for the generator, per-trial seed 42 + trial id for each model | reproducible only for the same task list, library versions and data; one seed hides how much results vary |
| (prevented) **thread count** | multi-threaded BLAS can split sums differently | 1 thread per process | parallelism must then come from processes |

The tiny example: `(1e16 + -1e16) + 1.0 = 1.0` but `1e16 + (-1e16 + 1.0) = 0.0`.

The random sleep (0–2 s) in the demo worker only makes the order change *every* time; on a busy shared server
the order changes anyway, just not reliably enough for a demo. We say this openly.

## Sections 11–12: results and reproducibility

- `outputs/results.json` is the single source of truth; the poster is generated from it.
- The notebook was run twice; `scripts/compare_results.py` confirms both runs agree apart from timings and the
  deliberately broken demos.

---

## Glossary (things a marker may ask)

- **Data parallelism:** same operation on different parts of the data at once (our months).
- **Task parallelism:** different, independent jobs at once (our feature blocks, our search fits).
- **Shard:** one piece of the data given to one worker (one month file).
- **Map / reduce:** apply a function to every shard (map), then combine the partial results (reduce).
- **Speedup / efficiency / Amdahl's law / p:** see section 9.
- **Overheads:** time spent on parallelism itself – starting processes, copying data, waiting.
- **Load imbalance:** some workers have more work than others, so some idle while the slowest finishes.
- **Leakage:** a feature that secretly contains information from the future (or the test set).
- **Lag feature:** the value k hours ago. **Rolling feature:** a summary (mean/std/max) of a recent window.
- **TimeSeriesSplit:** cross-validation where every validation block comes after its training block.
- **MAE / RMSE / WAPE:** average absolute miss / root of average squared miss / total miss as % of total demand.
- **Non-determinism:** same code and data giving different output on different runs.
- **Pickling:** how Python serialises objects to send them between processes.
- **GIL:** Python's global interpreter lock – only one thread runs Python bytecode at a time in a process.
