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
- **Which model won:** on the CV folds every top-7 setting was gradient boosting predicting the level
  directly. On the August validation month the best HGB (learning rate 0.1, 15 leaves, 500 iterations)
  had RMSE 378 vs 485 for the best Ridge and 760 for "same hour last week" → HGB chosen.
- **Test month (Sep 2026, scored once):**

  | Method | MAE (pickups/h) | RMSE | WAPE |
  |---|---|---|---|
  | **Gradient boosting** | **277** | **388** | **5.4%** |
  | Ridge (predicts change vs last week) | 415 | 562 | 8.2% |
  | Baseline: same hour last week | 698 | 987 | 13.7% |
  | Baseline: same hour yesterday | 908 | 1,392 | 17.9% |

- **We beat the baseline – but be ready to explain why, given the course example lost to it.** Our forecast is
  *one hour ahead*: the model knows the pickups of the last 1–3 hours (`lag_1`…`lag_3`). "Same hour last week"
  ignores what is happening *right now* (weather, events, a holiday, a slow day). The course example only had
  one month of data (384 training hours); we have ~22,000 training hours, so the boosting model can learn the
  hour × weekday shape and how today's level differs from last week's. MAE is **60% lower** than the best
  baseline.
- **Why Ridge is worse than HGB:** Ridge is linear – it cannot represent "8 am is busy on weekdays but quiet
  on Sundays" without hand-made interaction features. Trees learn such interactions automatically.
- **Why the 'diff168' target did not help HGB:** with lag_1 available, the previous hour is a better anchor
  than last week; for Ridge the diff target was slightly better.

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
- **Results** (benchmark job on 8 Slurm CPUs; mean of 3 timed runs after a warm-up; std below 2%):

  | Workload | Sequential | 1-worker pool | 2 | 4 | 8 workers | Speedup (8) | Efficiency (8) | Amdahl p |
  |---|---|---|---|---|---|---|---|---|
  | Data parallel: clean 33 months | 125.8 s | 132.5 s | 69.3 s | 37.5 s | 21.8 s | **5.76×** | 0.72 | 0.945 |
  | Task parallel: 96 search fits | 90.4 s | 91.6 s | 46.1 s | 23.9 s | 13.1 s | **6.92×** | 0.87 | 0.978 |
  | Task parallel: 3 feature blocks | 18 ms | 110 ms | 133 ms | 189 ms | 303 ms | 0.06× | 0.01 | < 0 |

- **How to read it:**
  - The **1-worker pool** is slower than the plain loop (132.5 vs 125.8 s): that 5% is pure overhead (starting a
    process, pickling each month's result back).
  - **Search scales best** (p ≈ 0.98, efficiency 0.87): 96 tasks, each 0.01–2.4 s of pure computation, and the
    data is sent once per worker. What is lost is mostly the **tail** – at the end a few long HGB fits run while
    other workers idle – plus pool start-up.
  - **Cleaning scales less** (p ≈ 0.945, efficiency 0.72): every worker reads Parquet from the **same disk** and
    needs ~2 GB of memory bandwidth, and 33 shards on 8 workers leave a final round with only one busy worker.
    Amdahl says even infinite workers would cap at ~18×.
  - **Features get slower** with more workers: 18 ms of work vs ~100 ms just to start processes and copy the
    series. p comes out negative – the formula's way of saying the overhead is bigger than the work.
    Lesson: only parallelise work that is large compared with the overhead.
  - **Shared server:** load average ~124 on 64 threads during the benchmark – other users' processes
    compete with ours, so our speedups are probably slightly pessimistic, and they could vary on another day.
  - Pool start-up measured directly: 0.04 s (1 worker) to 0.10 s (8 workers).

## Section 10: non-determinism

| Source | Broken version | Fix | Limits of the fix |
|---|---|---|---|
| **Completion order** | `submit` + `as_completed`, concatenating months as they finish → rows in a different order each run (different hash / first hour) | sort by month before combining (or use `map`, which returns input order) | must wait for all shards before combining; costs a sort |
| **Float reduction order** | sum of squared fares (needed for the fare standard deviation) = 24,096 hourly partial sums added in completion order → last digits change | add in a fixed (month) order; or `math.fsum`, which is exactly rounded so order doesn't matter | fixed order = repeatable, not more accurate; `fsum` is slower and only for sums |
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

## Code walkthrough – the six functions to know

1. **`cleaning.apply_rules(df, month)`** – applies the 5 rules in order. A small inner function `drop(mask, rule)`
   records `mask.sum()` (how many rows the rule removes) and keeps `df.loc[~mask]`. Because each rule runs on
   what the previous rule left, every row is counted once, under the first rule it breaks.
2. **`cleaning.clean_month(path)`** – the data-parallel worker: read 10 columns → `apply_rules` → categories →
   `hourly_counts` (floor to the hour, `value_counts`, re-index to every hour of the month, fill 0) → returns
   `(series, stats)`. It only depends on its file, so any process gives the same answer.
3. **`parallel.run_parallel(func, items, workers)`** – `ProcessPoolExecutor(max_workers=workers,
   initializer=pin_single_thread)` and `list(ex.map(func, items, chunksize=1))`. `map` returns results in the
   order of `items`; `chunksize=1` hands out one item at a time (better balance for uneven tasks).
4. **`cleaning.combine_months(results)`** – the reduce step: sort by month, `pd.concat` the series, stack the
   stats into a table.
5. **`features.rolling_features(y)`** – `past = y.shift(1)` first, then `past.rolling(w).mean()/std()/max()` –
   the window for hour t covers t−w … t−1, never t itself.
6. **`models.fit_and_score(task)`** – one search task: look up the fold's training/validation row numbers, fit
   the model (Ridge pipeline or HGB, level or diff target), predict the validation block, return
   `task_id, rmse, mae, seconds`. `run_search` runs all 96 with `map` and sorts by `task_id`.

Bonus: **`nondeterminism.combine_broken` vs `combine_fixed`** – identical except for one line:
`results.sort(key=lambda r: r[1]["month"])` before combining.

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

**Why the sum of *squared* fares, and three broken runs?** We first used the total fare from 33 monthly
subtotals. In one run the two "broken" totals happened to be identical, so we **measured** it with 3,000 random
month orders: 33 monthly subtotals gave only 5 distinct totals (fares are whole cents, so most orders round the
same way); 24,096 hourly sums of squared fares give a ~1% chance that two orders agree. The notebook therefore
uses the sum of squares (a real quantity: it is what you need for the standard deviation of all fares) and three
broken runs. Lesson for the Q&A: non-determinism does not have to show up in every run to be a bug – a program
that is only *usually* repeatable is not reproducible.
