# Q&A prep – likely marker questions with model answers

Answers use the numbers in `outputs/results.json`. Keep each answer short; point at the poster panel.

---

## Data and cleaning

**1. What data did you use?**
All 33 monthly yellow-taxi files on Rockhill, Jan 2024 to Sep 2026: 123.5 million trips. After cleaning we
kept 115.4 million (93.4%) and counted pickups per hour, giving 24,096 hourly observations, about 4,787 pickups
per hour on average.

**2. How did you load Parquet efficiently?**
Three ways. We read the footer metadata first (rows, row groups, schema, even null counts) without reading data.
We read only 10 of the 19–21 columns (column pruning). And we showed `iter_batches` streaming gives exactly the
same hourly counts for a month, which is what we would use if months were too big for memory.

**3. Why those cleaning thresholds?**
Under 1 minute is a meter started by mistake; over 3 hours is a meter left running. Zero distance means no trip,
and over 100 miles includes meter errors like the 281,576-mile trip. Fares of zero or less are refunds and voids.
Pickups outside the file's month are spill-over or broken clocks (some say 2001). Order matters: a row is counted
under the first rule it breaks.

**4. What did you do with Flex Fare / missing rate codes?**
Flex Fare trips are `payment_type 0`, and they are exactly the rows with null rate codes. They are real
pickups, so for a *demand* forecast we kept them (20.6 M trips). Dropping them would remove ~20–30% of recent
demand. Rate code became a category with an explicit 'unknown' level.

**5. How did you handle missing hours?**
We re-indexed to every hour and filled gaps with 0, because no recorded pickup means zero demand. Only 3 hours
were missing: 02:00 on each US daylight-saving spring-forward day, an hour that does not exist on the clock.

## Data parallelism

**6. Why shard by month?**
That is how the data is stored, so no worker reads across files. Each shard keeps its time order and months
don't overlap in time, so combining is just concatenating in month order. And the shards are similar sizes.

**7. Is it balanced?**
Not perfectly. Month sizes differ by up to 1.55×, and 33 shards on 8 workers means the last "round" has one
worker cleaning the 33rd month while seven wait. Smaller shards (month × row group) would balance better but add
more overhead; we kept the simple version and measured the effect (efficiency 0.72 at 8 workers).

**8. How do you know the parallel result equals the sequential one?**
We compare them with `.equals()`: the full hourly series and the cleaning statistics table, both `True` in the
notebook. The benchmark script asserts equality on every single timed run (and the 96 search scores and the
feature table too). If any run differed, the benchmark would stop with an error.

**9. How much memory does it use?**
About 2.3 GB peak per worker (one month of 10 columns plus copies made while cleaning), so 8 workers need
~18 GB; we asked Slurm for 32 GB. With bigger files we would stream with `iter_batches`.

## Task parallelism

**10. Why are your search tasks independent?**
Each task is one (setting, fold) pair: it fits a brand-new model on its own training block and scores its own
validation block. No task reads another's result, so they can run in any order on any worker. We sort results by
task id afterwards.

**11. Why did feature-extraction parallelism not speed up?**
The three tasks take 18 ms together. Starting worker processes and copying the series to them costs ~100 ms.
When the work is smaller than the overhead, parallel is slower – Amdahl with a big serial part. With per-zone
features (265 zones) the work would be big enough to pay off.

**12. Did you see load imbalance in the search?**
Yes. A Ridge fit takes ~0.01 s and a 500-iteration boosting fit up to ~2.4 s. Handing out tasks one at a time
(`chunksize=1`) keeps workers busy, but at the end a few long fits run while others idle. That is part of why
efficiency is 0.87 rather than 1.

## Speedup and Amdahl

**13. What speedups did you get?**
On 8 workers: 6.9× for the hyperparameter search, 5.8× for cleaning, and 0.06× (slower) for the tiny feature
tasks. Each is the mean of 3 timed runs after a discarded warm-up.

**14. Why is speedup below ideal?**
Serial parts (creating the pool, combining results), process start-up and pickling, all workers reading from the
same disk, load imbalance, and a busy shared server – the load average was ~124 on 64 threads.

**15. What is p and what does it mean?**
p is the fraction of the work that runs in parallel, from Amdahl's law S = 1 / ((1 − p) + p/k). From our
speedups: p ≈ 0.98 for the search and 0.945 for cleaning. So even with infinite workers cleaning could not beat
~18× – the remaining 5.5% (overheads included) limits it.

**16. Why time a 1-worker pool as well?**
It does the same work as the sequential loop but through the pool, so the difference is pure overhead: cleaning
went from 125.8 s to 132.5 s (+5%).

**17. Why one warm-up run?**
The first run pays one-off costs: reading files from disk into the operating system's cache and importing
libraries in new processes. Discarding it makes every configuration start from the same state.

**18. Why single-threaded BLAS inside the workers?**
Otherwise each of 8 workers could start dozens of maths threads, oversubscribing the 8 cores and making timings
meaningless. Also a multi-threaded sum can split the work differently depending on the thread count, which can
change the last digits – a source of non-determinism.

## Model and time series

**19. Why not a random train/test split?**
Neighbouring hours are almost identical, so a random split lets the model "see" the hour right next to each test
hour – it looks great but fails in real use. We split by time: train to July 2026, validate August, test
September, and the search folds also always validate on later weeks.

**20. Where could leakage creep in, and how did you stop it?**
Rolling windows that include the current hour (we shift first, then roll), features from the predicted hour like
fares (we use none), a scaler fitted on all data (it is inside a Pipeline, so refitted per fold), and choosing
the model on the test month (we choose on August, test once). The notebook asserts that `lag_1` is the previous
hour and rolling windows end at t−1.

**21. Did you beat the baseline? The course example didn't.**
Yes: 5.4% error vs 13.7% for "same hour last week" (MAE 277 vs 698 pickups/hour). We forecast one hour ahead,
so the model knows the last few hours, which captures today's conditions; the baseline only knows last week.
We also had 22,000 training hours instead of 384, so gradient boosting could learn the daily × weekly pattern.

**22. Why gradient boosting over Ridge?**
Ridge is linear, so it can't represent interactions like "8 am is busy on weekdays but not Sundays" without extra
features. Trees learn them. On the August validation month HGB had RMSE 378 vs 485 for Ridge.

**23. Why MAE, RMSE and WAPE?**
MAE is the average miss in pickups per hour – easy to explain. RMSE punishes big misses more, which matters for
fleet planning. WAPE (total miss ÷ total pickups) gives a percentage that compares across busy and quiet hours,
and it's the measure the course example used.

## Non-determinism

**24. Which sources did you show and how did you fix them?**
(1) Completion order: combining months as they finish gives a different row order each run → sort by month (or use
`map`). (2) Float reduction order: adding the 24,096 hourly sums of squared fares in the order months finish changes the last digits → fixed
order, or `math.fsum`. (3) Unseeded random search → base seed 42 and a per-trial seed 42 + trial id. Each was
shown twice broken (different) and twice fixed (identical).

**25. What are the limits of your fixes?**
Sorting means waiting for all shards before combining and costs a sort. A fixed sum order is repeatable but not
more accurate; `fsum` is exact but slower. Seeds give the same result only for the same task list, library
versions and data, and one seed hides how much results really vary.

**26. You added a random sleep in the demo – isn't that cheating?**
It only makes the effect visible every time. On a busy shared server workers finish in different orders anyway;
the sleep just guarantees a demo. The fixed version has the same sleep and is still identical.

## Bigger picture

**27. What would change with 100× more data?**
Months would not fit in memory, so we would stream with `iter_batches` and add partial hourly counts (map–reduce
with sums instead of concatenation), shard by row group for balance, and probably use more machines. The search
would stay the same size because the hourly series barely grows.

**28. What would you do with more time?**
Forecast per zone or per borough (much better task-parallel workload), add weather and event data, forecast more
than one hour ahead, and run the benchmarks at a quieter time or with more repeats.

**29. Is the notebook reproducible?**
Yes: one config file, fixed seeds, single-threaded maths, ordered combination. We ran it twice through Slurm and
a script confirmed identical results apart from timings and the deliberately broken demos.

**30. Did you use AI?**
Yes, as the module allows: an AI coding agent wrote much of the code under our direction from a brief we wrote.
We reviewed every checkpoint, and `docs/AI_USE.md` describes it. *(Each of you: be ready to say what you
personally checked or changed.)*
