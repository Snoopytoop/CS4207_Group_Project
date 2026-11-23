# 5-minute presentation script (~650 words)

Panels are numbered 1–7 on the poster. Practise with a timer: **5:00 is a hard limit**. Speak to the marker,
point at the panel, and do not read numbers you can't explain (see `QA_PREP.md`).

| Time | Speaker | Part |
|---|---|---|
| 0:00–0:30 | Saif | Problem |
| 0:30–1:15 | Samuel | Data and cleaning (panels 1–2) |
| 1:15–2:45 | Simon | Parallel design and split (panels 3–4) |
| 2:45–3:30 | Iwan | Speedup (panel 6) |
| 3:30–4:45 | Kevin | Results and non-determinism (panels 5, 7) |
| 4:45–5:00 | Saif | Close |

---

**Saif – Problem (0:30)**
"How many taxis will New York need in the next hour? If a fleet knows that, it can move drivers *before* demand
arrives. So our task is time-series forecasting: predict citywide yellow-taxi pickups one hour ahead. And because
this is an advanced programming module, we built the whole pipeline twice – sequentially and in parallel – and
measured the difference."

**Samuel – Data and cleaning (0:45)** *(point at panels 1 and 2)*
"We used all 33 monthly files on Rockhill, January 2024 to September 2026: 123.5 million trips. We read the
Parquet footers first to get sizes without loading data, and we read only 10 columns. The cleaning rules run in
this order: pickups outside the file's month, trips under a minute or over three hours, zero distance or over
100 miles – one trip claimed 281,000 miles – and zero or negative fares. That keeps 115.4 million trips.
One deliberate choice: Flex Fare trips have no rate code, but they're real pickups, so we kept them as demand.
Then we count pickups per hour: 24,096 hours."

**Simon – Parallel design and split (1:30)** *(point at panel 4, then 3)*
"Here's where the parallelism is. Blue is **data parallelism**: the same function, `clean_month`, runs on every
month in its own process – one shard per month, because that's how the data is stored and months don't overlap
in time. We merge the results sorted by month, and we check the merged series is *identical* to the sequential
one.
Orange is **task parallelism**. First, three independent feature tasks – lags, rolling windows, calendar – joined
back in a fixed column order. Second, the hyperparameter search: 24 settings times 4 folds is 96 independent
fits; each one trains its own model, so they can run in any order.
And panel 3: the split is by time, never random. We train to July 2026, choose the model on August, and test on
September exactly once. Every feature only looks at earlier hours – we shift before computing rolling windows."

**Iwan – Speedup (0:45)** *(point at panel 6)*
"On 8 workers the search runs 6.9 times faster and cleaning 5.8 times – each is the mean of three runs after a
warm-up, and every run was checked equal to sequential. Why not 8? Amdahl's law: from the speedup, about 98% of
the search is parallel, 94.5% of cleaning – which caps cleaning at about 18 times however many workers we add.
The rest is process start-up, copying results, all workers reading the same disk, the last month running on its
own, and a busy shared server. And the feature tasks got *slower* – 18 milliseconds of work can't pay for starting
processes."

**Kevin – Results and non-determinism (1:15)** *(point at panel 5, then 7)*
"Results on the September test month: gradient boosting misses by 5.4% of actual volume. The best no-model
baseline, 'same hour last week', misses by 13.7%. So our average error is 60% lower. The model wins because it
knows the last few hours – what's happening today – while the baseline only knows last week.
Panel 7: non-determinism. If we combine months in the order workers finish, the series comes out in a different
order every run, and the total fare changes in its last digits, because floating-point addition isn't
associative. Fix: sort by month first. An unseeded random search picks a different 'best' model each run; fix:
seed 42 plus the trial number. After each fix, two re-runs were identical."

**Saif – Close (0:15)**
"So: parallel results identical to sequential, close to 7 times faster where the work is big enough, and a
forecast that clearly beats the baseline. Every number on this poster comes from our notebook. Thank you – happy
to take questions."
