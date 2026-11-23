# Rubric checklist – Group 11 (ticked with the notebook section that proves it)

Source: `reference/rubric.md`. **Results claimed on the poster must be traceable to the notebook.**

## 1. Data cleaning and preprocessing – 10%
- [x] Parquet loaded efficiently: footer inspection (§2, and footer null counts §3), column pruning (§3), `iter_batches` verified equal (§3)
- [x] Meter errors handled (281,576-mile trip) with stated threshold of 100 miles (§3)
- [x] Negative/zero fares removed (§3)
- [x] Zero-distance trips removed (§3)
- [x] Zero/negative/implausible durations removed (< 1 min, > 3 h) (§3)
- [x] Out-of-month spill-over pickups removed (§3)
- [x] Flex Fare nulls treated deliberately and justified: payment_type 0 = null RatecodeID, kept as demand (§3)
- [x] Zone and rate codes encoded as categories (§3, `encode_categories`)
- [x] Before/after row counts per rule reported (§4 cleaning table, `outputs/cleaning_table.csv`, per month in `cleaning_by_month.csv`)
- [x] Missing hours handled and justified (3 DST hours filled with 0) (§3, §4)
- [x] Time-based train/val/test split; no leakage (asserted) (§5)

## 2. Sequential vs parallel – 15%
- [x] Identical work; outputs checked equal, `True` printed (§4, §6, §7; asserted on every benchmark run)
- [x] Wall-clock timing, warm-up discarded, mean of 3 repeats (§9, `scripts/benchmark.py`)
- [x] Speedup and efficiency for 2, 4, 8 workers (+ 1-worker pool) (§9)
- [x] Chart of speedup vs workers with ideal line (`figures/speedup.png`)
- [x] Amdahl's law: p per workload, max speedup, overheads discussed (§9)

## 3. Data parallelism – 10%
- [x] Sharding by month, justified, balance discussed (§4)
- [x] Same function (`clean_month`) on every shard (§4)
- [x] Combined in fixed order (sorted concat, stats table) (§4)
- [x] Verified equal to sequential (§4)
- [x] Memory considered: peak RSS per worker, column pruning, `iter_batches` (§3, §4)

## 4. Task parallelism (any 2 of 3) – 20%
- [x] Feature extraction: 3 independent tasks, fixed column order, verified, timed (§6)
- [x] Hyperparameter search: 96 (setting × fold) tasks, best chosen on validation, timed, verified (§7)
- [ ] (Optional) Gradient computation – not done
- [x] Each part explains why the work is independent (§6, §7)
- [x] Load imbalance discussed with a task duration table/chart (§7, `figures/task_durations.png`)

## 5. Non-determinism (any 2) and fixes – 15%
- [x] Completion order: 2 differing runs → sort by month → identical (§10)
- [x] Float reduction order: 2 differing runs → fixed order / fsum → identical (§10)
- [x] Unseeded randomness: 2 differing runs → seeds → identical (§10)
- [x] Thread-count BLAS: prevented by pinning 1 thread (§10, §1)
- [x] Each fix explained with its limits (§10)

## 6. Reproducible notebook – 10%
- [x] Runs end to end under Slurm without errors; run twice, `compare_results.py` → identical (§12)
- [x] Python and library versions; `requirements.txt`; data source and file names (§1, §2, §12)
- [x] Seeds, worker counts, paths, thresholds in `config.py` (§1)
- [x] Section headings, markdown explanations, reusable functions, tidy outputs
- [x] Run time and hardware noted (§1, §12)
- [ ] Results match the poster – re-check after any re-run (Phase 11)

## 7. Poster presentation – 20%
- [x] Problem and dataset stated (panel 1)
- [x] Pipeline diagram with data/task parallelism colour-coded (panel 4)
- [x] Speedup chart and key results (panels 5, 6)
- [x] Non-determinism findings and fixes (panel 7)
- [ ] Readable layout – humans to review the printed/PDF version
- [x] Every number generated from `outputs/results.json` by `scripts/make_poster.py`
- [ ] Group can answer questions confidently (study EXPLAINER + QA_PREP, rehearse to 5:00)
