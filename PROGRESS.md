# PROGRESS – Group 11 coding challenge

Read this first when resuming. Code lives in `project/` (lab PC) and is mirrored to `~/cs4207-group11-demand` on Rockhill.

## Mon 23 Nov 2026 – Phase 0: setup
- SSH key `~/.ssh/id_ed25519` created on the lab PC and installed on Rockhill by a human. Key login works (`ssh -o BatchMode=yes group11@rockhill.csis.ul.ie`).
- Rockhill: AMD EPYC 9124 (2 x 16 cores, 64 threads), 503 GB RAM, Python 3.12.3, soffice + pdftoppm available.
- Venv `~/venv` created: pandas 3.1.1, pyarrow 26.0.1, numpy 2.5.5, scikit-learn 1.9.1, matplotlib 3.11.2, jupytext 1.19.7, python-pptx 1.0.2, holidays 0.109, threadpoolctl 3.8.0.
- **Data differs from the brief:** 33 months on disk (2024-01 … 2026-09, ~123.5 M rows, ~2 GB), not 9. The brief says use what exists → all 33 months. Split: train Jan 2024–Jul 2026, validation Aug 2026, test Sep 2026.
- **Slurm partition:** `student` partition is up but the scheduler would not start any job before ~20:40 today (node in PLANNED state); the course partition `cs4207` (group11 is in group cs4207) starts jobs immediately. All jobs use `--partition=cs4207` with `--mem` and `--time`. Slurm here does not pin cores (jobs see all 64 CPUs), so the code reads `SLURM_CPUS_PER_TASK`.
- **Server is busy:** load average ~124 on 64 threads from long-running non-Slurm jobs of other users. This will reduce parallel speedups – recorded in the benchmark metadata.
- Pre-existing items in the group11 home (`~/envs`, `~/project`, `~/week9*`, `remote-*.sh`) were left untouched.
- Still waiting for from humans: GitHub repo URL, member names, poster size (using A1 portrait by default).

## Mon 23 Nov – Phases 1–5 (code + smoke test)
- Footer inspection of all 33 files; schema drift: `cbd_congestion_fee` from 2025-01, `request_source` from 2026-06 (not used).
- Flex Fare = `payment_type 0`; those rows are exactly the ones with null RatecodeID / passenger_count / surcharges. Kept as demand (RatecodeID → category with 'unknown').
- April 2026 cleaning reproduces the course worked example exactly (12 / 89,425 / 66,135 / 14,040 removed, 3,661,628 kept).
- All months: 123,471,721 raw → 115,358,059 kept; 24,096 hourly observations; only missing hours are the 3 DST spring-forward 02:00 hours.
- Smoke test: parallel cleaning (8 workers) 22 s vs ~140 s sequential estimate; HGB clearly beats lag-168 baseline on the last CV fold.
- Benchmark job 2011 submitted (cs4207, 8 CPUs, 32 GB, 1 h 30 limit).
