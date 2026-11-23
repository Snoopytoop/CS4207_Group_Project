#!/bin/bash
# Runs the timing benchmarks (sequential vs 1/2/4/8 workers, warm-up + >=3 repeats).
# Usage (on Rockhill, inside ~/cs4207-group11-demand):  mkdir -p logs && sbatch run_benchmark.slurm
#SBATCH --job-name=g11_bench
#SBATCH --partition=cs4207
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err

set -euo pipefail
source ~/venv/bin/activate
cd ~/cs4207-group11-demand

# Single-threaded maths inside each worker so the comparison is fair and deterministic
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "Host: $(hostname)  CPUs: ${SLURM_CPUS_PER_TASK:-?}  Start: $(date)"
lscpu | grep -E "Model name|^CPU\(s\)" || true
echo "Queue load at start:"; squeue | wc -l
OMP_NUM_THREADS=1 python scripts/benchmark.py
echo "Finished: $(date)"
