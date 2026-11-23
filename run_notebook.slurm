#!/bin/bash
# Executes the submission notebook on Rockhill through Slurm (CPU only, no GPU).
# Usage (on Rockhill, inside ~/cs4207-group11-demand):  mkdir -p logs && sbatch run_notebook.slurm
#SBATCH --job-name=g11_notebook
#SBATCH --partition=cs4207
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/notebook_%j.out
#SBATCH --error=logs/notebook_%j.err

set -euo pipefail
source ~/venv/bin/activate
cd ~/cs4207-group11-demand

echo "Host: $(hostname)  CPUs allocated: ${SLURM_CPUS_PER_TASK:-?}  Start: $(date)"
START=$(date +%s)
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3000 demand_forecast.ipynb
echo "Finished: $(date)  Wall time: $(( $(date +%s) - START )) s"
