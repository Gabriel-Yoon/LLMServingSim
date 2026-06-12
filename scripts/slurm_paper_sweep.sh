#!/bin/bash
#SBATCH --job-name=llm_paper_sweep
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=logs/paper_sweep_%j.out
#SBATCH --error=logs/paper_sweep_%j.err

# ── Paper sweep for ASP-DAC 2027 ─────────────────────────────────────────────
# Runs all Exp 1/1s/2/3 combinations sequentially.
# Each (topology, N) run takes ~18 min on HPC → total ~15 hours for full sweep.
# Wall time 24h provides margin.
#
# Usage:
#   sbatch scripts/slurm_paper_sweep.sh              # all experiments
#   sbatch scripts/slurm_paper_sweep.sh 1            # Exp 1 only
#   sbatch scripts/slurm_paper_sweep.sh 1 2          # Exp 1 and 2
#
# Resume / partial: already-completed runs are skipped (results CSV tracks them).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs

echo "=== LLMServingSim paper sweep ==="
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Start      : $(date)"
echo "  Repo root  : $REPO_ROOT"
echo "  Git commit : $(git rev-parse --short HEAD)"
echo "  CPUs       : $SLURM_CPUS_PER_TASK"
echo ""

# Activate conda/venv if needed — adjust to your HPC environment
# source /path/to/venv/bin/activate
# conda activate llmservingsim

# Determine which experiments to run (default: all)
if [[ $# -gt 0 ]]; then
    EXP_ARGS=("$@")
else
    EXP_ARGS=()
fi

# Build experiment selector string
if [[ ${#EXP_ARGS[@]} -gt 0 ]]; then
    EXP_FLAG="--exp ${EXP_ARGS[*]}"
else
    EXP_FLAG=""
fi

echo "Running: python scripts/sweep_paper.py $EXP_FLAG"
echo ""

python scripts/sweep_paper.py $EXP_FLAG

echo ""
echo "=== Done: $(date) ==="
