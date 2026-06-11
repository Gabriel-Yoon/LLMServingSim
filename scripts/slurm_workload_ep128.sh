#!/bin/bash
# SLURM job script for EP=128 workload sweep on HPC.
#
# Usage:
#   sbatch scripts/slurm_workload_ep128.sh chat
#   sbatch scripts/slurm_workload_ep128.sh coding
#   sbatch scripts/slurm_workload_ep128.sh agentic
#   sbatch scripts/slurm_workload_ep128.sh all   # all workloads in one job
#
# Estimated wall times (EP=128, NVL72 + FB_4x4):
#   chat    ~10hr  (NUM_REQ=3, OSL=256)
#   coding  ~36hr  (NUM_REQ=3, OSL=1024)
#   agentic ~10hr  (NUM_REQ=3, OSL=256, prefix=64k)
#
# Adjust --time, --partition, --gres according to your cluster.
# This script does NOT require a GPU — the simulator is CPU-only.

#SBATCH --job-name=llmsim_ep128
#SBATCH --output=outputs/slurm_%x_%j.log
#SBATCH --error=outputs/slurm_%x_%j.err
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu      # change to your cluster's CPU partition

# ── Environment ──────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKLOAD="${1:-all}"   # chat | coding | agentic | all

echo "=== LLMServingSim EP=128 Workload Sweep ==="
echo "Repo     : $REPO_ROOT"
echo "Workload : $WORKLOAD"
echo "Start    : $(date)"
echo ""

cd "$REPO_ROOT" || { echo "Cannot cd to $REPO_ROOT"; exit 1; }

# ── Activate environment ──────────────────────────────────────────────────────
# Option A: conda
# conda activate llmservingsim

# Option B: Docker (if available on HPC)
# docker exec servingsim_docker bash -c "cd /app/LLMServingSim && python scripts/sweep_workload_ep128.py --workload $WORKLOAD"
# exit $?

# Option C: native venv (most common on HPC)
if [ -f "scripts/setup_env.sh" ]; then
    source scripts/setup_env.sh
fi

# Make sure outputs/ and results/ exist
mkdir -p outputs/sim_workload results

# ── Run sweep ─────────────────────────────────────────────────────────────────
python scripts/sweep_workload_ep128.py \
    --workload "$WORKLOAD" \
    --results-csv results/dse_workload_ep128.csv \
    --output-dir outputs/sim_workload

STATUS=$?
echo ""
echo "End : $(date)"
echo "Exit: $STATUS"
exit $STATUS
