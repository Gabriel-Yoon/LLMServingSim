#!/bin/bash
# SLURM: Throughput vs Interactivity sweep for EP=32 (FB-4x4 vs NVL72).
# CPU-only, no GPU required.
#
# Usage:
#   sbatch scripts/slurm_tpot_sweep.sh          # both topologies
#   sbatch scripts/slurm_tpot_sweep.sh fb        # FB only
#   sbatch scripts/slurm_tpot_sweep.sh nvl72     # NVL72 only
#
# Estimated wall time: ~3-4 hr (16 runs, N=1..128, OSL=50)

#SBATCH --job-name=llmsim_tpot
#SBATCH --output=outputs/slurm_%x_%j.log
#SBATCH --error=outputs/slurm_%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOPO="${1:-both}"

APPTAINER_IMAGE="/storage/project/r-syu334-0/syoon351/containers/llmservingsim.sif"
SCRATCH="/storage/home/hcoda1/8/syoon351/scratch"
VENV="$SCRATCH/deps/apptainer_home/venvs/llmservingsim"

echo "=== TPOT Throughput-Interactivity Sweep ==="
echo "Repo  : $REPO_ROOT"
echo "Topo  : $TOPO"
echo "Start : $(date)"

mkdir -p "$REPO_ROOT/outputs" "$REPO_ROOT/results"

apptainer exec \
    --cleanenv \
    --home "$SCRATCH/deps/apptainer_home":/home/syoon351 \
    --bind "$REPO_ROOT":/app/LLMServingSim \
    --bind "$SCRATCH":"$SCRATCH" \
    --pwd /app/LLMServingSim \
    "$APPTAINER_IMAGE" \
    bash -lc "
        source $VENV/bin/activate
        python scripts/sweep_tpot.py --topo $TOPO
    "

echo "End : $(date)"
echo "Results: $REPO_ROOT/results/exp_tpot_throughput_ep32.csv"
