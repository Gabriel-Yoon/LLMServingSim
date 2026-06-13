#!/bin/bash
# HPC sanity check — runs EP=32 mini test (ISL=10, OSL=5, N=1) inside Apptainer
# before committing to a full paper sweep.
#
# Usage (from repo root on PACE):
#   bash scripts/hpc_sanity_check.sh
#
# Pass:  exits 0, prints [PASS] with elapsed / TPOT
# Fail:  exits non-zero, prints [FAIL] with error detail

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

APPTAINER_IMAGE="/storage/project/r-syu334-0/syoon351/containers/llmservingsim.sif"
SCRATCH="/storage/home/hcoda1/8/syoon351/scratch"
VENV="$SCRATCH/deps/apptainer_home/venvs/llmservingsim"

echo "=== LLMServingSim HPC Sanity Check ==="
echo "Repo       : $REPO_ROOT"
echo "Git commit : $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "Image      : $APPTAINER_IMAGE"
echo "VENV       : $VENV"
echo "Start      : $(date)"
echo ""

mkdir -p "$REPO_ROOT/outputs"

apptainer exec \
    --cleanenv \
    --home "$SCRATCH/deps/apptainer_home":/home/syoon351 \
    --bind "$REPO_ROOT":/app/LLMServingSim \
    --bind "$SCRATCH":"$SCRATCH" \
    --pwd /app/LLMServingSim \
    "$APPTAINER_IMAGE" \
    bash -lc "
        set -euo pipefail
        source $VENV/bin/activate
        echo '--- Python / simulator versions ---'
        python --version
        python -c 'import serving; print(\"serving module OK\")'  2>/dev/null || true
        echo ''
        echo '--- Running EP=32 mini test (ISL=10, OSL=5, N=1) ---'
        python scripts/test_exp1_mini.py --topo nvl72_ep32
    "

echo ""
echo "End : $(date)"
