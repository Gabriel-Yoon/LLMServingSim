#!/bin/bash
# HPC sanity check — runs EP=32 mini test (ISL=10, OSL=5, N=1).
# Run this from inside Apptainer with venv already activated.
#
# Usage (from repo root):
#   bash scripts/hpc_sanity_check.sh
#
# Pass:  exits 0, prints [PASS] with elapsed / TPOT
# Fail:  exits non-zero, prints [FAIL] with error detail

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== LLMServingSim HPC Sanity Check ==="
echo "Repo       : $REPO_ROOT"
echo "Git commit : $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "Start      : $(date)"
echo ""

mkdir -p "$REPO_ROOT/outputs"

cd "$REPO_ROOT"
python scripts/test_exp1_mini.py --topo nvl72_ep32

echo ""
echo "End : $(date)"
