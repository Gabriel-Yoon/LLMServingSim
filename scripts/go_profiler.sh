#!/bin/bash
# Activate the vLLM profiling environment on a bare-metal compute node.
#
# Do NOT run inside Apptainer — the vllm_venv relies on the host CUDA
# toolkit (nvcc via Spack) for FlashInfer JIT compilation, which is not
# accessible inside the container.
#
# Usage (on HPC compute node, after salloc):
#   source ./scripts/go_profiler.sh
#   SKIP_SKEW=1 ./profiler/profile_h200_new.sh
#   SKIP_SKEW=1 ./profiler/profile_h100_new.sh
#
# Or run a single model:
#   source ./scripts/go_profiler.sh && MODEL_FILTER=deepseek SKIP_SKEW=1 ./profiler/profile_h200_new.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

if [[ -z "${VLLM_VENV:-}" || ! -f "$VLLM_VENV/bin/activate" ]]; then
    echo "ERROR: vllm_venv not found at $VLLM_VENV"
    echo "  Expected: $SCRATCH/deps/vllm_venv"
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "$VLLM_VENV/bin/activate"

cd "$LLMSIM_REPO"

echo "=== Profiling environment ready ==="
echo "  venv   : $VLLM_VENV"
echo "  repo   : $LLMSIM_REPO"
echo "  python : $(python3 --version)"
echo "  nvcc   : $(nvcc --version 2>/dev/null | grep 'release' || echo 'NOT FOUND — FlashInfer JIT will fail')"
echo ""
echo "Run:"
echo "  SKIP_SKEW=1 ./profiler/profile_h200_new.sh"
echo "  SKIP_SKEW=1 ./profiler/profile_h100_new.sh"
