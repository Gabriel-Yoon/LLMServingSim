#!/bin/bash
#SBATCH --job-name=llmsim-profile
#SBATCH --account=REPLACE_ME           # e.g. gts-<pi-username> (check: pace-quota)
#SBATCH --partition=gpu-h100           # or gpu-a100 / gpu-rtx6000 (pace-check-queue)
#SBATCH --gres=gpu:1                   # ONE GPU is enough (TP emulated via shapes)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00                # 4 models x tp{1,2,4,8}; halve with SKIP_SKEW=1
#SBATCH --output=profile_%j.out

# Profile the Krishna-trend model set on a PACE GPU node.
#
# One-time setup (login node, before sbatch):
#   git clone <your-private-repo-url> LLMServingSim   # NO --recurse-submodules:
#   cd LLMServingSim && git checkout hybrid-fabric    # profiling never touches astra-sim
#   ./scripts/install-vllm.sh                         # bare-metal uv venv (no Docker on PACE)
#
# Submit:      sbatch profiler/pace-profile.sbatch
# Faster pass: sbatch --export=ALL,SKIP_SKEW=1 profiler/pace-profile.sbatch
# One model:   sbatch --export=ALL,ONLY="Qwen/Qwen3-235B-A22B" profiler/pace-profile.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?run via sbatch from the LLMServingSim repo root}"

# vLLM venv from scripts/install-vllm.sh (uv). Adjust if you used conda.
source .venv/bin/activate 2>/dev/null || source vllm-venv/bin/activate

# HARDWARE label = perf/ output folder name; derive from the allocated GPU.
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
case "$GPU_NAME" in
  *H200*) export HARDWARE=H200 ;;
  *H100*) export HARDWARE=H100 ;;
  *A100*) export HARDWARE=A100 ;;
  *"RTX 6000"*|*RTX6000*) export HARDWARE=RTX6000Ada ;;
  *) export HARDWARE="$(echo "$GPU_NAME" | tr -d ' ')" ;;
esac
echo "GPU: $GPU_NAME -> HARDWARE=$HARDWARE"

export TP_DEGREES="${TP_DEGREES:-1,2,4,8}"
./profiler/profile-krishna.sh

echo "Done. Commit the results:"
echo "  git add profiler/perf/$HARDWARE && git commit -m 'Add $HARDWARE profiles' && git push"
