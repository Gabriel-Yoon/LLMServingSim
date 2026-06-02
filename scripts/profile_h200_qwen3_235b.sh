#!/bin/bash
set -euo pipefail

module load cuda/12.6.1
module load uv/0.9.17

source /storage/home/hcoda1/8/syoon351/scratch/deps/vllm_venv/bin/activate

export UV_CACHE_DIR=/storage/home/hcoda1/8/syoon351/scratch/uv_cache
export HF_HOME=/storage/home/hcoda1/8/syoon351/scratch/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub

cd /storage/scratch1/8/syoon351/repos/LLMServingSim

MODEL="Qwen/Qwen3-235B-A22B" \
HARDWARE="H200" \
TP_DEGREES="1,2" \
./profiler/profile.sh
