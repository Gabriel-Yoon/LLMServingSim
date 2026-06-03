#!/bin/bash
set -euo pipefail

module load cuda/12.6.1
module load uv/0.9.17

source /storage/home/hcoda1/8/syoon351/scratch/deps/vllm_venv/bin/activate

export UV_CACHE_DIR=/storage/home/hcoda1/8/syoon351/scratch/uv_cache
export HF_HOME=/storage/home/hcoda1/8/syoon351/scratch/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python3 -m profiler profile "mistralai/Mixtral-8x7B-v0.1" \
  --hardware "H200" --tp "1" \
  --max-num-batched-tokens 8192 --max-num-seqs 256 \
  --attention-max-kv 16384 --attention-chunk-factor 2.0 \
  --attention-kv-factor 2.0 --measurement-iterations 3 \
  --skew-n-factor 2.0 --skew-pc-factor 2.0 \
  --skew-kp-factor 2.0 --skew-kvs-factor 2.0
