#!/bin/bash
# TP=8 profiling — Qwen/Qwen3-235B-A22B on H200 (ViBE-aligned TP=8+EP=8).
# Skew is SEPARATED into two passes (run pass 1, then pass 2):
#   SKIP_SKEW=1 scripts/profile_h200_qwen3_235b_tp8.sh   # pass 1: dense/attention/moe (fast)
#   ONLY_SKEW=1 scripts/profile_h200_qwen3_235b_tp8.sh   # pass 2: skew sweep + alpha fit
# Resume-safe: tp1 exists and is skipped; only tp8 is fired.
# NOTE: Qwen3-235B has num_key_value_heads=4 < tp=8. The profiler now emulates
# vLLM's GQA KV replication (1 KV head/rank) for this case (engine.py); per-rank
# shape becomes 8 Q : 1 KV. No action needed, just be aware it is replicated.
set -euo pipefail
export HARDWARE="H200"
export MODEL="Qwen/Qwen3-235B-A22B"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/profiler/profile_tp8_run.sh"
