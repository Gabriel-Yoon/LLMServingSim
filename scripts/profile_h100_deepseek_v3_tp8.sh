#!/bin/bash
# TP=8 profiling — deepseek-ai/DeepSeek-V3-0324 on H100 (ViBE-aligned TP=8+EP=8).
# Skew is SEPARATED into two passes (run pass 1, then pass 2):
#   SKIP_SKEW=1 scripts/profile_h100_deepseek_v3_tp8.sh   # pass 1: dense/attention/moe (fast)
#   ONLY_SKEW=1 scripts/profile_h100_deepseek_v3_tp8.sh   # pass 2: skew sweep + alpha fit
# Resume-safe: tp1 exists and is skipped; only tp8 is fired.
set -euo pipefail
export HARDWARE="H100"
export MODEL="deepseek-ai/DeepSeek-V3-0324"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/profiler/profile_tp8_run.sh"
