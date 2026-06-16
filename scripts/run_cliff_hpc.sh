#!/usr/bin/env bash
# H2 reach-cliff sweep for HPC (native x86, large RAM).
#
# WHY HPC: the MoE all-to-all Chakra .et graph grows super-linearly with NPU
# count under MOE_ALLTOALL. Measured peak: 32 NPU ~1.4 GiB, 64 NPU ~7.1 GiB;
# EP=128 (the cliff point, first EP>rack64) extrapolates to ~20-35 GiB and OOMs
# on the 7.65 GiB local Docker container even at max (~14 GiB). Run it where
# RAM >= ~32 GiB and ASTRA runs native x86 (no Rosetta).
#
# The cliff: NVL72 caps its NVLink domain at rack=64; EP>64 crosses inter-rack
# IB (50 GB/s) -> all-to-all explodes. Glass-FB panels stay on optical
# (512 GB/s) -> flat. Expect NVL72 TPOT/exposed to jump at EP=128 while glass
# holds. Plot with: python scripts/plot_reach.py --csv <out> --rack 64
#
# Local-validated in-domain anchor (EP 8-32, Qwen3-235B, batch4/isl256/osl24):
#   glass  ~20.4 ms (flat) ; nvl72 20.8 -> 24.1 ms (rises, latency-bound).
# This HPC run extends the same sweep through the EP=128 cliff.
set -euo pipefail
cd "$(dirname "$0")/.."

EP_LIST="8 16 32 64 128"
PANEL="4 4"          # glass panel 4x4 (16 GPUs); WG cap 5
WG=5
RACK=64              # NVL72 NVLink-domain boundary (real)
BATCH=4
ISL=256
OSL=24               # enough steady-decode samples for tpot_gt; keep small for speed
TIMEOUT=10800        # 3h per config

run_model () {
  local model="$1" memgb="$2" tag="$3"
  echo "=== reach cliff: $model (npu-mem ${memgb}GB) ==="
  MOE_ALLTOALL=1 python scripts/sweep_panel_dse.py --sweep epscale \
    --ep-list $EP_LIST --nvl72-rack $RACK --epscale-panel $PANEL --fixed-wg $WG \
    --batch-per-instance $BATCH --isl $ISL --osl $OSL --mode controlled \
    --model "$model" --hardware H100 --tp 1 --npu-mem-gb "$memgb" \
    --out "outputs/panel_dse/reach_${tag}.csv" --timeout $TIMEOUT
  python scripts/plot_reach.py --csv "outputs/panel_dse/reach_${tag}.csv" --rack $RACK \
    --name "f3_reach_${tag}"
}

# ViBE headline models (both have tp1 profiles; TP=1 reach path avoids the
# TP>1 + dp_group ASTRA deadlock). DeepSeek-V3 is heavy -> needs >=1024 GB npu-mem.
run_model "Qwen/Qwen3-235B-A22B"          1024 "qwen235b"
run_model "deepseek-ai/DeepSeek-V3-0324"  1024 "deepseek_v3"
