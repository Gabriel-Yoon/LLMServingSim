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

# H100-CONSISTENT baseline by DEFAULT (compute is H100-profiled): NVLink4 450 GB/s
# unidir, 8-GPU HGX NVLink island, cross-island IB 50 -> cliff at EP>8. Glass panel
# 4x4 (16 GPU) on optical 512 GB/s (--fixed-wg 8 = per-dir 4, feasible floor).
# GB200 sensitivity: RACK=64 NVLBW=900 WG=10 bash scripts/run_cliff_hpc.sh
EP_LIST="${EP_LIST:-8 16 32 64 128}"
PANEL="${PANEL:-4 4}"            # glass panel 4x4 (16 GPUs)
WG="${WG:-8}"                    # TOTAL bundle (even); per-dir 4 = 512 GB/s (4x4 floor)
RACK="${RACK:-8}"                # H100 NVLink island (GB200 sensitivity: 64)
NVLBW="${NVLBW:-450}"            # NVLink4 unidir (GB200: 900)
BATCH="${BATCH:-16}"             # decode concurrency/instance; larger -> bigger a2a -> clearer cliff
ISL="${ISL:-256}"
OSL="${OSL:-24}"                 # enough steady-decode samples for tpot_gt
MEM="${MEM:-1024}"               # fake-large to isolate the interconnect (fabric study)
TIMEOUT="${TIMEOUT:-10800}"

run_model () {
  local model="$1" tag="$2"
  echo "=== reach cliff: $model  (rack=$RACK nvlbw=$NVLBW wg=$WG) ==="
  MOE_ALLTOALL=1 python scripts/sweep_panel_dse.py --sweep epscale \
    --ep-list $EP_LIST --nvl72-rack $RACK --nvl72-bw $NVLBW --agg-bw \
    --epscale-panel $PANEL --fixed-wg $WG \
    --batch-per-instance $BATCH --isl $ISL --osl $OSL --mode controlled \
    --model "$model" --hardware H100 --tp 1 --npu-mem-gb "$MEM" \
    --out "outputs/panel_dse/reach_${tag}.csv" --timeout $TIMEOUT
  python scripts/plot_reach.py --csv "outputs/panel_dse/reach_${tag}.csv" --rack $RACK \
    --name "f3_reach_${tag}" || true
}

# ViBE headline models (both have tp1 profiles; TP=1 reach path avoids the
# TP>1 + dp_group ASTRA deadlock). MLA fix means DeepSeek fits at far less memory.
run_model "Qwen/Qwen3-235B-A22B"          "qwen235b"
run_model "deepseek-ai/DeepSeek-V3-0324"  "deepseek_v3"
