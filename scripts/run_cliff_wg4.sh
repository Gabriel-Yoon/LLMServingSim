#!/usr/bin/env bash
# Robustness check for the H2 reach cliff at the CONSERVATIVE feasible WG cap.
#
# 4x4 panel micro-bump budget = 9 waveguides/pair (total, both directions), so
# N_WG <= 4.5 per direction. The headline used wg=5 (the ceil -> 10 WG/pair =
# slightly AGGRESSIVE, above the 9 budget). This re-runs at wg=4:
#   wg=4 per direction = 4 TX + 4 RX = 8 WG/pair <= 9 budget  => strictly FEASIBLE
#   and an even/symmetric TX:RX split.
#
# Convention: --fixed-wg is N_WG PER DIRECTION; intra_opt_bw = wg x 128 GB/s.
#   wg=4 -> 512 GB/s intra-panel optical (vs 640 at wg=5). Inter-panel optical
#   stays 512 GB/s. Reach is cross-domain-dominated, so the cliff should hold;
#   this nails down "glass wins even at the conservative feasible WG cap."
#
# Outputs to *_wg4.csv (does NOT overwrite the wg5 headline CSVs) and prints the
# check_reach verdict + saves a plot per model.
set -euo pipefail
cd "$(dirname "$0")/.."

EP_LIST="8 16 32 64 128"
PANEL="4 4"
WG=4                 # per-direction N_WG (feasible floor cap for 4x4; even TX/RX).
                     # This is now the PRIMARY/physical config (wg5 is odd = non-physical;
                     # see feedback_wg_even). The wg5 framing in older comments is historical.
RACK=64
BATCH=64             # decode concurrency/instance. batch16 leaves the MoE a2a ~3% of the
                     # step (compute-bound) -> cliff is MUTED in TPOT; batch64 lifts the
                     # exposed-comm fraction (~15-25%) so the EP>rack cliff is visible.
ISL=256
OSL=24
TIMEOUT=10800

run_model () {
  local model="$1" tag="$2"
  echo "=== reach cliff @ wg=4 (feasible cap): $model ==="
  MOE_ALLTOALL=1 python scripts/sweep_panel_dse.py --sweep epscale \
    --ep-list $EP_LIST --nvl72-rack $RACK --epscale-panel $PANEL --fixed-wg $WG \
    --batch-per-instance $BATCH --isl $ISL --osl $OSL --mode controlled \
    --model "$model" --hardware H100 --tp 1 --npu-mem-gb 1024 \
    --out "outputs/panel_dse/reach_${tag}_wg4.csv" --timeout $TIMEOUT
  python scripts/check_reach.py --csv "outputs/panel_dse/reach_${tag}_wg4.csv" --rack $RACK
  python scripts/plot_reach.py --csv "outputs/panel_dse/reach_${tag}_wg4.csv" --rack $RACK \
    --name "f3_reach_${tag}_wg4"
}

run_model "Qwen/Qwen3-235B-A22B"          qwen235b
run_model "deepseek-ai/DeepSeek-V3-0324"  deepseek_v3

echo "=== wg=4 robustness done. Compare *_wg4.csv vs the wg5 headline CSVs. ==="
echo "Expect the cliff to HOLD (glass < NVL72 at EP128); glass intra drops 640->512"
echo "GB/s but reach is set by the cross-domain link, so the verdict should persist."
