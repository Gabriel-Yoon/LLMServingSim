#!/usr/bin/env bash
# PREFILL reach — the comm-heavier regime. Decode (run_cliff_wg4.sh) is
# compute-bound for big MoE, so the IB cliff is muted in decode TPOT. Prefill
# moves a large MoE all-to-all per chunk (here a single 2048-token chunk, ~32x
# the decode message), and weight loading is amortised, so communication is a
# bigger fraction -> the cross-domain link shows up in TTFT (prefill latency).
#
# Two passes, so we separate the two open questions:
#   per-link  (default): ASTRA's congestion-unaware per-pair bandwidth. NVLink 900
#     (per-GPU AGGREGATE, modelled as per-pair) vs glass 512 (one WG bundle) ->
#     under-credits glass's many-link aggregate.
#   AGG=1 (--agg-bw): glass within-panel bw = per-link x degree = aggregate egress,
#     comparable to NVL72's aggregate-900. Tests whether the per-pair model was the
#     thing hiding glass's bandwidth advantage in prefill.
#
# PREREQUISITE: bash scripts/apply_astra_overlay.sh
set -euo pipefail
cd "$(dirname "$0")/.."

EP_LIST="8 32 64 128"
PANEL="4 4"; WG=8; RACK=64   # --fixed-wg = TOTAL bundle (x8 = per-direction 4 = 512 GB/s, feasible 4x4 floor)
ISL=2048          # large prompt -> large prefill MoE all-to-all
OSL=4             # few decode steps (prefill is the focus; TTFT is the metric)
MAXTOK=8192       # prefill chunk budget >= ISL -> one big chunk -> full 2048-token a2a
BATCH=8
MEM=1024
TIMEOUT=10800

run_model () {
  local model="$1" tag="$2" aggflag="$3" suffix="$4"
  echo "=== PREFILL reach ($suffix): $model ==="
  MOE_ALLTOALL=1 python scripts/sweep_panel_dse.py --sweep epscale \
    --ep-list $EP_LIST --nvl72-rack $RACK --epscale-panel $PANEL --fixed-wg $WG \
    --batch-per-instance $BATCH --isl $ISL --osl $OSL --max-tokens $MAXTOK \
    --mode controlled $aggflag \
    --model "$model" --hardware H100 --tp 1 --npu-mem-gb $MEM \
    --out "outputs/panel_dse/prefill_${tag}_${suffix}.csv" --timeout $TIMEOUT
}

for m in "Qwen/Qwen3-235B-A22B:qwen235b" "deepseek-ai/DeepSeek-V3-0324:deepseek_v3"; do
  model="${m%%:*}"; tag="${m#*:}"
  run_model "$model" "$tag" ""          "perlink"   # per-pair bw (as the decode reach)
  run_model "$model" "$tag" "--agg-bw"  "aggbw"     # aggregate-egress fairness
done

echo "=== prefill reach done. Compare TTFT (=prefill latency) glass vs NVL72 across EP."
echo "Key: does NVL72 TTFT jump at EP128 (IB), and does glass beat it under --agg-bw?"
echo "CSVs: outputs/panel_dse/prefill_*_{perlink,aggbw}.csv"
