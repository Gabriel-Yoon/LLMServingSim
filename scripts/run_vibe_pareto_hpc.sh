#!/bin/bash
# -----------------------------------------------------------------------------
# TP=8 ViBE single-instance throughput-vs-interactivity Pareto (REALISTIC absolute
# numbers). The topology/fabric sweeps are TP=1 (multi-instance, TP>1 deadlocks) so
# their TPOT is ~8x inflated; this run uses the SINGLE-INSTANCE TP=8/EP=8 path
# (configs/cluster/vibe_tp8_ep8_*.json) which runs cleanly and gives production-
# realistic TPOT/interactivity for the AIConfigurator-axes Pareto (plot_pareto.py).
#
# At EP=8 BOTH glass (512/100) and NVLink4 (450/500) are IN-DOMAIN -> fabric PARITY
# (no IB cliff; the cliff is EP>domain, which at TP=8 needs the multi-instance path).
# So this traces ONE realistic frontier (glass~=nvl4 overlap) -> the "where we
# operate" serving figure; the fabric gap stays the analytical collective/topology fig.
#
# Sweeps concurrency B (= total concurrent requests on the 8-GPU instance). For each
# B: max_num_seqs=B, num_reqs=B, controlled burst (arrival t=0), prefix-cache OFF for
# a clean per-request compute. We record MEDIAN TPOT + TTFT into a plot_pareto CSV
# with per_device_batch=B; plot_pareto --tp 8 then computes tok/s/GPU = B/8 * 1000/TPOT
# (TotalGPUs=8 since TP and EP SHARE the 8 GPUs). Use --decode-only (controlled TTFT
# is queue-inflated).  Run on HPC (sim container, bare-metal python).
#
#   python scripts/plot_pareto.py --osl <OSL> --tp 8 --decode-only \
#     outputs/panel_dse/vibe_pareto_deepseek.csv \
#     --out outputs/paper_figures/fig_vibe_pareto_tp8_deepseek.png
# -----------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL_TAG="${MODEL_TAG:-deepseek}"                  # deepseek | qwen (config suffix)
GLASS_CFG="configs/cluster/vibe_tp8_ep8_${MODEL_TAG}_glass.json"
NVL_CFG="configs/cluster/vibe_tp8_ep8_${MODEL_TAG}.json"
ISL="${ISL:-1024}"; OSL="${OSL:-128}"               # ViBE Sonnet-like
read -r -a BATCHES <<< "${BATCHES:-8 16 32 64 128}"
TIMEOUT="${TIMEOUT:-3600}"
OUT="outputs/panel_dse/vibe_pareto_${MODEL_TAG}.csv"
mkdir -p outputs/panel_dse "$(dirname "$GLASS_CFG")"

# glass config = nvl config with link 512/100 (auto-derive if missing)
if [ ! -s "$GLASS_CFG" ]; then
  sed 's/"link_bw": 450/"link_bw": 512/; s/"link_latency": 500/"link_latency": 100/' \
    "$NVL_CFG" > "$GLASS_CFG"
fi

echo "label,fabric,ep,per_device_batch,isl,osl,tpot_gt_ms,ttft_ms,status" > "$OUT"

WL="workloads/_vibe_wl.jsonl"   # repo-relative (serving prefixes ../ from astra-sim/)
gen_workload() {  # $1=B  -> writes $WL (B reqs, isl/osl, t=0)
  python - "$1" "$ISL" "$OSL" "$WL" <<'PY'
import json, sys
B, isl, osl, wl = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
with open(wl, "w") as f:
    for i in range(B):
        # unique-ish ids so prefix cache (off anyway) can't merge; lengths must match toks
        inp = list(range(i*isl+1, i*isl+1+isl))
        out = list(range(1, osl+1))
        f.write(json.dumps({"input_toks": isl, "output_toks": osl,
                            "arrival_time_ns": 0, "input_tok_ids": inp,
                            "output_tok_ids": out}) + "\n")
PY
}

for B in "${BATCHES[@]}"; do
  gen_workload "$B"
  for pair in "glass:$GLASS_CFG" "nvl72:$NVL_CFG"; do
    fab="${pair%%:*}"; cfg="${pair#*:}"
    echo "=== $MODEL_TAG $fab B=$B (isl$ISL/osl$OSL) ==="
    log=$(timeout "$TIMEOUT" python -m serving --cluster-config "$cfg" \
      --dtype bfloat16 --block-size 16 --dataset "$WL" \
      --max-num-seqs "$B" --num-reqs "$B" --no-enable-prefix-caching \
      --output "/tmp/vibe_${fab}_b${B}.csv" --log-level WARNING 2>&1)
    tpot=$(echo "$log" | grep -m1 "Median TPOT" | grep -oE "[0-9]+\.[0-9]+")
    ttft=$(echo "$log" | grep -m1 "Median TTFT" | grep -oE "[0-9]+\.[0-9]+")
    if [ -n "$tpot" ]; then
      echo "vibe_${fab}_b${B},$fab,8,$B,$ISL,$OSL,$tpot,${ttft:-0},ok" >> "$OUT"
      echo "  -> TPOT=$tpot ms  TTFT=${ttft:-?} ms"
    else
      echo "vibe_${fab}_b${B},$fab,8,$B,$ISL,$OSL,0,0,fail" >> "$OUT"
      echo "  -> FAIL"; echo "$log" | tail -3
    fi
  done
done

echo ""; echo "=== done -> $OUT ==="
echo "  python scripts/plot_pareto.py --osl $OSL --tp 8 --decode-only $OUT \\"
echo "    --out outputs/paper_figures/fig_vibe_pareto_tp8_${MODEL_TAG}.png"
