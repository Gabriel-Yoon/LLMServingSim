#!/usr/bin/env bash
# STEP A — decisive: is the small exposed% (a) genuinely compute/memory-bound, or
# (b) suppressed by small batch? Sweep the per-instance batch (= decode tokens/step)
# and watch exposed% rise as weight-loading amortises. If exposed crosses ~30% at
# some feasible B AND glass end-to-end TPOT < NVL72 there -> a comm-bound regime
# exists (CASE 1). If it stays small to B=8192 -> parity (CASE 2); headline = energy
# + collective-a2a (10x) + topology.
#
# TP=1 only (TP>1 + dp_group deadlocks in ASTRA — collective stream misalign; an
# amplifier we can't run). So this characterises the TP=1 comm/compute plateau; TP>1
# (production) would shift it higher. State that in the paper.
#
# PREREQUISITE: bash scripts/apply_astra_overlay.sh
# Glass uses --agg-bw (fair aggregate-egress within-panel; inter 512 [8'']-realistic).
set -euo pipefail
cd "$(dirname "$0")/.."

# H100-CONSISTENT baseline: NVLink4 450 GB/s unidir, 8-GPU HGX NVLink island (rack=8),
# IB 50 cross-island. Matches the H100 compute profile (vs the GB200/NVL72 mismatch).
WG=8; PANEL="4 4"; RACK=8; NVLBW=450
BATCH_LIST="256 1024 2048 4096 8192"
MAXTOK=16384      # >= max batch so a DECODE step isn't capped by the token budget
MEM=1024
TIMEOUT=14400

run () {  # model tag ep phase isl osl
  local model="$1" tag="$2" ep="$3" phase="$4" isl="$5" osl="$6"
  local out="outputs/panel_dse/expB_${tag}_${phase}_ep${ep}.csv"
  if [ -s "$out" ] && [ "$(wc -l < "$out")" -gt 1 ]; then echo "SKIP $out"; return; fi
  echo "=== exposed-vs-batch: $tag $phase EP=$ep (isl$isl/osl$osl) ==="
  MOE_ALLTOALL=1 python scripts/sweep_panel_dse.py --sweep batch \
    --batch-ep "$ep" --batch-list $BATCH_LIST --epscale-panel $PANEL --fixed-wg $WG \
    --nvl72-rack $RACK --nvl72-bw $NVLBW --agg-bw --isl "$isl" --osl "$osl" --max-tokens $MAXTOK \
    --model "$model" --hardware H100 --tp 1 --npu-mem-gb $MEM --timeout $TIMEOUT \
    --out "$out"
}

for m in "deepseek-ai/DeepSeek-V3-0324:deepseek_v3" "Qwen/Qwen3-235B-A22B:qwen235b"; do
  model="${m%%:*}"; tag="${m#*:}"
  # EP=8 (in-domain control: 1 glass panel, 1 H100 island, no IB). EP=128 (cliff,
  # NVL72 crosses IB since rack=8) — the regime of interest.
  # NOTE: EP=128 x B=8192 is heavy; if it OOMs, drop 8192 (and 4096) for EP=128.
  run "$model" "$tag" 8   decode  256  24
  run "$model" "$tag" 128 decode  256  24
  run "$model" "$tag" 128 prefill 2048 4
done

echo "=== done. Analyse: python scripts/exposed_verdict.py outputs/panel_dse/expB_*.csv ==="
