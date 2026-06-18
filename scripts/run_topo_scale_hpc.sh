#!/usr/bin/env bash
# T1 topology comparison AT SCALE — measured replacement for the analytical
# projection (topo_project.py). A single glass panel physically caps DIRECT
# hardware to 16/32 GPUs, but the SIMULATOR can model 64-256 — the only local
# limit was .et graph memory (256 NPU OOMs the ~14 GiB local box). On HPC (big
# RAM) we run ASTRA-sim topo_compare at 64/128/256, turning T1 from
# "16/32 measured + 64-256 projected" into "16-256 ALL measured" -> kills the
# "it's just a projection" objection.
#
# PREREQUISITE: bash scripts/apply_astra_overlay.sh   (Mesh2D/Torus2D/Dragonfly
#   C++ classes must be compiled in, plus the ring->direct fix).
#
# CLEAN COMPARISON: each EP runs on a SINGLE PANEL of that exact size (panel = N),
# so every topology is ONE N-node structure. If panel < EP the builder adds a
# FullyConnected inter-panel dim (mixes the topologies) and muddies the compare.
#   EP 16 -> 4x4,  32 -> 4x8,  64 -> 8x8,  128 -> 8x16,  256 -> 16x16
# Ring/Dragonfly span all N regardless of panel.
#
# Two modes:
#   iso-budget (default): equal per-GPU WG budget split by degree (cost-fair).
#   iso-bandwidth (--equal-link-bw): fixed per-link BW (isolates pure diameter).
# batch 64: enough to lift the exposed-comm fraction so topology shows; keeps the
# .et manageable (EP256 x batch64 is still tens of GiB -> needs HPC).
set -euo pipefail
cd "$(dirname "$0")/.."

# Headline = a BIG MoE (the paper's models). Topology ordering is model-agnostic
# (diameter), but the figure must be on a credible large model. DeepSeek-V3 (MLA,
# bandwidth-bound) by default; override for Qwen3-235B / Kimi-K2:
#   MODEL=Qwen/Qwen3-235B-A22B MEM=141 bash scripts/run_topo_scale_hpc.sh
#   MODEL=moonshotai/Kimi-K2-Instruct MEM=200 PANELS="64:8,8 128:8,16" ...
# A big model only fits at higher EP (weights shard by EP): DeepSeek EP16=113GB,
# EP32=67, EP64=38 GB. Default measures EP 16/32/64 and you PROJECT 128/256 with
# topo_project.py (panel physics caps direct scale anyway). For the FULL measured
# 16->256 curve use the light model + full panel list (PANELS tuples are "EP:rows,cols",
# space-separated):
#   MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 MEM=96 PANELS="16:4,4 32:4,8 64:8,8 128:8,16 256:16,16" ...
MODEL="${MODEL:-deepseek-ai/DeepSeek-V3-0324}"
TOPOS="${TOPOS:-fb dragonfly torus mesh ring}"
WGB="${WGB:-60}"      # per-GPU WG budget (52x34 full-tile PIC, from wg_budget.py)
BATCH="${BATCH:-64}"  # lifts exposed-comm so topology shows (batch=4 hides it)
MEM="${MEM:-141}"
TIMEOUT="${TIMEOUT:-14400}"
EQBW="${EQBW:-}"      # set EQBW=512 to run the iso-bandwidth ablation instead

# EP -> single-panel (rows cols) so panel size == EP (no inter-panel mixing).
# Default = the big-model-feasible subset (16/32/64); override PANELS for full scale.
read -r -a PANELS <<< "${PANELS:-16:4,4 32:4,8 64:8,8}"
PANELS=("${PANELS[@]//,/ }")

run () {
  local ep="$1" rows="$2" cols="$3"
  local tag="isobudget"; local eqarg=""
  if [ -n "$EQBW" ]; then tag="isobw_${EQBW}"; eqarg="--equal-link-bw $EQBW"; fi
  local out="outputs/panel_dse/topo_scale_ep${ep}_${tag}.csv"
  if [ -s "$out" ] && [ "$(wc -l < "$out")" -gt 1 ]; then
    echo "=== SKIP (exists): $out ==="; return
  fi
  echo "=== topo_scale EP=$ep panel ${rows}x${cols} ($tag) ==="
  MOE_ALLTOALL=1 python scripts/sweep_panel_dse.py --sweep topo_compare \
    --topologies $TOPOS --ep-list "$ep" --epscale-panel "$rows" "$cols" \
    --wg-budget $WGB --batch-per-instance $BATCH $eqarg \
    --model "$MODEL" --hardware H100 --tp 1 --npu-mem-gb $MEM --timeout $TIMEOUT \
    --out "$out"
}

for pc in "${PANELS[@]}"; do
  ep="${pc%%:*}"; rc="${pc#*:}"
  run "$ep" ${rc}
done

echo "=== topo_scale done. CSVs: outputs/panel_dse/topo_scale_ep*_*.csv ==="
echo "Plot/validate: python scripts/topo_project.py --measured outputs/panel_dse/topo_scale_ep*_isobudget.csv"
echo "(re-run with EQBW=512 for the iso-bandwidth ablation set.)"
