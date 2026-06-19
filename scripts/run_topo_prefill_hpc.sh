#!/bin/bash
# -----------------------------------------------------------------------------
# Topology comparison in the PREFILL regime — the operating point where switchless
# topology DIAMETER is most exposed. A prefill step processes a full max_tokens chunk
# -> a large MoE all-to-all (e.g. 2048 tok x hidden) -> the per-hop/diameter cost of
# Ring/Mesh vs FB stands out far more than in compute-bound decode (where comm hides).
#
# Same clean setup as run_topo_scale_hpc.sh: each EP runs on a SINGLE PANEL of that
# size (panel == EP), so every topology is ONE pure N-node structure (no inter-panel
# mixing). EP sweep shows diameter scaling. The metric is the PREFILL phase
# (prefill_step_ms / prefill_exposed_frac from the phase-aware parser) -- NOT batch
# (the prefill chunk = max_tokens, batch-independent), so we do NOT sweep batch here.
#
# DeepSeek-V3 default (headline); MODEL/MEM env-overridable (Qwen3-235B / Kimi-K2 /
# Qwen3-30B). Big models fit at EP>=16 (weights shard by EP).
#
# Run on HPC (sim container). Resumable. Then plot:
#   python scripts/plot_step_breakdown.py --x ep --phase prefill \
#     outputs/panel_dse/topo_prefill_deepseek_v3_0324_ep*.csv \
#     --out outputs/paper_figures/fig_topology_prefill_breakdown.png
#   python scripts/plot_collective.py --metric exposed outputs/panel_dse/topo_prefill_*.csv   # (decode exposed col)
# -----------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-deepseek-ai/DeepSeek-V3-0324}"
TOPOS="${TOPOS:-fb dragonfly torus mesh ring}"
WGB="${WGB:-60}"
MEM="${MEM:-141}"
ISL="${ISL:-2048}"            # prefill chunk = max_tokens -> one big a2a per step
OSL="${OSL:-2}"               # short decode tail (phase parser isolates the prefill chunk)
MTOK="${MTOK:-2048}"
BPI="${BPI:-8}"               # a few requests to fill the pipeline (prefill cost is per-chunk)
TIMEOUT="${TIMEOUT:-14400}"
EQBW="${EQBW:-}"              # set EQBW=512 for the iso-per-link ablation (pure diameter)
export MOE_ALLTOALL="${MOE_ALLTOALL:-1}"
MTAG=$(echo "$MODEL" | sed 's#.*/##; s/[^A-Za-z0-9]/_/g' | tr 'A-Z' 'a-z')

# EP -> single-panel (rows cols) so panel size == EP (pure N-node topology).
# Default = big-model-feasible (EP 16/32/64); override PANELS ("EP:rows,cols" space-sep).
read -r -a PANELS <<< "${PANELS:-16:4,4 32:4,8 64:8,8}"
PANELS=("${PANELS[@]//,/ }")

for pc in "${PANELS[@]}"; do
  ep="${pc%%:*}"; rc="${pc#*:}"
  tag="isobudget"; eqarg=""
  if [ -n "$EQBW" ]; then tag="isobw_${EQBW}"; eqarg="--equal-link-bw $EQBW"; fi
  out="outputs/panel_dse/topo_prefill_${MTAG}_ep${ep}_${tag}.csv"
  if [ -s "$out" ] && [ "$(wc -l < "$out")" -gt 1 ]; then echo "SKIP $out"; continue; fi
  echo "=== topo PREFILL EP=$ep panel ${rc} ($MTAG, chunk=$MTOK, $tag) {$TOPOS} ==="
  python scripts/sweep_panel_dse.py --sweep topo_compare \
    --topologies $TOPOS --ep-list "$ep" --epscale-panel ${rc} --wg-budget $WGB $eqarg \
    --batch-per-instance $BPI --isl "$ISL" --osl "$OSL" --max-tokens "$MTOK" \
    --model "$MODEL" --hardware H100 --tp 1 --npu-mem-gb $MEM --timeout $TIMEOUT \
    --out "$out" 2>&1 | grep -iE "topo_|error|exceed|Waiting" | tail -8
done

echo ""
echo "=== topo PREFILL done. Plot the diameter signal (prefill exposed, all 5 topologies): ==="
echo "  python scripts/plot_collective.py --metric prefill_exposed \\"
echo "    outputs/panel_dse/topo_prefill_${MTAG}_ep*_${tag}.csv \\"
echo "    --title 'Topology prefill exposed comm vs scale ($MTAG, $tag)' --out outputs/paper_figures/fig_topo_prefill_${MTAG}_${tag}.png"
