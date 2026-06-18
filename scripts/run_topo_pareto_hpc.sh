#!/bin/bash
# -----------------------------------------------------------------------------
# Topology throughput-interactivity Pareto (FRESH data, current code) — the
# measured topology figure. Generates, for each glass topology (FB / Dragonfly /
# Torus / Mesh / Ring) at an EQUAL per-GPU WG budget, a batch-swept frontier so
# plot_pareto.py can draw tokens/s/GPU vs tokens/s/user (AIConfigurator axes).
#
# WHY a fresh run: the old topo_compare_*.csv predate the MoE-a2a volume partition,
# the MLA-KV fix, and the ring->direct collective fix -- their numbers are stale.
# This regenerates with the current simulator.
#
# Design:
#  - Fixed EP in the CROSS-DOMAIN regime (EP > panel) so topology DIAMETER matters
#    (in-domain EP<=panel hides it). EP=32 = 2 panels, EP=64 = 4 panels of 4x4.
#  - Sweep per-device batch to TRACE the frontier (one point per batch). Capped at
#    the profiling ceiling (max_num_seqs=256) so attention isn't extrapolated.
#  - Equal WG budget split by each topology's degree (fair: same silicon, the only
#    difference is how the wires are arranged) -> isolates diameter/hop-count.
#  - Light MoE model (Qwen3-30B): topology comparison is model-agnostic and this
#    keeps EP64 x batch tractable. DECODE (the steady regime).
#  - topo_compare takes ONE batch per call, so we loop batch in the shell.
#
# Run on HPC (sim container). Resumable. Then plot:
#   python scripts/plot_pareto.py --osl 16 --decode-only outputs/panel_dse/topo_pareto_ep32_b*.csv \
#     --title 'Topology Pareto EP32 (cross-domain)' --out outputs/paper_figures/fig_topology_pareto_ep32.png
# -----------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/.."

TOPOS="${TOPOS:-fb dragonfly torus mesh ring}"
EPS="${EPS:-32 64}"
BATCHES="${BATCHES:-16 32 64 128 256}"
WGB="${WGB:-60}"                  # total per-GPU WG budget, split by degree
PANEL="4 4"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
MEM="${MEM:-96}"; ISL="${ISL:-256}"; OSL="${OSL:-16}"; MTOK="${MTOK:-2048}"
TIMEOUT="${TIMEOUT:-10800}"
export MOE_ALLTOALL="${MOE_ALLTOALL:-1}"

for ep in $EPS; do
  for b in $BATCHES; do
    out="outputs/panel_dse/topo_pareto_ep${ep}_b${b}.csv"
    if [ -s "$out" ] && [ "$(wc -l < "$out")" -gt 1 ]; then echo "SKIP $out"; continue; fi
    echo "=== topo Pareto EP=$ep batch=$b  {$TOPOS} ==="
    python scripts/sweep_panel_dse.py --sweep topo_compare \
      --topologies $TOPOS --ep-list "$ep" --epscale-panel $PANEL --wg-budget $WGB \
      --batch-per-instance "$b" --isl "$ISL" --osl "$OSL" --max-tokens "$MTOK" \
      --model "$MODEL" --hardware H100 --tp 1 --npu-mem-gb $MEM --timeout $TIMEOUT \
      --out "$out" 2>&1 | grep -iE "topo_|error|exceed|Waiting" | tail -8
  done
done

echo ""
echo "=== done. Plot the per-EP frontiers: ==="
for ep in $EPS; do
  echo "  python scripts/plot_pareto.py --osl $OSL --decode-only outputs/panel_dse/topo_pareto_ep${ep}_b*.csv \\"
  echo "    --title 'Topology Pareto EP${ep} (cross-domain)' --out outputs/paper_figures/fig_topology_pareto_ep${ep}.png"
done
