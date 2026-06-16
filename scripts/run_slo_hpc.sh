#!/usr/bin/env bash
# H1/H3 SLO sweep for HPC (native x86, large RAM) — the serving-paper headline:
#   H1  goodput-interactivity Pareto under SLO   (AIC++ Pareto + ViBE Fig8/9)
#   H3  SLO-attainment / latency-percentiles vs QPS  (ViBE Fig8, Fig14)
#
# Driver: scripts/slo_eval.py (Poisson arrivals, ViBE SLO, goodput = SLO-compliant
# rate, prefill/decode split). Our axis is the INTERCONNECT (glass-FB vs NVL72),
# vs ViBE's placement axis. ViBE SLO (Table 2b): ShareGPT TTFT 250ms / Sonnet 350ms;
# TPOT 125ms (DeepSeek) / 100ms (Qwen).
#
# WHY HPC: QPS sweep = many runs x big models; EP=128 hits the super-linear MoE
# all-to-all .et memory that OOMs the 7.65GiB local Docker. Needs RAM>=~32GiB.
#
# WHY two EPs: EP=8 (ViBE's 8-GPU baseline, in-domain) shows glass≈NVL72 — the
# honest "no in-domain advantage" anchor. EP=128 (cliff, EP>rack64) is where
# NVL72's inter-rack IB collapses goodput and glass sustains it = the headline.
#
# NOTE: absolute QPS differs from ViBE (we run TP=1, not TP=8), so the 90%-goodput
# crossover QPS is unknown a priori. Run COARSE first (the lists below), read where
# goodput crosses 0.9 from slo_eval's stdout, then re-run a FINE list around it.
set -euo pipefail
cd "$(dirname "$0")/.."

PANEL="4 4"; WG=5; RACK=64; INTER_BW=512
NREQ=96           # requests per QPS point (sim cost ~ NREQ x MAXOSL iterations)
MAXOSL=24         # cap decode tokens: steady TPOT is reached in a few tokens, so
                  # this bounds per-run iterations without changing per-token latency.
                  # (Sonnet's native 128-out x 256 reqs x low QPS = ~10^4 iters = effective hang.)
TIMEOUT=10800     # 3h per (model,ds,ep,pd,qps-list) invocation
MEMGB=1024

# (model, tpot_slo is auto from slo_eval; we just pass model/dataset/qps)
# QPS lists are COARSE starting points — tune after the first pass.
run () {
  local model="$1" tag="$2" ds="$3" pd="$4" ep="$5" qps="$6"
  echo "=== SLO: $tag ds=$ds pd=$pd ep=$ep qps=[$qps] ==="
  MOE_ALLTOALL=1 python scripts/slo_eval.py \
    --model "$model" --hardware H100 --tp 1 \
    --dataset "$ds" --pd-mode "$pd" --ep "$ep" \
    --panel $PANEL --fixed-wg $WG --inter-opt-bw $INTER_BW --nvl72-rack $RACK \
    --fabrics glass nvl72 --qps-list $qps --n-req $NREQ --max-osl $MAXOSL \
    --npu-mem-gb $MEMGB --timeout $TIMEOUT \
    --out "outputs/slo_eval/${tag}_${ds}_${pd}_ep${ep}.csv"
}

# ── DeepSeek-V3 (256 experts; TPOT SLO 125ms) ──
for ep in 8 128; do
  run "deepseek-ai/DeepSeek-V3-0324" deepseek_v3 sonnet   decode  $ep "0.5 1 2 4 8"
  run "deepseek-ai/DeepSeek-V3-0324" deepseek_v3 sharegpt decode  $ep "2 4 8 16 32"
  run "deepseek-ai/DeepSeek-V3-0324" deepseek_v3 sonnet   prefill $ep "0.5 1 2 4 8"
done

# ── Qwen-3 235B (128 experts; TPOT SLO 100ms) ──
for ep in 8 128; do
  run "Qwen/Qwen3-235B-A22B" qwen235b sonnet   decode  $ep "1 2 4 8 16"
  run "Qwen/Qwen3-235B-A22B" qwen235b sharegpt decode  $ep "4 8 16 32 64"
  run "Qwen/Qwen3-235B-A22B" qwen235b sonnet   prefill $ep "1 2 4 8 16"
done

echo "=== done. CSVs in outputs/slo_eval/ ==="
echo "Next: a percentile-vs-QPS / goodput-vs-QPS plot over these CSVs (ViBE Fig8/Fig14"
echo "style) + the goodput-interactivity Pareto (H1). glass should sustain higher QPS"
echo "at >=90% goodput than NVL72 at ep=128 (cliff); near-equal at ep=8 (in-domain)."
