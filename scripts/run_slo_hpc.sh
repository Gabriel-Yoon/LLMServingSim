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

HARDWARE="${HARDWARE:-H200}"   # H200 (141GB) lets bf16 fit at the EPs below (no fake mem)
PANEL="4 4"; WG=8; RACK=8; NVLBW=450; INTER_BW=512   # NVLink4 450, HGX/H200 SXM domain 8   # --fixed-wg = TOTAL bundle (even); x8 = per-dir 4 = 512 GB/s, feasible 4x4 floor
NREQ="${NREQ:-96}"        # requests per QPS point (sim cost ~ NREQ x MAXOSL iterations)
MAXOSL="${MAXOSL:-24}"    # cap decode tokens. MAXOSL=24 = fast steady-TPOT view (bounds
                  # iterations); MAXOSL=128 = full Sonnet output -> requests linger ~5x
                  # longer -> ViBE-comparable sustainable QPS (slower; A/B with 24).
TIMEOUT="${TIMEOUT:-10800}"     # 3h per (model,ds,ep,pd,qps-list) invocation
MEMGB="${MEMGB:-141}"    # REALISTIC H200 (141GB) so KV-admission limits concurrency like ViBE
                  # (fake-large hides the KV/queueing TTFT failure mode). bf16 per-GPU fits:
                  # DeepSeek EP16=113 / EP32=67 / EP128=38; Qwen EP8=58. DeepSeek EP8 (199GB)
                  # does NOT fit -> not run in bf16 (would need FP8 or fake); Qwen EP8 anchors
                  # the in-domain parity instead.

# QPS lists below are the EP=8 baseline (cluster-wide). They are SCALED by ep/8
# inside run() so the PER-INSTANCE load stays constant across EP. Without this,
# ep128 gets n_req(96) < 128 instances and a tiny cluster QPS -> most instances
# are idle -> the DP-sync barrier makes every idle instance emit dummy batches
# each iteration -> dummy explosion -> effective hang. (ep8 ran fine; the cliff
# ep128 stalled purely from this starvation, not prefill — ep8 did full prefill.)
run () {
  local model="$1" tag="$2" ds="$3" pd="$4" ep="$5" qps="$6"
  local out="outputs/slo_eval/${tag}_${ds}_${pd}_ep${ep}_osl${MAXOSL}.csv"
  # Resumable: skip configs whose CSV already exists (header + >=1 row), so a
  # re-run only does the missing ones and never overwrites good data. ep8 params
  # are unchanged by the EP-scaling (f=1), so those CSVs stay valid. FORCE=1 to
  # re-run everything.
  if [ -z "${FORCE:-}" ] && [ -s "$out" ] && [ "$(wc -l < "$out")" -gt 1 ]; then
    echo "=== SKIP (exists): $out  (FORCE=1 to redo) ==="
    return
  fi
  local f=$(( ep / 8 )); [ "$f" -lt 1 ] && f=1     # per-instance-load scale (ep8 base)
  local sqps=""; for q in $qps; do sqps="$sqps $(awk "BEGIN{print $q*$f}")"; done
  local nreq=$(( NREQ * f ))
  echo "=== SLO: $tag ds=$ds pd=$pd ep=$ep qps=[$sqps] n_req=$nreq (scaled x$f) ==="
  MOE_ALLTOALL=1 python scripts/slo_eval.py \
    --model "$model" --hardware $HARDWARE --tp 1 \
    --dataset "$ds" --pd-mode "$pd" --ep "$ep" \
    --panel $PANEL --fixed-wg $WG --inter-opt-bw $INTER_BW --nvl72-rack $RACK --nvl72-bw $NVLBW \
    --fabrics glass nvl72 --qps-list $sqps --n-req $nreq --max-osl $MAXOSL \
    --npu-mem-gb $MEMGB --timeout $TIMEOUT \
    --out "$out"
}

# ── DeepSeek-V3 (256 experts; TPOT SLO 125ms). bf16 fits H200 at EP>=16; EP8 (199GB)
#    omitted (would need FP8/fake). EP16 = cliff onset (>domain 8 -> NVL72 on IB),
#    EP128 = production/deep cliff. ──
DS_EPS="${DS_EPS:-16 32 128}"
for ep in $DS_EPS; do
  run "deepseek-ai/DeepSeek-V3-0324" deepseek_v3 sonnet   decode  $ep "0.5 1 2 4 8"
  run "deepseek-ai/DeepSeek-V3-0324" deepseek_v3 sharegpt decode  $ep "2 4 8 16 32"
  run "deepseek-ai/DeepSeek-V3-0324" deepseek_v3 sonnet   prefill $ep "0.5 1 2 4 8"
done

# ── Qwen-3 235B (128 experts; TPOT SLO 100ms). bf16 EP8=58GB fits H200 -> EP8 is the
#    in-domain parity anchor; EP128 = cliff. ──
QW_EPS="${QW_EPS:-8 128}"
for ep in $QW_EPS; do
  run "Qwen/Qwen3-235B-A22B" qwen235b sonnet   decode  $ep "1 2 4 8 16"
  run "Qwen/Qwen3-235B-A22B" qwen235b sharegpt decode  $ep "4 8 16 32 64"
  run "Qwen/Qwen3-235B-A22B" qwen235b sonnet   prefill $ep "1 2 4 8 16"
done

echo "=== done. CSVs in outputs/slo_eval/ ==="
echo "Next: a percentile-vs-QPS / goodput-vs-QPS plot over these CSVs (ViBE Fig8/Fig14"
echo "style) + the goodput-interactivity Pareto (H1). glass should sustain higher QPS"
echo "at >=90% goodput than NVL72 at ep=128 (cliff); near-equal at ep=8 (in-domain)."
