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
# Batch is capped by the PROFILING sweep: DeepSeek/Qwen were profiled at
# max_num_seqs=256, so decode attention is only gridded to n_decode=256. Beyond that
# the 4D lookup extrapolates and TPOT blows up (b2048 -> 4.4 s artifact). So the
# reliable decode batch sweep is <=256; the EP axis (8 control vs 128 cliff) is the
# real lever (raising re-profiling beyond 256 is out of scope -- perf data is frozen).
BATCH_LIST="64 128 256"
MAXTOK=2048       # matches the profiled max_num_batched_tokens
# Realistic GPU memory now that the MLA KV fix removed the phantom 57x KV over-count
# (no more --npu-mem 32TB workaround). H200=141. NOTE: with correct KV, DeepSeek-V3 bf16
# needs EP>=16 to fit weights (EP8 = 217 GB); the in-domain control is EP=16 (= 1 glass
# panel), not EP=8. Qwen3-235B is GQA -> KV fix is a NO-OP, its results are UNCHANGED
# (no re-run needed); only the MLA models (DeepSeek/Kimi) change.
MEM=141
TIMEOUT=10800

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

# PREFILL: the per-step work is set by the prefill CHUNK (= --max-tokens), NOT by the
# per-device batch. parse_steady_decode picks the MODE iteration; with osl=4 that mode
# is the prefill chunk, so a BATCH sweep returns identical numbers (the chunk is
# batch-invariant). To characterise prefill we sweep MAX_TOKENS (the chunk/message
# size) at a fixed small batch instead. osl=2 keeps the mode firmly on the prefill
# chunk; isl >= max_tokens so each chunk is full.
MT_LIST="512 1024 2048 4096 8192"
run_prefill () {  # model tag ep
  local model="$1" tag="$2" ep="$3"
  local out="outputs/panel_dse/expB_${tag}_prefill_ep${ep}.csv"
  if [ -s "$out" ] && [ "$(wc -l < "$out")" -gt 1 ]; then echo "SKIP $out"; return; fi
  echo "=== prefill chunk sweep (max-tokens): $tag EP=$ep ==="
  for mt in $MT_LIST; do
    MOE_ALLTOALL=1 python scripts/sweep_panel_dse.py --sweep batch \
      --batch-ep "$ep" --batch-list 8 --epscale-panel $PANEL --fixed-wg $WG \
      --nvl72-rack $RACK --nvl72-bw $NVLBW --agg-bw --isl "$mt" --osl 2 --max-tokens "$mt" \
      --model "$model" --hardware H100 --tp 1 --npu-mem-gb $MEM --timeout $TIMEOUT \
      --out "outputs/panel_dse/expB_${tag}_prefill_ep${ep}_mt${mt}.csv"
  done
}

for m in "deepseek-ai/DeepSeek-V3-0324:deepseek_v3" "Qwen/Qwen3-235B-A22B:qwen235b"; do
  model="${m%%:*}"; tag="${m#*:}"
  # EP=16 (in-domain control: 1 glass panel; DeepSeek-V3 bf16 fits ~132 GB here, not at
  # EP8). EP=128 (cliff, NVL72 crosses IB since rack=8) — the regime of interest.
  run "$model" "$tag" 16  decode  256  24
  run "$model" "$tag" 128 decode  256  24
  run_prefill "$model" "$tag" 128
done

echo "=== done. Analyse: python scripts/exposed_verdict.py outputs/panel_dse/expB_*.csv ==="
