#!/bin/bash
# -----------------------------------------------------------------------------
# Add TP=8 profiles (ViBE-aligned: TP=8 + EP=8). Run from inside the vLLM Docker
# (scripts/docker-vllm.sh) at /workspace, on a GPU:
#
#     ./profiler/profile-tp8.sh
#
# Each TP degree is emulated on a SINGLE GPU by dividing SHARD_FIELDS
# (hidden/heads/intermediate/vocab) by TP via hf_overrides -- so this needs one
# GPU, not eight. Resume is ON by default: tp1/2/4 already exist and are skipped;
# only tp8 shots are fired. tp1 MUST stay in the list so the writer can replicate
# tp_stable layers (layernorms, sampler) into tp8/.
#
# MODEL SUPPORT (the profiler errors if a SHARD_FIELD is not divisible by TP):
#   DeepSeek-V3-0324: 128 heads / 128 KV (MLA) / 256 experts -> all ÷8 clean. OK.
#   Qwen3-235B-A22B : num_key_value_heads=4 -> 4 % 8 != 0 -> ValueError. TP=8 is
#                     NOT possible without a profiler patch (real vLLM replicates
#                     KV heads when TP>num_kv_heads; the naive divide doesn't). Use
#                     TP=4 for Qwen (4 KV ÷4 = 1, clean) or ask for the max(1,·) patch.
# -----------------------------------------------------------------------------
set -euo pipefail

HARDWARE="${HARDWARE:-H100}"
# Keep these identical to the existing tp1/2/4 sweep so the tp8 grid lines up.
TP_DEGREES="${TP_DEGREES:-1,2,4,8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
ATTENTION_MAX_KV="${ATTENTION_MAX_KV:-16384}"
ATTENTION_CHUNK_FACTOR="${ATTENTION_CHUNK_FACTOR:-2.0}"
ATTENTION_KV_FACTOR="${ATTENTION_KV_FACTOR:-2.0}"
MEASUREMENT_ITERATIONS="${MEASUREMENT_ITERATIONS:-3}"

# DeepSeek-V3 is the clean ÷8 case (and our bandwidth-bound headline model).
MODELS=(
    "deepseek-ai/DeepSeek-V3-0324"
)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

for MODEL in "${MODELS[@]}"; do
    cmd=(python3 -m profiler profile "$MODEL" --hardware "$HARDWARE")
    cmd+=(--tp "$TP_DEGREES")
    cmd+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
    cmd+=(--max-num-seqs "$MAX_NUM_SEQS")
    cmd+=(--attention-max-kv "$ATTENTION_MAX_KV")
    cmd+=(--attention-chunk-factor "$ATTENTION_CHUNK_FACTOR")
    cmd+=(--attention-kv-factor "$ATTENTION_KV_FACTOR")
    cmd+=(--measurement-iterations "$MEASUREMENT_ITERATIONS")
    [[ -n "${SKIP_SKEW:-}" ]]              && cmd+=(--skip-skew)
    [[ -n "${SKEW_N_FACTOR:-}" ]]          && cmd+=(--skew-n-factor "$SKEW_N_FACTOR")
    [[ -n "${SKEW_PC_FACTOR:-}" ]]         && cmd+=(--skew-pc-factor "$SKEW_PC_FACTOR")
    [[ -n "${SKEW_KP_FACTOR:-}" ]]         && cmd+=(--skew-kp-factor "$SKEW_KP_FACTOR")
    [[ -n "${SKEW_KVS_FACTOR:-}" ]]        && cmd+=(--skew-kvs-factor "$SKEW_KVS_FACTOR")
    [[ -n "${ONLY_SKEW:-}" ]]              && cmd+=(--only-skew)
    [[ -n "${FORCE:-}" ]]                  && cmd+=(--force)
    [[ -n "${DTYPE:-}" ]]                  && cmd+=(--dtype "$DTYPE")
    [[ -n "${KV_CACHE_DTYPE:-}" ]]         && cmd+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
    [[ -n "${VARIANT:-}" ]]                && cmd+=(--variant "$VARIANT")
    [[ -n "${VERBOSITY:-}" ]]              && cmd+=($VERBOSITY)

    echo "=== profiling $MODEL  tp=$TP_DEGREES (resume; only tp8 is new) ==="
    "${cmd[@]}"
done

echo
echo "Done. tp8 added under perf/$HARDWARE/<model>/<variant>/tp8/ (existing tp1/2/4 untouched)."
echo "NOTE: profiling tp8 unblocks the DATA; the SIMULATOR still deadlocks at TP>1"
echo "      (dp_group collective stream misalign) -- that must be fixed to RUN tp8+ep8."
