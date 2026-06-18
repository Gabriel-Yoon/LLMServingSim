#!/bin/bash
# -----------------------------------------------------------------------------
# Core TP=8 profiling runner (ViBE-aligned TP=8 + EP=8). Parameterised by
# HARDWARE + MODEL; the per-target wrappers under scripts/ set those and call
# this. Single GPU is enough: each TP degree is emulated by dividing
# SHARD_FIELDS by TP via hf_overrides (no real 8-GPU job).
#
# SKEW IS SEPARATED into its own pass (skew costs 1-2 h / TP):
#   SKIP_SKEW=1 ...   # pass 1: dense/attention/moe/per_sequence (fast)
#   ONLY_SKEW=1 ...   # pass 2: skew sweep + alpha fit only
# Run pass 1 first, then pass 2. ONLY_SKEW needs TP=1 in the list, which is why
# TP_DEGREES defaults to "1,8" (tp1 already exists -> resume-skipped, kept for
# tp_stable replication + the skew pass; only tp8 is actually fired).
#
# Resume is ON: existing CSVs are preserved; only missing shots fire. Set FORCE=1
# to re-profile from scratch.
# -----------------------------------------------------------------------------
set -euo pipefail

: "${HARDWARE:?set HARDWARE (H100 / H200)}"
: "${MODEL:?set MODEL (HF id, e.g. deepseek-ai/DeepSeek-V3-0324)}"

TP_DEGREES="${TP_DEGREES:-1,8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
ATTENTION_MAX_KV="${ATTENTION_MAX_KV:-16384}"
ATTENTION_CHUNK_FACTOR="${ATTENTION_CHUNK_FACTOR:-2.0}"
ATTENTION_KV_FACTOR="${ATTENTION_KV_FACTOR:-2.0}"
MEASUREMENT_ITERATIONS="${MEASUREMENT_ITERATIONS:-3}"
SKEW_N_FACTOR="${SKEW_N_FACTOR:-2.0}"
SKEW_PC_FACTOR="${SKEW_PC_FACTOR:-2.0}"
SKEW_KP_FACTOR="${SKEW_KP_FACTOR:-2.0}"
SKEW_KVS_FACTOR="${SKEW_KVS_FACTOR:-2.0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Env handling mirrors profile_h200_new.sh (which runs fine in the Apptainer shell):
# auto-activate the vLLM venv only if vLLM isn't already importable. NOTE: nvcc/CUDA
# come from however the shell was set up (the Apptainer image / a previously loaded
# module) -- this script does NOT gate on nvcc, exactly like profile_h200_new.sh.
if ! python3 -c "import vllm" >/dev/null 2>&1; then
    ENV_SH="$REPO_ROOT/scripts/env.sh"
    if [[ -f "$ENV_SH" ]]; then
        # shellcheck disable=SC1090
        source "$ENV_SH"
        if [[ -f "${VLLM_VENV:-}/bin/activate" ]]; then
            # shellcheck disable=SC1091
            source "$VLLM_VENV/bin/activate"
            echo "[env] activated $VLLM_VENV"
        else
            echo "ERROR: vllm_venv not found at ${VLLM_VENV:-<unset>}; source scripts/env.sh" >&2
            exit 1
        fi
    fi
fi

echo "========================================"
echo "  TP=8 profile  |  $MODEL  @  $HARDWARE"
echo "  TP sweep : $TP_DEGREES  (resume: only missing tp fired)"
echo "  MNBT/MSQ : $MAX_NUM_BATCHED_TOKENS / $MAX_NUM_SEQS    maxKV: $ATTENTION_MAX_KV"
if [[ -n "${SKIP_SKEW:-}" ]]; then _phase="SKIP_SKEW (main only)"; elif [[ -n "${ONLY_SKEW:-}" ]]; then _phase="ONLY_SKEW (skew only)"; else _phase="full (main+skew)"; fi
echo "  Phase    : $_phase"
echo "========================================"
ls "profiler/perf/$HARDWARE/$MODEL"/*/ -d 2>/dev/null | while read -r v; do
    echo "  existing: $(basename "$(dirname "$v")")/$(basename "$v") -> $(ls "$v" | grep '^tp' | tr '\n' ' ')"
done

cmd=(python3 -m profiler profile "$MODEL" --hardware "$HARDWARE")
cmd+=(--tp "$TP_DEGREES")
cmd+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
cmd+=(--max-num-seqs "$MAX_NUM_SEQS")
cmd+=(--attention-max-kv "$ATTENTION_MAX_KV")
cmd+=(--attention-chunk-factor "$ATTENTION_CHUNK_FACTOR")
cmd+=(--attention-kv-factor "$ATTENTION_KV_FACTOR")
cmd+=(--measurement-iterations "$MEASUREMENT_ITERATIONS")
cmd+=(--skew-n-factor "$SKEW_N_FACTOR")
cmd+=(--skew-pc-factor "$SKEW_PC_FACTOR")
cmd+=(--skew-kp-factor "$SKEW_KP_FACTOR")
cmd+=(--skew-kvs-factor "$SKEW_KVS_FACTOR")
[[ -n "${SKIP_SKEW:-}" ]]      && cmd+=(--skip-skew)
[[ -n "${ONLY_SKEW:-}" ]]      && cmd+=(--only-skew)
[[ -n "${FORCE:-}" ]]          && cmd+=(--force)
[[ -n "${DTYPE:-}" ]]          && cmd+=(--dtype "$DTYPE")
[[ -n "${KV_CACHE_DTYPE:-}" ]] && cmd+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
[[ -n "${VARIANT:-}" ]]        && cmd+=(--variant "$VARIANT")
[[ -n "${VERBOSITY:-}" ]]      && cmd+=($VERBOSITY)

echo "CMD: ${cmd[*]}"
"${cmd[@]}"
echo "=== done: perf/$HARDWARE/$MODEL/<variant>/tp8/ ==="
