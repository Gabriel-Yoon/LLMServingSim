#!/bin/bash
# Profile new and supplement existing models on H200.
#
# Safety: resume mode — existing CSV data is never overwritten.
#
# Run from inside the vLLM Docker at /workspace:
#   ./profiler/profile_h200_new.sh
#
# Single-model run:
#   MODEL_FILTER=deepseek  ./profiler/profile_h200_new.sh
#   MODEL_FILTER=Kimi      ./profiler/profile_h200_new.sh
#   MODEL_FILTER=gpt-oss   ./profiler/profile_h200_new.sh
#
# Skip skew (faster first pass):
#   SKIP_SKEW=1 ./profiler/profile_h200_new.sh
#
# ─────────────────────────────────────────────────────────────
# H200 status as of 2026-06-10:
#
#   ✅ Qwen/Qwen3-30B-A3B-Instruct-2507  tp1 tp2  — supplement tp4 below
#   ✅ Qwen/Qwen3-235B-A22B              tp1      — supplement tp2,tp4 below
#   ✅ deepseek-ai/DeepSeek-V3           tp1      — supplement tp2,tp4 below
#   ✅ mistralai/Mixtral-8x7B-v0.1       tp1      — supplement tp2 below
#   ✅ mistralai/Mixtral-8x22B-v0.1      tp1      — supplement tp2 below
#
#   ❌ deepseek-ai/DeepSeek-V3-0324      not started — ready
#   ❌ moonshotai/Kimi-K2-Instruct       not started — ready
#   ❌ openai/gpt-oss-120b               not started — yaml ready (class names TBD)
# ─────────────────────────────────────────────────────────────
#
# HPC salloc recommendation:
#   salloc --nodes=1 --gpus=4 --partition=gpu-h200 --time=24:00:00

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Auto-activate vllm_venv if vLLM is not already importable.
if ! python3 -c "import vllm" &>/dev/null 2>&1; then
    ENV_SH="$REPO_ROOT/scripts/env.sh"
    if [[ -f "$ENV_SH" ]]; then
        source "$ENV_SH"
        if [[ -f "${VLLM_VENV:-}/bin/activate" ]]; then
            # shellcheck disable=SC1091
            source "$VLLM_VENV/bin/activate"
            echo "[env] activated $VLLM_VENV"
        else
            echo "ERROR: vllm_venv not found at ${VLLM_VENV:-<unset>}" >&2
            echo "  source scripts/env.sh and check VLLM_VENV." >&2
            exit 1
        fi
    fi
fi

# ─────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────
HARDWARE="${HARDWARE:-H200}"
TP_DEGREES="${TP_DEGREES:-1,2,4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
ATTENTION_MAX_KV="${ATTENTION_MAX_KV:-32768}"
ATTENTION_CHUNK_FACTOR="${ATTENTION_CHUNK_FACTOR:-2.0}"
ATTENTION_KV_FACTOR="${ATTENTION_KV_FACTOR:-2.0}"
MEASUREMENT_ITERATIONS="${MEASUREMENT_ITERATIONS:-3}"
SKIP_SKEW="${SKIP_SKEW:-}"
MODEL_FILTER="${MODEL_FILTER:-}"

PERF_ROOT="profiler/perf/$HARDWARE"

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
_config_ready() {
    local cfg="configs/model/${1}.json"
    if [[ ! -f "$cfg" ]]; then
        echo "[SKIP] $1 — config not found: $cfg" >&2; return 1
    fi
    if grep -q '"hidden_size": 0\|"TODO"\|"TODO_' "$cfg" 2>/dev/null; then
        echo "[SKIP] $1 — config has unfilled TODO fields." >&2; return 1
    fi
    return 0
}

_profile() {
    local model_id="$1"
    local tp="$2"

    if [[ -n "$MODEL_FILTER" && "$model_id" != *"$MODEL_FILTER"* ]]; then
        echo "[SKIP] $model_id (MODEL_FILTER=$MODEL_FILTER)"
        return
    fi

    _config_ready "$model_id" || return

    echo ""
    echo "========================================"
    echo "  Model   : $model_id"
    echo "  Hardware: $HARDWARE   TP: $tp"
    echo "  Resume  : existing CSV data preserved"
    echo "========================================"

    local cmd=(python3 -m profiler profile "$model_id" --hardware "$HARDWARE")
    cmd+=(--tp "$tp")
    cmd+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
    cmd+=(--max-num-seqs "$MAX_NUM_SEQS")
    cmd+=(--attention-max-kv "$ATTENTION_MAX_KV")
    cmd+=(--attention-chunk-factor "$ATTENTION_CHUNK_FACTOR")
    cmd+=(--attention-kv-factor "$ATTENTION_KV_FACTOR")
    cmd+=(--measurement-iterations "$MEASUREMENT_ITERATIONS")
    [[ -n "$SKIP_SKEW" ]] && cmd+=(--skip-skew)

    echo "CMD: ${cmd[*]}"
    "${cmd[@]}"
}

# ─────────────────────────────────────────────────────────────
# Pre-flight
# ─────────────────────────────────────────────────────────────
echo "=== H200 Profile Run ==="
echo "Hardware : $HARDWARE"
echo "TP sweep : $TP_DEGREES"
echo "MNBT/MSQ : $MAX_NUM_BATCHED_TOKENS / $MAX_NUM_SEQS"
echo "Max KV   : $ATTENTION_MAX_KV"
echo "Skip skew: ${SKIP_SKEW:-no (full profile)}"
echo ""
echo "Existing $HARDWARE profiles:"
for meta in $(find "$PERF_ROOT" -name "meta.yaml" 2>/dev/null | sort); do
    model=$(echo "$meta" | sed "s|$PERF_ROOT/||" | sed 's|/bf16/meta.yaml||')
    tp_dirs=$(ls "$(dirname "$meta")" | grep "^tp" | tr '\n' ' ')
    echo "  ✅ $model  →  $tp_dirs"
done
echo ""

# ─────────────────────────────────────────────────────────────
# Phase 1: New models (not profiled on H200 yet)
# ─────────────────────────────────────────────────────────────
echo "--- Phase 1: New models ---"

_profile "deepseek-ai/DeepSeek-V3-0324" "$TP_DEGREES"
_profile "moonshotai/Kimi-K2-Instruct"  "$TP_DEGREES"

# ─────────────────────────────────────────────────────────────
# Phase 2: Supplement existing tp1-only profiles with tp2, tp4
# ─────────────────────────────────────────────────────────────
echo ""
echo "--- Phase 2: Supplement existing H200 profiles (tp2, tp4) ---"

_profile "deepseek-ai/DeepSeek-V3"          "2,4"
_profile "Qwen/Qwen3-235B-A22B"             "2,4"
_profile "Qwen/Qwen3-30B-A3B-Instruct-2507" "4"       # tp1,tp2 exist; add tp4
_profile "mistralai/Mixtral-8x7B-v0.1"     "2"        # tp1 exists; add tp2
_profile "mistralai/Mixtral-8x22B-v0.1"    "2,4"

# ─────────────────────────────────────────────────────────────
# Phase 3: GPT-OSS-120B (verify vLLM class names first)
# ─────────────────────────────────────────────────────────────
echo ""
echo "--- Phase 3: openai/gpt-oss-120b ---"
echo "  NOTE: gpt_oss.yaml class names are inferred, not verified."
echo "  Run with MODEL_FILTER=gpt-oss after confirming class names."
echo ""

_profile "openai/gpt-oss-120b" "$TP_DEGREES"

# ─────────────────────────────────────────────────────────────
echo ""
echo "=== Done. Output: $PERF_ROOT/ ==="
