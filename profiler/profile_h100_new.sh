#!/bin/bash
# Profile models that are NOT yet fully profiled on H100.
#
# Safety: resume mode only — never overwrites existing CSV data.
# The profiler's default behavior loads existing entries and only runs
# missing shots. FORCE=1 is intentionally absent from this script.
#
# Run from inside the vLLM Docker at /workspace:
#   ./profiler/profile_h100_new.sh
#
# To run a single model:
#   MODEL_FILTER=deepseek ./profiler/profile_h100_new.sh
#
# To skip skew profiling (faster first pass, ~1/3 of total time):
#   SKIP_SKEW=1 ./profiler/profile_h100_new.sh
#
# ─────────────────────────────────────────────────────────────
# Status as of 2026-06-10 (H100):
#
#   ✅ Qwen/Qwen3-30B-A3B-Instruct-2507              tp1 tp2 tp4  — COMPLETE
#   ✅ mistralai/Mixtral-8x7B-v0.1                   tp1 tp2      — COMPLETE
#   ⚠️  Qwen/Qwen3-235B-A22B                         tp1 only     — tp2,tp4 added below
#   ❌  deepseek-ai/DeepSeek-V3-0324                 not started  — ready
#   ❌  moonshotai/Kimi-K2-Instruct                  not started  — ready
#   ❌  openai/gpt-oss-120b                          not started  — yaml ready (class names TBD)
#   🚫  nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16  Mamba-2 hybrid — profiler incompatible
# ─────────────────────────────────────────────────────────────
#
# HPC salloc recommendation:
#   salloc --nodes=1 --gpus=4 --partition=gpu-h100 ...
#   (TP_DEGREES=1,2,4 needs at most 4 GPUs)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ─────────────────────────────────────────────────────────────
# Settings (override via env)
# ─────────────────────────────────────────────────────────────
HARDWARE="${HARDWARE:-H100}"
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

# Check whether a config has unfilled TODO fields.
_config_ready() {
    local cfg="configs/model/${1}.json"
    if [[ ! -f "$cfg" ]]; then
        echo "[SKIP] $1 — config not found: $cfg" >&2; return 1
    fi
    if grep -q '"hidden_size": 0\|"TODO"\|"TODO_' "$cfg" 2>/dev/null; then
        echo "[SKIP] $1 — config has unfilled TODO fields. Run ./profiler/fetch_hf_configs.sh first." >&2
        return 1
    fi
    return 0
}

# Run profiler for one model with specified TP degrees.
# Does NOT pass --force, so existing CSV data is always preserved.
_profile() {
    local model_id="$1"
    local tp="$2"       # comma-separated, e.g. "1,2,4"
    local key="${model_id##*/}"

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
# Pre-flight summary
# ─────────────────────────────────────────────────────────────
echo "=== H100 New-Model Profile Run ==="
echo "Hardware : $HARDWARE"
echo "TP sweep : $TP_DEGREES"
echo "MNBT/MSQ : $MAX_NUM_BATCHED_TOKENS / $MAX_NUM_SEQS"
echo "Max KV   : $ATTENTION_MAX_KV"
echo "Skip skew: ${SKIP_SKEW:-no (full profile)}"
echo ""
echo "Existing H100 profiles:"
for meta in $(find "$PERF_ROOT" -name "meta.yaml" 2>/dev/null | sort); do
    model=$(echo "$meta" | sed "s|$PERF_ROOT/||" | sed 's|/bf16/meta.yaml||')
    tp_dirs=$(ls "$(dirname "$meta")" | grep "^tp" | tr '\n' ' ')
    echo "  ✅ $model  →  $tp_dirs"
done
echo ""

# ─────────────────────────────────────────────────────────────
# Phase 1: Models ready to run now
# ─────────────────────────────────────────────────────────────
echo "--- Phase 1: Unstarted models (config + yaml ready) ---"

# DeepSeek-V3-0324: same architecture as V3, new weights.
_profile "deepseek-ai/DeepSeek-V3-0324" "$TP_DEGREES"

# Kimi K2: MLA MoE, same vLLM class as DeepSeek-V3.
_profile "moonshotai/Kimi-K2-Instruct"  "$TP_DEGREES"

# ─────────────────────────────────────────────────────────────
# Phase 2: Supplement — Qwen3-235B tp2/tp4 (tp1 already done)
# ─────────────────────────────────────────────────────────────
echo ""
echo "--- Phase 2: Qwen3-235B-A22B — supplement tp2, tp4 (tp1 exists, skip) ---"

# Only run tp2 and tp4; tp1 CSV already complete.
# Resume mode ensures tp1 data is untouched even if re-listed.
_profile "Qwen/Qwen3-235B-A22B" "2,4"

# ─────────────────────────────────────────────────────────────
# Phase 3: NVIDIA gated models (need fetch first)
# ─────────────────────────────────────────────────────────────
echo ""
echo "--- Phase 3: OpenAI GPT-OSS-120B ---"
echo "  gpt_oss.yaml is present; verify vLLM class names before running."
echo ""

_profile "openai/gpt-oss-120b" "$TP_DEGREES"

# nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 is a Mamba-2 + MoE hybrid
# (model_type: nemotron_h). SSM layers are not supported — skip.

# ─────────────────────────────────────────────────────────────
echo ""
echo "=== Done. Output: $PERF_ROOT/ ==="
