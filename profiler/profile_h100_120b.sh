#!/bin/bash
# Profile all 5 large-scale models on H100 GPUs.
#
# Models:
#   Qwen/Qwen3-235B-A22B            (MoE, qwen3_moe, 128 experts)  — ready
#   deepseek-ai/DeepSeek-V3-0324    (MoE, deepseek_v3, 256 experts) — ready
#   moonshotai/Kimi-K2-Instruct     (MoE, kimi_k2, 384 experts)    — ready
#   openai/gpt-oss-120b             (MoE, gpt_oss, 128 experts)    — yaml ready; verify vLLM class names
#
# NOT profileable (Mamba-2 hybrid — SSM layers unsupported):
#   nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16  (model_type: nemotron_h)
#
# Usage (inside vLLM Docker at /workspace):
#   # Profile a single model:
#   MODEL_FILTER=qwen3_235b ./profiler/profile_h100_120b.sh
#
#   # Profile all models sequentially:
#   ./profiler/profile_h100_120b.sh
#
#   # Profile with custom TP (e.g. 8-GPU node):
#   TP_DEGREES=1,2,4,8 ./profiler/profile_h100_120b.sh
#
# salloc tip (HPC): request enough GPUs for max TP degree.
#   salloc --nodes=1 --gpus=8 --partition=gpu ...

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# =============================================================================
# Global settings (override via env if needed)
# =============================================================================
HARDWARE="${HARDWARE:-H100}"
TP_DEGREES="${TP_DEGREES:-1,2,4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
ATTENTION_MAX_KV="${ATTENTION_MAX_KV:-32768}"
ATTENTION_CHUNK_FACTOR="${ATTENTION_CHUNK_FACTOR:-2.0}"
ATTENTION_KV_FACTOR="${ATTENTION_KV_FACTOR:-2.0}"
MEASUREMENT_ITERATIONS="${MEASUREMENT_ITERATIONS:-3}"
# Set SKIP_SKEW=1 for a quick first pass (saves 1-2h per model per TP).
SKIP_SKEW="${SKIP_SKEW:-}"

# Optional: restrict to one model by name substring for single-model runs.
MODEL_FILTER="${MODEL_FILTER:-}"

# =============================================================================
# Model list
# =============================================================================
declare -A MODELS
MODELS["qwen3_235b"]="Qwen/Qwen3-235B-A22B"
MODELS["deepseek_v3_0324"]="deepseek-ai/DeepSeek-V3-0324"
MODELS["kimi_k2"]="moonshotai/Kimi-K2-Instruct"
MODELS["gpt_oss_120b"]="openai/gpt-oss-120b"
# nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 — Mamba-2 hybrid, profiler incompatible

# =============================================================================
# Helper
# =============================================================================
_profile() {
    local key="$1"
    local model_id="$2"

    # Skip if MODEL_FILTER is set and doesn't match
    if [[ -n "$MODEL_FILTER" && "$key" != *"$MODEL_FILTER"* ]]; then
        echo "[SKIP] $key (MODEL_FILTER=$MODEL_FILTER)"
        return
    fi

    # Check config exists
    local cfg_path="configs/model/${model_id}.json"
    if [[ ! -f "$cfg_path" ]]; then
        echo "[SKIP] $key — config not found: $cfg_path"
        echo "       Run ./profiler/fetch_hf_configs.sh first."
        return
    fi

    # Check for TODO markers (unfilled template)
    if grep -q '"TODO"' "$cfg_path" 2>/dev/null || grep -q '"hidden_size": 0' "$cfg_path" 2>/dev/null; then
        echo "[SKIP] $key — config has unfilled TODO fields: $cfg_path"
        echo "       Run ./profiler/fetch_hf_configs.sh and fill in dimensions."
        return
    fi

    echo ""
    echo "========================================"
    echo "  Profiling: $model_id"
    echo "  Hardware : $HARDWARE"
    echo "  TP       : $TP_DEGREES"
    echo "========================================"

    local cmd=(python3 -m profiler profile "$model_id" --hardware "$HARDWARE")
    cmd+=(--tp "$TP_DEGREES")
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

# =============================================================================
# Run
# =============================================================================
echo "=== H100 Large-Model Profile Sweep ==="
echo "Hardware : $HARDWARE"
echo "TP sweep : $TP_DEGREES"
echo "MNBT     : $MAX_NUM_BATCHED_TOKENS   MSQ: $MAX_NUM_SEQS"
echo "Max KV   : $ATTENTION_MAX_KV"
echo "Skip skew: ${SKIP_SKEW:-no}"
echo ""

_profile "qwen3_235b"       "Qwen/Qwen3-235B-A22B"
_profile "deepseek_v3_0324" "deepseek-ai/DeepSeek-V3-0324"
_profile "kimi_k2"          "moonshotai/Kimi-K2-Instruct"
_profile "gpt_oss_120b"     "openai/gpt-oss-120b"

echo ""
echo "=== Profile sweep complete. ==="
echo "Output: profiler/perf/$HARDWARE/"
