#!/bin/bash
# -----------------------------------------------------------------------------
# Profile the Krishna-group (GT Synergy Lab) inference-paper model set.
#
# Models chosen by frequency across their 2024-26 LLM inference papers
# (GenZ, MIST, AFD, MoE-scaling, Chakra, STAGE, TurboAttention...):
#   1. Llama-3.1-70B   (dense GQA large  - their most common workload)
#   2. Mixtral-8x7B    (canonical MoE)
#   3. Qwen3-235B-A22B (modern fine-grained MoE; AFD/MoE-scaling/ThunderAgent)
#   4. Llama-3.1-8B    (small dense calibration anchor)
# DeepSeek-V3 (MLA) and GPT-OSS-120B need new architecture yamls under
# profiler/models/ before they can be profiled - see docs/profiling-guide.md.
#
# Run from the LLMServingSim repo root INSIDE the vLLM container
# (scripts/docker-vllm.sh) or a bare-metal vLLM venv
# (scripts/install-vllm.sh). One GPU is enough: every TP degree is
# emulated on a single GPU via hf_overrides shape division; weights are
# dummy-loaded (no checkpoint download).
#
# Usage:
#   HARDWARE=H100 ./profiler/profile-krishna.sh            # all four models
#   HARDWARE=H100 ONLY="Qwen/Qwen3-235B-A22B" ./profiler/profile-krishna.sh
# -----------------------------------------------------------------------------

set -euo pipefail

HARDWARE="${HARDWARE:?set HARDWARE to the GPU name, e.g. H100 / H200 / A100}"
TP_DEGREES="${TP_DEGREES:-1,2,4,8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
ATTENTION_MAX_KV="${ATTENTION_MAX_KV:-32768}"

MODELS=(
  "NousResearch/Meta-Llama-3.1-70B"
  "mistralai/Mixtral-8x7B-Instruct-v0.1"
  "Qwen/Qwen3-235B-A22B"
  "meta-llama/Llama-3.1-8B"
)

for MODEL in "${MODELS[@]}"; do
  if [ -n "${ONLY:-}" ] && [ "$MODEL" != "$ONLY" ]; then
    continue
  fi
  echo "=================================================================="
  echo "PROFILING $MODEL  (hardware=$HARDWARE, tp=$TP_DEGREES)"
  echo "=================================================================="
  python3 -m profiler profile "$MODEL" \
    --hardware "$HARDWARE" \
    --tp "$TP_DEGREES" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --attention-max-kv "$ATTENTION_MAX_KV"
done

echo "All done. Outputs under profiler/perf/$HARDWARE/."
