#!/bin/bash
# -----------------------------------------------------------------------------
# ONE-SHOT HPC profiling: the full priority list, resume-safe, with a
# git checkpoint after every model so an interrupted session loses at
# most one model's work.
#
#   git clone <repo> LLMServingSim && cd LLMServingSim
#   git checkout profiling
#   ./scripts/install-vllm.sh && source .venv/bin/activate
#   tmux new -s profile
#   ./profiler/profile-all-hpc.sh 2>&1 | tee profile-all.log
#
# Behavior:
#   - GPU is auto-detected -> HARDWARE label (H100/H200/A100/...).
#   - Skew sweeps are SKIPPED by default (simulator falls back to a
#     constant alpha; rerun specific models later with ONLY_SKEW=1).
#     Set WITH_SKEW=1 to include them (~2x time).
#   - Already-profiled TP degrees (existing perf/<HW>/<model>/bf16/tpN)
#     are pruned from each run -> safe to re-run after interruption.
#   - After each model: commit + push (set PUSH=0 to disable).
#   - A model failure logs FAIL and moves on (GPT-OSS yaml is a draft).
#
# Priority order and rationale: docs/profiling-guide.md in the research
# workspace. Rough total on one H100/H200, skew skipped: 10-16 h.
# -----------------------------------------------------------------------------

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PUSH="${PUSH:-1}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
case "$GPU_NAME" in
  *H200*) HARDWARE=H200 ;;
  *H100*) HARDWARE=H100 ;;
  *A100*) HARDWARE=A100 ;;
  *"RTX 6000"*|*RTX6000*) HARDWARE=RTX6000Ada ;;
  *) HARDWARE="$(echo "$GPU_NAME" | tr -d ' ')" ;;
esac
echo "==> GPU: $GPU_NAME  ->  HARDWARE=$HARDWARE  (branch: $BRANCH)"

SKEW_FLAG="--skip-skew"
[ "${WITH_SKEW:-0}" = "1" ] && SKEW_FLAG=""

run_one() {  # <hf-model-id> <tp-list>
  local model="$1" want_tps="$2"
  local mdir="profiler/perf/$HARDWARE/$model/bf16"
  # prune TP degrees that already have results (resume safety)
  local tps=""
  for tp in ${want_tps//,/ }; do
    if [ -d "$mdir/tp$tp" ] && [ -s "$mdir/tp$tp/dense.csv" ]; then
      echo "    tp$tp already profiled — skipping"
    else
      tps="${tps:+$tps,}$tp"
    fi
  done
  if [ -z "$tps" ]; then
    echo "==> $model: all requested TP degrees present — skip"
    return 0
  fi
  echo "=================================================================="
  echo "==> PROFILING $model  (tp=$tps, $SKEW_FLAG)"
  echo "=================================================================="
  if python3 -m profiler profile "$model" \
        --hardware "$HARDWARE" --tp "$tps" \
        --max-num-batched-tokens 2048 --max-num-seqs 256 \
        --attention-max-kv 32768 $SKEW_FLAG; then
    git add "profiler/perf/$HARDWARE" 2>/dev/null
    git commit -q -m "Add $HARDWARE profiles: $model (tp $tps)" 2>/dev/null \
      && echo "==> committed"
    if [ "$PUSH" = "1" ]; then
      git push origin "$BRANCH" && echo "==> pushed" \
        || echo "==> PUSH FAILED (results are committed locally)"
    fi
  else
    echo "==> FAIL $model — continuing with the next model"
  fi
}

# ---- priority list (see docs/profiling-guide.md) ----
run_one "NousResearch/Meta-Llama-3.1-70B"        "1,2,4,8"   # P1: dense large
run_one "Qwen/Qwen3-235B-A22B"                   "1,2,4,8"   # P2: modern MoE
run_one "meta-llama/Llama-3.1-8B"                "1,2,4"     # P3: anchor
run_one "mistralai/Mixtral-8x7B-Instruct-v0.1"   "1,2,4,8"   # P4: classic MoE
run_one "deepseek-ai/DeepSeek-V3-0324"           "8"         # P5: extend MLA to tp8
run_one "openai/gpt-oss-120b"                    "1,2,4"     # P6: yaml draft validation

echo "==> ALL DONE. perf tree:"
find "profiler/perf/$HARDWARE" -maxdepth 3 -type d | sort
