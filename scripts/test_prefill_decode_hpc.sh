#!/bin/bash
# -----------------------------------------------------------------------------
# Robustness matrix: does PREFILL and DECODE run correctly across many cases?
# Exercises every architecture code path (MLA / GQA / dense-attn), EP scaling
# (in-domain <= panel and cross-domain > panel), both phases, and both fabrics
# (glass-FB + NVL72 via the sweep's _fabric_pair). This is a SMOKE/SANITY matrix
# -- it checks COMPLETION + sane finite metrics (no hang/crash/OOM/NaN), not
# accuracy. TP=1 only (TP>1 multi-instance dp_group deadlocks in ASTRA).
#
# Run on HPC (simulator container/venv). Resumable: completed case CSVs are skipped.
#   bash scripts/test_prefill_decode_hpc.sh
#   MODELS_FILTER=deepseek bash scripts/test_prefill_decode_hpc.sh   # one model
# At the end it prints a PASS/FAIL table and exits non-zero if any case failed.
# -----------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=outputs/panel_dse/robust_test
mkdir -p "$OUT"
WG=8; PANEL="4 4"; RACK=8; NVLBW=450; TIMEOUT="${TIMEOUT:-7200}"
BPI="${BPI:-4}"                     # batch per instance (small: robustness, not measurement)
MOE_ALLTOALL="${MOE_ALLTOALL:-1}"; export MOE_ALLTOALL

# matrix rows: "HF_id:tag:arch:npu_mem:ep_list"
#   arch is informational (the path is auto-selected by config fields).
#   ep_list chosen so weights FIT npu_mem and EP divides the expert count, spanning
#   in-domain (<=16) and cross-domain (>16) where possible.
CASES=(
  "deepseek-ai/DeepSeek-V3-0324:deepseek:MLA:141:16 32 128"
  "moonshotai/Kimi-K2-Instruct:kimi:MLA:141:32 128"
  "Qwen/Qwen3-235B-A22B:qwen235b:GQA:141:8 16 128"
  "Qwen/Qwen3-30B-A3B-Instruct-2507:qwen30b:GQA:96:8 32 128"
  "mistralai/Mixtral-8x7B-v0.1:mixtral:dense:96:2 4 8"
)
# phases: "name:isl:osl:max_tokens"
PHASES=("decode:256:16:2048" "prefill:2048:2:2048")

run_case () {  # model tag mem ep_list phase isl osl mtok
  local model="$1" tag="$2" mem="$3" eps="$4" phase="$5" isl="$6" osl="$7" mtok="$8"
  local out="$OUT/${tag}_${phase}.csv"
  if [ -s "$out" ] && [ "$(wc -l < "$out")" -gt 1 ]; then echo "  SKIP $out (exists)"; return; fi
  echo "  === $tag $phase  EP={$eps} isl$isl/osl$osl mem${mem} ==="
  python scripts/sweep_panel_dse.py --sweep epscale \
    --ep-list $eps --epscale-panel $PANEL --fixed-wg $WG \
    --nvl72-rack $RACK --nvl72-bw $NVLBW --agg-bw \
    --isl "$isl" --osl "$osl" --max-tokens "$mtok" --batch-per-instance $BPI \
    --model "$model" --hardware H100 --tp 1 --npu-mem-gb "$mem" \
    --timeout "$TIMEOUT" --out "$out" 2>&1 | grep -iE "_ep[0-9]|error|exceed|Waiting" | tail -12
}

echo "=== Prefill/Decode robustness matrix (TP=1, glass+NVL72) ==="
for row in "${CASES[@]}"; do
  IFS=':' read -r model tag arch mem eps <<< "$row"
  [ -n "${MODELS_FILTER:-}" ] && [[ "$tag" != *"$MODELS_FILTER"* ]] && continue
  echo ""; echo "### $tag ($arch)  experts-EP {$eps}"
  for ph in "${PHASES[@]}"; do
    IFS=':' read -r pname isl osl mtok <<< "$ph"
    run_case "$model" "$tag" "$mem" "$eps" "$pname" "$isl" "$osl" "$mtok"
  done
done

# ---------------- PASS/FAIL summary ----------------
echo ""; echo "================= SUMMARY ================="
python3 - "$OUT" <<'PY'
import csv, glob, os, sys, math
out = sys.argv[1]
total = ok = bad = 0
fails = []
for f in sorted(glob.glob(os.path.join(out, "*.csv"))):
    for r in csv.DictReader(open(f)):
        total += 1
        st = r.get("status")
        tp = r.get("tpot_gt_ms"); ef = r.get("exposed_frac")
        finite = True
        try:
            for v in (tp, ef):
                if v not in (None, "") and not math.isfinite(float(v)): finite = False
        except (TypeError, ValueError):
            finite = False
        if st == "ok" and finite:
            ok += 1
        else:
            bad += 1
            fails.append((os.path.basename(f), r.get("label"), st, (r.get("error") or "")[:60]))
print(f"cases: {total}   PASS: {ok}   FAIL: {bad}")
for fn, lbl, st, err in fails:
    print(f"  FAIL  {fn:28s} {lbl:24s} [{st}] {err}")
sys.exit(1 if bad else 0)
PY
rc=$?
echo "=========================================="
[ $rc -eq 0 ] && echo "ALL PASS" || echo "SOME FAILED (see above)"
exit $rc
