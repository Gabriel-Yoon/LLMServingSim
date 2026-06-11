#!/bin/bash
# SLURM job: warm cache SLO test for Chat / Coding / Agentic.
#
# Runs 3 workloads × 2 topologies (NVL72 + FB-4x4) at EP=32 sequentially.
# NUM_REQ=3: req0=cold-start, req1/req2=warm-cache (prefix cached).
#
# Usage:
#   sbatch scripts/slurm_warm_cache.sh
#   sbatch scripts/slurm_warm_cache.sh fb        # FB 4x4 only
#   sbatch scripts/slurm_warm_cache.sh nvl72     # NVL72 only
#
# SLO targets: TTFT < 50ms (Chat), 100ms (Coding), 150ms (Agentic)
#              TPOT < 15ms (all)
# Results: results/exp_warm_cache_ep32.csv

#SBATCH --job-name=llmsim_warm
#SBATCH --output=outputs/slurm_%x_%j.log
#SBATCH --error=outputs/slurm_%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOPO_FILTER="${1:-both}"   # nvl72 | fb | both

APPTAINER_IMAGE="/storage/project/r-syu334-0/syoon351/containers/llmservingsim.sif"
SCRATCH="/storage/home/hcoda1/8/syoon351/scratch"
VENV="$SCRATCH/deps/apptainer_home/venvs/llmservingsim"

echo "=== Warm Cache SLO Test ==="
echo "Repo   : $REPO_ROOT"
echo "Topo   : $TOPO_FILTER"
echo "Start  : $(date)"

mkdir -p "$REPO_ROOT/outputs" "$REPO_ROOT/results"

# ── Topology configs (H100 profiling data, EP=32) ────────────────────────────
declare -A CONFIGS
CONFIGS[nvl72]="configs/cluster/h100_ep32.json"
CONFIGS[fb]="configs/cluster/h100_fb_4x4_ep32.json"

# ── Workloads ─────────────────────────────────────────────────────────────────
declare -A DATASETS
DATASETS[chat]="workloads/chat_fixed.jsonl"
DATASETS[coding]="workloads/coding_fixed.jsonl"
DATASETS[agentic]="workloads/agentic_hpc.jsonl"

# ── Output CSV ────────────────────────────────────────────────────────────────
RESULTS_CSV="$REPO_ROOT/results/exp_warm_cache_ep32.csv"

_run_sim() {
    local topo="$1" wl="$2" config="$3" dataset="$4"
    local label="${topo}_ep32_${wl}"
    local out_csv="$REPO_ROOT/outputs/warm_${label}.csv"

    echo ""
    echo "[RUN] $label"
    echo "      config  : $config"
    echo "      dataset : $dataset"

    apptainer exec \
        --cleanenv \
        --home "$SCRATCH/deps/apptainer_home":/home/syoon351 \
        --bind "$REPO_ROOT":/app/LLMServingSim \
        --bind "$SCRATCH":"$SCRATCH" \
        --pwd /app/LLMServingSim \
        "$APPTAINER_IMAGE" \
        bash -lc "
            source $VENV/bin/activate
            python -m serving \
                --cluster-config $config \
                --dtype bfloat16 \
                --block-size 16 \
                --dataset $dataset \
                --output $out_csv \
                --num-req 3 \
                --log-level WARNING
        "

    echo "      done → $out_csv"

    # Append summary row to results CSV
    python3 - <<PYEOF
import csv, os

out_csv = "$out_csv"
results_csv = "$RESULTS_CSV"
label = "$label"
topo = "$topo"
wl = "$wl"

if not os.path.exists(out_csv):
    print(f"  [WARN] output CSV not found: {out_csv}")
    exit()

rows = list(csv.DictReader(open(out_csv)))
if not rows:
    print("  [WARN] empty output CSV")
    exit()

def ns_to_ms(v):
    try: return float(v) / 1e6
    except: return None

# warm rows = req_id >= 1
warm = [r for r in rows if int(r.get("request id", r.get("request_id", 0))) >= 1]
all_rows = rows

def avg_ms(field, subset):
    vals = [ns_to_ms(r[field]) for r in subset if r.get(field)]
    return sum(v for v in vals if v is not None) / len([v for v in vals if v is not None]) if vals else None

summary = {
    "label": label, "topology": topo, "workload": wl, "ep": 32,
    "ttft_cold_ms":  avg_ms("TTFT", rows[:1]),
    "tpot_cold_ms":  avg_ms("TPOT", rows[:1]),
    "ttft_warm_ms":  avg_ms("TTFT", warm),
    "tpot_warm_ms":  avg_ms("TPOT", warm),
    "n_warm": len(warm),
}

fields = list(summary.keys())
write_header = not os.path.exists(results_csv)
with open(results_csv, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    if write_header:
        w.writeheader()
    w.writerow(summary)

print(f"  TTFT cold={summary['ttft_cold_ms']:.1f}ms  warm={summary['ttft_warm_ms']:.1f}ms  TPOT={summary['tpot_warm_ms']:.2f}ms")
PYEOF
}

# ── Run matrix ────────────────────────────────────────────────────────────────
for wl in chat coding agentic; do
    for topo in nvl72 fb; do
        [[ "$TOPO_FILTER" != "both" && "$TOPO_FILTER" != "$topo" ]] && continue
        cfg="${CONFIGS[$topo]}"
        ds="${DATASETS[$wl]}"
        _run_sim "$topo" "$wl" "$cfg" "$ds"
    done
done

echo ""
echo "=== Done. Results: $RESULTS_CSV ==="
echo "End: $(date)"
