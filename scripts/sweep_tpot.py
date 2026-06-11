"""
Throughput vs Interactivity sweep for FB-4x4 and NVL72 at EP=32.

For each batch size N, all N requests arrive simultaneously and skip
prefill (--skip-prefill), giving a pure decode batch of size N.

  Throughput   [tokens/s]       = N / TPOT_avg
  Interactivity [tokens/s/user] = 1 / TPOT_avg   (= TPOT^-1)

SLO boundary: TPOT < 15 ms  →  interactivity > 66.7 tokens/s/user

Results: results/exp_tpot_throughput_ep32.csv  (git-tracked)
Per-request CSVs: outputs/tpot_sweep/<label>.csv (gitignored)

Usage:
  python scripts/sweep_tpot.py               # full sweep, both topologies
  python scripts/sweep_tpot.py --topo fb     # FB only
  python scripts/sweep_tpot.py --topo nvl72  # NVL72 only
  python scripts/sweep_tpot.py --dry-run     # print plan without running

  # Override sweep parameters
  python scripts/sweep_tpot.py --n-values 1,4,16,64 --isl 512 --osl 50

Docker (local):
  docker exec servingsim_docker bash -c \\
    "cd /app/LLMServingSim && python scripts/sweep_tpot.py"

HPC (via Apptainer):
  sbatch scripts/slurm_tpot_sweep.sh
"""

import argparse
import csv
import json
import os
import random
import subprocess
import tempfile
import time

# ─── Sweep configuration ────────────────────────────────────────────────────
DEFAULT_N_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]
ISL        = 512    # initial KV context tokens (chat-like)
OSL        = 50     # output tokens per request (steady-state TPOT)
VOCAB_SIZE = 151936  # Qwen3 vocab size

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
HARDWARE   = "H100"
BLOCK_SIZE = 16
LOG_LEVEL  = "WARNING"

# ─── Topology configs ────────────────────────────────────────────────────────
TOPOLOGIES = {
    "nvl72": "configs/cluster/h100_ep32.json",      # NVLink 1800 GB/s, 1000 ns
    "fb":    "configs/cluster/h100_fb_4x4_ep32.json",  # Optical 512 GB/s, 300/2000 ns
}

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV = os.path.join(REPO_ROOT, "results", "exp_tpot_throughput_ep32.csv")
OUTPUT_DIR  = os.path.join(REPO_ROOT, "outputs", "tpot_sweep")

CSV_FIELDS = [
    "label", "status", "topology", "n_requests", "isl", "osl",
    "tpot_avg_ms", "tpot_p50_ms", "tpot_p99_ms",
    "throughput_toks_per_s", "interactivity_toks_per_s_per_user",
    "step_time_ms",
    "n_completed", "elapsed_s", "error",
]

TPOT_SLO_MS = 15.0


# ─── Workload generation ─────────────────────────────────────────────────────
def _make_workload(n: int, isl: int, osl: int, seed: int = 0) -> str:
    """Write JSONL with N requests arriving 1ms apart.

    Returns a path RELATIVE to REPO_ROOT so the simulator (cwd=astra-sim/)
    resolves it correctly via its internal '../' prefix.
    """
    rng = random.Random(seed)
    lines = []
    for i in range(n):
        inp = [rng.randint(0, VOCAB_SIZE - 1) for _ in range(isl)]
        out = [rng.randint(0, VOCAB_SIZE - 1) for _ in range(osl)]
        lines.append(json.dumps({
            "input_toks": isl,
            "output_toks": osl,
            "arrival_time_ns": i * 1_000_000,  # 1 ms stagger
            "input_tok_ids": inp,
            "output_tok_ids": out,
        }))
    abs_path = os.path.join(OUTPUT_DIR, f"wl_n{n}_isl{isl}_osl{osl}.jsonl")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(abs_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    # return relative to REPO_ROOT (simulator prefixes ../ internally)
    return os.path.relpath(abs_path, REPO_ROOT)


# ─── Results helpers ─────────────────────────────────────────────────────────
def _load_done(results_csv: str) -> set:
    done = set()
    if not os.path.exists(results_csv):
        return done
    with open(results_csv) as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok":
                done.add(row["label"])
    return done


def _append_row(results_csv: str, row: dict):
    write_header = not os.path.exists(results_csv)
    with open(results_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def _parse_output(out_csv: str, n: int, osl: int):
    if not os.path.exists(out_csv):
        return None
    rows = list(csv.DictReader(open(out_csv)))
    if not rows:
        return None

    def ns_to_ms(v):
        try:
            return float(v) / 1e6
        except Exception:
            return None

    tpots = [ns_to_ms(r["TPOT"]) for r in rows if r.get("TPOT")]
    tpots = [v for v in tpots if v is not None]
    if not tpots:
        return None

    tpots_sorted = sorted(tpots)
    tpot_avg  = sum(tpots) / len(tpots)
    tpot_p50  = tpots_sorted[len(tpots_sorted) // 2]
    tpot_p99  = tpots_sorted[min(int(len(tpots_sorted) * 0.99), len(tpots_sorted) - 1)]

    throughput     = n / (tpot_avg / 1000.0)   # tokens/s (n tokens per step_time)
    interactivity  = 1000.0 / tpot_avg          # tokens/s/user = 1/TPOT_s

    return {
        "tpot_avg_ms":   tpot_avg,
        "tpot_p50_ms":   tpot_p50,
        "tpot_p99_ms":   tpot_p99,
        "throughput_toks_per_s":               throughput,
        "interactivity_toks_per_s_per_user":   interactivity,
        "step_time_ms":  tpot_avg,  # decode: step_time = TPOT
        "n_completed":   len(rows),
    }


# ─── Simulation runner ───────────────────────────────────────────────────────
def run_sim(label, config_path, workload_path, n, run_dir,
            dry_run=False, timeout=7200):
    os.makedirs(run_dir, exist_ok=True)
    # Absolute path for local file-existence checks; relative for subprocess args.
    # config_builder, router, and scheduler all prepend '../' internally (cwd=astra-sim/).
    out_csv_abs = os.path.join(run_dir, f"{label}.csv")
    out_csv_rel = os.path.relpath(out_csv_abs, REPO_ROOT)

    cmd = [
        "python", "-m", "serving",
        "--cluster-config", config_path,    # relative to REPO_ROOT
        "--dtype", "bfloat16",
        "--block-size", str(BLOCK_SIZE),
        "--max-num-seqs", "128",
        "--max-num-batched-tokens", "2048",
        "--dataset", workload_path,          # relative to REPO_ROOT
        "--output", out_csv_rel,             # relative to REPO_ROOT
        "--num-req", str(n),
        "--skip-prefill",
        "--log-level", LOG_LEVEL,
    ]

    if dry_run:
        print(f"  [dry] {label:40s}  n={n:3d}  config={os.path.basename(config_path)}")
        return {"status": "dry", **{f: 0 for f in CSV_FIELDS if f not in
                                    ("label", "status", "topology", "n_requests",
                                     "isl", "osl", "error")}, "error": ""}

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=REPO_ROOT, timeout=timeout,
        )
        elapsed = time.time() - t0
        if proc.returncode != 0:
            return {"status": "error", "elapsed_s": elapsed,
                    "error": proc.stderr.strip()[-300:],
                    **{f: 0 for f in CSV_FIELDS
                       if f not in ("label","status","topology","n_requests",
                                    "isl","osl","elapsed_s","error")}}
        metrics = _parse_output(out_csv_abs, n, OSL) or {}
        return {"status": "ok", "elapsed_s": elapsed, "error": "", **metrics}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "elapsed_s": timeout,
                "error": f"exceeded {timeout}s",
                **{f: 0 for f in CSV_FIELDS
                   if f not in ("label","status","topology","n_requests",
                                "isl","osl","elapsed_s","error")}}


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run",  action="store_true")
    ap.add_argument("--topo",     choices=["nvl72", "fb", "both"], default="both")
    ap.add_argument("--n-values", default=None,
                    help="Comma-separated batch sizes, e.g. 1,4,16,64")
    ap.add_argument("--isl",      type=int, default=ISL)
    ap.add_argument("--osl",      type=int, default=OSL)
    ap.add_argument("--results-csv", default=RESULTS_CSV)
    ap.add_argument("--output-dir",  default=OUTPUT_DIR)
    args = ap.parse_args()

    n_values = (
        [int(x) for x in args.n_values.split(",")]
        if args.n_values else DEFAULT_N_VALUES
    )
    topo_list = {"both": ["nvl72", "fb"], "nvl72": ["nvl72"], "fb": ["fb"]}[args.topo]

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)

    done = _load_done(args.results_csv)

    print("=== Throughput vs Interactivity Sweep ===")
    print(f"Topologies : {topo_list}")
    print(f"N values   : {n_values}")
    print(f"ISL / OSL  : {args.isl} / {args.osl}")
    print(f"Results    : {args.results_csv}")
    print(f"SLO        : TPOT < {TPOT_SLO_MS} ms  "
          f"(interactivity > {1000/TPOT_SLO_MS:.1f} tokens/s/user)\n")

    for topo in topo_list:
        config_path = TOPOLOGIES[topo]  # relative to REPO_ROOT; simulator prefixes ../ internally
        for n in n_values:
            label = f"{topo}_ep32_n{n:03d}"
            if label in done:
                print(f"[SKIP] {label}")
                continue

            wl_path = _make_workload(n, args.isl, args.osl)
            run_dir = os.path.join(args.output_dir, label)
            print(f"[RUN ] {label:40s}  n={n:3d}", flush=True)

            result = run_sim(
                label=label,
                config_path=config_path,
                workload_path=wl_path,
                n=n,
                run_dir=run_dir,
                dry_run=args.dry_run,
            )

            row = {
                "label": label, "topology": topo,
                "n_requests": n, "isl": args.isl, "osl": args.osl,
                **result,
            }
            if not args.dry_run:
                _append_row(args.results_csv, row)

            tpot   = result.get("tpot_avg_ms", 0)
            tput   = result.get("throughput_toks_per_s", 0)
            inter  = result.get("interactivity_toks_per_s_per_user", 0)
            slo_ok = "✅" if tpot < TPOT_SLO_MS else "❌"
            print(f"         → TPOT={tpot:.2f}ms  "
                  f"throughput={tput:.0f} tok/s  "
                  f"interactivity={inter:.1f} tok/s/user  {slo_ok}")

    if not args.dry_run:
        print(f"\nDone → {args.results_csv}")


if __name__ == "__main__":
    main()
