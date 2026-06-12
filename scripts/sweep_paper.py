"""
ASP-DAC 2027 paper experiment sweep.

CONTROLLED COMPARISON CONTRACT
  Same model:    Qwen/Qwen3-30B-A3B-Instruct-2507  (H100 compute profile)
  Same dtype:    bfloat16
  Same ISL/OSL:  512 / 256 tokens  (Chat use case)
  Same scheduler: max_num_seqs=128, max_num_batched_tokens=2048
  Same workload: deterministic JSONL (seed=0), arrivals 1ms apart
  Only variable: topology config (→ interconnect BW + latency)
  Goodput:       max throughput at TPOT ≤ 15 ms (SLO_TPOT_MS)

NVL72-like baseline = "NVLink-switch-class high-BW interconnect + H100 compute"
  NOT a reproduction of actual GB200 NVL72 hardware.

Experiments
  1   T-I curve: FB 4x4 vs NVL72-like baseline (lat=0), EP=32, N=1..128
  1s  NVL72 latency sensitivity: lat=100,300,500,1000 ns, EP=32, N=16..128
  2   EP scalability: FB 4x4 + NVL72, EP=16,32,64,128, N=32,64,128
  3   Panel DSE: 4x4 vs ps16-in900 vs 6x6-4c vs NVL72, EP=32, N=32,64,128

Usage:
  python scripts/sweep_paper.py                   # all experiments
  python scripts/sweep_paper.py --exp 1           # Exp 1 only
  python scripts/sweep_paper.py --exp 1s          # latency sensitivity
  python scripts/sweep_paper.py --exp 2           # EP scaling
  python scripts/sweep_paper.py --exp 3           # panel DSE
  python scripts/sweep_paper.py --exp 1 --dry-run # print plan

Docker:
  docker exec servingsim_docker bash -c \\
    "cd /app/LLMServingSim && python scripts/sweep_paper.py --exp 1"

HPC:
  sbatch scripts/slurm_paper_sweep.sh
"""

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time

# ─── Constants ────────────────────────────────────────────────────────────────
VOCAB_SIZE             = 151936
BLOCK_SIZE             = 16
LOG_LEVEL              = "WARNING"
TPOT_SLO_MS            = 15.0      # goodput boundary
MAX_NUM_SEQS           = 128
MAX_NUM_BATCHED_TOKENS = 2048
ISL                    = 512
OSL                    = 256    # Chat use case: enough for steady-state TPOT with staggered arrivals

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "paper_sweep")

CSV_FIELDS = [
    "exp", "label", "status",
    "topology", "ep", "n_requests", "isl", "osl",
    "tpot_avg_ms", "tpot_p50_ms", "tpot_p99_ms",
    "throughput_toks_per_s", "interactivity_toks_per_s_per_user",
    "goodput_toks_per_s",      # throughput at TPOT ≤ SLO (else 0)
    "within_slo",              # 1 if TPOT ≤ SLO_TPOT_MS
    "n_completed", "elapsed_s", "error",
]

# ─── Experiment definitions ───────────────────────────────────────────────────
_FB4X4     = "configs/cluster/h100_fb_4x4_ep32.json"
_NVL72_L0  = "configs/cluster/h100_ep32.json"

EXPERIMENTS = {
    "1": {
        "desc": "T-I curve: FB 4x4 vs NVL72-like (lat=0), EP=32, N=1..128",
        "topologies": {
            "fb_4x4_ep32":     "configs/cluster/h100_fb_4x4_ep32.json",
            "nvl72_lat0_ep32": "configs/cluster/h100_ep32.json",
        },
        "n_values": [1, 2, 4, 8, 16, 32, 64, 128],
        "results_csv": "results/paper_exp1_ti.csv",
    },
    "1s": {
        "desc": "NVL72 latency sensitivity: lat=100,300,500,1000 ns, EP=32",
        "topologies": {
            "nvl72_lat100_ep32":  "configs/cluster/h100_ep32_lat100.json",
            "nvl72_lat300_ep32":  "configs/cluster/h100_ep32_lat300.json",
            "nvl72_lat500_ep32":  "configs/cluster/h100_ep32_lat500.json",
            "nvl72_lat1000_ep32": "configs/cluster/h100_ep32_lat1000.json",
        },
        "n_values": [16, 32, 64, 128],
        "results_csv": "results/paper_exp1s_lat_sensitivity.csv",
    },
    "2": {
        "desc": "EP scalability: FB 4x4 + NVL72, EP=16,32,64,128, N=32,64,128",
        "topologies": {
            "fb_4x4_ep16":      "configs/cluster/h100_fb_4x4_ep16.json",
            "fb_4x4_ep32":      "configs/cluster/h100_fb_4x4_ep32.json",
            "fb_4x4_ep64":      "configs/cluster/h100_fb_4x4_ep64.json",
            "fb_4x4_ep128":     "configs/cluster/h100_fb_4x4_ep128.json",
            "nvl72_lat0_ep16":  "configs/cluster/h100_ep16.json",
            "nvl72_lat0_ep32":  "configs/cluster/h100_ep32.json",
            "nvl72_lat0_ep64":  "configs/cluster/h100_ep64.json",
            "nvl72_lat0_ep128": "configs/cluster/h100_ep128.json",
        },
        "n_values": [32, 64, 128],
        "results_csv": "results/paper_exp2_ep_scale.csv",
    },
    "3": {
        "desc": "Panel DSE: fb_4x4 vs ps16-in900 vs 6x6-4c vs NVL72, EP=32",
        "topologies": {
            "fb_4x4_in400_ep32":  "configs/cluster/h100_fb_4x4_ep32.json",
            "fb_ps16_in900_ep32": "configs/cluster/h100_fb_ps16_ep32_in900_ex200.json",
            "fb_6x6_4c_ep32":     "configs/cluster/h100_fb_6x6_4c_ep32.json",
            "nvl72_lat0_ep32":    "configs/cluster/h100_ep32.json",
        },
        "n_values": [32, 64, 128],
        "results_csv": "results/paper_exp3_panel_dse.csv",
    },
}


# ─── Workload generation ──────────────────────────────────────────────────────
def _make_workload(n: int, isl: int, osl: int, seed: int = 0) -> str:
    """Return path RELATIVE to REPO_ROOT (simulator prepends ../ internally)."""
    rng = random.Random(seed)
    lines = []
    for i in range(n):
        inp = [rng.randint(0, VOCAB_SIZE - 1) for _ in range(isl)]
        out = [rng.randint(0, VOCAB_SIZE - 1) for _ in range(osl)]
        lines.append(json.dumps({
            "input_toks": isl, "output_toks": osl,
            "arrival_time_ns": i * 1_000_000,
            "input_tok_ids": inp, "output_tok_ids": out,
        }))
    abs_path = os.path.join(OUTPUT_DIR, "workloads", f"wl_n{n}_isl{isl}_osl{osl}.jsonl")
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return os.path.relpath(abs_path, REPO_ROOT)


# ─── Results helpers ──────────────────────────────────────────────────────────
def _load_done(csv_path: str) -> set:
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok":
                done.add(row["label"])
    return done


def _append_row(csv_path: str, row: dict):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def _parse_output(out_csv_abs: str, n: int):
    if not os.path.exists(out_csv_abs):
        return None
    rows = list(csv.DictReader(open(out_csv_abs)))
    if not rows:
        return None

    def _ms(v):
        try:
            return float(v) / 1e6
        except Exception:
            return None

    tpots = [_ms(r["TPOT"]) for r in rows if r.get("TPOT")]
    tpots = [v for v in tpots if v is not None]
    if not tpots:
        return None

    tpots.sort()
    tpot_avg = sum(tpots) / len(tpots)
    tpot_p50 = tpots[len(tpots) // 2]
    tpot_p99 = tpots[min(int(len(tpots) * 0.99), len(tpots) - 1)]

    throughput    = n / (tpot_avg / 1000.0)
    interactivity = 1000.0 / tpot_avg
    within_slo    = 1 if tpot_avg <= TPOT_SLO_MS else 0
    goodput       = throughput if within_slo else 0.0

    return {
        "tpot_avg_ms": tpot_avg, "tpot_p50_ms": tpot_p50, "tpot_p99_ms": tpot_p99,
        "throughput_toks_per_s": throughput,
        "interactivity_toks_per_s_per_user": interactivity,
        "goodput_toks_per_s": goodput,
        "within_slo": within_slo,
        "n_completed": len(rows),
    }


# ─── Simulation runner ────────────────────────────────────────────────────────
_ZERO_METRICS = {
    "tpot_avg_ms": 0, "tpot_p50_ms": 0, "tpot_p99_ms": 0,
    "throughput_toks_per_s": 0, "interactivity_toks_per_s_per_user": 0,
    "goodput_toks_per_s": 0, "within_slo": 0, "n_completed": 0,
}

def _run_one(label, config_path, workload_path, n, run_dir,
             dry_run=False, timeout=14400, debug=False):
    os.makedirs(run_dir, exist_ok=True)
    out_csv_abs = os.path.join(run_dir, f"{label}.csv")
    out_csv_rel = os.path.relpath(out_csv_abs, REPO_ROOT)

    cmd = [
        "python", "-m", "serving",
        "--cluster-config", config_path,
        "--dtype", "bfloat16",
        "--block-size", str(BLOCK_SIZE),
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--max-num-batched-tokens", str(MAX_NUM_BATCHED_TOKENS),
        "--dataset", workload_path,
        "--output", out_csv_rel,
        "--num-req", str(n),
        "--skip-prefill",
        "--log-level", "INFO" if debug else LOG_LEVEL,
    ]

    if dry_run:
        print(f"    [dry] n={n:<4d}  cfg={os.path.basename(config_path)}")
        return {"status": "dry", "error": ""}

    if debug:
        print(f"\n[DEBUG] $ {' '.join(cmd)}\n", flush=True)

    t0 = time.time()
    try:
        if debug:
            # Stream stdout/stderr directly to terminal — no capture
            proc = subprocess.run(cmd, cwd=REPO_ROOT, timeout=timeout)
            elapsed = time.time() - t0
            if proc.returncode != 0:
                return {"status": "error", "elapsed_s": elapsed,
                        "error": f"exit code {proc.returncode} (see streamed output above)",
                        **_ZERO_METRICS}
        else:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, cwd=REPO_ROOT, timeout=timeout,
            )
            elapsed = time.time() - t0
            if proc.returncode != 0:
                stderr_tail = proc.stderr.strip()[-1200:]
                stdout_tail = proc.stdout.strip()[-600:]
                combined = (stderr_tail + "\n---stdout---\n" + stdout_tail).strip()[-1200:]
                return {"status": "error", "elapsed_s": elapsed, "error": combined,
                        **_ZERO_METRICS}

        metrics = _parse_output(out_csv_abs, n) or {}
        return {"status": "ok", "elapsed_s": elapsed, "error": "", **metrics}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "elapsed_s": timeout,
                "error": f"timeout>{timeout}s", **_ZERO_METRICS}


# ─── EP extraction ────────────────────────────────────────────────────────────
def _ep_from_name(topo_name: str) -> int:
    """Extract EP size from topology label (e.g. fb_4x4_ep32 → 32)."""
    import re
    m = re.search(r"ep(\d+)", topo_name)
    return int(m.group(1)) if m else 0


# ─── Run one experiment ───────────────────────────────────────────────────────
def run_experiment(exp_id: str, dry_run: bool = False, debug: bool = False):
    exp = EXPERIMENTS[exp_id]
    results_csv = os.path.join(REPO_ROOT, exp["results_csv"])
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)

    isl, osl = exp.get("isl", ISL), exp.get("osl", OSL)

    done = _load_done(results_csv)

    print(f"\n{'='*70}")
    print(f"Exp {exp_id}: {exp['desc']}")
    print(f"  Topologies : {list(exp['topologies'].keys())}")
    print(f"  N values   : {exp['n_values']}")
    print(f"  ISL/OSL    : {isl}/{osl}")
    print(f"  SLO        : TPOT ≤ {TPOT_SLO_MS} ms")
    print(f"  Results    : {results_csv}")
    print(f"  Already done: {len(done)} runs")
    if debug:
        print(f"  [DEBUG] streaming mode — log_level=INFO, output not captured")

    for topo_name, config_path in exp["topologies"].items():
        ep = _ep_from_name(topo_name)
        config_full = os.path.join(REPO_ROOT, config_path)
        if not os.path.exists(config_full):
            print(f"  [SKIP] Config not found: {config_path}")
            continue

        for n in exp["n_values"]:
            label = f"exp{exp_id}_{topo_name}_n{n:03d}"
            if label in done:
                print(f"  [DONE] {label}")
                continue

            wl_path  = _make_workload(n, isl, osl)
            run_dir  = os.path.join(OUTPUT_DIR, f"exp{exp_id}", topo_name)
            print(f"  [RUN ] {label}", flush=True)

            result = _run_one(
                label=label,
                config_path=config_path,
                workload_path=wl_path,
                n=n, run_dir=run_dir,
                dry_run=dry_run,
                debug=debug,
            )

            row = {
                "exp": exp_id, "label": label, "topology": topo_name,
                "ep": ep, "n_requests": n, "isl": isl, "osl": osl,
                **result,
            }
            if not dry_run:
                _append_row(results_csv, row)

            tpot  = result.get("tpot_avg_ms", 0)
            tput  = result.get("throughput_toks_per_s", 0)
            slo   = "✅" if result.get("within_slo") else "❌"
            st    = result.get("status", "?")
            if st == "ok":
                print(f"  → TPOT={tpot:.2f}ms  tput={tput:.0f} tok/s  {slo}")
            elif st == "dry":
                print()
            else:
                print(f"  → [{st}] {result.get('error','')[-300:]}")

    if not dry_run:
        print(f"\n  ✓ Saved: {results_csv}")


# ─── Summary printer ──────────────────────────────────────────────────────────
def print_summary(exp_id: str):
    exp = EXPERIMENTS[exp_id]
    results_csv = os.path.join(REPO_ROOT, exp["results_csv"])
    if not os.path.exists(results_csv):
        print(f"  No results yet: {results_csv}")
        return

    rows = [r for r in csv.DictReader(open(results_csv)) if r.get("status") == "ok"]
    if not rows:
        print("  No ok rows.")
        return

    print(f"\n{'─'*90}")
    print(f"{'label':50s}  {'N':>4}  {'TPOT':>8}  {'tput':>10}  {'interact':>12}  SLO")
    print(f"{'─'*90}")
    for r in sorted(rows, key=lambda x: (x["topology"], int(x["n_requests"]))):
        n    = int(r["n_requests"])
        tpot = float(r["tpot_avg_ms"])
        tput = float(r["throughput_toks_per_s"])
        intr = float(r["interactivity_toks_per_s_per_user"])
        slo  = "✅" if tpot <= TPOT_SLO_MS else "❌"
        print(f"{r['label']:50s}  {n:4d}  {tpot:7.2f}ms  {tput:9.0f}  {intr:11.1f}  {slo}")

    # Goodput summary
    print(f"\n{'─'*50}")
    print("Goodput (max throughput within TPOT SLO per topology):")
    from collections import defaultdict
    by_topo = defaultdict(list)
    for r in rows:
        by_topo[r["topology"]].append(r)
    for topo, topo_rows in sorted(by_topo.items()):
        slo_rows = [r for r in topo_rows if float(r["tpot_avg_ms"]) <= TPOT_SLO_MS]
        if slo_rows:
            best = max(slo_rows, key=lambda x: float(x["throughput_toks_per_s"]))
            gput = float(best["throughput_toks_per_s"])
            n_at = int(best["n_requests"])
            print(f"  {topo:45s}  goodput={gput:.0f} tok/s  (N={n_at})")
        else:
            print(f"  {topo:45s}  no points within SLO")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp",      nargs="+", choices=list(EXPERIMENTS.keys()) + ["all"],
                    default=["all"], help="Which experiments to run (default: all)")
    ap.add_argument("--dry-run",  action="store_true")
    ap.add_argument("--debug",    action="store_true",
                    help="Stream simulation stdout/stderr to terminal (log_level=INFO). "
                         "Use with --exp 1 to monitor a single experiment.")
    ap.add_argument("--summary",  action="store_true",
                    help="Print summary table for completed experiments and exit")
    args = ap.parse_args()

    exp_ids = list(EXPERIMENTS.keys()) if "all" in args.exp else args.exp

    if args.summary:
        for eid in exp_ids:
            print(f"\nExp {eid}: {EXPERIMENTS[eid]['desc']}")
            print_summary(eid)
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for eid in exp_ids:
        run_experiment(eid, dry_run=args.dry_run, debug=args.debug)

    print("\n=== All requested experiments complete ===")
    for eid in exp_ids:
        print_summary(eid)


if __name__ == "__main__":
    main()
