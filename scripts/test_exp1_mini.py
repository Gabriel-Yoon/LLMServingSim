"""
Minimal one-run sanity test for Exp 1 (sweep_paper.py) on HPC / Docker.

Runs a single simulation with:
  - topology : h100_ep32.json  (NVL72-like, EP=32)
  - N=1 request, ISL=10, OSL=5  (5 decode steps → ~22s on HPC, ~4 min in Docker)
  - --skip-prefill

Pass: simulator exits 0, outputs/test_exp1_mini.csv written, TPOT printed.
Fail: non-zero exit or missing CSV.

Usage (inside Apptainer or Docker, from repo root):
  python scripts/test_exp1_mini.py

Optional: test FB 4x4 topology instead of NVL72
  python scripts/test_exp1_mini.py --topo fb
"""

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCAB_SIZE = 151936

TOPOLOGIES = {
    "nvl72":      "configs/cluster/h100_ep32.json",
    "fb":         "configs/cluster/h100_fb_4x4_ep32.json",
    "nvl72_ep8":  "configs/cluster/h100_ep8.json",
    "nvl72_ep16": "configs/cluster/h100_ep16.json",
    "nvl72_ep32": "configs/cluster/h100_ep32.json",
    "nvl72_ep64": "configs/cluster/h100_ep64.json",
    "nvl72_ep128": "configs/cluster/h100_ep128.json",
}

ISL = 10
OSL = 5
N   = 1


def make_workload() -> str:
    rng   = random.Random(42)
    inp   = [rng.randint(0, VOCAB_SIZE - 1) for _ in range(ISL)]
    out   = [rng.randint(0, VOCAB_SIZE - 1) for _ in range(OSL)]
    entry = {"input_toks": ISL, "output_toks": OSL,
             "arrival_time_ns": 0,
             "input_tok_ids": inp, "output_tok_ids": out}
    wl_path = os.path.join(REPO_ROOT, "outputs", "test_exp1_mini_workload.jsonl")
    os.makedirs(os.path.dirname(wl_path), exist_ok=True)
    with open(wl_path, "w") as f:
        f.write(json.dumps(entry) + "\n")
    return os.path.relpath(wl_path, REPO_ROOT)


def run(topo_name: str):
    config_path = TOPOLOGIES[topo_name]
    config_full = os.path.join(REPO_ROOT, config_path)
    if not os.path.exists(config_full):
        print(f"[FAIL] Config not found: {config_full}")
        sys.exit(1)

    wl_rel  = make_workload()
    out_csv = os.path.join(REPO_ROOT, "outputs", "test_exp1_mini.csv")
    out_rel = os.path.relpath(out_csv, REPO_ROOT)

    cmd = [
        "python", "-m", "serving",
        "--cluster-config", config_path,
        "--dtype", "bfloat16",
        "--block-size", "16",
        "--max-num-seqs", "128",
        "--max-num-batched-tokens", "2048",
        "--dataset", wl_rel,
        "--output", out_rel,
        "--num-req", str(N),
        "--log-level", "INFO",
    ]

    print(f"Topology : {topo_name}  ({config_path})")
    print(f"Workload : ISL={ISL}, OSL={OSL}, N={N}  → 1 prefill + {OSL} decode steps")
    print(f"Command  : {' '.join(cmd)}")
    print()

    t0   = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"\n[FAIL] exit code {proc.returncode}  ({elapsed:.1f}s)")
        sys.exit(1)

    if not os.path.exists(out_csv):
        print(f"\n[FAIL] output CSV not found: {out_csv}")
        sys.exit(1)

    rows = list(csv.DictReader(open(out_csv)))
    if not rows:
        print(f"\n[FAIL] output CSV is empty")
        sys.exit(1)

    tpots = [float(r["TPOT"]) / 1e6 for r in rows if r.get("TPOT")]
    tpot_avg = sum(tpots) / len(tpots) if tpots else 0.0

    print(f"\n[PASS]  elapsed={elapsed:.1f}s  n_completed={len(rows)}  TPOT_avg={tpot_avg:.2f}ms")
    print(f"        output: {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", choices=["nvl72", "fb", "nvl72_ep8", "nvl72_ep16",
                                       "nvl72_ep32", "nvl72_ep64", "nvl72_ep128"], default="nvl72_ep8",
                    help="Topology to test (default: nvl72_ep8)")
    args = ap.parse_args()
    run(args.topo)


if __name__ == "__main__":
    main()
