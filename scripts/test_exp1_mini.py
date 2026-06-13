"""
Minimal one-run sanity test for Exp 1 (sweep_paper.py) on HPC / Docker.

Default: topology=nvl72_ep8, ISL=10, OSL=5, N=1 (fast smoke test).

To reproduce the exact sweep_paper.py exp1 N=16 scenario on HPC:
  python scripts/test_exp1_mini.py --topo fb --num-req 16 --isl 512 --osl 5 --skip-prefill

Pass: simulator exits 0, output CSV written, TPOT printed.
Fail: non-zero exit or missing CSV.

Usage (inside Apptainer or Docker, from repo root):
  python scripts/test_exp1_mini.py [--topo TOPO] [--num-req N] [--isl ISL] [--osl OSL] [--skip-prefill]
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
    "nvl72":       "configs/cluster/h100_ep32.json",
    "fb":          "configs/cluster/h100_fb_4x4_ep32.json",
    "nvl72_ep8":   "configs/cluster/h100_ep8.json",
    "nvl72_ep16":  "configs/cluster/h100_ep16.json",
    "nvl72_ep32":  "configs/cluster/h100_ep32.json",
    "nvl72_ep64":  "configs/cluster/h100_ep64.json",
    "nvl72_ep128": "configs/cluster/h100_ep128.json",
}


def make_workload(n: int, isl: int, osl: int) -> str:
    rng   = random.Random(42)
    lines = []
    for i in range(n):
        inp = [rng.randint(0, VOCAB_SIZE - 1) for _ in range(isl)]
        out = [rng.randint(0, VOCAB_SIZE - 1) for _ in range(osl)]
        lines.append(json.dumps({
            "input_toks": isl, "output_toks": osl,
            "arrival_time_ns": i * 1_000_000,
            "input_tok_ids": inp, "output_tok_ids": out,
        }))
    wl_path = os.path.join(REPO_ROOT, "outputs", f"test_mini_n{n}_isl{isl}_osl{osl}.jsonl")
    os.makedirs(os.path.dirname(wl_path), exist_ok=True)
    with open(wl_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return os.path.relpath(wl_path, REPO_ROOT)


def run(topo_name: str, n: int, isl: int, osl: int, skip_prefill: bool):
    config_path = TOPOLOGIES[topo_name]
    config_full = os.path.join(REPO_ROOT, config_path)
    if not os.path.exists(config_full):
        print(f"[FAIL] Config not found: {config_full}")
        sys.exit(1)

    wl_rel  = make_workload(n, isl, osl)
    tag     = f"test_mini_{topo_name}_n{n}_isl{isl}_osl{osl}{'_skip' if skip_prefill else ''}"
    out_csv = os.path.join(REPO_ROOT, "outputs", f"{tag}.csv")
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
        "--num-req", str(n),
        "--log-level", "INFO",
    ]
    if skip_prefill:
        cmd.append("--skip-prefill")

    mode = "--skip-prefill (decode only)" if skip_prefill else "prefill + decode"
    print(f"Topology     : {topo_name}  ({config_path})")
    print(f"Workload     : ISL={isl}, OSL={osl}, N={n}  mode={mode}")
    print(f"Decode steps : {osl} × {n} requests")
    print(f"Command      : {' '.join(cmd)}")
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
    ap = argparse.ArgumentParser(
        description="Minimal serving sanity test. Default: nvl72_ep8, ISL=10, OSL=5, N=1."
    )
    ap.add_argument("--topo", choices=list(TOPOLOGIES.keys()), default="nvl72_ep8",
                    help="Topology to test (default: nvl72_ep8)")
    ap.add_argument("--num-req", type=int, default=1,
                    help="Number of requests (default: 1)")
    ap.add_argument("--isl", type=int, default=10,
                    help="Input sequence length (default: 10)")
    ap.add_argument("--osl", type=int, default=5,
                    help="Output sequence length (default: 5)")
    ap.add_argument("--skip-prefill", action="store_true",
                    help="Pass --skip-prefill to the simulator (matches sweep_paper.py)")
    args = ap.parse_args()
    run(args.topo, args.num_req, args.isl, args.osl, args.skip_prefill)


if __name__ == "__main__":
    main()
