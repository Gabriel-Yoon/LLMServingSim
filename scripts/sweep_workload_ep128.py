"""
EP=128 Workload DSE: Chat / Coding / AgenticCoding × NVL72 / FB-4×4.

Follows the workload definition from:
  Hanjiang Wu et al. "How Far Can Disaggregation Go?", arXiv 2605.28302

  UseCase       Prefix    ISL    OSL   (OSL capped at 256 for TPOT measurement)
  ─────────────────────────────────────────────────────────────────────────────
  Chat           4 096    512    256
  Coding         2 048  4 096  1 024
  AgenticCoding 65 536    256    256   (paper uses 524k; we use 64k within H100
                                        KV-cache budget; OSL truncated to 256
                                        for practical simulation time while
                                        still measuring steady-state TPOT)

Topologies (both at EP=128):
  nvl72   — hierarchical_fb, tile=64, panel=128, NVLink 1800 GB/s / 1000 ns
  fb_4x4  — fb_2d, 4×4 panel (16 GPUs/panel, 8 panels), 512 GB/s / 300 ns intra,
             512 GB/s / 2000 ns inter

Results are written to results/dse_workload_ep128.csv (git-tracked).
Per-request CSVs land in outputs/sim_workload/<label>/ (gitignored).

Usage:
  # Dry-run: print planned runs without executing
  python scripts/sweep_workload_ep128.py --dry-run

  # Full sweep (NVL72 + FB, all workloads)
  python scripts/sweep_workload_ep128.py

  # Single topology
  python scripts/sweep_workload_ep128.py --topology nvl72
  python scripts/sweep_workload_ep128.py --topology fb

  # Single workload
  python scripts/sweep_workload_ep128.py --workload chat
  python scripts/sweep_workload_ep128.py --workload coding
  python scripts/sweep_workload_ep128.py --workload agentic

  # Override per-workload NUM_REQ
  python scripts/sweep_workload_ep128.py --num-req-chat 5 --num-req-coding 3 --num-req-agentic 2

SLURM example (per-workload separate jobs):
  sbatch scripts/slurm_workload_ep128.sh chat
  sbatch scripts/slurm_workload_ep128.sh coding
  sbatch scripts/slurm_workload_ep128.sh agentic
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time

# ─────────────────────────────────────────────────────────────
# Topology parameters
# ─────────────────────────────────────────────────────────────
EP = 128

# NVL72: NVLink intra-tile (64 GPUs) + InfiniBand inter-tile
NVL72_ELEC_BW       = 1800.0   # GB/s  NVLink
NVL72_IB_BW         = 50.0     # GB/s  InfiniBand (kicks in at EP > 64)
NVL72_LAT           = 1000.0   # ns    (same for all levels in our model)

# FB 4×4 glass panel
FB_PANEL_ROWS    = 4
FB_PANEL_COLS    = 4
FB_INTRA_BW      = 512.0   # GB/s  intra-panel optical
FB_INTRA_LAT     = 300.0   # ns    intra-panel optical (target quality)
FB_INTER_BW      = 512.0   # GB/s  inter-panel fiber
FB_INTER_LAT     = 2000.0  # ns    inter-panel (baseline)

# ─────────────────────────────────────────────────────────────
# Workload definitions
# ─────────────────────────────────────────────────────────────
WORKLOADS = {
    "chat":    {"file": "workloads/chat_fixed.jsonl",    "num_req": 3,
                "desc": "Chat (prefix=4k, ISL=512, OSL=256)"},
    "coding":  {"file": "workloads/coding_fixed.jsonl",  "num_req": 3,
                "desc": "Coding (prefix=2k, ISL=4096, OSL=1024)"},
    "agentic": {"file": "workloads/agentic_hpc.jsonl",   "num_req": 3,
                "desc": "AgenticCoding (prefix=64k, ISL=256, OSL=256)"},
}

# ─────────────────────────────────────────────────────────────
# Hardware / model
# ─────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
HARDWARE   = "H100"
NPU_MEM    = {"mem_size": 80, "mem_bw": 3350, "mem_latency": 0}
CPU_MEM    = {"mem_size": 1024, "mem_bw": 512, "mem_latency": 0}

BLOCK_SIZE  = 16
MAX_SEQS    = 128
MAX_TOKENS  = 2048
LOG_LEVEL   = "WARNING"

REPO_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV    = os.path.join(REPO_ROOT, "results", "dse_workload_ep128.csv")
OUTPUT_DIR     = os.path.join(REPO_ROOT, "outputs", "sim_workload")
WORKLOAD_CACHE = os.path.join(REPO_ROOT, "astra-sim", "inputs", "workload")

CSV_FIELDS = [
    "label", "status", "topology", "workload", "ep",
    "tpot_ms", "ttft_ms", "lat_ms",
    "tpot_p50_ms", "ttft_p50_ms",
    "n_reqs", "elapsed_s", "stderr", "error",
]


# ─────────────────────────────────────────────────────────────
# Config builders
# ─────────────────────────────────────────────────────────────
def _instance(ep):
    return {
        "model_name": MODEL_NAME, "hardware": HARDWARE,
        "npu_mem": NPU_MEM, "num_npus": 1, "tp_size": 1,
        "ep_size": ep, "dp_group": "A", "pd_type": None,
    }

def _node(ep):
    return {
        "num_instances": ep,
        "cpu_mem": CPU_MEM,
        "instances": [_instance(ep) for _ in range(ep)],
    }

def make_nvl72_config(ep=EP):
    return {
        "num_nodes": 1,
        "topology_config": {
            "type":              "hierarchical_fb",
            "panel_size":        128,
            "tile_size":         64,
            "elec_bw":           NVL72_ELEC_BW,
            "intra_opt_bw":      NVL72_IB_BW,
            "inter_bw":          0.0,
            "elec_latency":      NVL72_LAT,
            "intra_opt_latency": NVL72_LAT,
            "inter_latency":     NVL72_LAT,
        },
        "nodes": [_node(ep)],
    }

def make_fb_config(ep=EP):
    return {
        "num_nodes": 1,
        "topology_config": {
            "type":       "fb_2d",
            "panel_rows": FB_PANEL_ROWS,
            "panel_cols": FB_PANEL_COLS,
            "intra_bw":   FB_INTRA_BW,
            "intra_lat":  FB_INTRA_LAT,
            "inter_bw":   FB_INTER_BW,
            "inter_lat":  FB_INTER_LAT,
        },
        "nodes": [_node(ep)],
    }


# ─────────────────────────────────────────────────────────────
# Sweep helpers
# ─────────────────────────────────────────────────────────────
def _load_done(results_csv):
    done = set()
    if not os.path.exists(results_csv):
        return done
    with open(results_csv) as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok":
                done.add(row["label"])
    return done

def _append_row(results_csv, row):
    write_header = not os.path.exists(results_csv)
    with open(results_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)

def _cleanup_workload_cache():
    if not os.path.isdir(WORKLOAD_CACHE):
        return
    for entry in os.listdir(WORKLOAD_CACHE):
        path = os.path.join(WORKLOAD_CACHE, entry)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)

def _parse_results(out_csv):
    if not os.path.exists(out_csv):
        return None
    rows = []
    with open(out_csv) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return None
    def avg(field):
        vals = [float(r[field]) for r in rows if r.get(field)]
        return sum(vals) / len(vals) if vals else 0.0
    def med(field):
        vals = sorted(float(r[field]) for r in rows if r.get(field))
        return vals[len(vals)//2] if vals else 0.0
    return {
        "tpot_ms":     avg("TPOT") / 1e6,
        "ttft_ms":     avg("TTFT") / 1e6,
        "lat_ms":      avg("latency") / 1e6,
        "tpot_p50_ms": med("TPOT") / 1e6,
        "ttft_p50_ms": med("TTFT") / 1e6,
        "n_reqs":      len(rows),
    }

def run_sim(label, config, workload_file, num_req, run_dir,
            dry_run=False, timeout=172800):  # 48hr timeout
    os.makedirs(run_dir, exist_ok=True)
    cfg_path = os.path.join(run_dir, f"{label}.json")
    out_csv  = os.path.join(run_dir, f"{label}.csv")
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)

    cmd = [
        "python", "-m", "serving",
        "--cluster-config", cfg_path,
        "--dtype", "bfloat16",
        "--block-size", str(BLOCK_SIZE),
        "--max-num-seqs", str(MAX_SEQS),
        "--max-num-batched-tokens", str(MAX_TOKENS),
        "--dataset", workload_file,
        "--output", out_csv,
        "--num-req", str(num_req),
        "--log-level", LOG_LEVEL,
    ]

    if dry_run:
        tc = config.get("topology_config", {})
        topo = tc.get("type", "?")
        print(f"  [dry] {label:55s}  topo={topo:15s}  nr={num_req}")
        return {"status": "dry", "tpot_ms": 0, "ttft_ms": 0, "lat_ms": 0,
                "tpot_p50_ms": 0, "ttft_p50_ms": 0, "n_reqs": num_req,
                "elapsed_s": 0, "stderr": "", "error": ""}

    _cleanup_workload_cache()
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=REPO_ROOT, timeout=timeout,
        )
        elapsed = time.time() - t0
        stderr = proc.stderr.strip()[-500:] if proc.stderr else ""
        if proc.returncode != 0:
            return {"status": "error", "elapsed_s": elapsed,
                    "error": proc.stderr.strip()[-200:], "stderr": stderr,
                    "tpot_ms": 0, "ttft_ms": 0, "lat_ms": 0,
                    "tpot_p50_ms": 0, "ttft_p50_ms": 0, "n_reqs": 0}
        metrics = _parse_results(out_csv) or {}
        return {"status": "ok", "elapsed_s": elapsed,
                "stderr": stderr, "error": "", **metrics}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "elapsed_s": timeout,
                "error": f"exceeded {timeout}s", "stderr": "",
                "tpot_ms": 0, "ttft_ms": 0, "lat_ms": 0,
                "tpot_p50_ms": 0, "ttft_p50_ms": 0, "n_reqs": 0}


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print planned runs without executing.")
    ap.add_argument("--topology", choices=["nvl72", "fb", "both"], default="both",
                    help="Which topology to run. Default: both.")
    ap.add_argument("--workload", choices=["chat", "coding", "agentic", "all"],
                    default="all", help="Which workload to run. Default: all.")
    ap.add_argument("--num-req-chat",    type=int, default=None, dest="nr_chat")
    ap.add_argument("--num-req-coding",  type=int, default=None, dest="nr_coding")
    ap.add_argument("--num-req-agentic", type=int, default=None, dest="nr_agentic")
    ap.add_argument("--results-csv", default=RESULTS_CSV, dest="results_csv",
                    help=f"Output summary CSV. Default: {RESULTS_CSV}")
    ap.add_argument("--output-dir", default=OUTPUT_DIR, dest="output_dir")
    args = ap.parse_args()

    # Override num_req if specified
    if args.nr_chat    is not None: WORKLOADS["chat"]["num_req"]    = args.nr_chat
    if args.nr_coding  is not None: WORKLOADS["coding"]["num_req"]  = args.nr_coding
    if args.nr_agentic is not None: WORKLOADS["agentic"]["num_req"] = args.nr_agentic

    # Build run list
    topo_filter    = {"both": ["nvl72", "fb"], "nvl72": ["nvl72"], "fb": ["fb"]}[args.topology]
    wl_filter      = list(WORKLOADS.keys()) if args.workload == "all" else [args.workload]

    runs = []
    for wl in wl_filter:
        for topo in topo_filter:
            label = f"{topo}_ep{EP}_{wl}"
            cfg   = make_nvl72_config() if topo == "nvl72" else make_fb_config()
            runs.append((label, topo, wl, cfg))

    done = _load_done(args.results_csv)

    print(f"=== EP={EP} Workload Sweep ===")
    print(f"Topologies : {topo_filter}")
    print(f"Workloads  : {wl_filter}")
    print(f"Total runs : {len(runs)}  |  already done: {len(done & {r[0] for r in runs})}")
    print(f"Results    : {args.results_csv}\n")

    for label, topo, wl, cfg in runs:
        wl_cfg = WORKLOADS[wl]
        if label in done:
            print(f"[SKIP] {label}")
            continue

        nr = wl_cfg["num_req"]
        run_dir = os.path.join(args.output_dir, label)
        print(f"[RUN ] {label:45s}  nr={nr}  {wl_cfg['desc']}", flush=True)

        result = run_sim(
            label=label,
            config=cfg,
            workload_file=os.path.join(REPO_ROOT, wl_cfg["file"]),
            num_req=nr,
            run_dir=run_dir,
            dry_run=args.dry_run,
        )

        row = {"label": label, "topology": topo, "workload": wl, "ep": EP, **result}
        if not args.dry_run:
            _append_row(args.results_csv, row)

        status = result.get("status", "?")
        tpot   = result.get("tpot_ms", 0)
        ttft   = result.get("ttft_ms", 0)
        elapsed = result.get("elapsed_s", 0)
        print(f"         → status={status}  TTFT={ttft:.2f}ms  TPOT={tpot:.2f}ms  ({elapsed:.0f}s)")

    if not args.dry_run:
        print(f"\nDone → {args.results_csv}")


if __name__ == "__main__":
    main()
