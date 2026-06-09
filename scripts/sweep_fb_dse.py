"""
Flattened Butterfly Glass Panel DSE Sweep.

Panel designs:
  4x4    : panel_size=16, tile_size=4  (16 GPUs per panel)
  6x6-4c : panel_size=32, tile_size=4  (6x6 minus 4 corner GPUs = 32 GPUs per panel)

Sweep axes:
  EP          : [16, 32, 64, 128]
  elec_bw     : electrical BW within tile (GB/s)
  intra_opt_bw: optical BW within panel between tiles (GB/s)
  inter_bw    : inter-panel optical I/O BW (GB/s)

Baselines:
  flat_900  : H100 NVLink flat FC 900 GB/s
  flat_1800 : 2× H100 NVLink 1800 GB/s

Usage (inside Docker, from /app/LLMServingSim):
  python scripts/sweep_fb_dse.py [--dry-run] [--parallel N] [--ep 64 128]
  python scripts/sweep_fb_dse.py --quick          # fast representative subset
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

# ─────────────────────────────────────────────────────────────
# Fixed hardware parameters
# ─────────────────────────────────────────────────────────────
MODEL_NAME   = "Qwen/Qwen3-30B-A3B-Instruct-2507"
HARDWARE     = "H100"
NPU_MEM_SIZE = 80
NPU_MEM_BW   = 3350
CPU_MEM_SIZE = 1024
CPU_MEM_BW   = 512

# ─────────────────────────────────────────────────────────────
# Sweep parameters  (can be overridden via CLI)
# ─────────────────────────────────────────────────────────────
EP_SIZES = [16, 32, 64, 128]

# Electrical BW (neighboring GPUs within a tile)
ELEC_BWS = [900, 1800, 3600]          # GB/s

# Optical BW within panel (tile-to-tile on same glass panel)
INTRA_OPT_BWS = [200, 400, 800]       # GB/s

# Inter-panel optical I/O
INTER_BWS = [50, 100, 200, 400]       # GB/s

# Latency estimates (ns) — kept fixed; sweep BW only
ELEC_LAT      = 100.0   # ~electrical trace on glass
INTRA_OPT_LAT = 300.0   # ~photonic waveguide within panel
INTER_LAT     = 2000.0  # ~inter-panel optical I/O

# Panel topologies to evaluate
PANEL_CONFIGS = {
    "4x4":    {"panel_size": 16, "tile_size": 4},  # 4×4 grid, 4 tiles of 4
    "6x6_4c": {"panel_size": 32, "tile_size": 4},  # 6×6 minus 4 corners = 32 GPUs, 8 tiles of 4
}

# Flat baselines
FLAT_BASELINES = [
    {"name": "flat_900",  "link_bw": 900,  "link_latency": 0},
    {"name": "flat_1800", "link_bw": 1800, "link_latency": 0},
]

# Simulation settings
WORKLOAD   = "workloads/example_trace.jsonl"
NUM_REQ    = 2
BLOCK_SIZE = 16
MAX_SEQS   = 128
MAX_TOKENS = 2048
LOG_LEVEL  = "WARNING"

# ─────────────────────────────────────────────────────────────
# Config generation helpers
# ─────────────────────────────────────────────────────────────

def _instance(ep_size):
    return {
        "model_name": MODEL_NAME, "hardware": HARDWARE,
        "npu_mem": {"mem_size": NPU_MEM_SIZE, "mem_bw": NPU_MEM_BW, "mem_latency": 0},
        "num_npus": 1, "tp_size": 1,
        "ep_size": ep_size, "dp_group": "A", "pd_type": None,
    }


def make_fb_config(ep, panel_cfg, elec_bw, intra_opt_bw, inter_bw):
    panel_size = panel_cfg["panel_size"]
    tile_size  = panel_cfg["tile_size"]
    # EP must be divisible by tile_size; if EP > panel_size, also by panel_size
    if ep % tile_size != 0:
        return None
    if ep > panel_size and ep % panel_size != 0:
        return None
    instances = [_instance(ep) for _ in range(ep)]
    return {
        "num_nodes": 1,
        "topology_config": {
            "type":            "hierarchical_fb",
            "panel_size":      panel_size,
            "tile_size":       tile_size,
            "elec_bw":         float(elec_bw),
            "intra_opt_bw":    float(intra_opt_bw),
            "inter_bw":        float(inter_bw),
            "elec_latency":    ELEC_LAT,
            "intra_opt_latency": INTRA_OPT_LAT,
            "inter_latency":   INTER_LAT,
        },
        "nodes": [{
            "num_instances": ep,
            "cpu_mem": {"mem_size": CPU_MEM_SIZE, "mem_bw": CPU_MEM_BW, "mem_latency": 0},
            "instances": instances,
        }],
    }


def make_flat_config(ep, link_bw, link_latency):
    instances = [_instance(ep) for _ in range(ep)]
    return {
        "num_nodes": 1,
        "link_bw": float(link_bw), "link_latency": float(link_latency),
        "nodes": [{
            "num_instances": ep,
            "cpu_mem": {"mem_size": CPU_MEM_SIZE, "mem_bw": CPU_MEM_BW, "mem_latency": 0},
            "instances": instances,
        }],
    }


# ─────────────────────────────────────────────────────────────
# Run a single simulation
# ─────────────────────────────────────────────────────────────

def run_sim(config_dict, label, output_dir, dry_run=False, timeout=3600):
    os.makedirs(output_dir, exist_ok=True)
    cfg_path = os.path.join(output_dir, f"{label}.json")
    out_csv  = os.path.join(output_dir, f"{label}.csv")

    with open(cfg_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    cmd = [
        "python", "-m", "serving",
        "--cluster-config", cfg_path,
        "--dtype",         "bfloat16",
        "--block-size",    str(BLOCK_SIZE),
        "--max-num-seqs",  str(MAX_SEQS),
        "--max-num-batched-tokens", str(MAX_TOKENS),
        "--dataset",       WORKLOAD,
        "--output",        out_csv,
        "--num-req",       str(NUM_REQ),
        "--log-level",     LOG_LEVEL,
    ]

    if dry_run:
        tc = config_dict.get("topology_config", {})
        print(f"[dry-run] {label}  tc={tc.get('type','flat')} "
              f"ep={config_dict['nodes'][0]['num_instances']} "
              f"elec={tc.get('elec_bw','?')} in={tc.get('intra_opt_bw','?')} "
              f"ex={tc.get('inter_bw','?')}")
        return {"label": label, "status": "dry-run"}

    t0 = time.time()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, cwd="/app/LLMServingSim")
        elapsed = time.time() - t0
        if res.returncode != 0:
            return {"label": label, "status": "error", "elapsed_s": elapsed,
                    "stderr": res.stderr[-400:]}
        return _parse_csv(out_csv, label, elapsed)
    except subprocess.TimeoutExpired:
        return {"label": label, "status": "timeout", "elapsed_s": timeout}
    except Exception as e:
        return {"label": label, "status": "exception", "error": str(e)}


def _parse_csv(csv_path, label, elapsed):
    if not os.path.exists(csv_path):
        return {"label": label, "status": "no_output", "elapsed_s": elapsed}
    rows, ttfts, tpots, lats = [], [], [], []
    try:
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                try:
                    ttfts.append(float(row.get("TTFT", 0)) / 1e6)
                    tpots.append(float(row.get("TPOT", 0)) / 1e6)
                    lats.append(float(row.get("latency", 0)) / 1e6)
                    rows.append(row)
                except (ValueError, KeyError):
                    pass
    except Exception as e:
        return {"label": label, "status": "parse_error", "error": str(e)}

    n = len(ttfts)
    if n == 0:
        return {"label": label, "status": "empty_csv", "elapsed_s": elapsed}

    # Parse label to extract axes
    meta = _parse_label(label)
    return {
        "label":       label,
        "status":      "ok",
        "elapsed_s":   elapsed,
        "n_reqs":      n,
        "ttft_ms":     sum(ttfts) / n,
        "tpot_ms":     sum(tpots) / n,
        "lat_ms":      sum(lats) / n,
        "ttft_p50_ms": sorted(ttfts)[n // 2],
        "tpot_p50_ms": sorted(tpots)[n // 2],
        **meta,
    }


def _parse_label(label):
    """Extract sweep axes from label string."""
    # fb_4x4_ep64_el1800_in400_ex200
    m = re.match(r"fb_(\w+)_ep(\d+)_el(\d+)_in(\d+)_ex(\d+)", label)
    if m:
        return {"topology": "fb", "panel": m.group(1), "ep": int(m.group(2)),
                "elec_bw": int(m.group(3)), "intra_opt_bw": int(m.group(4)),
                "inter_bw": int(m.group(5))}
    # flat_900_ep64
    m = re.match(r"flat_(\d+)_ep(\d+)", label)
    if m:
        return {"topology": "flat", "panel": "flat", "ep": int(m.group(2)),
                "elec_bw": int(m.group(1)), "intra_opt_bw": int(m.group(1)),
                "inter_bw": int(m.group(1))}
    return {}


# ─────────────────────────────────────────────────────────────
# Build run list
# ─────────────────────────────────────────────────────────────

def build_run_list(ep_sizes, elec_bws, intra_opt_bws, inter_bws,
                   panel_configs, flat_baselines, quick=False):
    runs = []

    # Quick mode: single representative point per panel × EP
    if quick:
        for panel_name, panel_cfg in panel_configs.items():
            for ep in ep_sizes:
                cfg = make_fb_config(ep, panel_cfg, 1800, 400, 200)
                if cfg:
                    label = f"fb_{panel_name}_ep{ep}_el1800_in400_ex200"
                    runs.append((label, cfg))
        for ep in ep_sizes:
            for fb in flat_baselines:
                label = f"{fb['name']}_ep{ep}"
                runs.append((label, make_flat_config(ep, fb["link_bw"], fb["link_latency"])))
        return runs

    # Full sweep
    for panel_name, panel_cfg in panel_configs.items():
        for ep, eb, ib, xb in product(ep_sizes, elec_bws, intra_opt_bws, inter_bws):
            cfg = make_fb_config(ep, panel_cfg, eb, ib, xb)
            if cfg is None:
                continue
            label = f"fb_{panel_name}_ep{ep}_el{eb}_in{ib}_ex{xb}"
            runs.append((label, cfg))

    for ep in ep_sizes:
        for fb in flat_baselines:
            label = f"{fb['name']}_ep{ep}"
            runs.append((label, make_flat_config(ep, fb["link_bw"], fb["link_latency"])))

    return runs


# ─────────────────────────────────────────────────────────────
# CSV writer
# ─────────────────────────────────────────────────────────────

def _write_results(results, path):
    if not results:
        return
    fieldnames = ["label", "status", "topology", "panel", "ep",
                  "elec_bw", "intra_opt_bw", "inter_bw",
                  "ttft_ms", "tpot_ms", "lat_ms",
                  "ttft_p50_ms", "tpot_p50_ms",
                  "n_reqs", "elapsed_s", "stderr", "error"]
    # include any extra keys
    extra = sorted({k for r in results for k in r if k not in fieldnames})
    fieldnames += extra
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FB DSE sweep — 4x4 / 6x6-4corners panels")
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--quick",      action="store_true", help="Run one representative point per config")
    parser.add_argument("--parallel",   type=int,   default=1)
    parser.add_argument("--output-dir", default="outputs/dse")
    parser.add_argument("--results-csv",default="outputs/dse_results.csv")
    parser.add_argument("--timeout",    type=int,   default=3600)
    parser.add_argument("--ep",         type=int,   nargs="+")
    parser.add_argument("--elec-bw",    type=float, nargs="+")
    parser.add_argument("--intra-bw",   type=float, nargs="+")
    parser.add_argument("--inter-bw",   type=float, nargs="+")
    parser.add_argument("--panel",      type=str,   nargs="+",
                        choices=list(PANEL_CONFIGS.keys()),
                        help="Panel configs to run (default: all)")
    args = parser.parse_args()

    ep_sizes     = args.ep       or EP_SIZES
    elec_bws     = args.elec_bw  or ELEC_BWS
    intra_bws    = args.intra_bw or INTRA_OPT_BWS
    inter_bws    = args.inter_bw or INTER_BWS
    panel_cfgs   = {k: v for k, v in PANEL_CONFIGS.items()
                    if args.panel is None or k in args.panel}

    runs = build_run_list(ep_sizes, elec_bws, intra_bws, inter_bws,
                          panel_cfgs, FLAT_BASELINES, quick=args.quick)

    print(f"Total runs: {len(runs)} | parallel: {args.parallel} | "
          f"dry-run: {args.dry_run} | quick: {args.quick}")

    os.makedirs(os.path.dirname(args.results_csv) or ".", exist_ok=True)
    results = []

    def _on_done(r, done, total):
        tc = r.get('tpot_ms')
        tf = r.get('ttft_ms')
        print(f"[{done:3d}/{total}] {r['label']:55s} "
              f"status={r.get('status','?'):8s} "
              f"TTFT={f'{tf:.2f}ms' if tf else '?':9s} "
              f"TPOT={f'{tc:.2f}ms' if tc else '?':9s} "
              f"({r.get('elapsed_s',0):.0f}s)")

    if args.parallel > 1 and not args.dry_run:
        with ProcessPoolExecutor(max_workers=args.parallel) as pool:
            futures = {
                pool.submit(run_sim, cfg, label, args.output_dir,
                            args.dry_run, args.timeout): label
                for label, cfg in runs
            }
            done = 0
            for fut in as_completed(futures):
                done += 1
                r = fut.result()
                results.append(r)
                _on_done(r, done, len(runs))
                _write_results(results, args.results_csv)
    else:
        for i, (label, cfg) in enumerate(runs):
            r = run_sim(cfg, label, args.output_dir, args.dry_run, args.timeout)
            results.append(r)
            _on_done(r, i + 1, len(runs))
            _write_results(results, args.results_csv)

    _write_results(results, args.results_csv)
    n_ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"\nDone. {n_ok}/{len(runs)} succeeded → {args.results_csv}")


if __name__ == "__main__":
    main()
