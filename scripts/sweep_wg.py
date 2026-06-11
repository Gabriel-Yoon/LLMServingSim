"""
Waveguide-group count sweep for 6×6-4c glass panel topology.

WG spec: 1 WG = 32λ × 32 Gb/s = 1.024 Tb/s = 128 GB/s
WG counts: [1, 2, 3, 4, 6, 8, 12] → intra_opt_bw: [128, 256, 384, 512, 768, 1024, 1536] GB/s

Three topology approximations tested:
  4x8  : tile_size=4 → [4,4] at EP=16, [4,8] at EP=32
  2x16 : tile_size=2 → [2,8] at EP=16, [2,16] at EP=32
  flat : 1D ring at intra_opt_bw (no hierarchy)

Baselines:
  nvl72: flat 1800 GB/s (H100 NVLink)

Results saved to outputs/dse_wg_sweep.csv after each run.

Usage (inside Docker, from /app/LLMServingSim):
  python scripts/sweep_wg.py [--dry-run] [--ep 32] [--topo 4x8 2x16 flat]
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time

# ─────────────────────────────────────────────────────────────
# Sweep axes
# ─────────────────────────────────────────────────────────────
WG_COUNTS = [1, 2, 3, 4, 6, 8, 12]
WG_PER_BW = 128  # GB/s per WG
INTRA_OPT_BWS = [wg * WG_PER_BW for wg in WG_COUNTS]  # [128..1536] GB/s

EP_SIZES = [16, 32]

ELEC_BW = 1800.0      # GB/s (electrical within tile, fixed)
ELEC_LAT = 100.0      # ns
INTRA_OPT_LAT = 300.0 # ns
INTER_BW = 0.0        # EP≤32 fits in 1 panel → no inter-panel traffic
INTER_LAT = 2000.0    # ns (unused for EP≤panel_size)

PANEL_SIZE = 32       # 6×6-4c: 32 GPUs per panel
TILE_SIZES = {
    "4x8":  4,   # [4,4] at EP=16, [4,8] at EP=32
    "2x16": 2,   # [2,8] at EP=16, [2,16] at EP=32
}

NVL72_LINK_BW = 1800.0

# Fixed hardware
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
HARDWARE = "H100"
NPU_MEM = {"mem_size": 80, "mem_bw": 3350, "mem_latency": 0}
CPU_MEM = {"mem_size": 1024, "mem_bw": 512, "mem_latency": 0}

WORKLOAD = "workloads/example_trace.jsonl"
NUM_REQ = 2
BLOCK_SIZE = 16
MAX_SEQS = 128
MAX_TOKENS = 2048
LOG_LEVEL = "WARNING"


# ─────────────────────────────────────────────────────────────
# Config builders
# ─────────────────────────────────────────────────────────────
def _instance(ep_size):
    return {
        "model_name": MODEL_NAME, "hardware": HARDWARE,
        "npu_mem": NPU_MEM,
        "num_npus": 1, "tp_size": 1,
        "ep_size": ep_size, "dp_group": "A", "pd_type": None,
    }


def make_fb_config(ep, tile_size, intra_opt_bw):
    """Hierarchical FB config for 6×6-4c panel (panel_size=32)."""
    if ep % tile_size != 0:
        return None
    # EP fits in one panel (32 GPUs), no inter-panel needed
    if ep > PANEL_SIZE:
        return None
    instances = [_instance(ep) for _ in range(ep)]
    return {
        "num_nodes": 1,
        "topology_config": {
            "type": "hierarchical_fb",
            "panel_size": PANEL_SIZE,
            "tile_size": tile_size,
            "elec_bw": ELEC_BW,
            "intra_opt_bw": float(intra_opt_bw),
            "inter_bw": INTER_BW,
            "elec_latency": ELEC_LAT,
            "intra_opt_latency": INTRA_OPT_LAT,
            "inter_latency": INTER_LAT,
        },
        "nodes": [{
            "num_instances": ep,
            "cpu_mem": CPU_MEM,
            "instances": instances,
        }],
    }


def make_flat_config(ep, link_bw, link_latency=0.0):
    """Flat FC config (1D ring) at given BW."""
    instances = [_instance(ep) for _ in range(ep)]
    return {
        "num_nodes": 1,
        "link_bw": float(link_bw),
        "link_latency": float(link_latency),
        "nodes": [{
            "num_instances": ep,
            "cpu_mem": CPU_MEM,
            "instances": instances,
        }],
    }


# ─────────────────────────────────────────────────────────────
# Run simulation
# ─────────────────────────────────────────────────────────────
def run_sim(config, label, output_dir, dry_run=False, timeout=3600):
    os.makedirs(output_dir, exist_ok=True)
    cfg_path = os.path.join(output_dir, f"{label}.json")
    out_csv = os.path.join(output_dir, f"{label}.csv")

    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)

    cmd = [
        "python", "-m", "serving",
        "--cluster-config", cfg_path,
        "--dtype", "bfloat16",
        "--block-size", str(BLOCK_SIZE),
        "--max-num-seqs", str(MAX_SEQS),
        "--max-num-batched-tokens", str(MAX_TOKENS),
        "--dataset", WORKLOAD,
        "--output", out_csv,
        "--num-req", str(NUM_REQ),
        "--log-level", LOG_LEVEL,
    ]

    if dry_run:
        tc = config.get("topology_config", {})
        print(f"[dry] {label:55s}  type={tc.get('type','flat'):15s}"
              f"  ep={config['nodes'][0]['num_instances']:3d}"
              f"  in_opt={tc.get('intra_opt_bw', config.get('link_bw','?'))}")
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
    ttfts, tpots, lats = [], [], []
    try:
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                try:
                    ttfts.append(float(row["TTFT"]) / 1e6)
                    tpots.append(float(row["TPOT"]) / 1e6)
                    lats.append(float(row["latency"]) / 1e6)
                except (ValueError, KeyError):
                    pass
    except Exception as e:
        return {"label": label, "status": "parse_error", "error": str(e), "elapsed_s": elapsed}

    n = len(ttfts)
    if n == 0:
        return {"label": label, "status": "empty_csv", "elapsed_s": elapsed}

    meta = _parse_label(label)
    return {
        "label": label, "status": "ok", "elapsed_s": elapsed, "n_reqs": n,
        "ttft_ms": sum(ttfts) / n,
        "tpot_ms": sum(tpots) / n,
        "lat_ms": sum(lats) / n,
        "ttft_p50_ms": sorted(ttfts)[n // 2],
        "tpot_p50_ms": sorted(tpots)[n // 2],
        **meta,
    }


def _parse_label(label):
    # wg_4x8_ep32_wg4_bw512
    m = re.match(r"wg_(\w+)_ep(\d+)_wg(\d+)_bw(\d+)", label)
    if m:
        return {"topology": m.group(1), "ep": int(m.group(2)),
                "wg_count": int(m.group(3)), "intra_opt_bw": int(m.group(4))}
    # wg_flat_ep32_wg4_bw512 (flat at intra_opt_bw)
    m = re.match(r"wg_flat_ep(\d+)_wg(\d+)_bw(\d+)", label)
    if m:
        return {"topology": "flat", "ep": int(m.group(1)),
                "wg_count": int(m.group(2)), "intra_opt_bw": int(m.group(3))}
    # nvl72_ep32
    m = re.match(r"nvl72_ep(\d+)", label)
    if m:
        return {"topology": "nvl72", "ep": int(m.group(1)), "wg_count": 14,
                "intra_opt_bw": NVL72_LINK_BW}  # 1800/128 ≈ 14 WGs equiv
    return {}


# ─────────────────────────────────────────────────────────────
# Build run list
# ─────────────────────────────────────────────────────────────
def build_run_list(ep_sizes, wg_counts, topo_names):
    runs = []

    # NVL72 baselines
    for ep in ep_sizes:
        label = f"nvl72_ep{ep}"
        runs.append((label, make_flat_config(ep, NVL72_LINK_BW, link_latency=0.0)))

    # FB approximations sweep
    for ep in ep_sizes:
        for wg in wg_counts:
            bw = wg * WG_PER_BW

            if "4x8" in topo_names:
                tile = TILE_SIZES["4x8"]
                cfg = make_fb_config(ep, tile, bw)
                if cfg:
                    runs.append((f"wg_4x8_ep{ep}_wg{wg}_bw{bw}", cfg))

            if "2x16" in topo_names:
                tile = TILE_SIZES["2x16"]
                cfg = make_fb_config(ep, tile, bw)
                if cfg:
                    runs.append((f"wg_2x16_ep{ep}_wg{wg}_bw{bw}", cfg))

            if "flat" in topo_names:
                cfg = make_flat_config(ep, bw, link_latency=INTRA_OPT_LAT)
                runs.append((f"wg_flat_ep{ep}_wg{wg}_bw{bw}", cfg))

    return runs


# ─────────────────────────────────────────────────────────────
# CSV writer
# ─────────────────────────────────────────────────────────────
FIELDNAMES = ["label", "status", "topology", "ep", "wg_count", "intra_opt_bw",
              "ttft_ms", "tpot_ms", "lat_ms", "ttft_p50_ms", "tpot_p50_ms",
              "n_reqs", "elapsed_s", "stderr", "error"]


def write_results(results, path):
    if not results:
        return
    extra = sorted({k for r in results for k in r if k not in FIELDNAMES})
    fields = FIELDNAMES + extra
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="WG count sweep — 6×6-4c panel")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ep", type=int, nargs="+", default=EP_SIZES)
    parser.add_argument("--wg", type=int, nargs="+", default=WG_COUNTS,
                        help="WG counts to sweep (default: 1 2 3 4 6 8 12)")
    parser.add_argument("--topo", type=str, nargs="+", default=["4x8", "2x16", "flat"],
                        choices=["4x8", "2x16", "flat"])
    parser.add_argument("--output-dir", default="outputs/wg_runs")
    parser.add_argument("--results-csv", default="outputs/dse_wg_sweep.csv")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    runs = build_run_list(args.ep, args.wg, args.topo)
    print(f"Total runs: {len(runs)} | dry-run: {args.dry_run}")
    print(f"WG counts: {args.wg}  → BW: {[w*WG_PER_BW for w in args.wg]} GB/s")
    print(f"Topologies: {args.topo}")

    os.makedirs(os.path.dirname(args.results_csv) or ".", exist_ok=True)
    results = []

    for i, (label, cfg) in enumerate(runs):
        r = run_sim(cfg, label, args.output_dir, args.dry_run, args.timeout)
        results.append(r)
        tc = r.get("tpot_ms"); tf = r.get("ttft_ms")
        print(f"[{i+1:3d}/{len(runs)}] {label:55s}  status={r.get('status','?'):8s}"
              f"  TTFT={f'{tf:.2f}ms' if tf else '?':9s}"
              f"  TPOT={f'{tc:.2f}ms' if tc else '?':9s}"
              f"  ({r.get('elapsed_s',0):.0f}s)")
        write_results(results, args.results_csv)

    write_results(results, args.results_csv)
    n_ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"\nDone. {n_ok}/{len(runs)} succeeded → {args.results_csv}")


if __name__ == "__main__":
    main()
