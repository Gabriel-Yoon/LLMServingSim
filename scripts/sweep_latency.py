"""
Latency DSE: intra-panel and inter-panel link latency sensitivity sweep.

This sweep answers two questions for the glass-panel FB topology paper:

  Sweep A — Intra-panel latency sensitivity (what optical link quality is needed?)
    Fixes inter_lat=2000 ns and intra_bw=512 GB/s.
    Sweeps intra_lat: [100, 200, 300, 500, 750, 1000, 1500, 2000] ns.
    EP: [8, 16] (fits on single 4×4 panel) and EP=32 (crosses one panel boundary).
    NVL72 baseline at each EP for comparison.
    Key insight: EP ≤ 16 only sees intra_lat; EP=32 also pays inter_lat once.

  Sweep B — Inter-panel latency sensitivity (multi-panel scaling cost)
    Fixes intra_lat=300 ns (baseline good optical quality) and intra_bw=512 GB/s.
    Sweeps inter_lat: [500, 1000, 2000, 3000, 5000] ns.
    EP: [32, 64, 128] (2, 4, 8 panels respectively).
    NVL72 baseline at each EP.

Topology: 4×4 glass panel (fb_2d, panel_rows=4, panel_cols=4, panel_size=16).
Hardware:  H100, Qwen3-30B-A3B-Instruct-2507.
Output:    outputs/dse_latency_sweep.csv

Usage:
  python scripts/sweep_latency.py --dry-run        # print planned runs
  python scripts/sweep_latency.py                  # run both sweeps
  python scripts/sweep_latency.py --sweep A        # intra-panel sweep only
  python scripts/sweep_latency.py --sweep B        # inter-panel sweep only
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
# Sweep axes
# ─────────────────────────────────────────────────────────────
INTRA_LAT_VALUES = [100, 200, 300, 500, 750, 1000, 1500, 2000]  # ns
INTER_LAT_VALUES = [500, 1000, 2000, 3000, 5000]                # ns

SWEEP_A_EPS = [8, 16, 32]    # EP=8,16 single-panel; EP=32 crosses panel boundary
SWEEP_B_EPS = [32, 64, 128]  # 2, 4, 8 panels

# Fixed parameters during each sub-sweep
INTRA_BW_REF  = 512.0   # GB/s (= 4 WG × 128 GB/s)
INTER_BW_REF  = 512.0   # GB/s inter-panel fiber (same as sweep_128gpu.py)
INTRA_LAT_REF = 300.0   # ns  (baseline quality, fixed during sweep B)
INTER_LAT_REF = 2000.0  # ns  (fixed during sweep A)

PANEL_ROWS = 4
PANEL_COLS = 4
PANEL_SIZE = PANEL_ROWS * PANEL_COLS  # 16

# NVL72 parameters (from sweep_128gpu.py)
NVL72_ELEC_BW = 1800.0   # GB/s NVLink intra-rack
NVL72_IB_BW   = 50.0     # GB/s InfiniBand inter-rack
NVL72_LAT     = 1000.0   # ns

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
HARDWARE   = "H100"
NPU_MEM    = {"mem_size": 80, "mem_bw": 3350, "mem_latency": 0}
CPU_MEM    = {"mem_size": 1024, "mem_bw": 512, "mem_latency": 0}

WORKLOAD   = "workloads/example_trace.jsonl"
NUM_REQ    = 2
BLOCK_SIZE = 16
MAX_SEQS   = 128
MAX_TOKENS = 2048
LOG_LEVEL  = "WARNING"
REPO_ROOT  = "/app/LLMServingSim"
WORKLOAD_CACHE = os.path.join(REPO_ROOT, "astra-sim/inputs/workload")


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


def make_fb_config(ep, intra_lat, inter_lat, intra_bw=INTRA_BW_REF, inter_bw=INTER_BW_REF):
    """4×4 glass panel: fb_2d topology.
    Returns None if EP cannot be mapped onto this panel configuration.
    """
    if ep > PANEL_SIZE and ep % PANEL_SIZE != 0:
        return None
    return {
        "num_nodes": 1,
        "topology_config": {
            "type":       "fb_2d",
            "panel_rows": PANEL_ROWS,
            "panel_cols": PANEL_COLS,
            "intra_bw":   float(intra_bw),
            "intra_lat":  float(intra_lat),
            "inter_bw":   float(inter_bw),
            "inter_lat":  float(inter_lat),
        },
        "nodes": [_node(ep)],
    }


def make_nvl72_config(ep):
    """NVL72: hierarchical_fb with tile_size=64, panel_size=128.
    EP ≤ 64: NVLink only (1800 GB/s, 1000 ns).
    EP = 128: NVLink intra-rack + InfiniBand inter-rack.
    """
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


# ─────────────────────────────────────────────────────────────
# Run helpers
# ─────────────────────────────────────────────────────────────

def _cleanup_workload_cache():
    if not os.path.isdir(WORKLOAD_CACHE):
        return
    for entry in os.listdir(WORKLOAD_CACHE):
        path = os.path.join(WORKLOAD_CACHE, entry)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)


def run_sim(config, label, output_dir, dry_run=False, timeout=7200):
    os.makedirs(output_dir, exist_ok=True)
    cfg_path = os.path.join(output_dir, f"{label}.json")
    out_csv  = os.path.join(output_dir, f"{label}.csv")
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
        ep = config["nodes"][0]["num_instances"]
        print(f"  [dry] {label:65s}  ep={ep:3d}  "
              f"type={tc.get('type', '?'):16s}  "
              f"intra_lat={tc.get('intra_lat', tc.get('elec_latency', '?'))}")
        return {"label": label, "status": "dry-run"}

    t0 = time.time()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, cwd=REPO_ROOT)
        elapsed = time.time() - t0
        if res.returncode == 0:
            result = _parse_csv(out_csv, label, elapsed)
        else:
            result = {"label": label, "status": "error", "elapsed_s": elapsed,
                      "stderr": res.stderr[-400:]}
    except subprocess.TimeoutExpired:
        result = {"label": label, "status": "timeout", "elapsed_s": timeout}
    except Exception as e:
        result = {"label": label, "status": "exception", "error": str(e)}

    _cleanup_workload_cache()
    return result


def _parse_csv(csv_path, label, elapsed):
    if not os.path.exists(csv_path):
        return {"label": label, "status": "no_output", "elapsed_s": elapsed}
    ttfts, tpots, lats = [], [], []
    try:
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                ttfts.append(float(row["TTFT"]) / 1e6)
                tpots.append(float(row["TPOT"]) / 1e6)
                lats.append(float(row["latency"]) / 1e6)
    except Exception as e:
        return {"label": label, "status": "parse_error", "error": str(e), "elapsed_s": elapsed}
    n = len(ttfts)
    if n == 0:
        return {"label": label, "status": "empty_csv", "elapsed_s": elapsed}
    meta = _parse_label(label)
    return {
        "label": label, "status": "ok", "elapsed_s": elapsed, "n_reqs": n,
        "ttft_ms":     sum(ttfts) / n,
        "tpot_ms":     sum(tpots) / n,
        "lat_ms":      sum(lats)  / n,
        "ttft_p50_ms": sorted(ttfts)[n // 2],
        "tpot_p50_ms": sorted(tpots)[n // 2],
        **meta,
    }


def _parse_label(label):
    # nvl72_ep32
    m = re.match(r"nvl72_ep(\d+)$", label)
    if m:
        ep = int(m.group(1))
        return {"sweep": "nvl72", "topology": "nvl72", "ep": ep,
                "intra_lat": NVL72_LAT, "inter_lat": NVL72_LAT, "intra_bw": NVL72_ELEC_BW,
                "inter_bw": NVL72_IB_BW}
    # fb4x4_swA_ep16_ilat300
    m = re.match(r"fb4x4_swA_ep(\d+)_ilat(\d+)$", label)
    if m:
        return {"sweep": "A", "topology": "fb_4x4", "ep": int(m.group(1)),
                "intra_lat": int(m.group(2)), "inter_lat": INTER_LAT_REF,
                "intra_bw": INTRA_BW_REF, "inter_bw": INTER_BW_REF}
    # fb4x4_swB_ep32_xlat2000
    m = re.match(r"fb4x4_swB_ep(\d+)_xlat(\d+)$", label)
    if m:
        return {"sweep": "B", "topology": "fb_4x4", "ep": int(m.group(1)),
                "intra_lat": INTRA_LAT_REF, "inter_lat": int(m.group(2)),
                "intra_bw": INTRA_BW_REF, "inter_bw": INTER_BW_REF}
    return {}


# ─────────────────────────────────────────────────────────────
# Run list builder
# ─────────────────────────────────────────────────────────────

def build_run_list(sweep_a, sweep_b, intra_lat_values, inter_lat_values,
                   sweep_a_eps, sweep_b_eps):
    runs = []

    # NVL72 baselines — deduplicated across both sweeps
    all_eps = set()
    if sweep_a:
        all_eps |= set(sweep_a_eps)
    if sweep_b:
        all_eps |= set(sweep_b_eps)
    for ep in sorted(all_eps):
        runs.append((f"nvl72_ep{ep}", make_nvl72_config(ep)))

    # Sweep A: intra-panel latency sweep
    if sweep_a:
        for ep in sweep_a_eps:
            for ilat in intra_lat_values:
                cfg = make_fb_config(ep, intra_lat=ilat, inter_lat=INTER_LAT_REF)
                if cfg is not None:
                    runs.append((f"fb4x4_swA_ep{ep}_ilat{ilat}", cfg))

    # Sweep B: inter-panel latency sweep
    if sweep_b:
        for ep in sweep_b_eps:
            for xlat in inter_lat_values:
                cfg = make_fb_config(ep, intra_lat=INTRA_LAT_REF, inter_lat=xlat)
                if cfg is not None:
                    runs.append((f"fb4x4_swB_ep{ep}_xlat{xlat}", cfg))

    return runs


# ─────────────────────────────────────────────────────────────
# CSV writer
# ─────────────────────────────────────────────────────────────

FIELDS = ["label", "status", "sweep", "topology", "ep", "intra_lat", "inter_lat",
          "intra_bw", "inter_bw", "tpot_ms", "ttft_ms", "lat_ms",
          "tpot_p50_ms", "ttft_p50_ms", "n_reqs", "elapsed_s", "stderr", "error"]


def write_results(results, path):
    extra = sorted({k for r in results for k in r if k not in FIELDS})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + extra, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Latency DSE: intra- and inter-panel link latency sweep (4×4 FB panel)")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print planned runs without executing")
    parser.add_argument("--sweep",       choices=["A", "B", "AB"], default="AB",
                        help="A=intra-panel sweep, B=inter-panel sweep, AB=both (default)")
    parser.add_argument("--intra-lat",   type=int, nargs="+", default=INTRA_LAT_VALUES,
                        help="Intra-panel latency values in ns (Sweep A)")
    parser.add_argument("--inter-lat",   type=int, nargs="+", default=INTER_LAT_VALUES,
                        help="Inter-panel latency values in ns (Sweep B)")
    parser.add_argument("--ep-a",        type=int, nargs="+", default=SWEEP_A_EPS,
                        help="EP values for Sweep A")
    parser.add_argument("--ep-b",        type=int, nargs="+", default=SWEEP_B_EPS,
                        help="EP values for Sweep B")
    parser.add_argument("--output-dir",  default="outputs/sim_latency",
                        help="Directory for per-run JSON configs and output CSVs")
    parser.add_argument("--results-csv", default="results/exp_latency_fb4x4_intra_inter_sweep.csv",
                        help="Aggregated results CSV path")
    parser.add_argument("--timeout",     type=int, default=7200)
    args = parser.parse_args()

    sweep_a = args.sweep in ("A", "AB")
    sweep_b = args.sweep in ("B", "AB")

    runs = build_run_list(sweep_a, sweep_b, args.intra_lat, args.inter_lat,
                          args.ep_a, args.ep_b)

    all_eps = set()
    if sweep_a: all_eps |= set(args.ep_a)
    if sweep_b: all_eps |= set(args.ep_b)
    total_a = len(args.intra_lat) * len(args.ep_a) if sweep_a else 0
    total_b = len(args.inter_lat) * len(args.ep_b) if sweep_b else 0

    print(f"=== Latency DSE Sweep ===")
    print(f"Sweep A (intra_lat): {'on' if sweep_a else 'off'}"
          f"  EP={args.ep_a}  lat={args.intra_lat} ns  → {total_a} runs")
    print(f"Sweep B (inter_lat): {'on' if sweep_b else 'off'}"
          f"  EP={args.ep_b}  lat={args.inter_lat} ns  → {total_b} runs")
    print(f"NVL72 baselines    : EP={sorted(all_eps)}")
    print(f"Fixed params       : intra_bw={INTRA_BW_REF} GB/s  inter_bw={INTER_BW_REF} GB/s")
    print(f"Total runs         : {len(runs)}  |  dry-run: {args.dry_run}")
    print(f"Results            : {args.results_csv}")
    print()

    if args.dry_run:
        for label, cfg in runs:
            run_sim(cfg, label, args.output_dir, dry_run=True)
        print(f"\nDry-run complete: {len(runs)} planned runs.")
        return

    os.makedirs(os.path.dirname(args.results_csv) or ".", exist_ok=True)

    # Resume: skip already-completed runs
    done_labels = set()
    results = []
    if os.path.exists(args.results_csv):
        with open(args.results_csv) as f:
            for row in csv.DictReader(f):
                if row.get("status") in ("ok", "timeout", "error"):
                    done_labels.add(row["label"])
                    results.append(row)
        if done_labels:
            print(f"Resuming: {len(done_labels)} runs already done, skipping them.")

    for i, (label, cfg) in enumerate(runs):
        if label in done_labels:
            print(f"[{i+1:3d}/{len(runs)}] SKIP  {label}")
            continue
        r = run_sim(cfg, label, args.output_dir, dry_run=False, timeout=args.timeout)
        results.append(r)
        tc = r.get("tpot_ms")
        tf = r.get("ttft_ms")
        print(f"[{i+1:3d}/{len(runs)}] {label:65s}"
              f"  status={r.get('status', '?'):8s}"
              f"  TTFT={f'{tf:.2f}ms' if tf else '?':9s}"
              f"  TPOT={f'{tc:.2f}ms' if tc else '?':9s}"
              f"  ({r.get('elapsed_s', 0):.0f}s)")
        write_results(results, args.results_csv)

    write_results(results, args.results_csv)
    n_ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"\nDone. {n_ok}/{len(runs)} succeeded → {args.results_csv}")


if __name__ == "__main__":
    main()
