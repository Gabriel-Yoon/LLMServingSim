"""
128-GPU EP sweep: NVL72 vs Glass FB 6×6-4c vs Glass FB 4×4.

System configurations (CORRECTED model)
─────────────────────────────────────────
[NVL72]  Modeled as hierarchical_fb with tile_size=64, panel_size=128
  EP ≤ 64  → dims [EP],     BW=1800 GB/s flat (one NVLink rack)
  EP=128   → dims [64,2],   dim0=1800 GB/s (NVLink) / dim1=50 GB/s (IB)
  lat: 1000 ns at all dims

[6×6-4c] fb_2d with panel_rows=4, panel_cols=8 (32 nodes per panel)
  Physical: 6×6 grid with 4 corner GPUs inactive = 32 active GPUs.
  ASTRA-Sim rectangular approx: [4×8]=32 nodes.
  Both dims use uniform optical BW = N_WG×128 GB/s, lat=300 ns.
  (Conservative: ASTRA-Sim single BW per dim; electrical adjacent connections
   are NOT modeled separately — optical BW is the conservative bound.)
  Multi-panel: dim2 = inter-panel fiber, 512 GB/s, 5000 ns.
  EP values: 4(→[4]), 8(→[2,4]), 16(→[4,4]), 32(→[4,8]),
             64(→[4,8,2]), 128(→[4,8,4])
  Note: [6,6]=36 is incompatible — 36 does not divide 128 model experts.

[4×4]    fb_2d with panel_rows=4, panel_cols=4 (16 nodes per panel)
  Physical: 4×4 grid, all 16 GPUs active.
  Both dims use uniform optical BW = N_WG×128 GB/s, lat=300 ns.
  Multi-panel: dim2 = inter-panel fiber, 512 GB/s, 5000 ns.
  EP values: 4(→[4]), 8(→[2,4]), 16(→[4,4]), 32(→[4,4,2]),
             64(→[4,4,4]), 128(→[4,4,8])

Sweep parameters
────────────────
  EP:   [4, 8, 16, 32, 64, 128]
  N_WG: [2, 4, 6, 8]  →  intra_opt_bw: [256, 512, 768, 1024] GB/s

Output: outputs/dse_128gpu_4x4_vs_6x6.csv
"""

import argparse
import csv
import json
import os
import re
import subprocess
import time

# ─────────────────────────────────────────────────────────────
# Sweep axes
# ─────────────────────────────────────────────────────────────
EP_SIZES = [4, 8, 16, 32, 64, 128]
WG_COUNTS = [2, 4, 6, 8]
WG_BW = 128                        # GB/s per WG

# Physical constants
INTRA_OPT_LAT = 300.0    # ns   intra-panel optical
INTER_BW      = 512.0    # GB/s inter-panel fiber
INTER_LAT     = 5000.0   # ns

NVL72_ELEC_BW = 1800.0   # GB/s NVLink (intra-rack)
NVL72_IB_BW   = 50.0     # GB/s InfiniBand (inter-rack)
NVL72_LAT     = 1000.0   # ns

# Hardware
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


# ─────────────────────────────────────────────────────────────
# EP validation helpers
# ─────────────────────────────────────────────────────────────
def _6x6_ep_actual(ep_nominal):
    """6×6-4c panel: [4×8]=32 nodes per panel. EP maps exactly (32 divides 128).
    Returns ep_nominal unchanged, or None if incompatible.
    """
    panel = 32  # 4×8 rectangular approx of 6×6-4c (32 active GPUs)
    if ep_nominal <= 0:
        return None
    if ep_nominal <= panel:
        return ep_nominal
    elif ep_nominal % panel == 0:
        return ep_nominal
    return None


def _4x4_ep_actual(ep_nominal):
    """4×4 panel: [4×4]=16 nodes per panel. EP maps exactly (16 divides 128)."""
    panel = 16
    if ep_nominal <= 0:
        return None
    if ep_nominal <= panel:
        return ep_nominal
    elif ep_nominal % panel == 0:
        return ep_nominal
    return None


def _6x6_full_ep_actual(ep_nominal):
    """6×6 full panel: 36 nodes (6×6 FlattenedButterfly).
    Valid single-panel EP: any value that tiles as r×c with r≤6 and c≤6.
    Valid multi-panel EP: multiples of 36.
    Returns ep_nominal if compatible, None otherwise.
    """
    panel = 36
    if ep_nominal <= 0:
        return None
    if ep_nominal <= panel:
        if ep_nominal <= 6:  # fits in one row (1×EP)
            return ep_nominal
        if ep_nominal % 6 == 0 and ep_nominal // 6 <= 6:  # full-column: N/6 rows × 6 cols
            return ep_nominal
        for r in range(2, 7):  # most-square fallback within 6×6 bounds
            if ep_nominal % r == 0 and ep_nominal // r <= 6:
                return ep_nominal
        return None  # no valid r×c within 6×6 (e.g., EP=32 on 6×6 panel)
    elif ep_nominal % panel == 0:
        return ep_nominal
    return None


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


def make_nvl72_config(ep):
    """NVL72: hierarchical_fb with tile_size=64, panel_size=128."""
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


def make_6x6_4c_config(ep_nominal, intra_opt_bw):
    """6×6-4c glass panel: fb_2d with panel_rows=4, panel_cols=8 (32 nodes).
    Physical: 6×6 grid minus 4 corners = 32 GPUs.
    ASTRA-Sim model: [4×8]=32 rectangular approximation, uniform optical BW.
    Note: [6,6]=36 is unusable — 36 does not divide 128 model experts.
    """
    ep = _6x6_ep_actual(ep_nominal)
    if ep is None:
        return None, None
    return ep, {
        "num_nodes": 1,
        "topology_config": {
            "type":       "fb_2d",
            "panel_rows": 4,
            "panel_cols": 8,
            "intra_bw":   float(intra_opt_bw),
            "intra_lat":  INTRA_OPT_LAT,
            "inter_bw":   INTER_BW,
            "inter_lat":  INTER_LAT,
        },
        "nodes": [_node(ep)],
    }


def make_4x4_config(ep_nominal, intra_opt_bw):
    """4×4 glass panel: fb_2d with panel_rows=4, panel_cols=4 (16 nodes).
    Uniform optical BW for both intra-panel dims.
    """
    ep = _4x4_ep_actual(ep_nominal)
    if ep is None:
        return None, None
    return ep, {
        "num_nodes": 1,
        "topology_config": {
            "type":       "fb_2d",
            "panel_rows": 4,
            "panel_cols": 4,
            "intra_bw":   float(intra_opt_bw),
            "intra_lat":  INTRA_OPT_LAT,
            "inter_bw":   INTER_BW,
            "inter_lat":  INTER_LAT,
        },
        "nodes": [_node(ep)],
    }


def make_6x6_full_config(ep_nominal, intra_opt_bw):
    """6×6 full glass panel: fb_2d with panel_rows=6, panel_cols=6 (36 nodes).
    Physical: complete 6×6 grid, all 36 GPUs active.
    ASTRA-Sim model: FlattenedButterfly with rows=6, cols=6.
      EP grid layouts: EP=6→1×6, EP=12→2×6, EP=18→3×6, EP=24→4×6, EP=30→5×6,
                       EP=36→6×6 (full), EP=8→2×4, EP=9→3×3, EP=16→4×4, EP=25→5×5.
    Multi-panel: EP=72→[36,2], EP=108→[36,3], etc.
    Note: EP=32 is incompatible (32 cannot tile a 6×6 panel).
    """
    ep = _6x6_full_ep_actual(ep_nominal)
    if ep is None:
        return None, None
    return ep, {
        "num_nodes": 1,
        "topology_config": {
            "type":       "fb_2d",
            "panel_rows": 6,
            "panel_cols": 6,
            "intra_bw":   float(intra_opt_bw),
            "intra_lat":  INTRA_OPT_LAT,
            "inter_bw":   INTER_BW,
            "inter_lat":  INTER_LAT,
        },
        "nodes": [_node(ep)],
    }


# ─────────────────────────────────────────────────────────────
# Run helpers
# ─────────────────────────────────────────────────────────────
WORKLOAD_CACHE = "astra-sim/inputs/workload"  # relative to repo root


def _cleanup_workload_cache():
    """Delete ASTRA-Sim temporary .et workload files to free disk space."""
    import shutil
    cache = os.path.join("/app/LLMServingSim", WORKLOAD_CACHE)
    for entry in os.listdir(cache) if os.path.isdir(cache) else []:
        path = os.path.join(cache, entry)
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
        print(f"[dry] {label:60s}  ep={ep:3d}  "
              f"type={tc.get('type','flat'):15s}  "
              f"in_bw={tc.get('intra_bw', tc.get('intra_opt_bw', tc.get('elec_bw','?')))}")
        return {"label": label, "status": "dry-run"}

    t0 = time.time()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, cwd="/app/LLMServingSim")
        elapsed = time.time() - t0
        result = _parse_csv(out_csv, label, elapsed) if res.returncode == 0 else \
                 {"label": label, "status": "error", "elapsed_s": elapsed, "stderr": res.stderr[-400:]}
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
    # nvl72_ep128
    m = re.match(r"nvl72_ep(\d+)", label)
    if m:
        return {"topology": "nvl72", "panel": "nvl72",
                "ep_nominal": int(m.group(1)), "ep": int(m.group(1)),
                "wg_count": 14, "intra_opt_bw": NVL72_ELEC_BW}
    # fb_6x6_4c_ep64n_ep72_wg6_bw768
    m = re.match(r"fb_(\w+)_ep(\d+)n_ep(\d+)_wg(\d+)_bw(\d+)", label)
    if m:
        return {"topology": "fb", "panel": m.group(1),
                "ep_nominal": int(m.group(2)), "ep": int(m.group(3)),
                "wg_count": int(m.group(4)), "intra_opt_bw": int(m.group(5))}
    return {}


# ─────────────────────────────────────────────────────────────
# Build run list
# ─────────────────────────────────────────────────────────────
def build_run_list(ep_sizes, wg_counts, panels):
    runs = []

    # NVL72 baselines
    for ep in ep_sizes:
        label = f"nvl72_ep{ep}"
        runs.append((label, ep, make_nvl72_config(ep)))

    # Glass FB panels
    for ep_nom in ep_sizes:
        for wg in wg_counts:
            bw = wg * WG_BW

            if "6x6_4c" in panels:
                ep_act, cfg = make_6x6_4c_config(ep_nom, bw)
                if cfg is not None:
                    label = f"fb_6x6_4c_ep{ep_nom}n_ep{ep_act}_wg{wg}_bw{bw}"
                    runs.append((label, ep_act, cfg))

            if "4x4" in panels:
                ep_act, cfg = make_4x4_config(ep_nom, bw)
                if cfg is not None:
                    label = f"fb_4x4_ep{ep_nom}n_ep{ep_act}_wg{wg}_bw{bw}"
                    runs.append((label, ep_act, cfg))

            if "6x6" in panels:
                ep_act, cfg = make_6x6_full_config(ep_nom, bw)
                if cfg is not None:
                    label = f"fb_6x6_ep{ep_nom}n_ep{ep_act}_wg{wg}_bw{bw}"
                    runs.append((label, ep_act, cfg))

    return runs


# ─────────────────────────────────────────────────────────────
# CSV writer
# ─────────────────────────────────────────────────────────────
FIELDS = ["label", "status", "topology", "panel", "ep_nominal", "ep", "wg_count",
          "intra_opt_bw", "ttft_ms", "tpot_ms", "lat_ms", "ttft_p50_ms", "tpot_p50_ms",
          "n_reqs", "elapsed_s", "stderr", "error"]


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
    parser = argparse.ArgumentParser(description="128-GPU EP sweep: NVL72 vs 4×4 vs 6×6-4c FB (corrected model)")
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--ep",          type=int,   nargs="+", default=EP_SIZES)
    parser.add_argument("--wg",          type=int,   nargs="+", default=WG_COUNTS)
    parser.add_argument("--panels",      type=str,   nargs="+", default=["6x6_4c", "4x4"],
                        choices=["6x6_4c", "4x4", "6x6"])
    parser.add_argument("--output-dir",  default="outputs/sim_128gpu")
    parser.add_argument("--results-csv", default="outputs/dse_128gpu_4x4_vs_6x6.csv")
    parser.add_argument("--timeout",     type=int,   default=7200)
    args = parser.parse_args()

    runs = build_run_list(args.ep, args.wg, args.panels)
    print(f"Total runs : {len(runs)} | dry-run: {args.dry_run}")
    print(f"EP (nominal): {args.ep}")
    print(f"N_WG       : {args.wg}  → BW: {[w*WG_BW for w in args.wg]} GB/s")
    print(f"Panels     : {args.panels}")
    print(f"6×6-4c: [4×8]=32 nodes/panel, EP exact (32 divides 128 experts)")
    print(f"4×4:    [4×4]=16 nodes/panel, EP exact (16 divides 128 experts)")
    print(f"6×6:    [6×6]=36 nodes/panel, valid EP: 4,6,8,9,12,16,18,24,25,30,36 (sub-grid within 6×6)")
    print(f"Results    : {args.results_csv}")

    os.makedirs(os.path.dirname(args.results_csv) or ".", exist_ok=True)

    # Load already-completed results for resume
    done_labels = set()
    results = []
    if os.path.exists(args.results_csv) and not args.dry_run:
        with open(args.results_csv) as f:
            for row in csv.DictReader(f):
                if row.get("status") in ("ok", "timeout", "error"):
                    done_labels.add(row["label"])
                    results.append(row)
        if done_labels:
            print(f"Resuming: {len(done_labels)} runs already done, skipping them.")

    for i, (label, ep_act, cfg) in enumerate(runs):
        if label in done_labels:
            print(f"[{i+1:3d}/{len(runs)}] SKIP (already done): {label}")
            continue
        r = run_sim(cfg, label, args.output_dir, args.dry_run, args.timeout)
        results.append(r)
        tc  = r.get("tpot_ms"); tf = r.get("ttft_ms")
        print(f"[{i+1:3d}/{len(runs)}] {label:65s}"
              f"  status={r.get('status','?'):8s}"
              f"  TTFT={f'{tf:.2f}ms' if tf else '?':9s}"
              f"  TPOT={f'{tc:.2f}ms' if tc else '?':9s}"
              f"  ({r.get('elapsed_s',0):.0f}s)")
        write_results(results, args.results_csv)

    write_results(results, args.results_csv)
    n_ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"\nDone. {n_ok}/{len(runs)} succeeded → {args.results_csv}")


if __name__ == "__main__":
    main()
