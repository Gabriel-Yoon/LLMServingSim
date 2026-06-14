"""
Glass-panel intra-/inter-panel bandwidth DSE (ASP-DAC 2027).

Uses the heterogeneous FlattenedButterfly link model added in this branch:
  - electrical RDL    for adjacent tiles  (grid distance == 1)  — FIXED
  - optical waveguide for far same-row/col tiles                — SWEPT (intra)
  - inter-panel Ring  (glass-panel perimeter optical I/O)       — SWEPT (inter)

WG spec: 1 waveguide group = 32λ × 32 Gb/s = 1.024 Tb/s = 128 GB/s.

Two sweeps
----------
intra : single-panel (EP == panel_size). Fix electrical RDL, sweep wg_count
        (optical far-link bw = wg_count × 128). Question: "how many WGs to
        match NVL72 within one panel?"
inter : multi-panel (EP == panel_size × k). Fix intra at the chosen sweet-spot
        WG count, sweep inter_bw (panel egress). Question: "how much panel
        egress bandwidth to match NVL72?"

Baselines: NVL72 modeled at 1800 GB/s bidirectional NVLink (consistent with
the electrical RDL convention), one config per EP.

Workload modes
--------------
controlled : every request arrives at t=0 (N == EP, one per instance, no dummy
             waves). Isolates the network: reports steady-state decode TPOT
             (median inter-token interval, warmup tokens dropped). DEFAULT.
realistic  : staggered 1ms arrivals (matches sweep_paper.py); use to validate
             the controlled sweet-spot under a realistic arrival pattern.

Usage (inside Docker, from /app/LLMServingSim):
  python scripts/sweep_panel_dse.py --sweep intra --dry-run
  python scripts/sweep_panel_dse.py --sweep intra --panels 4x4 6x6_4c
  python scripts/sweep_panel_dse.py --sweep inter --fixed-wg 8 --panels 4x4
  python scripts/sweep_panel_dse.py --sweep intra --mode realistic
"""

import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import sys
import time

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCAB_SIZE = 151936

# ─────────────────────────── Fixed hardware / model ───────────────────────────
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
HARDWARE   = "H100"
NPU_MEM    = {"mem_size": 80, "mem_bw": 3350, "mem_latency": 0}
CPU_MEM    = {"mem_size": 1024, "mem_bw": 512, "mem_latency": 0}

BLOCK_SIZE = 16
MAX_SEQS   = 128
MAX_TOKENS = 2048
ISL        = 512
OSL        = 64    # enough decode steps to reach steady state in controlled mode
WARMUP_TOK = 8     # ITL indices dropped before taking the steady-state median

# ─────────────────────────── Link model constants ─────────────────────────────
WG_BW          = 128.0    # GB/s per waveguide group
ELEC_BW        = 1800.0   # GB/s electrical RDL (adjacent tiles)  — FIXED
ELEC_LAT       = 100.0    # ns
INTRA_OPT_LAT  = 300.0    # ns  optical waveguide latency
INTER_LAT      = 5000.0   # ns  inter-panel fiber latency

# NVL72 baseline (1800 GB/s bidirectional NVLink convention).
# Latency held at 300 ns — on par with the optical waveguide level so the
# comparison reflects bandwidth, not a latency-assumption mismatch (real
# NVLink latency is a few hundred ns, not the legacy 1000 ns placeholder).
NVL72_ELEC_BW = 1800.0
NVL72_IB_BW   = 50.0      # inter-rack InfiniBand (only used for EP > 64)
NVL72_LAT     = 300.0

# ─────────────────────────── Sweep axes ───────────────────────────────────────
WG_COUNTS_DEFAULT  = [1, 2, 3, 4, 6, 8, 12]
INTER_BW_DEFAULT   = [64, 128, 256, 512, 1024]

# panel_name -> (panel_rows, panel_cols)
PANELS = {
    "4x4":    (4, 4),   # 16 nodes
    "6x6_4c": (4, 8),   # 32 nodes (rectangular approx of 6×6 minus 4 corners)
    "6x6":    (6, 6),   # 36 nodes
}


def _instance(ep):
    return {"model_name": MODEL_NAME, "hardware": HARDWARE, "npu_mem": NPU_MEM,
            "num_npus": 1, "tp_size": 1, "ep_size": ep, "dp_group": "A", "pd_type": None}


def _node(ep):
    return {"num_instances": ep, "cpu_mem": CPU_MEM, "instances": [_instance(ep) for _ in range(ep)]}


def make_panel_config(rows, cols, ep, wg_count, inter_bw):
    """fb_2d config with the electrical/optical split. inter_bw=0 → single panel."""
    return {
        "num_nodes": 1,
        "topology_config": {
            "type": "fb_2d",
            "panel_rows": rows, "panel_cols": cols,
            "elec_bw": ELEC_BW, "elec_latency": ELEC_LAT,        # adjacent (fixed)
            "wg_count": wg_count, "wg_bw": WG_BW,                # optical far (swept)
            "intra_opt_latency": INTRA_OPT_LAT,
            "inter_bw": float(inter_bw), "inter_lat": INTER_LAT,  # inter-panel (swept)
        },
        "nodes": [_node(ep)],
    }


NVL72_RACK = 64    # NVLink domain size used as the rack boundary (largest pow2 <=72 dividing 256/384)

def make_nvl72_config(ep):
    """NVL72 baseline: a 64-GPU NVLink domain (1800 GB/s); beyond it, EP spans
    racks over InfiniBand (~50 GB/s). EP<=64 → flat NVLink; EP>64 → [64, ep/64]
    with the cross-rack dim at IB bandwidth. This inter-rack IB cliff is the
    weak point large models hit, since they need EP >> 64."""
    return {
        "num_nodes": 1,
        "topology_config": {
            "type": "hierarchical_fb", "panel_size": NVL72_RACK, "tile_size": NVL72_RACK,
            "elec_bw": NVL72_ELEC_BW, "intra_opt_bw": NVL72_ELEC_BW, "inter_bw": NVL72_IB_BW,
            "elec_latency": NVL72_LAT, "intra_opt_latency": NVL72_LAT, "inter_latency": NVL72_LAT,
        },
        "nodes": [_node(ep)],
    }


# ─────────────────────────── Workload ─────────────────────────────────────────
def make_workload(ep, batch_per_inst, mode, isl, osl):
    """N = ep × batch_per_inst requests. With controlled (all arrive t=0) and a
    short ISL, the requests accumulate so that ~batch_per_inst of them decode
    concurrently on each instance — driving total_len (and thus the EXPOSED MoE
    all-to-all message) up. That is the regime where inter-GPU bandwidth
    (FlattenedButterfly vs NVL72) actually lands on the critical path."""
    rng = random.Random(0)
    n = ep * batch_per_inst
    lines = []
    for i in range(n):
        arrival = 0 if mode == "controlled" else i * 1_000_000
        lines.append(json.dumps({
            "input_toks": isl, "output_toks": osl, "arrival_time_ns": arrival,
            "input_tok_ids": [rng.randint(0, VOCAB_SIZE - 1) for _ in range(isl)],
            "output_tok_ids": [rng.randint(0, VOCAB_SIZE - 1) for _ in range(osl)],
        }))
    path = os.path.join(REPO_ROOT, "outputs", "panel_dse", "workloads",
                        f"wl_{mode}_ep{ep}_b{batch_per_inst}_isl{isl}_osl{osl}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return os.path.relpath(path, REPO_ROOT), n


# ─────────────────────────── Parse ────────────────────────────────────────────
def parse_metrics(out_csv_abs):
    """Return steady-state TPOT (median ITL past warmup) + avg TPOT + throughput."""
    if not os.path.exists(out_csv_abs):
        return None
    rows = list(csv.DictReader(open(out_csv_abs)))
    if not rows:
        return None
    steady, tpots, ttfts, e2es = [], [], [], []
    for r in rows:
        if r.get("TPOT"):
            tpots.append(float(r["TPOT"]) / 1e6)
        if r.get("TTFT"):
            ttfts.append(float(r["TTFT"]) / 1e6)
        if r.get("latency"):
            e2es.append(float(r["latency"]) / 1e6)
        try:
            itl = [v / 1e6 for v in eval(r["ITL"])]   # ns -> ms
            steady.extend(itl[WARMUP_TOK:])
        except Exception:
            pass
    if not tpots:
        return None
    tpot_avg = sum(tpots) / len(tpots)
    # Steady-state per-token cost = MEAN of pooled post-warmup inter-token
    # intervals (= total steady decode time / steady token count). Robust to
    # the lockstep ITL clustering that all-simultaneous arrival induces, where
    # a per-request median would collapse onto the clustered near-zero values.
    tpot_ss = (sum(steady) / len(steady)) if steady else tpot_avg
    return {
        "ttft_ms": (sum(ttfts) / len(ttfts)) if ttfts else 0.0,
        "e2e_latency_ms": (sum(e2es) / len(e2es)) if e2es else 0.0,
        "tpot_avg_ms": tpot_avg,
        "tpot_steady_ms": tpot_ss,
        "interactivity_tok_s_user": (1000.0 / tpot_ss) if tpot_ss > 0 else 0.0,
        "n_completed": len(rows),
    }


# ─────────────────────────── Run one config ───────────────────────────────────
CSV_FIELDS = ["label", "sweep", "mode", "topology", "panel", "ep", "wg_count",
              "intra_opt_bw", "inter_bw", "status", "ttft_ms", "e2e_latency_ms",
              "tpot_avg_ms", "tpot_steady_ms", "interactivity_tok_s_user",
              "n_completed", "elapsed_s", "error"]


def run_one(label, cfg, ep, meta, mode, isl, osl, max_tokens, batch_per_inst, out_dir, dry_run, timeout):
    cfg_path = os.path.join(out_dir, "configs", f"{label}.json")
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=1)
    cfg_rel = os.path.relpath(cfg_path, REPO_ROOT)
    out_csv = os.path.join(out_dir, "runs", f"{label}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_rel = os.path.relpath(out_csv, REPO_ROOT)
    wl_rel, n_req = make_workload(ep, batch_per_inst, mode, isl, osl)

    # Prefill is INCLUDED (no --skip-prefill): the prefill MoE all-gather /
    # reduce-scatter moves ~MB-scale messages, making the run bandwidth-bound so
    # the optical WG-count sweep is meaningful. TTFT captures the
    # bandwidth-sensitive prefill cost; TPOT is the (latency-bound) decode.
    # ``max_tokens`` (--max-tokens) sets how many tokens a prefill step batches:
    # larger value → larger per-collective MoE message → more bandwidth-bound,
    # which is what makes the FlattenedButterfly bandwidth advantage visible.
    cmd = ["python", "-m", "serving", "--cluster-config", cfg_rel, "--dtype", "bfloat16",
           "--block-size", str(BLOCK_SIZE), "--max-num-seqs", str(MAX_SEQS),
           "--max-num-batched-tokens", str(max_tokens), "--dataset", wl_rel,
           "--output", out_rel, "--num-req", str(n_req), "--log-level", "WARNING"]

    if dry_run:
        print(f"  [dry] {label:38s} ep={ep:<4d} {meta}")
        return {"label": label, "status": "dry", **meta}

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, timeout=timeout)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            # collapse the traceback to a single CSV-safe line (last error line)
            err = proc.stderr.strip().replace("\n", " | ").replace(",", ";")[-300:]
            return {"label": label, "status": "error", "elapsed_s": elapsed,
                    "error": err, **meta}
        m = parse_metrics(out_csv) or {}
        return {"label": label, "status": "ok", "elapsed_s": elapsed, "error": "", **meta, **m}
    except subprocess.TimeoutExpired:
        return {"label": label, "status": "timeout", "elapsed_s": timeout,
                "error": f"timeout>{timeout}s", **meta}


def append_row(csv_path, row):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


# ─────────────────────────── Build run lists ──────────────────────────────────
def build_intra_runs(panels, wg_counts, mode, isl, osl):
    runs = []
    for pname in panels:
        rows, cols = PANELS[pname]
        ep = rows * cols  # full single panel
        for wg in wg_counts:
            cfg = make_panel_config(rows, cols, ep, wg_count=wg, inter_bw=0.0)
            meta = {"sweep": "intra", "mode": mode, "topology": pname, "panel": ep,
                    "ep": ep, "wg_count": wg, "intra_opt_bw": int(wg * WG_BW), "inter_bw": 0}
            runs.append((f"intra_{pname}_ep{ep}_wg{wg}", cfg, ep, meta))
        # NVL72 baseline at this EP
        runs.append((f"intra_{pname}_ep{ep}_nvl72", make_nvl72_config(ep), ep,
                     {"sweep": "intra", "mode": mode, "topology": "nvl72", "panel": ep,
                      "ep": ep, "wg_count": 0, "intra_opt_bw": int(NVL72_ELEC_BW), "inter_bw": 0}))
    return runs


def build_epscale_runs(panel, wg, inter_opt_bw, ep_list, mode):
    """EP-scaling: sweep EP, compare glass-FB (multi-panel optical) vs NVL72
    (NVLink rack + inter-rack IB cliff). The large-model story — at EP >> 64
    NVL72 falls to IB while the glass panel's optical inter-panel BW holds."""
    rows, cols = panel
    panel_size = rows * cols
    runs = []
    for ep in ep_list:
        # glass FB: single panel if EP fits, else multi-panel over optical fiber
        multi = ep > panel_size
        cfg = make_panel_config(rows, cols, ep, wg_count=wg,
                                inter_bw=(inter_opt_bw if multi else 0.0))
        runs.append((f"epscale_fb_ep{ep}", cfg, ep,
                     {"sweep": "epscale", "mode": mode, "topology": "glass_fb", "panel": panel_size,
                      "ep": ep, "wg_count": wg, "intra_opt_bw": int(wg * WG_BW),
                      "inter_bw": (inter_opt_bw if multi else 0)}))
        runs.append((f"epscale_nvl72_ep{ep}", make_nvl72_config(ep), ep,
                     {"sweep": "epscale", "mode": mode, "topology": "nvl72", "panel": NVL72_RACK,
                      "ep": ep, "wg_count": 0, "intra_opt_bw": int(NVL72_ELEC_BW),
                      "inter_bw": (NVL72_IB_BW if ep > NVL72_RACK else 0)}))
    return runs


def build_inter_runs(panels, inter_bws, fixed_wg, mode, isl, osl):
    runs = []
    for pname in panels:
        rows, cols = PANELS[pname]
        panel = rows * cols
        for k in (2, 4):  # number of panels
            ep = panel * k
            if ep > 128:
                continue
            for ibw in inter_bws:
                cfg = make_panel_config(rows, cols, ep, wg_count=fixed_wg, inter_bw=ibw)
                meta = {"sweep": "inter", "mode": mode, "topology": pname, "panel": panel,
                        "ep": ep, "wg_count": fixed_wg, "intra_opt_bw": int(fixed_wg * WG_BW),
                        "inter_bw": ibw}
                runs.append((f"inter_{pname}_ep{ep}_wg{fixed_wg}_ib{ibw}", cfg, ep, meta))
            runs.append((f"inter_{pname}_ep{ep}_nvl72", make_nvl72_config(ep), ep,
                         {"sweep": "inter", "mode": mode, "topology": "nvl72", "panel": panel,
                          "ep": ep, "wg_count": 0, "intra_opt_bw": int(NVL72_ELEC_BW),
                          "inter_bw": NVL72_IB_BW}))
    return runs


def main():
    global MODEL_NAME, HARDWARE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", choices=["intra", "inter", "epscale"], required=True)
    ap.add_argument("--ep-list", nargs="+", type=int, default=[16, 32, 64, 128, 256],
                    help="epscale: EP degrees to sweep (must divide model experts)")
    ap.add_argument("--epscale-panel", nargs=2, type=int, default=[4, 8],
                    help="epscale: glass panel (rows cols), default 4x8=32 (6x6-4c)")
    ap.add_argument("--inter-opt-bw", type=int, default=512,
                    help="epscale: glass inter-panel optical egress BW (GB/s)")
    ap.add_argument("--mode", choices=["controlled", "realistic"], default="controlled")
    ap.add_argument("--panels", nargs="+", default=["4x4", "6x6_4c", "6x6"], choices=list(PANELS))
    ap.add_argument("--wg", nargs="+", type=int, default=WG_COUNTS_DEFAULT, help="intra: WG counts")
    ap.add_argument("--inter-bw", nargs="+", type=int, default=INTER_BW_DEFAULT, help="inter: egress BW (GB/s)")
    ap.add_argument("--fixed-wg", type=int, default=8, help="inter: intra optical WG count to hold fixed")
    ap.add_argument("--model", default=MODEL_NAME,
                    help="HF model id (must have configs/model/<id>.json and profiler/perf/<hw>/<id>/). "
                         "DeepSeek-V3 (hidden 7168, 256 experts, 61 layers) is far more bandwidth-bound than Qwen.")
    ap.add_argument("--hardware", default=HARDWARE)
    ap.add_argument("--npu-mem-gb", type=int, default=NPU_MEM["mem_size"],
                    help="per-NPU memory capacity (GB). Raise for large models like DeepSeek-V3 at tp=1 "
                         "(671B replicates dense weights per GPU); only gates the weight-fit/KV check, "
                         "not network timing.")
    ap.add_argument("--batch-per-instance", type=int, default=1,
                    help="concurrent requests per instance (N = EP × this). >1 raises decode total_len "
                         "so the EP all-to-all message is exposed — the regime where FB bandwidth matters.")
    ap.add_argument("--isl", type=int, default=ISL)
    ap.add_argument("--osl", type=int, default=OSL)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                    help="prefill batch size (tokens/step). Larger → larger MoE messages → more bandwidth-bound.")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=None, help="results CSV (default outputs/panel_dse/dse_<sweep>_<mode>.csv)")
    args = ap.parse_args()

    # Override the model/hardware globals used by the config builders.
    MODEL_NAME = args.model
    HARDWARE = args.hardware
    NPU_MEM["mem_size"] = args.npu_mem_gb

    out_dir = os.path.join(REPO_ROOT, "outputs", "panel_dse")
    results_csv = args.out or os.path.join(out_dir, f"dse_{args.sweep}_{args.mode}.csv")
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)

    if args.sweep == "intra":
        runs = build_intra_runs(args.panels, args.wg, args.mode, args.isl, args.osl)
    elif args.sweep == "epscale":
        runs = build_epscale_runs(tuple(args.epscale_panel), args.fixed_wg, args.inter_opt_bw,
                                  args.ep_list, args.mode)
    else:
        runs = build_inter_runs(args.panels, args.inter_bw, args.fixed_wg, args.mode, args.isl, args.osl)

    print(f"\n{'='*72}")
    print(f"Panel DSE — sweep={args.sweep}  mode={args.mode}  model={MODEL_NAME} hw={HARDWARE}")
    print(f"  electrical RDL (fixed): {ELEC_BW} GB/s / {ELEC_LAT} ns")
    if args.sweep == "intra":
        print(f"  WG counts: {args.wg}  (optical = WG×{WG_BW:.0f} GB/s)")
    elif args.sweep == "epscale":
        print(f"  EP list: {args.ep_list}  glass panel {args.epscale_panel} wg{args.fixed_wg}, "
              f"inter-panel optical {args.inter_opt_bw} GB/s  vs  NVL72 (NVLink {NVL72_ELEC_BW} / "
              f"inter-rack IB {NVL72_IB_BW} @EP>{NVL72_RACK})")
    else:
        print(f"  fixed intra WG: {args.fixed_wg} (={int(args.fixed_wg*WG_BW)} GB/s)  inter_bw sweep: {args.inter_bw}")
    print(f"  workload: N=EP×{args.batch_per_instance}, ISL/OSL {args.isl}/{args.osl}, "
          f"max_tokens={args.max_tokens}, arrivals={'t=0' if args.mode=='controlled' else '1ms staggered'}")
    print(f"  baseline: NVL72 {NVL72_ELEC_BW} GB/s bidirectional")
    print(f"  runs: {len(runs)}   results: {results_csv}")
    print(f"{'='*72}\n")

    for label, cfg, ep, meta in runs:
        row = run_one(label, cfg, ep, meta, args.mode, args.isl, args.osl, args.max_tokens,
                      args.batch_per_instance, out_dir, args.dry_run, args.timeout)
        if not args.dry_run:
            append_row(results_csv, row)
            tag = row.get("status")
            if row.get("ttft_ms") is not None and tag == "ok":
                print(f"  {label:40s} {tag:8s} "
                      f"TTFT={row['ttft_ms']:.3f}ms  TPOT_ss={row.get('tpot_steady_ms', 0):.4f}ms")
            else:
                print(f"  {label:40s} {tag}  {row.get('error','')[:80]}")

    print("\n=== done ===" + ("  (dry-run)" if args.dry_run else f"  → {results_csv}"))


if __name__ == "__main__":
    main()
