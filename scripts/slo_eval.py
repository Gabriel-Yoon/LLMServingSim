#!/usr/bin/env python3
"""ViBE-style SLO evaluation infrastructure for the glass-FB vs NVL72 study.

Adopts the experimental setup of ViBE (Go et al., "ViBE: Co-Optimizing Workload
Skew and Hardware Variability for MoE Serving", arXiv 2026):

  - Models      : DeepSeek-V3 (256 experts), Qwen-3 235B (128 experts), 8 routed.
  - SLO (Tab 2b): ShareGPT  TTFT 250 ms ; Sonnet TTFT 350 ms.
                  TPOT 125 ms (DeepSeek) / 100 ms (Qwen).
  - Datasets    : ShareGPT (in 219 / out 201 avg, variable) and
                  Sonnet (in 1024 / out 128, fixed).
  - Load        : Poisson arrivals at a target QPS; sweep QPS.
  - Metric      : goodput = fraction of SLO-compliant requests; report the max
                  sustainable QPS at >= 90 % goodput, plus TTFT/TPOT percentiles.
  - P/D split   : prefill isolated (1 output token) and decode isolated
                  (--skip-prefill) measured separately, as ViBE does.

Our axis of variation is the INTERCONNECT (glass-FB panel vs NVL72) at EP
scaling, orthogonal to ViBE's expert-placement axis. Cluster-config builders are
reused from sweep_panel_dse so the topology/bandwidth model stays identical.

Example (local, light model):
  MOE_ALLTOALL=1 python scripts/slo_eval.py \
      --model Qwen/Qwen3-30B-A3B-Instruct-2507 --hardware H100 --tp 1 \
      --dataset sonnet --pd-mode decode --ep 8 --panel 4 4 --fixed-wg 5 \
      --fabrics glass nvl72 --qps-list 1 2 4 --n-req 64 --npu-mem-gb 320 \
      --out outputs/slo_eval/qwen30b_sonnet_decode_ep8.csv
"""
import argparse, csv, json, math, os, random, statistics, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_panel_dse as S

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOODPUT_TARGET = 0.90   # ViBE: max sustainable QPS at >= 90 % goodput

# ── ViBE SLO table (Table 2b): per dataset (TTFT) and per model family (TPOT) ──
def get_slo(model, dataset):
    """Return (TTFT_ms, TPOT_ms) SLO per ViBE Table 2b."""
    ttft = 250.0 if dataset == "sharegpt" else 350.0     # sonnet -> 350
    fam_qwen = "qwen" in model.lower()
    tpot = 100.0 if fam_qwen else 125.0                  # qwen 100, deepseek 125
    return ttft, tpot

# ── dataset token-length specs (ViBE Table 2b) ──
DATASETS = {
    "sonnet":   dict(in_len=1024, out_len=128, fixed=True),
    "sharegpt": dict(in_len=219,  out_len=201, fixed=False),
}


def _sample_len(mean, fixed, rng, cv=0.6):
    """Fixed length for Sonnet; lognormal-spread length for ShareGPT (variable
    inputs drive routing variance). Clipped to >= 8 tokens."""
    if fixed:
        return int(mean)
    sigma = math.sqrt(math.log(1 + cv * cv))
    mu = math.log(max(mean, 2)) - 0.5 * sigma * sigma
    return max(8, int(rng.lognormvariate(mu, sigma)))


def make_poisson_workload(dataset, qps, n_req, pd_mode, seed=0, max_osl=None):
    """Poisson arrivals at `qps` req/s with ViBE dataset token lengths.

    pd_mode: 'full' (in/out as dataset), 'prefill' (out=1, TTFT-isolated),
    'decode' (out as dataset; run with --skip-prefill for TPOT-isolation).
    max_osl caps output_toks (steady-state TPOT is reached in a few tokens, so a
    small cap keeps the per-request decode-iteration count — and thus the sim
    runtime — bounded without changing the measured per-token latency).
    """
    spec = DATASETS[dataset]
    rng = random.Random(seed)
    V = S.VOCAB_SIZE
    t = 0.0
    lines = []
    for _ in range(n_req):
        t += rng.expovariate(qps)            # inter-arrival ~ Exp(qps) [seconds]
        arrival_ns = int(t * 1e9)
        il = _sample_len(spec["in_len"], spec["fixed"], rng)
        ol = _sample_len(spec["out_len"], spec["fixed"], rng)
        if max_osl is not None:
            ol = min(ol, max_osl)
        if pd_mode == "prefill":
            ol = 1                            # prefill isolated: single output token
        lines.append(json.dumps({
            "input_toks": il, "output_toks": ol, "arrival_time_ns": arrival_ns,
            "input_tok_ids": [rng.randint(0, V - 1) for _ in range(il)],
            "output_tok_ids": [rng.randint(0, V - 1) for _ in range(ol)],
        }))
    path = os.path.join(REPO, "outputs", "slo_eval", "workloads",
                        f"{dataset}_{pd_mode}_qps{qps:g}_n{n_req}_s{seed}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return os.path.relpath(path, REPO)


def build_config(fabric, ep, panel, fixed_wg, inter_opt_bw, nvl72_rack):
    """Glass-FB panel or NVL72 cluster config for `ep` ranks (reuses sweep
    builders; TP/model/hardware come from the sweep_panel_dse globals)."""
    rows, cols = panel
    panel_size = rows * cols
    if fabric == "glass":
        multi = ep > panel_size
        return S.make_panel_config(rows, cols, ep, wg_count=fixed_wg,
                                   inter_bw=(inter_opt_bw if multi else 0.0))
    elif fabric == "nvl72":
        return S.make_nvl72_config(ep)
    raise ValueError(f"unknown fabric {fabric}")


def run_serving(cfg, wl_rel, out_csv_rel, pd_mode, max_seqs, max_tokens, timeout):
    """Run one serving simulation; returns (ok, log)."""
    cfg_rel = os.path.join("configs", "cluster", "_slo_eval_tmp.json")
    with open(os.path.join(REPO, cfg_rel), "w") as f:
        json.dump(cfg, f, indent=2)
    cmd = ["python", "-m", "serving", "--cluster-config", cfg_rel,
           "--dtype", "bfloat16", "--block-size", "16",
           "--dataset", wl_rel, "--output", out_csv_rel,
           "--max-num-seqs", str(max_seqs),
           "--max-num-batched-tokens", str(max_tokens),
           "--log-level", "WARNING"]
    if pd_mode == "decode":
        cmd.append("--skip-prefill")          # decode isolated (prefix-cache warm)
    env = {**os.environ, "MOE_ALLTOALL": "1"}
    try:
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                           env=env, timeout=timeout)
        return (p.returncode == 0), (p.stdout + p.stderr)
    except subprocess.TimeoutExpired as e:
        return False, f"TIMEOUT\n{(e.stdout or '')}{(e.stderr or '')}"


def _pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[k]


def compute_goodput(out_csv_abs, ttft_slo, tpot_slo, pd_mode):
    """Goodput = fraction of requests meeting the applicable SLO(s), plus
    TTFT/TPOT percentiles. Decode mode checks TPOT only; prefill checks TTFT only."""
    if not os.path.exists(out_csv_abs):
        return None
    rows = list(csv.DictReader(open(out_csv_abs)))
    if not rows:
        return None
    ttfts, tpots, ok = [], [], 0
    for r in rows:
        ttft = float(r["TTFT"]) / 1e6 if r.get("TTFT") not in (None, "", "0") else None
        tpot = float(r["TPOT"]) / 1e6 if r.get("TPOT") not in (None, "", "0") else None
        if ttft is not None:
            ttfts.append(ttft)
        if tpot is not None:
            tpots.append(tpot)
        good = True
        if pd_mode != "decode" and ttft is not None:
            good &= ttft <= ttft_slo
        if pd_mode != "prefill" and tpot is not None:
            good &= tpot <= tpot_slo
        ok += int(good)
    return {
        "n": len(rows), "goodput": ok / len(rows),
        "ttft_p50": _pct(ttfts, 0.50), "ttft_p90": _pct(ttfts, 0.90), "ttft_p99": _pct(ttfts, 0.99),
        "tpot_p50": _pct(tpots, 0.50), "tpot_p90": _pct(tpots, 0.90), "tpot_p99": _pct(tpots, 0.99),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--hardware", default="H100")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--dataset", choices=list(DATASETS), default="sonnet")
    ap.add_argument("--pd-mode", choices=["full", "prefill", "decode"], default="full")
    ap.add_argument("--ep", type=int, default=8)
    ap.add_argument("--panel", type=int, nargs=2, default=[4, 4], metavar=("ROWS", "COLS"))
    ap.add_argument("--fixed-wg", type=int, default=5)
    ap.add_argument("--inter-opt-bw", type=float, default=512.0)
    ap.add_argument("--nvl72-rack", type=int, default=64)
    ap.add_argument("--nvl72-bw", type=float, default=None,
                    help="NVLink within-domain unidir BW (GB/s). H100-consistent: 450 with "
                         "--nvl72-rack 8; default (None) keeps the module value (GB200 900).")
    ap.add_argument("--fabrics", nargs="+", default=["glass", "nvl72"])
    ap.add_argument("--qps-list", type=float, nargs="+", default=[1, 2, 4])
    ap.add_argument("--n-req", type=int, default=128)
    ap.add_argument("--max-osl", type=int, default=None,
                    help="cap output tokens per request (bounds decode iterations / runtime)")
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-num-batched-tokens", type=int, default=8192)
    ap.add_argument("--npu-mem-gb", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default="outputs/slo_eval/slo_eval.csv")
    args = ap.parse_args()

    # push run-wide settings into the shared cluster-config builders
    S.MODEL_NAME = args.model
    S.HARDWARE = args.hardware
    S.TP = args.tp
    S.NPU_MEM["mem_size"] = args.npu_mem_gb
    # mem_bw was H100's 3350 even for H200 (cosmetic for decode: LOCAL weights are
    # compute-bound/profiled, so this does not change TPOT; fixed for correctness of
    # offloading / analytical weight-load terms). H200 SXM ~4.8 TB/s.
    S.NPU_MEM["mem_bw"] = 4800 if args.hardware.upper() == "H200" else 3350
    S.NVL72_RACK = args.nvl72_rack
    if args.nvl72_bw is not None:
        S.NVL72_ELEC_BW = args.nvl72_bw

    ttft_slo, tpot_slo = get_slo(args.model, args.dataset)
    print(f"# model={args.model} hw={args.hardware} tp={args.tp} ds={args.dataset} "
          f"pd={args.pd_mode} ep={args.ep} panel={args.panel}")
    print(f"# SLO: TTFT<={ttft_slo}ms TPOT<={tpot_slo}ms  goodput target={GOODPUT_TARGET:.0%}")

    fields = ["fabric", "dataset", "pd_mode", "ep", "tp", "qps", "n", "goodput",
              "ttft_p50", "ttft_p90", "ttft_p99", "tpot_p50", "tpot_p90", "tpot_p99",
              "ttft_slo", "tpot_slo", "status"]
    out_abs = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    results = []

    for fabric in args.fabrics:
        max_qps_ok = 0.0
        for qps in args.qps_list:
            cfg = build_config(fabric, args.ep, args.panel, args.fixed_wg,
                               args.inter_opt_bw, args.nvl72_rack)
            wl = make_poisson_workload(args.dataset, qps, args.n_req, args.pd_mode,
                                       args.seed, args.max_osl)
            run_csv = f"outputs/slo_eval/runs/{fabric}_{args.dataset}_{args.pd_mode}_ep{args.ep}_qps{qps:g}.csv"
            os.makedirs(os.path.dirname(os.path.join(REPO, run_csv)), exist_ok=True)
            ok, log = run_serving(cfg, wl, run_csv, args.pd_mode,
                                  args.max_num_seqs, args.max_num_batched_tokens, args.timeout)
            g = compute_goodput(os.path.join(REPO, run_csv), ttft_slo, tpot_slo, args.pd_mode)
            if g is None:
                row = {"fabric": fabric, "dataset": args.dataset, "pd_mode": args.pd_mode,
                       "ep": args.ep, "tp": args.tp, "qps": qps, "status": "error"}
                err = [l for l in log.splitlines() if l.strip()][-1:] if log else ["?"]
                print(f"  {fabric:<6} qps={qps:<5g} ERROR {err}")
            else:
                if g["goodput"] >= GOODPUT_TARGET:
                    max_qps_ok = max(max_qps_ok, qps)
                row = {"fabric": fabric, "dataset": args.dataset, "pd_mode": args.pd_mode,
                       "ep": args.ep, "tp": args.tp, "qps": qps,
                       "ttft_slo": ttft_slo, "tpot_slo": tpot_slo, "status": "ok", **g}
                print(f"  {fabric:<6} qps={qps:<5g} goodput={g['goodput']:.2f} "
                      f"TTFT(p50/p90/p99)={_fmt(g['ttft_p50'])}/{_fmt(g['ttft_p90'])}/{_fmt(g['ttft_p99'])} "
                      f"TPOT={_fmt(g['tpot_p50'])}/{_fmt(g['tpot_p90'])}/{_fmt(g['tpot_p99'])}")
            results.append(row)
        print(f"  >> {fabric}: max sustainable QPS at >=90% goodput = {max_qps_ok:g}")

    with open(out_abs, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"Saved: {args.out}")


def _fmt(v):
    return f"{v:.0f}" if v is not None else "-"


if __name__ == "__main__":
    main()
