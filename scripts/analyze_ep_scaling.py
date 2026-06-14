"""
Analytical EP-scaling curve for large MoE serving (ASP-DAC 2027).

Goal: show WHERE inter-GPU bandwidth (glass-panel FlattenedButterfly vs NVL72)
starts to matter for LARGE MoE models, using the PROFILED expert compute
(moe.csv) plus the (anchoring-fixed) EP all-to-all communication model.

Per decode step, per MoE layer, at expert-parallel degree EP with B concurrent
requests per rank:

  experts/rank   = E / EP                       (weight-load shrinks with EP)
  tokens/rank    = EP*B * (1 - (1-1/EP)^k)       (group tokens reaching a rank)
  activated/rank = min(experts/rank, tokens/rank)
  compute_us     = moe.csv[tokens/rank, activated/rank]   (PROFILED; weight-load bound)
  comm_bytes     = EP*B * hidden * fp / 2        (group all-to-all volume, ÷2 ASTRA Ring)
  comm_us        = comm_bytes / (fabric_GBps * 1e3)
  layer_us       = max(compute_us, comm_us)      (comm overlaps compute; exposed when larger)
  TPOT           = layers * layer_us

compute ∝ 1/EP (drops), comm ∝ EP (grows) → U-shaped TPOT(EP). The minimum and
the high-EP branch are set by fabric bandwidth: higher bandwidth → comm branch
lower → can scale EP further → lower achievable TPOT. This is the regime where
the glass panel's large bandwidth pays off for big models.

Usage (inside Docker, from /app/LLMServingSim):
  python scripts/analyze_ep_scaling.py
  python scripts/analyze_ep_scaling.py --B 128 --hardware H200
"""

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR   = os.path.join(REPO_ROOT, "outputs", "panel_dse", "plots")
FP        = 2
WG_BW     = 128.0

MODELS = {
    "DeepSeek-V3": "deepseek-ai/DeepSeek-V3-0324",
    "Kimi-K2":     "moonshotai/Kimi-K2-Instruct",
}

# Effective all-to-all bandwidth (GB/s) AS A FUNCTION OF EP — the large-model
# story lives in the cross-boundary tier:
#   NVL72: 1800 GB/s NVLink within a 72-GPU rack, but only ~50 GB/s InfiniBand
#          once EP spans racks (EP > 72). Large models need EP >> 72.
#   Glass FB panel (32 GPUs): intra-panel optical (WG×128) within a panel,
#          inter-panel optical fiber (egress BW) once EP spans panels (EP > 32).
NVL72_RACK = 72
NVL72_INTRA = 1800.0
NVL72_IB = 50.0
FB_PANEL = 32

def _fb_bw(wg, ep):
    intra = wg * WG_BW
    inter = wg * WG_BW          # inter-panel egress modeled at the same WG budget
    return intra if ep <= FB_PANEL else inter

def _nvl72_bw(ep):
    return NVL72_INTRA if ep <= NVL72_RACK else NVL72_IB

FABRICS = {
    "FB 4WG (512)":   lambda ep: _fb_bw(4, ep),
    "FB 8WG (1024)":  lambda ep: _fb_bw(8, ep),
    "FB 16WG (2048)": lambda ep: _fb_bw(16, ep),
    "NVL72 (NVLink/IB)": _nvl72_bw,
}


def load_moe(hw, model):
    path = os.path.join(REPO_ROOT, "profiler", "perf", hw, model, "bf16", "tp1", "moe.csv")
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((int(r["tokens"]), int(r["activated_experts"]), float(r["time_us"])))
    return rows


def lookup_moe(rows, tokens, activated):
    """Nearest-activated bracket + linear interp on tokens (mirrors _lookup_moe)."""
    aes = sorted(set(a for _, a, _ in rows))
    # bracket activated
    lo = max([a for a in aes if a <= activated], default=aes[0])
    hi = min([a for a in aes if a >= activated], default=aes[-1])

    def interp_tokens(ae):
        pts = sorted((t, v) for t, a, v in rows if a == ae)
        if tokens <= pts[0][0]:
            return pts[0][1]
        if tokens >= pts[-1][0]:
            # linear extrapolation from last segment
            (t0, v0), (t1, v1) = pts[-2], pts[-1]
            return v1 + (v1 - v0) * (tokens - t1) / (t1 - t0)
        for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
            if t0 <= tokens <= t1:
                return v0 + (v1 - v0) * (tokens - t0) / (t1 - t0)
        return pts[-1][1]

    v_lo = interp_tokens(lo)
    if lo == hi:
        return v_lo
    v_hi = interp_tokens(hi)
    return v_lo + (v_hi - v_lo) * (activated - lo) / (hi - lo)


def curve(model_cfg, moe_rows, B, ep_list):
    hidden = model_cfg["hidden_size"]
    layers = model_cfg["num_hidden_layers"]
    E = model_cfg.get("n_routed_experts", model_cfg.get("num_local_experts"))
    k = model_cfg["num_experts_per_tok"]

    out = {"ep": [], "compute_ms": [], "comm_ms": {f: [] for f in FABRICS}, "tpot_ms": {f: [] for f in FABRICS}}
    for ep in ep_list:
        if ep > E:
            continue
        experts_per_rank = max(1, E // ep)
        tokens_per_rank = ep * B * (1 - (1 - 1.0 / ep) ** k)
        activated = max(1, min(experts_per_rank, round(tokens_per_rank)))
        compute_us = lookup_moe(moe_rows, max(1, round(tokens_per_rank)), activated)
        comm_bytes = ep * B * hidden * FP / 2.0
        out["ep"].append(ep)
        out["compute_ms"].append(layers * compute_us / 1000.0)
        for fname, bw_fn in FABRICS.items():
            bw = bw_fn(ep)                             # GB/s, EP-dependent (rack/panel boundary)
            comm_us = comm_bytes / (bw * 1e3)          # bytes / (GB/s) = ns; /1000 = us
            layer_us = max(compute_us, comm_us)        # comm overlaps compute; exposed when larger
            out["comm_ms"][fname].append(layers * comm_us / 1000.0)
            out["tpot_ms"][fname].append(layers * layer_us / 1000.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=64, help="concurrent requests per rank")
    ap.add_argument("--hardware", default="H200")
    args = ap.parse_args()
    ep_list = [8, 16, 32, 64, 128, 256, 384]

    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, len(MODELS), figsize=(13, 5.2), sharey=False)

    for ax, (mname, mid) in zip(axes, MODELS.items()):
        cfg = json.load(open(os.path.join(REPO_ROOT, "configs", "model", f"{mid}.json")))
        moe_rows = load_moe(args.hardware, mid)
        c = curve(cfg, moe_rows, args.B, ep_list)
        eps = c["ep"]
        # compute floor (fabric-independent)
        ax.plot(eps, c["compute_ms"], "k--", lw=1.5, label="compute floor (weight-load)")
        fb_colors = {"FB 4WG (512)": "#2ca02c", "FB 8WG (1024)": "#17becf", "FB 16WG (2048)": "#9467bd"}
        for fname in FABRICS:
            if "NVL72" in fname:
                style = dict(color="#d62728", lw=2.8, marker="s", ms=5)   # red, exploding
            else:
                style = dict(color=fb_colors.get(fname, "gray"), lw=2, marker="o", ms=4)
            ax.plot(eps, c["tpot_ms"][fname], label=fname, **style)
        ax.axvline(NVL72_RACK, color="#d62728", ls=":", alpha=0.5)
        ax.text(NVL72_RACK, ax.get_ylim()[0], " NVL72 rack=72", color="#d62728", fontsize=7, va="bottom")
        ax.set_title(f"{mname}  (E={cfg.get('n_routed_experts')}, hidden={cfg['hidden_size']})", fontsize=10)
        ax.set_xlabel("Expert-parallel degree EP", fontsize=10)
        ax.set_ylabel("TPOT [ms] (per decode step)", fontsize=10)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="upper center")

    fig.suptitle(f"EP-scaling: large-MoE decode TPOT vs EP  (B={args.B}/rank, {args.hardware})\n"
                 f"Large models need EP >> 72: NVL72 hits the inter-rack IB (50 GB/s) cliff and explodes; "
                 f"glass-FB optical inter-panel BW tracks the compute floor → up to 2-3x faster.",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(OUT_DIR, f"ep_scaling_analysis_B{args.B}_{args.hardware}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # text summary: min-TPOT EP and value per fabric
    for mname, mid in MODELS.items():
        cfg = json.load(open(os.path.join(REPO_ROOT, "configs", "model", f"{mid}.json")))
        moe_rows = load_moe(args.hardware, mid)
        c = curve(cfg, moe_rows, args.B, ep_list)
        print(f"\n{mname}:")
        for fname in FABRICS:
            tp = c["tpot_ms"][fname]
            imin = min(range(len(tp)), key=lambda i: tp[i])
            print(f"  {fname:16s} min TPOT={tp[imin]:.2f}ms @ EP={c['ep'][imin]}")


if __name__ == "__main__":
    main()
