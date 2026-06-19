#!/usr/bin/env python3
"""
Fabric headline (Fig B) — analytical MoE all-to-all latency vs EP: glass-FB vs NVL72.

The simulator's measured end-to-end TPOT is fabric-PARITY (congestion-unaware ASTRA
divides collective volume by domain count, and decode a2a is ~5% of a compute-bound
step), so the inter-domain bandwidth gap does NOT show in measured TPOT. It shows in
the COLLECTIVE LATENCY itself, which is a deterministic alpha-beta quantity — computed
here from first principles (independent of the sim, like topo_project's hop model):

  per-layer a2a bytes = dispatch(local chunk) + combine(full buffer)
      dispatch = max(1, batch//EP) * (hidden + n_experts) * fp
      combine  = batch * hidden * fp
  a2a_us(EP) = (dispatch+combine) * n_layers / inter_bw / 1000
  inter_bw:  glass = optical 512 GB/s for all EP (intra-panel <=16, inter-panel >16)
             NVL72 = NVLink (450 H100 / 900 GB200) while EP <= domain, else IB 50.

The cliff is the RATIO nvl72/glass = bw_glass / bw_nvl72: ~512/50 ~ 10x once EP crosses
the NVLink domain (8 for H100-HGX, 64 for GB200), vs ~1x (parity) in-domain. Bytes
cancel in the ratio, so the cliff is robust to batch/model.

Run:
  python scripts/plot_fabric_cliff.py --model configs/model/deepseek-ai/DeepSeek-V3-0324.json \
    --out outputs/paper_figures/fig_fabric_a2a_cliff_deepseek.png
"""
import argparse, json, os

OPTICAL = 512.0      # glass inter/intra-panel optical GB/s
IB = 50.0            # inter-domain InfiniBand GB/s
BASELINES = {        # name -> (nvlink_bw, domain_gpus, label)
    "h100":  (450.0, 8,  "NVL72 baseline (H100 NVLink4, domain 8)"),
    "gb200": (900.0, 64, "NVL72 baseline (GB200 NVLink5, domain 64)"),
}


def a2a_bytes_per_layer(cfg, ep, batch, fp=2):
    hidden = cfg["hidden_size"]
    n_exp = cfg.get("num_local_experts") or cfg.get("n_routed_experts") or cfg.get("num_experts") or 0
    dispatch = max(1, batch // max(ep, 1)) * (hidden + n_exp) * fp
    combine = batch * hidden * fp
    return dispatch + combine


def a2a_us(cfg, ep, batch, inter_bw):
    b = a2a_bytes_per_layer(cfg, ep, batch) * cfg["num_hidden_layers"]
    return b / inter_bw / 1000.0     # bytes / (GB/s = B/ns) / 1000 = us


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to configs/model/.../*.json")
    ap.add_argument("--ep-list", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--baselines", nargs="+", default=["h100", "gb200"], choices=list(BASELINES))
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.model))
    label = args.label or os.path.basename(args.model).replace(".json", "")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5))
    eps = args.ep_list

    glass = [a2a_us(cfg, e, args.batch, OPTICAL) for e in eps]
    ax.plot(eps, glass, "o-", color="tab:green", lw=2.4, ms=8, label="glass-FB (optical 512)")

    cliff_notes = []
    colors = {"h100": "tab:red", "gb200": "tab:orange"}
    for bl in args.baselines:
        nvbw, dom, lab = BASELINES[bl]
        ys = [a2a_us(cfg, e, args.batch, nvbw if e <= dom else IB) for e in eps]
        ax.plot(eps, ys, "s--", color=colors[bl], lw=2.2, ms=7, label=lab)
        # cliff ratio at the largest EP
        e = eps[-1]; r = a2a_us(cfg, e, args.batch, IB) / a2a_us(cfg, e, args.batch, OPTICAL)
        cliff_notes.append((bl, dom, r, ys))

    # annotate the H100 cliff (domain 8) — the primary
    if "h100" in args.baselines:
        e = next((x for x in eps if x > 8), eps[-1])
        yv = a2a_us(cfg, e, args.batch, IB)
        ax.annotate(f"{OPTICAL/IB:.0f}x cliff\n(EP>8 -> IB)", xy=(e, yv),
                    xytext=(e * 1.1, yv * 0.45), fontsize=10, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="black"))

    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(eps); ax.set_xticklabels([str(e) for e in eps])
    ax.set_xlabel("GPUs (EP)"); ax.set_ylabel("MoE all-to-all latency per step (us)")
    ax.set_title(f"Fabric collective-latency cliff — glass-FB vs NVL72 ({label})\n"
                 "(analytical alpha-beta; glass optical stays flat, NVL72 falls to IB past its domain)")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.abspath(args.out); os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140); print(f"wrote {out}")
    print(f"  glass  :", " ".join(f"EP{e}={g:.0f}us" for e, g in zip(eps, glass)))
    for bl, dom, r, ys in cliff_notes:
        print(f"  {bl:<6}:", " ".join(f"EP{e}={y:.0f}us" for e, y in zip(eps, ys)), f"| cliff@EP{eps[-1]}={r:.1f}x (domain {dom})")


if __name__ == "__main__":
    main()
