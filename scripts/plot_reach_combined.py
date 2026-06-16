#!/usr/bin/env python3
"""H2 headline: decode TPOT + exposed-comm vs EP, glass-FB vs NVL72, 2 models.

2x2 grid: (top) decode TPOT vs EP, (bottom) exposed all-to-all vs EP, for
Qwen3-235B and DeepSeek-V3. Shows glass staying flat (optical keeps the MoE
all-to-all cheap as EP scales) while NVL72 rises (its exposed comm climbs as EP
crosses the NVLink rack into inter-rack IB). rack boundary marked.

Usage:
  python scripts/plot_reach_combined.py \
      --qwen outputs/panel_dse/reach_qwen235b.csv \
      --deepseek outputs/panel_dse/reach_deepseek_v3.csv --rack 64
"""
import argparse, csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs", "panel_dse", "plots")
C = {"glass": "#2ca02c", "nvl72": "#d62728"}
L = {"glass": "Glass-FB (optical)", "nvl72": "NVL72 (NVLink+IB)"}


def load(path):
    g, n = {}, {}
    for r in csv.DictReader(open(path)):
        if r.get("status") != "ok":
            continue
        d = g if r["fabric"].startswith("glass") else n
        d[int(r["ep"])] = (float(r["tpot_gt_ms"]), float(r["exposed_frac"]))
    return g, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", required=True)
    ap.add_argument("--deepseek", required=True)
    ap.add_argument("--rack", type=int, default=64)
    ap.add_argument("--name", default="H2_reach_combined")
    args = ap.parse_args()

    models = [("Qwen3-235B-A22B", args.qwen), ("DeepSeek-V3", args.deepseek)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for col, (title, path) in enumerate(models):
        g, n = load(path)
        eps = sorted(g)
        # (top) TPOT
        axT = axes[0][col]
        axT.plot(eps, [g[e][0] for e in eps], "-o", color=C["glass"], lw=2.4, label=L["glass"])
        axT.plot(eps, [n[e][0] for e in eps], "-s", color=C["nvl72"], lw=2.4, label=L["nvl72"])
        ratio = n[max(eps)][0] / g[max(eps)][0]
        axT.set_title(f"{title}: decode TPOT vs EP  (NVL72/glass = {ratio:.2f}x @EP{max(eps)})")
        axT.set_ylabel("decode TPOT [ms] (lower=better)")
        axT.set_ylim(bottom=0)
        # (bottom) exposed
        axE = axes[1][col]
        axE.plot(eps, [g[e][1] * 100 for e in eps], "-o", color=C["glass"], lw=2.4, label=L["glass"])
        axE.plot(eps, [n[e][1] * 100 for e in eps], "-s", color=C["nvl72"], lw=2.4, label=L["nvl72"])
        axE.set_title(f"{title}: exposed all-to-all vs EP")
        axE.set_ylabel("exposed comm [%]")
        axE.set_ylim(bottom=0)
        for ax in (axT, axE):
            ax.set_xscale("log", base=2)
            ax.set_xticks(eps)
            ax.set_xticklabels([str(e) for e in eps])
            ax.set_xlabel("Expert-parallel degree (EP)")
            ax.axvline(args.rack, color="grey", ls="--", lw=1.2, alpha=0.7)
            ax.annotate("NVLink rack\nboundary", xy=(args.rack, ax.get_ylim()[1]*0.5),
                        fontsize=8, color="grey", ha="center")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

    fig.suptitle("Reach: glass-FB optical sustains flat TPOT as EP scales past the NVLink rack; "
                 "NVL72 degrades on inter-rack IB", fontsize=12)
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUT, args.name + ".png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {os.path.relpath(path, REPO)}")


if __name__ == "__main__":
    main()
