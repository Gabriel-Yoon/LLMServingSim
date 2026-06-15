"""
F3 — reach crossover plot: TPOT vs EP, glass-FB vs NVL72.

Reads an epscale CSV (sweep_panel_dse.py --sweep epscale, ideally with
MOE_ALLTOALL=1). Plots glass_fb vs nvl72 steady-state TPOT against EP on a log-y
axis, shades the cross-domain region (EP > rack), and annotates the NVL72/glass
ratio at each EP. Output: outputs/panel_dse/plots/<name>.png.

Usage (inside Docker, from /app/LLMServingSim):
  python scripts/plot_reach.py --csv outputs/panel_dse/epscale_4x4_wg5_a2a.csv --rack 64
  python scripts/plot_reach.py --csv outputs/panel_dse/epscale_mini_a2a.csv --rack 4 --name f3_reach_mini
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path):
    d = {}
    for r in csv.DictReader(open(path)):
        if r.get("status") != "ok":
            continue
        d.setdefault(r["fabric"], {})[int(r["ep"])] = (
            float(r["tpot_steady_ms"]), float(r.get("exposed_frac") or 0) * 100)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--rack", type=int, default=64, help="NVLink rack size (cross-domain boundary)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--title", default="EP-scaling: glass-FB vs NVL72 (MoE all-to-all)")
    args = ap.parse_args()

    d = load(args.csv)
    fb, nv = d.get("glass_fb", {}), d.get("nvl72", {})
    eps = sorted(set(fb) & set(nv))
    if not eps:
        print("no overlapping EP rows in", args.csv)
        return

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(eps, [fb[e][0] for e in eps], "o-", color="#2ca02c", lw=2.2, ms=8,
            label="glass FB (optical)")
    ax.plot(eps, [nv[e][0] for e in eps], "s-", color="#d62728", lw=2.2, ms=8,
            label="NVL72 (NVLink + inter-rack IB)")

    # cross-domain region (EP > rack): NVL72 falls to IB
    if max(eps) > args.rack:
        ax.axvspan(args.rack, max(eps), color="gray", alpha=0.08)
        ax.axvline(args.rack, color="gray", ls=":", lw=1.3)
        ax.text(args.rack, ax.get_ylim()[1], f"  rack={args.rack}\n  (NVL72 → IB)",
                va="top", ha="left", fontsize=8, color="gray")

    # annotate NVL72/glass ratio
    for e in eps:
        ratio = nv[e][0] / fb[e][0]
        if ratio >= 1.2:
            ax.annotate(f"{ratio:.1f}x", xy=(e, nv[e][0]), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=8,
                        color="#d62728", fontweight="bold")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(eps)
    ax.set_xticklabels([str(e) for e in eps])
    ax.set_xlabel("Expert-parallel degree (EP)", fontsize=11)
    ax.set_ylabel("decode TPOT [ms]  (steady, lower = better)", fontsize=11)
    ax.set_title(args.title, fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10)

    out_dir = os.path.join(REPO, "outputs", "panel_dse", "plots")
    os.makedirs(out_dir, exist_ok=True)
    name = args.name or ("f3_reach_" + os.path.splitext(os.path.basename(args.csv))[0])
    path = os.path.join(out_dir, name + ".png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved:", path)
    print(f"{'EP':>5} {'glass':>9} {'NVL72':>9} {'NVL/glass':>10}")
    for e in eps:
        print(f"{e:>5} {fb[e][0]:>9.2f} {nv[e][0]:>9.2f} {nv[e][0]/fb[e][0]:>9.2f}x")


if __name__ == "__main__":
    main()
