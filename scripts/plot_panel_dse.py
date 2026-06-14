"""
Plot glass-panel intra-/inter-panel bandwidth DSE results.

Reads:  outputs/panel_dse/dse_intra_<mode>.csv  and  dse_inter_<mode>.csv
        (written by scripts/sweep_panel_dse.py)
Output: outputs/panel_dse/plots/   (NOT comparison_plots or dse_plots)
          intra_ttft_vs_wg.png     TTFT vs WG count, per panel, NVL72 ref line
          intra_tpot_vs_wg.png     decode TPOT_ss vs WG count (noisy; reference)
          inter_ttft_vs_bw.png     TTFT vs inter-panel egress BW, per panel/EP

The bandwidth-bound headline is TTFT (prefill). Decode TPOT is plotted for
completeness but is noisy under the controlled (all-arrive-at-t0) workload.

Usage:
  python scripts/plot_panel_dse.py                 # mode=controlled
  python scripts/plot_panel_dse.py --mode realistic
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DSE_DIR   = os.path.join(REPO_ROOT, "outputs", "panel_dse")
OUT_DIR   = os.path.join(DSE_DIR, "plots")

PANEL_STYLE = {
    "4x4":    dict(color="#d62728", marker="s", label="FB 4x4 (16)"),
    "6x6_4c": dict(color="#2ca02c", marker="^", label="FB 6x6-4c (32)"),
    "6x6":    dict(color="#9467bd", marker="D", label="FB 6x6 (36)"),
}
NVL72_COLOR = "#1f77b4"


def load(csv_path):
    rows = []
    if os.path.exists(csv_path):
        rows = [r for r in csv.DictReader(open(csv_path)) if r.get("status") == "ok"]
    return rows


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_intra(rows, metric, ylabel, title, fname):
    """metric: 'ttft_ms' or 'tpot_steady_ms'. One line per panel vs WG count."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    panels = [p for p in PANEL_STYLE if any(r["topology"] == p for r in rows)]
    for p in panels:
        sub = sorted([r for r in rows if r["topology"] == p], key=lambda r: int(r["wg_count"]))
        xs = [int(r["wg_count"]) for r in sub]
        ys = [float(r[metric]) for r in sub]
        s = PANEL_STYLE[p]
        ax.plot(xs, ys, color=s["color"], marker=s["marker"], linewidth=2, markersize=7, label=s["label"])
        # NVL72 reference for this panel's EP (horizontal line)
        nv = [r for r in rows if r["topology"] == "nvl72" and r["ep"] == sub[0]["ep"]]
        if nv:
            ax.axhline(float(nv[0][metric]), color=s["color"], linestyle=":", linewidth=1.5, alpha=0.7)
    ax.plot([], [], color="gray", linestyle=":", label="NVL72 ref (matching color)")
    ax.set_xlabel("Optical waveguide groups per axis  (1 WG = 128 GB/s)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    _save(fig, fname)


def plot_inter(rows, fname):
    """TTFT vs inter-panel egress BW. One line per (panel, EP)."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    keys = sorted({(r["topology"], int(r["ep"])) for r in rows if r["topology"] != "nvl72"})
    cmap = plt.cm.viridis
    for i, (p, ep) in enumerate(keys):
        sub = sorted([r for r in rows if r["topology"] == p and int(r["ep"]) == ep and int(r["inter_bw"]) > 0],
                     key=lambda r: int(r["inter_bw"]))
        if not sub:
            continue
        xs = [int(r["inter_bw"]) for r in sub]
        ys = [float(r["ttft_ms"]) for r in sub]
        c = cmap(i / max(len(keys) - 1, 1))
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=6, color=c, label=f"{p} EP={ep}")
        nv = [r for r in rows if r["topology"] == "nvl72" and int(r["ep"]) == ep]
        if nv:
            ax.axhline(float(nv[0]["ttft_ms"]), color=c, linestyle=":", linewidth=1.3, alpha=0.7)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Inter-panel egress bandwidth [GB/s]", fontsize=11)
    ax.set_ylabel("TTFT [ms]  (prefill; lower = better)", fontsize=11)
    ax.set_title("Inter-panel egress BW sweep (fixed intra WG)\nDotted = NVL72 ref per EP", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    _save(fig, fname)


def plot_epscale(rows, fname, title="EP-scaling: glass-FB vs NVL72"):
    """TPOT vs EP, glass_fb vs nvl72. NVL72 should cliff up at EP>64 (IB)."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    styles = {"glass_fb": dict(color="#2ca02c", marker="o", label="Glass FB (optical inter-panel)"),
              "nvl72": dict(color="#d62728", marker="s", label="NVL72 (NVLink + inter-rack IB)")}
    for topo, s in styles.items():
        sub = sorted([r for r in rows if r["topology"] == topo], key=lambda r: int(r["ep"]))
        if not sub:
            continue
        xs = [int(r["ep"]) for r in sub]
        ys = [float(r["tpot_steady_ms"]) for r in sub]
        ax.plot(xs, ys, lw=2.5, markersize=7, **s)
    ax.axvline(64, color="gray", ls=":", alpha=0.6)
    ax.text(64, ax.get_ylim()[0], " NVL72 rack=64", color="gray", fontsize=8, va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Expert-parallel degree EP", fontsize=11)
    ax.set_ylabel("TPOT [ms]  (decode; lower = better)", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(bottom=0)
    _save(fig, fname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="controlled", choices=["controlled", "realistic"])
    args = ap.parse_args()

    intra = load(os.path.join(DSE_DIR, f"dse_intra_{args.mode}.csv"))
    inter = load(os.path.join(DSE_DIR, f"dse_inter_{args.mode}.csv"))

    import glob
    for ep_csv in glob.glob(os.path.join(DSE_DIR, "dse_epscale*.csv")):
        rows = load(ep_csv)
        if rows:
            tag = os.path.basename(ep_csv).replace("dse_epscale_", "").replace(".csv", "")
            print(f"epscale[{tag}]: {len(rows)} ok rows")
            plot_epscale(rows, f"epscale_tpot_vs_ep_{tag}.png",
                         title=f"EP-scaling ({tag}): glass-FB vs NVL72 (IB cliff at EP>64)")

    if intra:
        print(f"intra: {len(intra)} ok rows")
        plot_intra(intra, "ttft_ms", "TTFT [ms]  (prefill; lower = better)",
                   "Intra-panel WG sweep: TTFT vs optical bandwidth\nDotted = NVL72 ref per panel EP",
                   "intra_ttft_vs_wg.png")
        plot_intra(intra, "tpot_steady_ms", "TPOT steady [ms]  (decode; noisy under controlled)",
                   "Intra-panel WG sweep: decode TPOT vs optical bandwidth",
                   "intra_tpot_vs_wg.png")
    else:
        print("no intra rows")

    if inter:
        print(f"inter: {len(inter)} ok rows")
        plot_inter(inter, "inter_ttft_vs_bw.png")
    else:
        print("no inter rows")


if __name__ == "__main__":
    main()
