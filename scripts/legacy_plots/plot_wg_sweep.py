"""
Plot WG count sweep results.

Reads outputs/dse_wg_sweep.csv (produced by sweep_wg.py).
Saves figures to outputs/wg_plots/ at dpi=300.

Figures:
  01_tpot_vs_wg_all.png    — TPOT vs WG count, all topologies + EP, NVL72 dashed
  02_ttft_vs_wg_all.png    — TTFT vs WG count
  03_tpot_ep16.png         — EP=16 only, breakeven annotations
  04_tpot_ep32.png         — EP=32 only, breakeven annotations
  05_speedup_vs_nvl72.png  — TPOT speedup over NVL72 vs WG count
  06_breakeven_summary.png — Summary: breakeven WG count per topology+EP
  07_topo_comparison_ep32.png — All three topologies at EP=32 side-by-side
  08_collective_cost_model.png — Theoretical collective cost vs WG count
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

OUTDIR = "outputs/wg_plots"
NVL72_EQUIV_WG = 14  # 1800/128 ≈ 14 WGs for reference line positioning
WG_PER_BW = 128.0
WG_COUNTS = [1, 2, 3, 4, 6, 8, 12]

TOPO_STYLES = {
    "4x8":  {"color": "tab:orange", "marker": "o", "ls": "-",  "label": "[4,8] tile_size=4"},
    "2x16": {"color": "tab:green",  "marker": "s", "ls": "--", "label": "[2,16] tile_size=2"},
    "flat": {"color": "tab:red",    "marker": "^", "ls": ":",  "label": "[32] flat optical"},
}
EP_ALPHA = {16: 0.6, 32: 1.0}
NVL72_STYLE = {"color": "navy", "ls": "--", "linewidth": 2}


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "ok"].copy()
    df["wg_count"] = pd.to_numeric(df["wg_count"], errors="coerce")
    df["intra_opt_bw"] = pd.to_numeric(df["intra_opt_bw"], errors="coerce")
    df["ep"] = pd.to_numeric(df["ep"], errors="coerce")
    df["tpot_ms"] = pd.to_numeric(df["tpot_ms"], errors="coerce")
    df["ttft_ms"] = pd.to_numeric(df["ttft_ms"], errors="coerce")
    return df


def nvl72_tpot(df, ep):
    rows = df[(df["topology"] == "nvl72") & (df["ep"] == ep)]
    if rows.empty:
        return None
    return rows["tpot_ms"].mean()


def nvl72_ttft(df, ep):
    rows = df[(df["topology"] == "nvl72") & (df["ep"] == ep)]
    if rows.empty:
        return None
    return rows["ttft_ms"].mean()


def find_breakeven(wg_arr, tpot_arr, baseline):
    """Return WG count where tpot_arr first drops below baseline, or None."""
    if baseline is None or len(wg_arr) < 2:
        return None
    for i in range(len(wg_arr) - 1):
        if tpot_arr[i] >= baseline >= tpot_arr[i + 1]:
            try:
                f = interp1d([tpot_arr[i], tpot_arr[i + 1]],
                             [wg_arr[i], wg_arr[i + 1]])
                return float(f(baseline))
            except Exception:
                return (wg_arr[i] + wg_arr[i + 1]) / 2
    if tpot_arr[-1] < baseline:
        return wg_arr[0]
    return None


def annotate_breakeven(ax, be_wg, baseline, color, label=""):
    if be_wg is None:
        return
    ax.axvline(be_wg, color=color, alpha=0.4, linewidth=1, linestyle=":")
    ax.annotate(f"  breakeven\n  {be_wg:.1f} WG",
                xy=(be_wg, baseline), xytext=(be_wg + 0.3, baseline * 0.95),
                fontsize=7.5, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0))


def savefig(fig, name, outdir=OUTDIR):
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────
def plot_01_tpot_all(df, outdir):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("TPOT vs WG Count — All Topologies & EP Sizes", fontweight="bold")

    for ep in [16, 32]:
        nvl = nvl72_tpot(df, ep)
        if nvl:
            ax.axhline(nvl, color="navy", ls="--", lw=1.5, alpha=0.5 + 0.5 * (ep == 32))
            ax.text(WG_COUNTS[-1] + 0.2, nvl, f"NVL72 EP={ep}\n{nvl:.2f}ms",
                    va="center", fontsize=8, color="navy")

        for topo, sty in TOPO_STYLES.items():
            sub = df[(df["topology"] == topo) & (df["ep"] == ep)].sort_values("wg_count")
            if sub.empty:
                continue
            wg = sub["wg_count"].values
            tpot = sub["tpot_ms"].values
            alpha = EP_ALPHA[ep]
            lbl = f"{sty['label']} EP={ep}"
            ax.plot(wg, tpot, color=sty["color"], marker=sty["marker"],
                    ls=sty["ls"], linewidth=2, alpha=alpha, label=lbl, markersize=7)

    ax.set_xlabel("WG count per tile pair  (1 WG = 128 GB/s)")
    ax.set_ylabel("TPOT (ms)")
    ax.set_xticks(WG_COUNTS)
    ax.set_xticklabels([f"{w}\n({w*WG_PER_BW:.0f}G)" for w in WG_COUNTS], fontsize=8)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    savefig(fig, "01_tpot_vs_wg_all.png", outdir)


def plot_02_ttft_all(df, outdir):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("TTFT vs WG Count — All Topologies & EP Sizes", fontweight="bold")

    for ep in [16, 32]:
        nvl = nvl72_ttft(df, ep)
        if nvl:
            ax.axhline(nvl, color="navy", ls="--", lw=1.5, alpha=0.5 + 0.5 * (ep == 32))
            ax.text(WG_COUNTS[-1] + 0.2, nvl, f"NVL72 EP={ep}", va="center",
                    fontsize=8, color="navy")

        for topo, sty in TOPO_STYLES.items():
            sub = df[(df["topology"] == topo) & (df["ep"] == ep)].sort_values("wg_count")
            if sub.empty:
                continue
            ax.plot(sub["wg_count"].values, sub["ttft_ms"].values,
                    color=sty["color"], marker=sty["marker"], ls=sty["ls"],
                    linewidth=2, alpha=EP_ALPHA[ep], label=f"{sty['label']} EP={ep}", markersize=7)

    ax.set_xlabel("WG count per tile pair  (1 WG = 128 GB/s)")
    ax.set_ylabel("TTFT (ms)")
    ax.set_xticks(WG_COUNTS)
    ax.set_xticklabels([f"{w}\n({w*WG_PER_BW:.0f}G)" for w in WG_COUNTS], fontsize=8)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    savefig(fig, "02_ttft_vs_wg_all.png", outdir)


def _plot_ep_tpot(df, ep, outdir, fname):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"TPOT vs WG Count — EP={ep}, 6×6-4c Panel", fontweight="bold")

    nvl = nvl72_tpot(df, ep)
    if nvl:
        ax.axhline(nvl, label=f"NVL72 EP={ep} ({nvl:.2f} ms)", **NVL72_STYLE)

    for topo, sty in TOPO_STYLES.items():
        sub = df[(df["topology"] == topo) & (df["ep"] == ep)].sort_values("wg_count")
        if sub.empty:
            continue
        wg = sub["wg_count"].values
        tpot = sub["tpot_ms"].values
        ax.plot(wg, tpot, color=sty["color"], marker=sty["marker"],
                ls=sty["ls"], linewidth=2.5, label=sty["label"], markersize=8)

        be = find_breakeven(wg, tpot, nvl)
        annotate_breakeven(ax, be, nvl, sty["color"])

    ax.set_xlabel("WG count per tile pair  (1 WG = 128 GB/s)")
    ax.set_ylabel("TPOT (ms)")
    ax.set_xticks(WG_COUNTS)
    ax.set_xticklabels([f"{w}\n({w*WG_PER_BW:.0f}G)" for w in WG_COUNTS], fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    savefig(fig, fname, outdir)


def plot_03_ep16(df, outdir):
    _plot_ep_tpot(df, 16, outdir, "03_tpot_ep16.png")


def plot_04_ep32(df, outdir):
    _plot_ep_tpot(df, 32, outdir, "04_tpot_ep32.png")


def plot_05_speedup(df, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle("TPOT Speedup over NVL72 Baseline", fontweight="bold")

    for ax, ep in zip(axes, [16, 32]):
        ax.set_title(f"EP={ep}")
        nvl = nvl72_tpot(df, ep)
        if nvl is None:
            continue
        ax.axhline(1.0, color="navy", ls="--", lw=2, label="NVL72 = 1.0×")
        ax.axhline(0.9, color="gray", ls=":", lw=1, label="−10% faster")

        for topo, sty in TOPO_STYLES.items():
            sub = df[(df["topology"] == topo) & (df["ep"] == ep)].sort_values("wg_count")
            if sub.empty:
                continue
            wg = sub["wg_count"].values
            speedup = nvl / sub["tpot_ms"].values
            ax.plot(wg, speedup, color=sty["color"], marker=sty["marker"],
                    ls=sty["ls"], linewidth=2.5, label=sty["label"], markersize=8)

            be_idx = np.where(speedup >= 1.0)[0]
            if len(be_idx) > 0:
                ax.annotate(f"≥NVL72\n@{wg[be_idx[0]]}WG",
                            xy=(wg[be_idx[0]], speedup[be_idx[0]]),
                            xytext=(wg[be_idx[0]] + 0.5, speedup[be_idx[0]] - 0.05),
                            fontsize=8, color=sty["color"],
                            arrowprops=dict(arrowstyle="->", color=sty["color"], lw=1))

        ax.set_xlabel("WG count per tile pair  (1 WG = 128 GB/s)")
        ax.set_ylabel("Speedup over NVL72 (higher = better)")
        ax.set_xticks(WG_COUNTS)
        ax.set_xticklabels([f"{w}\n({w*WG_PER_BW:.0f}G)" for w in WG_COUNTS], fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    savefig(fig, "05_speedup_vs_nvl72.png", outdir)


def plot_06_breakeven_summary(df, outdir):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("Breakeven WG Count: FB topology beats NVL72", fontweight="bold")

    topos = list(TOPO_STYLES.keys())
    eps = [16, 32]
    x_pos = np.arange(len(topos))
    width = 0.35

    for i, ep in enumerate(eps):
        nvl = nvl72_tpot(df, ep)
        be_vals = []
        for topo in topos:
            sub = df[(df["topology"] == topo) & (df["ep"] == ep)].sort_values("wg_count")
            if sub.empty or nvl is None:
                be_vals.append(None)
                continue
            be = find_breakeven(sub["wg_count"].values, sub["tpot_ms"].values, nvl)
            be_vals.append(be if be is not None else float("inf"))

        bars_data = [v if v is not None and v != float("inf") else 0 for v in be_vals]
        bars = ax.bar(x_pos + i * width - width / 2, bars_data, width,
                      label=f"EP={ep}", alpha=0.8,
                      color=["tab:orange", "tab:green", "tab:red"])
        for j, (bar, bv) in enumerate(zip(bars, be_vals)):
            if bv is None or bv == float("inf"):
                ax.text(bar.get_x() + bar.get_width() / 2, 0.5, "Never",
                        ha="center", va="bottom", fontsize=8, color="red")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f"{bv:.1f}WG\n({bv*WG_PER_BW:.0f}G)",
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([TOPO_STYLES[t]["label"] for t in topos], fontsize=9)
    ax.set_ylabel("Breakeven WG count (lower = better)")
    ax.set_xlabel("Topology approximation")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    savefig(fig, "06_breakeven_summary.png", outdir)


def plot_07_topo_comparison_ep32(df, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Topology Approximation Comparison at EP=32", fontweight="bold")

    for ax, metric, ylabel, nvl_fn in [
        (axes[0], "tpot_ms", "TPOT (ms)", nvl72_tpot),
        (axes[1], "ttft_ms", "TTFT (ms)", nvl72_ttft),
    ]:
        ep = 32
        nvl = nvl_fn(df, ep)
        if nvl:
            ax.axhline(nvl, label=f"NVL72 {nvl:.2f}ms", **NVL72_STYLE)

        for topo, sty in TOPO_STYLES.items():
            sub = df[(df["topology"] == topo) & (df["ep"] == ep)].sort_values("wg_count")
            if sub.empty:
                continue
            ax.plot(sub["wg_count"].values, sub[metric].values,
                    color=sty["color"], marker=sty["marker"], ls=sty["ls"],
                    linewidth=2.5, label=sty["label"], markersize=8)

        ax.set_xlabel("WG count (1 WG = 128 GB/s)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(WG_COUNTS)
        ax.set_xticklabels([f"{w}\n({w*WG_PER_BW:.0f}G)" for w in WG_COUNTS], fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    savefig(fig, "07_topo_comparison_ep32.png", outdir)


def plot_08_collective_cost_model(df, outdir):
    """Theoretical ASTRA-Sim collective cost."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Theoretical ALLTOALL Cost Model vs WG Count", fontweight="bold")

    ELEC_BW = 1800.0
    WG_SWEEP = np.linspace(1, 12, 100)
    OPT_BW = WG_SWEEP * WG_PER_BW

    for ax, ep in zip(axes, [16, 32]):
        if ep == 16:
            configs = {"4x8→[4,4]": (3, 3), "2x16→[2,8]": (1, 7), "flat→[16]": (0, 15)}
        else:
            configs = {"4x8→[4,8]": (3, 7), "2x16→[2,16]": (1, 15), "flat→[32]": (0, 31)}

        colors = ["tab:orange", "tab:green", "tab:red"]
        for (label, (e_steps, o_steps)), color in zip(configs.items(), colors):
            cost = e_steps / ELEC_BW + o_steps / OPT_BW
            cost_norm = cost / cost[0]
            ax.plot(WG_SWEEP, cost_norm, color=color, linewidth=2.5, label=label)

        # NVL72 ref: flat 1D at 1800 GB/s
        nvl_steps = ep - 1
        nvl_cost = nvl_steps / 1800.0
        ref_cost = (configs[list(configs.keys())[0]][0] / ELEC_BW +
                    configs[list(configs.keys())[0]][1] / (1 * WG_PER_BW))
        ax.axhline(nvl_cost / ref_cost, color="navy", ls="--", lw=2, label=f"NVL72 ({ep-1} steps @ 1800G)")

        ax.set_title(f"EP={ep}")
        ax.set_xlabel("WG count per tile pair")
        ax.set_ylabel("Normalized ALLTOALL latency (lower = faster)")
        ax.set_xticks([1, 2, 4, 6, 8, 12])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    savefig(fig, "08_collective_cost_model.png", outdir)


def main():
    parser = argparse.ArgumentParser(description="Plot WG sweep results")
    parser.add_argument("--results", default="outputs/dse_wg_sweep.csv")
    parser.add_argument("--out-dir", default=OUTDIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Theoretical plot never needs simulation data
    _df_empty = pd.DataFrame(columns=["topology", "ep", "wg_count", "tpot_ms", "ttft_ms", "status"])
    plot_08_collective_cost_model(_df_empty, args.out_dir)

    if not os.path.exists(args.results):
        print(f"No results CSV yet ({args.results}) — only theoretical plot generated.")
        return

    df = load_data(args.results)
    print(f"Loaded {len(df)} successful runs from {args.results}")
    if df.empty:
        print("No successful runs yet.")
        return
    print(f"Topologies: {sorted(df['topology'].unique())}")
    print(f"EP sizes:   {sorted(df['ep'].dropna().unique())}")
    print(f"WG counts:  {sorted(df['wg_count'].dropna().unique())}")

    plot_01_tpot_all(df, args.out_dir)
    plot_02_ttft_all(df, args.out_dir)
    plot_03_ep16(df, args.out_dir)
    plot_04_ep32(df, args.out_dir)
    plot_05_speedup(df, args.out_dir)
    plot_06_breakeven_summary(df, args.out_dir)
    plot_07_topo_comparison_ep32(df, args.out_dir)

    print(f"\nAll figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
