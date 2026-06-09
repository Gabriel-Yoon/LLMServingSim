"""
Plot 128-GPU EP sweep results (NVL72 vs 6×6-4c vs 4×4 FB panels).

Reads:  outputs/dse_128gpu_4x4_vs_6x6.csv
Saves:  outputs/comparison_plots/  (dpi=300)

Graph A — N_WG=6 fixed, EP sweep (x=EP, y=TPOT)
  Lines: NVL72 (red), 6×6-4c N_WG=6 (blue), 4×4 N_WG=6 (green)
  Annotations:
    EP=16 "4×4 panel boundary"   (green dashed)
    EP=32 "6×6-4c panel boundary" (blue dashed)
    EP=72 "NVL72 cliff"           (red dashed, between 64 and 128)

Graph B — EP=64 fixed, WG sweep (x=WG count, y=TPOT)
  Lines: 6×6-4c (blue), 4×4 (green)
  NVL72 reference (red horizontal dashed)
  Breakeven annotation

Additional graphs:
  C — Full heatmap: panel × EP × WG → TPOT
  D — TTFT version of Graph A
  E — Speedup over NVL72 (EP sweep, N_WG=6)

Usage:
  python scripts/plot_128gpu.py [--results CSV] [--out-dir DIR] [--wg-ref 6]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

OUTDIR = "outputs/comparison_plots"
WG_BW = 128.0   # GB/s per WG
NVL72_CLIFF_EP = 72   # annotation x position (between EP=64 and EP=128)

PANEL_STYLES = {
    "nvl72":    {"color": "tab:red",    "marker": "o", "ls": "-",  "label": "NVL72 baseline"},
    "6x6_4c":   {"color": "tab:blue",   "marker": "s", "ls": "-",  "label": "6×6-4c FB panel"},
    "4x4":      {"color": "tab:green",  "marker": "^", "ls": "-",  "label": "4×4 FB panel"},
    "6x6":      {"color": "tab:purple", "marker": "D", "ls": "-",  "label": "6×6 FB panel"},
}


def load(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "ok"].copy()
    # Use ep_nominal (intended EP) for x-axis if available, else fall back to ep
    if "ep_nominal" in df.columns:
        df["ep"] = pd.to_numeric(df["ep_nominal"], errors="coerce")
    for col in ["ep", "wg_count", "intra_opt_bw", "tpot_ms", "ttft_ms", "lat_ms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def nvl_tpot(df, ep):
    row = df[(df["panel"] == "nvl72") & (df["ep"] == ep)]
    return row["tpot_ms"].mean() if not row.empty else None


def nvl_ttft(df, ep):
    row = df[(df["panel"] == "nvl72") & (df["ep"] == ep)]
    return row["ttft_ms"].mean() if not row.empty else None


def breakeven_wg(wg_arr, tpot_arr, baseline):
    if baseline is None or len(wg_arr) < 2:
        return None
    for i in range(len(wg_arr) - 1):
        if tpot_arr[i] >= baseline >= tpot_arr[i + 1]:
            try:
                f = interp1d([tpot_arr[i], tpot_arr[i + 1]], [wg_arr[i], wg_arr[i + 1]])
                return float(f(baseline))
            except Exception:
                return (wg_arr[i] + wg_arr[i + 1]) / 2
    return None


def savefig(fig, name, outdir):
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────
# Graph A: N_WG fixed, EP sweep
# ─────────────────────────────────────────────────────────────
def plot_graph_A(df, wg_ref, outdir):
    ep_vals = sorted(df["ep"].dropna().unique())
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_title(f"Graph A — TPOT vs EP (N_WG={wg_ref} fixed)", fontweight="bold", fontsize=13)

    # NVL72 line
    nvl_ep, nvl_tpots = [], []
    for ep in ep_vals:
        v = nvl_tpot(df, ep)
        if v: nvl_ep.append(ep); nvl_tpots.append(v)
    ax.plot(nvl_ep, nvl_tpots, color="tab:red", marker="o", ls="-", linewidth=2.5,
            label="NVL72 baseline", markersize=8, zorder=5)

    # FB panel lines
    for panel, sty in [("6x6_4c", PANEL_STYLES["6x6_4c"]), ("4x4", PANEL_STYLES["4x4"]),
                       ("6x6", PANEL_STYLES["6x6"])]:
        sub = df[(df["panel"] == panel) & (df["wg_count"] == wg_ref)].sort_values("ep")
        if sub.empty:
            continue
        ax.plot(sub["ep"].values, sub["tpot_ms"].values,
                color=sty["color"], marker=sty["marker"], ls=sty["ls"],
                linewidth=2.5, label=f"{sty['label']} (N_WG={wg_ref})", markersize=8)

    # Vertical annotations
    ax.axvline(16, color="tab:green", ls="--", lw=1.5, alpha=0.7)
    ax.text(16, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 5,
            " EP=16\n 4×4 panel\n boundary", color="tab:green",
            fontsize=8.5, va="top", ha="left")

    ax.axvline(32, color="tab:blue", ls="--", lw=1.5, alpha=0.7)
    ax.text(32, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 5,
            "  EP=32\n  6×6-4c panel\n  boundary", color="tab:blue",
            fontsize=8.5, va="top", ha="left")

    # 6x6 panel only supports EP≤16 for single-panel
    ax.axvline(16, color="tab:purple", ls=":", lw=1.5, alpha=0.5)

    # NVL72 cliff: between EP=64 and EP=128
    ax.axvline(NVL72_CLIFF_EP, color="tab:red", ls=":", lw=2, alpha=0.5)
    ax.text(NVL72_CLIFF_EP, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0,
            "  EP=72\n  NVL72 cliff\n  (IB boundary)", color="tab:red",
            fontsize=8.5, va="bottom", ha="left")

    ax.set_xlabel("EP (Expert Parallelism degree)", fontsize=11)
    ax.set_ylabel("TPOT (ms)", fontsize=11)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ep_vals)
    ax.set_xticklabels([str(int(e)) for e in ep_vals])
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Add BW reference text
    ax.text(0.99, 0.02,
            f"Glass FB (corrected): uniform optical intra={wg_ref}×128={wg_ref*WG_BW:.0f}G/300ns | inter=512G/5000ns\n"
            f"6×6-4c: [4×8]=32 nodes/panel | 4×4: [4×4]=16 nodes/panel | 6×6: [6×6]=36 nodes/panel (EP≤16)\n"
            f"NVL72: intra=1800G/1000ns | inter=50G/1000ns (IB, kicks in at EP>64)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            style="italic", bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))

    savefig(fig, f"graph_A_tpot_ep_wg{wg_ref}.png", outdir)


# ─────────────────────────────────────────────────────────────
# Graph B: EP=64 fixed, WG sweep
# ─────────────────────────────────────────────────────────────
def plot_graph_B(df, ep_ref, outdir):
    wg_vals = sorted(df["wg_count"].dropna().unique())
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title(f"Graph B — TPOT vs WG count (EP={ep_ref} fixed)", fontweight="bold", fontsize=13)

    # NVL72 reference
    nvl = nvl_tpot(df, ep_ref)
    if nvl:
        ax.axhline(nvl, color="tab:red", ls="--", lw=2.5,
                   label=f"NVL72 baseline ({nvl:.2f} ms)", zorder=5)

    for panel, sty in [("6x6_4c", PANEL_STYLES["6x6_4c"]), ("4x4", PANEL_STYLES["4x4"]),
                       ("6x6", PANEL_STYLES["6x6"])]:
        sub = df[(df["panel"] == panel) & (df["ep"] == ep_ref)].sort_values("wg_count")
        if sub.empty:
            continue
        wg = sub["wg_count"].values
        tpot = sub["tpot_ms"].values
        ax.plot(wg, tpot, color=sty["color"], marker=sty["marker"], ls="-",
                linewidth=2.5, label=sty["label"], markersize=9)

        be = breakeven_wg(wg, tpot, nvl)
        if be is not None:
            ax.axvline(be, color=sty["color"], ls=":", lw=1.5, alpha=0.6)
            ax.annotate(f"Breakeven\n{be:.1f} WG\n({be*WG_BW:.0f} GB/s)",
                        xy=(be, nvl), xytext=(be + 0.4, nvl * 0.97),
                        fontsize=8.5, color=sty["color"],
                        arrowprops=dict(arrowstyle="->", color=sty["color"], lw=1.2))

    ax.set_xlabel(f"WG count (1 WG = {WG_BW:.0f} GB/s intra-panel)", fontsize=11)
    ax.set_ylabel("TPOT (ms)", fontsize=11)
    ax.set_xticks(wg_vals)
    ax.set_xticklabels([f"{w}\n({w*WG_BW:.0f}G)" for w in wg_vals])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    savefig(fig, f"graph_B_wg_sweep_ep{ep_ref}.png", outdir)


# ─────────────────────────────────────────────────────────────
# Graph C: Full TPOT heatmap (panel, EP, WG)
# ─────────────────────────────────────────────────────────────
def plot_graph_C(df, outdir):
    panels = [p for p in ["6x6_4c", "4x4", "6x6"] if p in df["panel"].values]
    wg_vals = sorted([w for w in df["wg_count"].dropna().unique() if w > 0])
    ep_vals = sorted(df["ep"].dropna().unique())

    fig, axes = plt.subplots(1, len(panels), figsize=(8 * len(panels), 6), sharey=True)
    if len(panels) == 1:
        axes = [axes]
    fig.suptitle("TPOT Heatmap: EP × N_WG for each Glass FB panel", fontweight="bold")

    for ax, panel in zip(axes, panels):
        mat = np.full((len(wg_vals), len(ep_vals)), np.nan)
        for i, wg in enumerate(wg_vals):
            for j, ep in enumerate(ep_vals):
                row = df[(df["panel"] == panel) & (df["ep"] == ep) & (df["wg_count"] == wg)]
                if not row.empty:
                    mat[i, j] = row["tpot_ms"].mean()

        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", origin="lower")
        ax.set_title(f"{panel.replace('_', ' ')} panel", fontweight="bold")
        ax.set_xticks(range(len(ep_vals)))
        ax.set_xticklabels([str(int(e)) for e in ep_vals])
        ax.set_yticks(range(len(wg_vals)))
        ax.set_yticklabels([f"{w} ({w*WG_BW:.0f}G)" for w in wg_vals])
        ax.set_xlabel("EP degree")
        ax.set_ylabel("N_WG (intra-panel BW)")
        plt.colorbar(im, ax=ax, label="TPOT (ms)")

        for i in range(len(wg_vals)):
            for j in range(len(ep_vals)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                            fontsize=8, color="white" if mat[i, j] > np.nanmedian(mat) else "black")

    savefig(fig, "graph_C_tpot_heatmap.png", outdir)


# ─────────────────────────────────────────────────────────────
# Graph D: TTFT EP sweep (same as A but TTFT)
# ─────────────────────────────────────────────────────────────
def plot_graph_D(df, wg_ref, outdir):
    ep_vals = sorted(df["ep"].dropna().unique())
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_title(f"Graph D — TTFT vs EP (N_WG={wg_ref} fixed)", fontweight="bold", fontsize=13)

    nvl_ep, nvl_ttfts = [], []
    for ep in ep_vals:
        v = nvl_ttft(df, ep)
        if v: nvl_ep.append(ep); nvl_ttfts.append(v)
    ax.plot(nvl_ep, nvl_ttfts, color="tab:red", marker="o", ls="-", linewidth=2.5,
            label="NVL72", markersize=8)

    for panel, sty in [("6x6_4c", PANEL_STYLES["6x6_4c"]), ("4x4", PANEL_STYLES["4x4"]),
                       ("6x6", PANEL_STYLES["6x6"])]:
        sub = df[(df["panel"] == panel) & (df["wg_count"] == wg_ref)].sort_values("ep")
        if sub.empty: continue
        ax.plot(sub["ep"].values, sub["ttft_ms"].values,
                color=sty["color"], marker=sty["marker"], ls="-",
                linewidth=2.5, label=f"{sty['label']} N_WG={wg_ref}", markersize=8)

    ax.axvline(16,  color="tab:green", ls="--", lw=1.5, alpha=0.7)
    ax.axvline(32,  color="tab:blue",  ls="--", lw=1.5, alpha=0.7)
    ax.axvline(NVL72_CLIFF_EP, color="tab:red", ls=":", lw=2, alpha=0.5)

    ax.set_xlabel("EP degree", fontsize=11)
    ax.set_ylabel("TTFT (ms)", fontsize=11)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ep_vals)
    ax.set_xticklabels([str(int(e)) for e in ep_vals])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    savefig(fig, f"graph_D_ttft_ep_wg{wg_ref}.png", outdir)


# ─────────────────────────────────────────────────────────────
# Graph E: Speedup over NVL72 (EP sweep, N_WG fixed)
# ─────────────────────────────────────────────────────────────
def plot_graph_E(df, wg_ref, outdir):
    ep_vals = sorted(df["ep"].dropna().unique())
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"Graph E — TPOT Speedup over NVL72 (N_WG={wg_ref})", fontweight="bold", fontsize=13)

    ax.axhline(1.0, color="tab:red", ls="--", lw=2, label="NVL72 = 1.0×")
    ax.axhline(1.1, color="gray",    ls=":",  lw=1, alpha=0.5, label="+10% faster")

    for panel, sty in [("6x6_4c", PANEL_STYLES["6x6_4c"]), ("4x4", PANEL_STYLES["4x4"]),
                       ("6x6", PANEL_STYLES["6x6"])]:
        sub = df[(df["panel"] == panel) & (df["wg_count"] == wg_ref)].sort_values("ep")
        if sub.empty: continue
        eps, speedups = [], []
        for _, row in sub.iterrows():
            nvl = nvl_tpot(df, row["ep"])
            if nvl:
                eps.append(row["ep"])
                speedups.append(nvl / row["tpot_ms"])
        ax.plot(eps, speedups, color=sty["color"], marker=sty["marker"], ls="-",
                linewidth=2.5, label=f"{sty['label']} N_WG={wg_ref}", markersize=8)

        # Mark where FB beats NVL72
        for i, (ep, sp) in enumerate(zip(eps, speedups)):
            if sp >= 1.0:
                ax.annotate(f">{1.0:.0f}×\nEP={int(ep)}",
                            xy=(ep, sp), xytext=(ep * 1.1, sp - 0.05),
                            fontsize=7.5, color=sty["color"],
                            arrowprops=dict(arrowstyle="->", color=sty["color"], lw=1))
                break

    ax.axvline(NVL72_CLIFF_EP, color="tab:red", ls=":", lw=1.5, alpha=0.4)
    ax.text(NVL72_CLIFF_EP * 1.02, 1.0, "NVL72\ncliff", fontsize=8, color="tab:red")

    ax.set_xlabel("EP degree", fontsize=11)
    ax.set_ylabel("Speedup over NVL72 (higher = better)", fontsize=11)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ep_vals)
    ax.set_xticklabels([str(int(e)) for e in ep_vals])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    savefig(fig, f"graph_E_speedup_ep_wg{wg_ref}.png", outdir)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Plot 128-GPU EP sweep results")
    parser.add_argument("--results",  default="outputs/dse_128gpu_4x4_vs_6x6.csv")
    parser.add_argument("--out-dir",  default=OUTDIR)
    parser.add_argument("--wg-ref",   type=int, default=6,
                        help="Fixed WG count for Graphs A, D, E (default: 6)")
    parser.add_argument("--ep-ref",   type=int, default=64,
                        help="Fixed EP for Graph B (default: 64)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.results):
        print(f"Results not found: {args.results}"); sys.exit(1)

    df = load(args.results)
    print(f"Loaded {len(df)} successful runs")
    print(f"Panels: {sorted(df['panel'].unique())}")
    print(f"EP    : {sorted(df['ep'].dropna().unique())}")
    print(f"WG    : {sorted(df['wg_count'].dropna().unique())}")

    plot_graph_A(df, args.wg_ref,  args.out_dir)
    plot_graph_B(df, args.ep_ref,  args.out_dir)
    plot_graph_C(df,               args.out_dir)
    plot_graph_D(df, args.wg_ref,  args.out_dir)
    plot_graph_E(df, args.wg_ref,  args.out_dir)

    print(f"\nAll figures → {args.out_dir}/")


if __name__ == "__main__":
    main()
