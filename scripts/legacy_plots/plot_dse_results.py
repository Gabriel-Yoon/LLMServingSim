"""
Comprehensive DSE results: 4 topology candidates comparison.

Reads:  outputs/dse_128gpu_4x4_vs_6x6.csv
Saves:  outputs/dse_plots/  (dpi=300)

Plots:
  A1 — TTFT vs EP (per panel, WG sweep as shaded band)
  A2 — TPOT vs EP (per panel, WG sweep as shaded band)
  B  — BW sensitivity: TPOT vs intra BW (one line per EP, each panel)
  C  — Breakeven analysis: min WG count to match NVL72 per EP
  D  — Latency heatmap: all 4 topologies × EP
  E  — Combined best-case comparison (best WG for each topology/EP)
  F  — TTFT + TPOT together (dual-axis, WG=6 reference)

Usage:
  python scripts/plot_dse_results.py [--results CSV] [--out-dir DIR]
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

OUTDIR = "outputs/dse_plots"
WG_BW  = 128.0   # GB/s per WG

PANEL_STYLES = {
    "nvl72":    {"color": "#D62728", "marker": "o", "ls": "-",  "label": "NVL72"},
    "6x6_4c":   {"color": "#1F77B4", "marker": "s", "ls": "-",  "label": "6×6-4c FB"},
    "4x4":      {"color": "#2CA02C", "marker": "^", "ls": "-",  "label": "4×4 FB"},
    "6x6":      {"color": "#9467BD", "marker": "D", "ls": "-",  "label": "6×6 FB"},
}

WG_ALPHA = {2: 0.35, 4: 0.55, 6: 0.75, 8: 0.95}


def savefig(fig, name, outdir):
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, name)
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")


def load(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "ok"].copy()
    if "ep_nominal" in df.columns:
        df["ep"] = pd.to_numeric(df["ep_nominal"], errors="coerce")
    for col in ["ep", "wg_count", "intra_opt_bw", "tpot_ms", "ttft_ms", "lat_ms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def nvl_val(df, ep, metric="tpot_ms"):
    row = df[(df["panel"] == "nvl72") & (df["ep"] == ep)]
    return row[metric].mean() if not row.empty else None


def breakeven_wg(wg_arr, val_arr, baseline):
    if baseline is None or len(wg_arr) < 2:
        return None
    for i in range(len(wg_arr) - 1):
        if val_arr[i] >= baseline >= val_arr[i + 1]:
            try:
                f = interp1d([val_arr[i], val_arr[i + 1]], [wg_arr[i], wg_arr[i + 1]])
                return float(f(baseline))
            except Exception:
                return (wg_arr[i] + wg_arr[i + 1]) / 2
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Plot A: TTFT and TPOT vs EP with WG shaded bands
# ─────────────────────────────────────────────────────────────────────────────
def plot_A(df, wg_ref, outdir):
    """TTFT and TPOT vs EP, WG sweep shown as individual lines (wg=wg_ref bold)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle(f"Serving Latency vs EP — 4 Topology Candidates\n"
                 f"Qwen3-30B-A3B (128 experts) · H100 · Bold line = N_WG={wg_ref}",
                 fontsize=12, fontweight="bold")

    wg_vals = sorted(df["wg_count"].dropna().unique())
    ep_vals = sorted(df["ep"].dropna().unique())

    for ax, metric, ylabel, title in [
        (ax1, "ttft_ms", "TTFT (ms)", "Time to First Token (TTFT)"),
        (ax2, "tpot_ms", "TPOT (ms)", "Time per Output Token (TPOT)"),
    ]:
        ax.set_title(title, fontsize=11, fontweight="bold")

        # NVL72 baseline
        nvl_ep, nvl_y = [], []
        for ep in ep_vals:
            v = nvl_val(df, ep, metric)
            if v is not None:
                nvl_ep.append(ep); nvl_y.append(v)
        ax.plot(nvl_ep, nvl_y,
                color=PANEL_STYLES["nvl72"]["color"],
                marker=PANEL_STYLES["nvl72"]["marker"],
                ls="-", lw=2.5, label="NVL72 baseline", zorder=6, ms=8)

        # Glass panels
        for panel in ["4x4", "6x6_4c", "6x6"]:
            sty = PANEL_STYLES[panel]
            for wg in wg_vals:
                sub = df[(df["panel"] == panel) & (df["wg_count"] == wg)].sort_values("ep")
                if sub.empty:
                    continue
                lw = 2.5 if wg == wg_ref else 1.0
                alpha = 0.9 if wg == wg_ref else 0.35
                lbl = f"{sty['label']} (N_WG={wg})" if wg == wg_ref else None
                ax.plot(sub["ep"].values, sub[metric].values,
                        color=sty["color"], marker=sty["marker"],
                        ls=sty["ls"], lw=lw, alpha=alpha,
                        ms=7 if wg == wg_ref else 4, zorder=4 if wg == wg_ref else 2,
                        label=lbl)

        ax.set_xlabel("EP (Expert Parallelism)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ep_vals)
        ax.set_xticklabels([str(int(e)) for e in ep_vals])
        ax.legend(fontsize=8.5, loc="upper left", ncol=1)
        ax.grid(True, alpha=0.3)

    savefig(fig, f"A_latency_vs_ep_wg{wg_ref}.png", outdir)


# ─────────────────────────────────────────────────────────────────────────────
# Plot B: BW sensitivity — TPOT vs intra BW
# ─────────────────────────────────────────────────────────────────────────────
def plot_B(df, outdir):
    """TPOT vs intra BW for each panel, EP-colored lines."""
    glass_panels = [p for p in ["4x4", "6x6_4c", "6x6"] if p in df["panel"].values]
    n_panels = len(glass_panels)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6), sharey=True)
    if n_panels == 1:
        axes = [axes]
    fig.suptitle("TPOT Sensitivity to Intra-Panel BW — Glass FB Panels\n"
                 "(NVL72 reference shown as horizontal dashed line)",
                 fontsize=12, fontweight="bold")

    ep_vals = sorted(df["ep"].dropna().unique())
    cmap = plt.get_cmap("viridis")
    ep_colors = {ep: cmap(i / max(1, len(ep_vals) - 1)) for i, ep in enumerate(ep_vals)}

    for ax, panel in zip(axes, glass_panels):
        sty = PANEL_STYLES[panel]
        ax.set_title(f"{sty['label']}", fontsize=11, color=sty["color"], fontweight="bold")

        panel_eps = sorted(df[df["panel"] == panel]["ep"].dropna().unique())
        for ep in panel_eps:
            sub = df[(df["panel"] == panel) & (df["ep"] == ep)].sort_values("intra_opt_bw")
            if sub.empty or len(sub) < 2:
                continue
            nvl = nvl_val(df, ep, "tpot_ms")
            color = ep_colors.get(ep, "gray")
            ax.plot(sub["intra_opt_bw"].values, sub["tpot_ms"].values,
                    marker="o", lw=2.0, color=color, ms=6,
                    label=f"EP={int(ep)}")
            if nvl is not None:
                ax.axhline(nvl, color=PANEL_STYLES["nvl72"]["color"],
                           ls="--", lw=1.2, alpha=0.5)
                ax.text(sub["intra_opt_bw"].min(), nvl * 1.02,
                        f"NVL72 EP={int(ep)}: {nvl:.2f}ms",
                        fontsize=6.5, color=PANEL_STYLES["nvl72"]["color"], alpha=0.8)

        ax.set_xlabel(f"Intra-panel BW (GB/s)\n= N_WG × {int(WG_BW)} GB/s", fontsize=10)
        ax.set_ylabel("TPOT (ms)", fontsize=10)
        ax.legend(fontsize=7.5, loc="upper right", ncol=2)
        ax.grid(True, alpha=0.3)

    savefig(fig, "B_tpot_vs_bw.png", outdir)


# ─────────────────────────────────────────────────────────────────────────────
# Plot C: Breakeven analysis
# ─────────────────────────────────────────────────────────────────────────────
def plot_C(df, outdir):
    """Min WG count to match NVL72 TPOT for each panel × EP."""
    glass_panels = [p for p in ["4x4", "6x6_4c", "6x6"] if p in df["panel"].values]
    if not glass_panels:
        return

    ep_vals = sorted(df["ep"].dropna().unique())
    wg_vals = sorted(df["wg_count"].dropna().unique())

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Breakeven N_WG to Match NVL72 TPOT\n"
                 "(lower = glass panel is more competitive)",
                 fontsize=12, fontweight="bold")

    x_pos = np.arange(len(ep_vals))
    width = 0.25
    offsets = np.linspace(-width, width, len(glass_panels))

    for offset, panel in zip(offsets, glass_panels):
        sty = PANEL_STYLES[panel]
        be_vals = []
        for ep in ep_vals:
            nvl = nvl_val(df, ep, "tpot_ms")
            sub = df[(df["panel"] == panel) & (df["ep"] == ep)].sort_values("wg_count")
            if sub.empty or nvl is None:
                be_vals.append(None)
                continue
            wg = sub["wg_count"].values
            tpot = sub["tpot_ms"].values
            be = breakeven_wg(list(wg), list(tpot), nvl)
            be_vals.append(be)

        xs = []
        ys = []
        for i, (ep, be) in enumerate(zip(ep_vals, be_vals)):
            if be is not None:
                xs.append(x_pos[i] + offset)
                ys.append(be)

        if xs:
            ax.bar(xs, ys, width=width * 0.85, color=sty["color"], alpha=0.8,
                   label=sty["label"])
            for x, y in zip(xs, ys):
                ax.text(x, y + 0.05, f"{y:.1f}", ha="center", va="bottom",
                        fontsize=7, color=sty["color"], fontweight="bold")

    # shade "better than NVL72 with WG=8" region
    ax.axhline(max(wg_vals), color="gray", ls="--", lw=1.2, alpha=0.6,
               label=f"Max tested WG={max(wg_vals)}")
    ax.axhline(min(wg_vals), color="gray", ls=":", lw=1.0, alpha=0.4,
               label=f"Min tested WG={min(wg_vals)}")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"EP={int(ep)}" for ep in ep_vals], fontsize=9)
    ax.set_ylabel("Breakeven N_WG (WG count to match NVL72 TPOT)", fontsize=10)
    ax.set_ylim(0, max(wg_vals) + 1.5)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    ax.text(0.5, 0.97,
            "Below dashed line = glass panel can match NVL72 within tested BW range\n"
            "Values shown = interpolated breakeven WG count",
            transform=ax.transAxes, ha="center", va="top", fontsize=8,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    savefig(fig, "C_breakeven_wg.png", outdir)


# ─────────────────────────────────────────────────────────────────────────────
# Plot D: TPOT heatmap — all 4 topologies
# ─────────────────────────────────────────────────────────────────────────────
def plot_D(df, wg_ref, outdir):
    """TPOT heatmap for all 4 topologies at fixed WG."""
    panels_ordered = ["nvl72", "4x4", "6x6_4c", "6x6"]
    panels_in_data = [p for p in panels_ordered if p in df["panel"].values]

    ep_vals = sorted(df["ep"].dropna().unique())
    n_panels = len(panels_in_data)

    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 5.5))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(f"TPOT Heatmap — All 4 Topologies (N_WG={wg_ref})\n"
                 "Qwen3-30B-A3B · 128 experts · H100 · 2 requests",
                 fontsize=12, fontweight="bold")

    # common color scale
    all_tpot = []
    for p in panels_in_data:
        if p == "nvl72":
            sub = df[df["panel"] == p]
        else:
            sub = df[(df["panel"] == p) & (df["wg_count"] == wg_ref)]
        all_tpot.extend(sub["tpot_ms"].dropna().tolist())
    vmin = min(all_tpot) * 0.95 if all_tpot else 0
    vmax = max(all_tpot) * 1.05 if all_tpot else 1

    for ax, panel in zip(axes, panels_in_data):
        sty = PANEL_STYLES[panel]
        if panel == "nvl72":
            sub = df[df["panel"] == panel]
            wg_vals_p = [None]  # single WG dimension for NVL72
        else:
            sub = df[(df["panel"] == panel) & (df["wg_count"] == wg_ref)]
            wg_vals_p = [wg_ref]

        # build EP × WG (here: NVL72 has no WG dim, use EP-only)
        ep_here = sorted(sub["ep"].dropna().unique())
        mat = np.full((1, len(ep_here)), np.nan)
        for j, ep in enumerate(ep_here):
            row = sub[sub["ep"] == ep]
            if not row.empty:
                mat[0, j] = row["tpot_ms"].mean()

        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r",
                       vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(f"{sty['label']}", fontsize=10, color=sty["color"],
                     fontweight="bold")
        ax.set_xticks(range(len(ep_here)))
        ax.set_xticklabels([str(int(e)) for e in ep_here], fontsize=8)
        ax.set_yticks([])
        ax.set_xlabel("EP degree", fontsize=9)

        for j in range(len(ep_here)):
            if not np.isnan(mat[0, j]):
                ax.text(j, 0, f"{mat[0, j]:.2f}", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if mat[0, j] > (vmin + vmax) / 2 else "black")

        plt.colorbar(im, ax=ax, label="TPOT (ms)", shrink=0.7)

    if panel != "nvl72":
        fig.text(0.5, 0.01,
                 f"Glass FB panels shown at N_WG={wg_ref} ({wg_ref*int(WG_BW)} GB/s intra)  |  "
                 f"NVL72: all-to-all NVLink (1800 GB/s)",
                 ha="center", fontsize=8.5, style="italic")

    plt.tight_layout()
    savefig(fig, f"D_tpot_heatmap_wg{wg_ref}.png", outdir)


# ─────────────────────────────────────────────────────────────────────────────
# Plot E: Best-case comparison (best WG per topology/EP)
# ─────────────────────────────────────────────────────────────────────────────
def plot_E(df, outdir):
    """Best achievable TPOT per EP for each topology (min over WG)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Best-Case TPOT & TTFT: Optimal BW per EP\n"
                 "(Glass FB = minimum TPOT over all tested N_WG values)",
                 fontsize=12, fontweight="bold")

    ep_vals = sorted(df["ep"].dropna().unique())

    for ax, metric, ylabel in [(ax1, "tpot_ms", "TPOT (ms)"), (ax2, "ttft_ms", "TTFT (ms)")]:
        # NVL72
        nvl_ep, nvl_y = [], []
        for ep in ep_vals:
            v = nvl_val(df, ep, metric)
            if v is not None:
                nvl_ep.append(ep); nvl_y.append(v)
        ax.plot(nvl_ep, nvl_y, color=PANEL_STYLES["nvl72"]["color"],
                marker="o", ls="-", lw=2.5, ms=9, zorder=6, label="NVL72 baseline")

        for panel in ["4x4", "6x6_4c", "6x6"]:
            sty = PANEL_STYLES[panel]
            eps, best_y = [], []
            for ep in ep_vals:
                sub = df[(df["panel"] == panel) & (df["ep"] == ep)]
                if sub.empty:
                    continue
                best = sub[metric].min()
                best_wg = sub.loc[sub[metric].idxmin(), "wg_count"]
                eps.append(ep)
                best_y.append(best)

            if eps:
                ax.plot(eps, best_y, color=sty["color"], marker=sty["marker"],
                        ls=sty["ls"], lw=2.5, ms=8, label=f"{sty['label']} (best WG)")

        ax.set_xlabel("EP degree", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ep_vals)
        ax.set_xticklabels([str(int(e)) for e in ep_vals])
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)

    savefig(fig, "E_best_case_comparison.png", outdir)


# ─────────────────────────────────────────────────────────────────────────────
# Plot F: Combined 4-metric panel at fixed WG
# ─────────────────────────────────────────────────────────────────────────────
def plot_F(df, wg_ref, outdir):
    """2×2 grid: TTFT, TPOT, Latency, Speedup-over-NVL72 at fixed WG."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(f"Serving Metrics: 4 Topology Candidates at N_WG={wg_ref}\n"
                 f"Qwen3-30B-A3B (128 MoE experts) · H100 · EP sweep",
                 fontsize=13, fontweight="bold")

    ep_vals = sorted(df["ep"].dropna().unique())
    panels_ordered = [("nvl72", None), ("4x4", wg_ref), ("6x6_4c", wg_ref), ("6x6", wg_ref)]

    def get_series(panel, wg, metric):
        if panel == "nvl72":
            sub = df[df["panel"] == panel].sort_values("ep")
        else:
            sub = df[(df["panel"] == panel) & (df["wg_count"] == wg)].sort_values("ep")
        if sub.empty:
            return [], []
        return sub["ep"].tolist(), sub[metric].tolist()

    plot_configs = [
        (axes[0, 0], "ttft_ms", "TTFT (ms)", "Time to First Token"),
        (axes[0, 1], "tpot_ms", "TPOT (ms)", "Time per Output Token"),
        (axes[1, 0], "lat_ms",  "Latency (ms)", "End-to-End Latency"),
    ]

    for ax, metric, ylabel, title in plot_configs:
        ax.set_title(title, fontsize=11, fontweight="bold")
        for panel, wg in panels_ordered:
            sty = PANEL_STYLES[panel]
            eps, vals = get_series(panel, wg, metric)
            if eps:
                lbl = sty["label"] if panel == "nvl72" else f"{sty['label']} N_WG={wg}"
                ax.plot(eps, vals, color=sty["color"], marker=sty["marker"],
                        ls=sty["ls"], lw=2.2, ms=7, label=lbl)
        ax.set_xlabel("EP degree", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ep_vals)
        ax.set_xticklabels([str(int(e)) for e in ep_vals], fontsize=8)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    # Speedup panel
    ax = axes[1, 1]
    ax.set_title("TPOT Speedup over NVL72", fontsize=11, fontweight="bold")
    ax.axhline(1.0, color=PANEL_STYLES["nvl72"]["color"], ls="--",
               lw=2, label="NVL72 = 1.0×")
    ax.axhline(1.1, color="gray", ls=":", lw=1, alpha=0.5)
    ax.text(ep_vals[-1] * 1.02, 1.1, "+10%", fontsize=7, color="gray", va="center")

    for panel, wg in panels_ordered:
        if panel == "nvl72":
            continue
        sty = PANEL_STYLES[panel]
        sub = df[(df["panel"] == panel) & (df["wg_count"] == wg)].sort_values("ep")
        if sub.empty:
            continue
        eps, speedups = [], []
        for _, row in sub.iterrows():
            nvl = nvl_val(df, row["ep"], "tpot_ms")
            if nvl:
                eps.append(row["ep"])
                speedups.append(nvl / row["tpot_ms"])
        if eps:
            ax.plot(eps, speedups, color=sty["color"], marker=sty["marker"],
                    ls=sty["ls"], lw=2.2, ms=7,
                    label=f"{sty['label']} N_WG={wg}")

    ax.set_xlabel("EP degree", fontsize=10)
    ax.set_ylabel("Speedup (NVL72 TPOT / Glass TPOT)", fontsize=10)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ep_vals)
    ax.set_xticklabels([str(int(e)) for e in ep_vals], fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.text(0.5, 0.01,
             f"Glass FB panels: intra BW = {wg_ref}×128 = {wg_ref*int(WG_BW)} GB/s / 300 ns  |  "
             f"inter BW = 512 GB/s / 5000 ns  |  NVL72: 1800 GB/s / 1000 ns",
             ha="center", fontsize=8.5, style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    savefig(fig, f"F_combined_metrics_wg{wg_ref}.png", outdir)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="DSE results comparison plots")
    parser.add_argument("--results",  default="outputs/dse_128gpu_4x4_vs_6x6.csv")
    parser.add_argument("--out-dir",  default=OUTDIR)
    parser.add_argument("--wg-ref",   type=int, default=6,
                        help="Reference WG count for fixed-BW plots (default: 6)")
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Results not found: {args.results}")
        sys.exit(1)

    df = load(args.results)
    print(f"Loaded {len(df)} ok rows")
    print(f"Panels  : {sorted(df['panel'].unique())}")
    print(f"EP      : {sorted(df['ep'].dropna().unique())}")
    print(f"WG count: {sorted(df['wg_count'].dropna().unique())}")

    plot_A(df, args.wg_ref, args.out_dir)
    plot_B(df, args.out_dir)
    plot_C(df, args.out_dir)
    plot_D(df, args.wg_ref, args.out_dir)
    plot_E(df, args.out_dir)
    plot_F(df, args.wg_ref, args.out_dir)

    print(f"\nAll DSE result figures → {args.out_dir}/")


if __name__ == "__main__":
    main()
