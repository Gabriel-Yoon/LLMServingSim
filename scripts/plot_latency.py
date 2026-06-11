"""
Plot latency DSE results: intra-panel and inter-panel link latency sweep.

Reads:  outputs/dse_latency_sweep.csv  (from sweep_latency.py)
Saves:  outputs/latency_plots/  (dpi=300)

Figure A — Intra-panel latency sensitivity (Sweep A)
  x-axis: intra_lat (ns, log scale)
  y-axis: TPOT (ms)
  Lines : EP=8, EP=16 (single 4×4 panel), EP=32 (crosses panel boundary)
  Per-EP dashed NVL72 reference.  Vertical annotation at 1000 ns = NVL72 latency.

Figure B — Inter-panel latency sensitivity (Sweep B)
  x-axis: EP (32, 64, 128) — number of panels increases as EP grows
  y-axis: TPOT (ms)
  Lines : inter_lat = 500, 1000, 2000, 3000, 5000 ns  +  NVL72 reference

Figure C — TPOT speedup over NVL72 vs intra_lat
  x-axis: intra_lat (ns, log scale)
  y-axis: speedup = NVL72_TPOT / FB_TPOT  (>1.0 means FB is faster)
  Lines : EP=8, 16, 32
  Annotates the breakeven intra_lat for each EP line.

Figure D — TPOT overhead from inter-panel latency (relative to inter_lat=500 ns)
  x-axis: inter_lat (ns)
  y-axis: TPOT increase (%) vs the 500 ns reference case
  Lines : EP=32 (2 panels), EP=64 (4 panels), EP=128 (8 panels)
  Shows "2× inter_lat → X% TPOT increase" directly.

Usage:
  python scripts/plot_latency.py [--results CSV] [--out-dir DIR]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTDIR = "outputs/latency_plots"

# These match sweep_latency.py constants (used for annotations only)
NVL72_LAT_REF   = 1000.0   # ns NVL72 link latency
INTER_LAT_REF   = 2000.0   # ns fixed during Sweep A
INTRA_LAT_REF   = 300.0    # ns fixed during Sweep B
INTRA_BW_REF    = 512.0    # GB/s

EP_COLORS = {
    8:   "tab:blue",
    16:  "tab:orange",
    32:  "tab:green",
    64:  "tab:purple",
    128: "tab:brown",
}

INTER_LAT_STYLE = {
    500:   {"ls": "-",           "marker": "o"},
    1000:  {"ls": "--",          "marker": "s"},
    2000:  {"ls": "-.",          "marker": "^"},
    3000:  {"ls": ":",           "marker": "D"},
    5000:  {"ls": (0, (5, 2)),   "marker": "v"},
}


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

def load(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "ok"].copy()
    for col in ["ep", "intra_lat", "inter_lat", "intra_bw", "inter_bw",
                "tpot_ms", "ttft_ms", "lat_ms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def nvl72_tpot(df, ep):
    rows = df[(df["topology"] == "nvl72") & (df["ep"] == ep)]
    return rows["tpot_ms"].mean() if not rows.empty else None


def savefig(fig, name, outdir):
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────
# Figure A: TPOT vs intra_lat, lines = EP
# ─────────────────────────────────────────────────────────────

def plot_figure_A(df, outdir):
    sub = df[df["sweep"] == "A"].copy()
    if sub.empty:
        print("No Sweep A data — skipping Figure A")
        return

    ep_vals  = sorted(sub["ep"].dropna().unique().astype(int))
    ilat_vals = sorted(sub["intra_lat"].dropna().unique())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(
        "Figure A — Intra-panel Link Latency Sensitivity\n"
        "4×4 FB Panel  |  intra_bw=512 GB/s  |  inter_lat=2000 ns (fixed)",
        fontweight="bold", fontsize=12)

    for ep in ep_vals:
        color = EP_COLORS.get(ep, "gray")
        ep_sub = sub[sub["ep"] == ep].sort_values("intra_lat")
        if ep_sub.empty:
            continue
        panel_label = "single panel" if ep <= 16 else "2 panels"
        ax.plot(ep_sub["intra_lat"], ep_sub["tpot_ms"],
                color=color, marker="o", linewidth=2.5, markersize=8,
                label=f"4×4 FB  EP={ep}  ({panel_label})")
        # NVL72 horizontal reference for this EP
        nvl = nvl72_tpot(df, ep)
        if nvl is not None:
            ax.axhline(nvl, color=color, ls="--", lw=1.5, alpha=0.55)
            ax.text(max(ilat_vals) * 1.01, nvl,
                    f" NVL72\n EP={ep}", color=color, fontsize=7.5, va="center")

    # NVL72 latency vertical reference
    ax.axvline(NVL72_LAT_REF, color="tab:red", ls=":", lw=2, alpha=0.75)
    y_bot = ax.get_ylim()[0]
    ax.text(NVL72_LAT_REF * 1.03, y_bot,
            "← NVL72\n   link lat\n   (1000 ns)",
            color="tab:red", fontsize=8.5, va="bottom")

    ax.set_xlabel("Intra-panel link latency (ns)", fontsize=11)
    ax.set_ylabel("TPOT (ms)", fontsize=11)
    ax.set_xscale("log")
    ax.set_xticks(ilat_vals)
    ax.set_xticklabels([str(int(v)) for v in ilat_vals])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    savefig(fig, "latency_A_intra_sensitivity.png", outdir)


# ─────────────────────────────────────────────────────────────
# Figure B: TPOT vs EP, lines = inter_lat
# ─────────────────────────────────────────────────────────────

def plot_figure_B(df, outdir):
    sub = df[df["sweep"] == "B"].copy()
    if sub.empty:
        print("No Sweep B data — skipping Figure B")
        return

    ep_vals   = sorted(sub["ep"].dropna().unique().astype(int))
    xlat_vals = sorted(sub["inter_lat"].dropna().unique().astype(int))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title(
        "Figure B — Inter-panel Link Latency Sensitivity\n"
        "4×4 FB Panel  |  intra_bw=512 GB/s  |  intra_lat=300 ns (fixed)",
        fontweight="bold", fontsize=12)

    for xlat in xlat_vals:
        sty = INTER_LAT_STYLE.get(xlat, {"ls": "-", "marker": "x"})
        xsub = sub[sub["inter_lat"] == xlat].sort_values("ep")
        if xsub.empty:
            continue
        ax.plot(xsub["ep"], xsub["tpot_ms"],
                ls=sty["ls"], marker=sty["marker"], linewidth=2.5, markersize=8,
                label=f"inter_lat = {xlat} ns")

    # NVL72 reference
    nvl_eps, nvl_tpots = [], []
    for ep in ep_vals:
        v = nvl72_tpot(df, ep)
        if v is not None:
            nvl_eps.append(ep); nvl_tpots.append(v)
    if nvl_eps:
        ax.plot(nvl_eps, nvl_tpots, color="tab:red", marker="o", ls="-",
                linewidth=2.5, markersize=9, label="NVL72 baseline", zorder=5)

    # Panel count annotations
    ylim_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 5
    for ep, n_panels in [(32, 2), (64, 4), (128, 8)]:
        ax.axvline(ep, color="gray", ls="--", lw=1, alpha=0.4)
        ax.text(ep, ylim_top * 0.97, f" {n_panels}\n panels", color="gray",
                fontsize=8, ha="left", va="top")

    ax.set_xlabel("EP (Expert Parallelism degree)", fontsize=11)
    ax.set_ylabel("TPOT (ms)", fontsize=11)
    ax.set_xticks(ep_vals)
    ax.set_xticklabels([str(e) for e in ep_vals])
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    savefig(fig, "latency_B_inter_sensitivity.png", outdir)


# ─────────────────────────────────────────────────────────────
# Figure C: speedup over NVL72 vs intra_lat
# ─────────────────────────────────────────────────────────────

def plot_figure_C(df, outdir):
    sub = df[df["sweep"] == "A"].copy()
    if sub.empty:
        print("No Sweep A data — skipping Figure C")
        return

    ep_vals   = sorted(sub["ep"].dropna().unique().astype(int))
    ilat_vals = sorted(sub["intra_lat"].dropna().unique())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(
        "Figure C — FB Panel Speedup over NVL72 vs Intra-panel Latency\n"
        "4×4 FB Panel  |  intra_bw=512 GB/s  |  inter_lat=2000 ns (fixed)",
        fontweight="bold", fontsize=12)

    # Shaded region where FB wins
    ax.fill_between([min(ilat_vals), max(ilat_vals)], [1.0, 1.0], [2.0, 2.0],
                    alpha=0.06, color="green")
    ax.axhline(1.0, color="tab:red", ls="--", lw=2, label="NVL72 parity (1.0×)", zorder=3)
    ax.axhline(1.1, color="gray",    ls=":",  lw=1, alpha=0.5, label="+10% faster than NVL72")
    ax.text(min(ilat_vals) * 1.05, 1.03, "FB faster →", color="green",
            fontsize=8.5, style="italic")
    ax.text(min(ilat_vals) * 1.05, 0.97, "NVL72 faster →", color="tab:red",
            fontsize=8.5, style="italic", va="top")

    for ep in ep_vals:
        color = EP_COLORS.get(ep, "gray")
        nvl = nvl72_tpot(df, ep)
        if nvl is None:
            continue
        ep_sub = sub[sub["ep"] == ep].sort_values("intra_lat")
        if ep_sub.empty:
            continue
        lats     = ep_sub["intra_lat"].values
        speedups = nvl / ep_sub["tpot_ms"].values
        panel_label = "single panel" if ep <= 16 else "2 panels"
        ax.plot(lats, speedups, color=color, marker="o", linewidth=2.5, markersize=8,
                label=f"EP={ep} ({panel_label})")

        # Breakeven annotation
        for i in range(len(speedups) - 1):
            hi, lo = max(speedups[i], speedups[i+1]), min(speedups[i], speedups[i+1])
            if lo <= 1.0 <= hi:
                be = float(np.interp(1.0,
                                     sorted([speedups[i], speedups[i+1]]),
                                     sorted([lats[i], lats[i+1]])))
                ax.annotate(f"breakeven\n{be:.0f} ns",
                            xy=(be, 1.0), xytext=(be * 1.15, 1.0 + 0.06),
                            color=color, fontsize=8.5,
                            arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
                break

    ax.axvline(NVL72_LAT_REF, color="tab:red", ls=":", lw=2, alpha=0.75)
    ax.text(NVL72_LAT_REF * 1.03, 1.0,
            "← NVL72\n   lat (1000 ns)", color="tab:red", fontsize=8.5, va="center")

    ax.set_xlabel("Intra-panel link latency (ns)", fontsize=11)
    ax.set_ylabel("TPOT speedup over NVL72  (higher = FB faster)", fontsize=11)
    ax.set_xscale("log")
    ax.set_xticks(ilat_vals)
    ax.set_xticklabels([str(int(v)) for v in ilat_vals])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    savefig(fig, "latency_C_speedup_over_nvl72.png", outdir)


# ─────────────────────────────────────────────────────────────
# Figure D: TPOT overhead (%) from inter_lat vs 500 ns reference
# ─────────────────────────────────────────────────────────────

def plot_figure_D(df, outdir):
    sub = df[df["sweep"] == "B"].copy()
    if sub.empty:
        print("No Sweep B data — skipping Figure D")
        return

    ep_vals   = sorted(sub["ep"].dropna().unique().astype(int))
    xlat_vals = sorted(sub["inter_lat"].dropna().unique().astype(int))
    base_xlat = xlat_vals[0]   # 500 ns reference

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title(
        f"Figure D — TPOT Overhead from Inter-panel Latency\n"
        f"Relative to {base_xlat} ns baseline  |  4×4 FB  |  intra_lat=300 ns (fixed)",
        fontweight="bold", fontsize=12)

    ax.axhline(0, color="gray", ls="-", lw=1, alpha=0.5, label=f"Baseline ({base_xlat} ns)")

    for ep in ep_vals:
        color = EP_COLORS.get(ep, "gray")
        base_rows = sub[(sub["ep"] == ep) & (sub["inter_lat"] == base_xlat)]
        if base_rows.empty:
            continue
        base_tpot = base_rows["tpot_ms"].mean()
        xlats, overheads = [], []
        for xlat in xlat_vals:
            rows = sub[(sub["ep"] == ep) & (sub["inter_lat"] == xlat)]
            if not rows.empty:
                xlats.append(xlat)
                overheads.append((rows["tpot_ms"].mean() / base_tpot - 1.0) * 100.0)
        n_panels = ep // 16
        ax.plot(xlats, overheads, color=color, marker="o", linewidth=2.5, markersize=8,
                label=f"EP={ep} ({n_panels} panels)")

        # Annotate overhead at worst inter_lat
        if xlats:
            ax.annotate(f"{overheads[-1]:+.1f}%",
                        xy=(xlats[-1], overheads[-1]),
                        xytext=(xlats[-1] * 0.93, overheads[-1] + 0.5),
                        color=color, fontsize=8.5,
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.0))

    ax.axvline(NVL72_LAT_REF, color="tab:red", ls=":", lw=2, alpha=0.7)
    ax.text(NVL72_LAT_REF * 1.02, 0,
            " NVL72 link\n lat (1000 ns)", color="tab:red", fontsize=8.5, va="bottom")

    ax.set_xlabel("Inter-panel link latency (ns)", fontsize=11)
    ax.set_ylabel(f"TPOT overhead vs {base_xlat} ns baseline (%)", fontsize=11)
    ax.set_xticks(xlat_vals)
    ax.set_xticklabels([str(v) for v in xlat_vals])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    savefig(fig, "latency_D_inter_overhead.png", outdir)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot latency DSE sweep results")
    parser.add_argument("--results",  default="results/exp_latency_fb4x4_intra_inter_sweep.csv")
    parser.add_argument("--out-dir",  default=OUTDIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if not os.path.exists(args.results):
        print(f"Results not found: {args.results}")
        sys.exit(1)

    df = load(args.results)
    print(f"Loaded {len(df)} successful runs from {args.results}")
    if "sweep" in df.columns:
        print(f"Sweeps   : {sorted(df['sweep'].dropna().unique())}")
    print(f"EP values: {sorted(df['ep'].dropna().unique().astype(int))}")
    if "intra_lat" in df.columns:
        a_lats = sorted(df[df.get("sweep", pd.Series()) == "A"]["intra_lat"].dropna().unique().astype(int))
        if a_lats:
            print(f"Sweep A intra_lat: {a_lats} ns")
    if "inter_lat" in df.columns:
        b_lats = sorted(df[df.get("sweep", pd.Series()) == "B"]["inter_lat"].dropna().unique().astype(int))
        if b_lats:
            print(f"Sweep B inter_lat: {b_lats} ns")
    print()

    plot_figure_A(df, args.out_dir)
    plot_figure_B(df, args.out_dir)
    plot_figure_C(df, args.out_dir)
    plot_figure_D(df, args.out_dir)

    print(f"\nAll figures → {args.out_dir}/")


if __name__ == "__main__":
    main()
