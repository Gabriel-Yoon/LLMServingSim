"""
Quick preview plot from existing dse_results.csv (6 valid runs).

Reads:  outputs/dse_results.csv  (old DSE quick sweep, no NVL72 baseline)
Saves:  outputs/preview_plots/   (dpi=300)

Figures:
  01_tpot_vs_ep.png     — TPOT vs EP for 4×4 and 6×6-4c
  02_ttft_vs_ep.png     — TTFT vs EP
  03_tpot_ratio.png     — TPOT(6×6-4c) / TPOT(4×4) ratio vs EP (>1 = 4×4 wins)
  04_lat_vs_ep.png      — end-to-end latency vs EP
  05_speedup_6x6_4c.png — 6×6-4c speedup over 4×4 (TPOT, <1 = 6×6-4c faster)
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SRC  = "outputs/dse_results.csv"
OUT  = "outputs/preview_plots"
DPI  = 300

STYLES = {
    "4x4":     {"color": "tab:green",  "marker": "^", "label": "4×4 FB panel",    "ls": "-"},
    "6x6_4c":  {"color": "tab:blue",   "marker": "s", "label": "6×6-4c FB panel", "ls": "-"},
}

NOTE = "Params: elec=1800 GB/s, intra=400 GB/s, inter=200 GB/s (old DSE sweep)"


def load(path):
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    for col in ["ep", "tpot_ms", "ttft_ms", "lat_ms"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["ep", "tpot_ms"])


def savefig(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")


def plot_metric(df, col, ylabel, title, fname):
    ep_vals = sorted(df["ep"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(title, fontweight="bold", fontsize=12)
    for panel, sty in STYLES.items():
        sub = df[df["panel"] == panel].sort_values("ep")
        if sub.empty:
            continue
        ax.plot(sub["ep"].values, sub[col].values,
                color=sty["color"], marker=sty["marker"], ls=sty["ls"],
                linewidth=2.5, label=sty["label"], markersize=9)
    ax.set_xlabel("EP (Expert Parallelism degree)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ep_vals)
    ax.set_xticklabels([str(int(e)) for e in ep_vals])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.99, 0.02, NOTE, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, style="italic",
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))
    savefig(fig, fname)


def plot_ratio(df):
    """TPOT ratio: 6×6-4c / 4×4 at each EP."""
    s6 = df[df["panel"] == "6x6_4c"].set_index("ep")["tpot_ms"]
    s4 = df[df["panel"] == "4x4"].set_index("ep")["tpot_ms"]
    common = sorted(set(s6.index) & set(s4.index))
    if not common:
        print("No common EP values for ratio plot, skipping.")
        return
    ratios = [s6[ep] / s4[ep] for ep in common]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("TPOT ratio: 6×6-4c / 4×4 (< 1 = 6×6-4c faster)", fontweight="bold", fontsize=12)
    ax.plot(common, ratios, color="tab:purple", marker="D", ls="-", linewidth=2.5,
            markersize=9, label="TPOT(6×6-4c) / TPOT(4×4)")
    ax.axhline(1.0, color="gray", ls="--", lw=1.5, label="Equal performance")
    for ep, r in zip(common, ratios):
        ax.annotate(f"{r:.3f}", (ep, r), textcoords="offset points",
                    xytext=(5, 5), fontsize=9)
    ax.set_xlabel("EP degree", fontsize=11)
    ax.set_ylabel("TPOT ratio (6×6-4c / 4×4)", fontsize=11)
    ax.set_xscale("log", base=2)
    ax.set_xticks(common)
    ax.set_xticklabels([str(int(e)) for e in common])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.99, 0.02, NOTE, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, style="italic",
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))
    savefig(fig, "03_tpot_ratio.png")


def plot_speedup(df):
    """6×6-4c speedup over 4×4 (TPOT): > 1 = 6×6-4c is faster."""
    s6 = df[df["panel"] == "6x6_4c"].set_index("ep")["tpot_ms"]
    s4 = df[df["panel"] == "4x4"].set_index("ep")["tpot_ms"]
    common = sorted(set(s6.index) & set(s4.index))
    if not common:
        return
    speedups = [s4[ep] / s6[ep] for ep in common]  # higher = 6×6-4c faster
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("6×6-4c TPOT Speedup over 4×4 (> 1 = 6×6-4c faster)",
                 fontweight="bold", fontsize=12)
    ax.plot(common, speedups, color="tab:blue", marker="s", ls="-",
            linewidth=2.5, markersize=9, label="TPOT(4×4) / TPOT(6×6-4c)")
    ax.axhline(1.0, color="gray", ls="--", lw=1.5, label="Equal performance")
    for ep, sp in zip(common, speedups):
        ax.annotate(f"{sp:.3f}×", (ep, sp), textcoords="offset points",
                    xytext=(5, 3), fontsize=9)
    ax.set_xlabel("EP degree", fontsize=11)
    ax.set_ylabel("Speedup (higher = 6×6-4c faster)", fontsize=11)
    ax.set_xscale("log", base=2)
    ax.set_xticks(common)
    ax.set_xticklabels([str(int(e)) for e in common])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.99, 0.02, NOTE, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, style="italic",
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))
    savefig(fig, "05_speedup_6x6_4c.png")


def main():
    if not os.path.exists(SRC):
        print(f"Not found: {SRC}"); sys.exit(1)
    df = load(SRC)
    print(f"Loaded {len(df)} rows: panels={list(df['panel'].unique())}, "
          f"EP={sorted(df['ep'].unique())}")

    plot_metric(df, "tpot_ms", "TPOT (ms)", "TPOT vs EP", "01_tpot_vs_ep.png")
    plot_metric(df, "ttft_ms", "TTFT (ms)", "TTFT vs EP", "02_ttft_vs_ep.png")
    plot_ratio(df)
    plot_metric(df, "lat_ms",  "E2E Latency (ms)", "End-to-end Latency vs EP", "04_lat_vs_ep.png")
    plot_speedup(df)
    print(f"\nAll figures → {OUT}/")


if __name__ == "__main__":
    main()
