"""
Plot Throughput vs Interactivity for FB-4x4 vs NVL72.

Reads: results/exp_tpot_throughput_ep32.csv
Output: outputs/tpot_plots/tpot_throughput_ep32.png  (NOT overwriting comparison_plots or dse_plots)

Usage:
  python scripts/plot_tpot.py
  python scripts/plot_tpot.py --csv results/exp_tpot_throughput_ep32.csv
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(REPO_ROOT, "results", "exp_tpot_throughput_ep32.csv")
OUT_DIR     = os.path.join(REPO_ROOT, "outputs", "tpot_plots")

TPOT_SLO_MS = 15.0
SLO_INTER   = 1000.0 / TPOT_SLO_MS  # tokens/s/user

TOPO_STYLE = {
    "nvl72": dict(color="#1f77b4", marker="o", label="NVL72 (NVLink 1800 GB/s)"),
    "fb":    dict(color="#d62728", marker="s", label="FB 4×4 (Optical 512 GB/s)"),
}


def load_data(csv_path):
    data = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            topo = row["topology"]
            if topo not in data:
                data[topo] = {"n": [], "tpot": [], "throughput": [], "interactivity": []}
            data[topo]["n"].append(int(row["n_requests"]))
            data[topo]["tpot"].append(float(row["tpot_avg_ms"]))
            data[topo]["throughput"].append(float(row["throughput_toks_per_s"]))
            data[topo]["interactivity"].append(float(row["interactivity_toks_per_s_per_user"]))
    # sort by n
    for topo in data:
        idx = sorted(range(len(data[topo]["n"])), key=lambda i: data[topo]["n"][i])
        for k in data[topo]:
            data[topo][k] = [data[topo][k][i] for i in idx]
    return data


def plot_throughput_vs_interactivity(data, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))

    for topo, d in data.items():
        style = TOPO_STYLE.get(topo, {})
        ax.plot(d["interactivity"], d["throughput"],
                color=style.get("color", "gray"),
                marker=style.get("marker", "o"),
                linewidth=2, markersize=7,
                label=style.get("label", topo))
        # annotate batch size at each point
        for inter, tput, n in zip(d["interactivity"], d["throughput"], d["n"]):
            ax.annotate(f"N={n}", (inter, tput),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color=style.get("color", "gray"), alpha=0.8)

    # SLO vertical line
    ax.axvline(x=SLO_INTER, color="black", linestyle="--", linewidth=1.5,
               label=f"TPOT SLO = {TPOT_SLO_MS} ms\n({SLO_INTER:.0f} tok/s/user)")
    ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1e6],
                     0, SLO_INTER, alpha=0.05, color="red", label="_nolegend_")

    ax.set_xlabel("Interactivity [tokens/s/user]  (higher = better)", fontsize=11)
    ax.set_ylabel("Throughput [tokens/s]  (higher = better)", fontsize=11)
    ax.set_title("EP=32  Throughput–Interactivity Tradeoff\nFB 4×4 vs NVL72  |  Qwen3-30B-A3B  |  H100",
                 fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tpot_throughput_ep32.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_tpot_vs_n(data, out_dir):
    """Secondary plot: TPOT vs batch size N with SLO line."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for topo, d in data.items():
        style = TOPO_STYLE.get(topo, {})
        ax.plot(d["n"], d["tpot"],
                color=style.get("color", "gray"),
                marker=style.get("marker", "o"),
                linewidth=2, markersize=7,
                label=style.get("label", topo))

    ax.axhline(y=TPOT_SLO_MS, color="black", linestyle="--", linewidth=1.5,
               label=f"TPOT SLO = {TPOT_SLO_MS} ms")
    ax.fill_between(ax.get_xlim() if ax.get_xlim()[1] > 0 else [0, 128],
                    TPOT_SLO_MS, ax.get_ylim()[1] if ax.get_ylim()[1] > TPOT_SLO_MS else TPOT_SLO_MS * 3,
                    alpha=0.07, color="red", label="_nolegend_")

    ax.set_xlabel("Decode Batch Size N  (concurrent requests)", fontsize=11)
    ax.set_ylabel("TPOT [ms]  (lower = better)", fontsize=11)
    ax.set_title("EP=32  TPOT vs Batch Size\nFB 4×4 vs NVL72  |  Qwen3-30B-A3B  |  H100",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tpot_vs_n_ep32.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_throughput_vs_n(data, out_dir):
    """Secondary plot: Throughput vs N."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for topo, d in data.items():
        style = TOPO_STYLE.get(topo, {})
        ax.plot(d["n"], d["throughput"],
                color=style.get("color", "gray"),
                marker=style.get("marker", "o"),
                linewidth=2, markersize=7,
                label=style.get("label", topo))

    ax.set_xlabel("Decode Batch Size N  (concurrent requests)", fontsize=11)
    ax.set_ylabel("Throughput [tokens/s]  (higher = better)", fontsize=11)
    ax.set_title("EP=32  Throughput vs Batch Size\nFB 4×4 vs NVL72  |  Qwen3-30B-A3B  |  H100",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "throughput_vs_n_ep32.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv",     default=DEFAULT_CSV)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV not found: {args.csv}")
        return

    data = load_data(args.csv)
    if not data:
        print("No 'ok' rows found in CSV.")
        return

    print(f"Loaded {sum(len(d['n']) for d in data.values())} data points "
          f"from {len(data)} topologies.")

    # Print summary table
    print(f"\n{'Label':45s}  {'N':>4}  {'TPOT':>8}  {'Throughput':>12}  {'Interactivity':>16}  SLO")
    print("-" * 100)
    for topo, d in sorted(data.items()):
        for n, tpot, tput, inter in zip(d["n"], d["tpot"], d["throughput"], d["interactivity"]):
            slo = "✅" if tpot < TPOT_SLO_MS else "❌"
            print(f"{topo+'_ep32_n'+str(n):45s}  {n:4d}  {tpot:7.2f}ms  "
                  f"{tput:11.0f}  {inter:15.1f}  {slo}")

    plot_throughput_vs_interactivity(data, args.out_dir)
    plot_tpot_vs_n(data, args.out_dir)
    plot_throughput_vs_n(data, args.out_dir)

    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
