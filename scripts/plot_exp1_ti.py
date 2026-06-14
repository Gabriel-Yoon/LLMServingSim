"""
Plot Exp 1 (ASP-DAC 2027) Throughput-Interactivity curve: FB 4x4 vs NVL72-like.

Reads:  results/paper_exp1_ti.csv  (written by scripts/sweep_paper.py --exp 1)
Output: outputs/paper_sweep/plots/  (NOT comparison_plots or dse_plots)
          exp1_ti_tradeoff.png   throughput vs interactivity (the T-I curve)
          exp1_tpot_vs_n.png     TPOT vs decode batch size N
          exp1_throughput_vs_n.png

Usage:
  python scripts/plot_exp1_ti.py
  python scripts/plot_exp1_ti.py --csv results/paper_exp1_ti.csv
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(REPO_ROOT, "results", "paper_exp1_ti.csv")
OUT_DIR     = os.path.join(REPO_ROOT, "outputs", "paper_sweep", "plots")

TPOT_SLO_MS = 15.0
SLO_INTER   = 1000.0 / TPOT_SLO_MS  # tokens/s/user

# Keyed on the topology names emitted by sweep_paper.py exp 1.
TOPO_STYLE = {
    "nvl72_lat0_ep32": dict(color="#1f77b4", marker="o",
                            label="NVL72-like (NVLink, lat=0)"),
    "fb_4x4_ep32":     dict(color="#d62728", marker="s",
                            label="FB 4x4 (optical panel)"),
}

TITLE_SUFFIX = "EP=32  |  Qwen3-30B-A3B  |  H100  |  ISL/OSL 512/256"


def load_data(csv_path):
    data = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            topo = row["topology"]
            d = data.setdefault(topo, {"n": [], "tpot": [],
                                       "throughput": [], "interactivity": []})
            d["n"].append(int(row["n_requests"]))
            d["tpot"].append(float(row["tpot_avg_ms"]))
            d["throughput"].append(float(row["throughput_toks_per_s"]))
            d["interactivity"].append(float(row["interactivity_toks_per_s_per_user"]))
    # sort each series by N
    for topo in data:
        idx = sorted(range(len(data[topo]["n"])), key=lambda i: data[topo]["n"][i])
        for k in data[topo]:
            data[topo][k] = [data[topo][k][i] for i in idx]
    return data


def _ordered(data):
    # stable, known topologies first for consistent legend ordering
    keys = [k for k in TOPO_STYLE if k in data] + \
           [k for k in data if k not in TOPO_STYLE]
    return [(k, data[k]) for k in keys]


def plot_ti_tradeoff(data, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for topo, d in _ordered(data):
        s = TOPO_STYLE.get(topo, {})
        ax.plot(d["interactivity"], d["throughput"],
                color=s.get("color", "gray"), marker=s.get("marker", "o"),
                linewidth=2, markersize=7, label=s.get("label", topo))
        for inter, tput, n in zip(d["interactivity"], d["throughput"], d["n"]):
            ax.annotate(f"N={n}", (inter, tput), textcoords="offset points",
                        xytext=(4, 4), fontsize=7,
                        color=s.get("color", "gray"), alpha=0.8)

    ax.axvline(x=SLO_INTER, color="black", linestyle="--", linewidth=1.5,
               label=f"TPOT SLO = {TPOT_SLO_MS:.0f} ms ({SLO_INTER:.0f} tok/s/user)")
    ax.set_xlabel("Interactivity [tokens/s/user]  (higher = better)", fontsize=11)
    ax.set_ylabel("Throughput [tokens/s]  (higher = better)", fontsize=11)
    ax.set_title(f"Throughput-Interactivity Tradeoff\n{TITLE_SUFFIX}", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    return _save(fig, out_dir, "exp1_ti_tradeoff.png")


def plot_tpot_vs_n(data, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for topo, d in _ordered(data):
        s = TOPO_STYLE.get(topo, {})
        ax.plot(d["n"], d["tpot"], color=s.get("color", "gray"),
                marker=s.get("marker", "o"), linewidth=2, markersize=7,
                label=s.get("label", topo))
    ax.axhline(y=TPOT_SLO_MS, color="black", linestyle="--", linewidth=1.5,
               label=f"TPOT SLO = {TPOT_SLO_MS:.0f} ms")
    ax.set_xlabel("Decode batch size N  (concurrent requests)", fontsize=11)
    ax.set_ylabel("TPOT [ms]  (lower = better)", fontsize=11)
    ax.set_title(f"TPOT vs Batch Size\n{TITLE_SUFFIX}", fontsize=11)
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    return _save(fig, out_dir, "exp1_tpot_vs_n.png")


def plot_throughput_vs_n(data, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for topo, d in _ordered(data):
        s = TOPO_STYLE.get(topo, {})
        ax.plot(d["n"], d["throughput"], color=s.get("color", "gray"),
                marker=s.get("marker", "o"), linewidth=2, markersize=7,
                label=s.get("label", topo))
    ax.set_xlabel("Decode batch size N  (concurrent requests)", fontsize=11)
    ax.set_ylabel("Throughput [tokens/s]  (higher = better)", fontsize=11)
    ax.set_title(f"Throughput vs Batch Size\n{TITLE_SUFFIX}", fontsize=11)
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    return _save(fig, out_dir, "exp1_throughput_vs_n.png")


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
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

    print(f"Loaded {sum(len(d['n']) for d in data.values())} points "
          f"from {len(data)} topologies.\n")
    print(f"{'topology':20s} {'N':>4}  {'TPOT(ms)':>9}  {'Tput(tok/s)':>12}  "
          f"{'Inter(t/s/u)':>13}  SLO")
    print("-" * 70)
    for topo, d in _ordered(data):
        for n, tpot, tput, inter in zip(d["n"], d["tpot"], d["throughput"], d["interactivity"]):
            slo = "ok " if tpot <= TPOT_SLO_MS else "OVER"
            print(f"{topo:20s} {n:4d}  {tpot:9.2f}  {tput:12.0f}  {inter:13.1f}  {slo}")
    print()

    plot_ti_tradeoff(data, args.out_dir)
    plot_tpot_vs_n(data, args.out_dir)
    plot_throughput_vs_n(data, args.out_dir)


if __name__ == "__main__":
    main()
