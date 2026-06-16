#!/usr/bin/env python3
"""
L1 — Latency-radix figure: decode TPOT / prefill TTFT vs optical link latency.

Reads a `--sweep latency` CSV from sweep_panel_dse.py (glass-FB at several link
latencies + one NVL72 reference). At fixed BW (WG) and EP the MoE collective is
latency-bound, so this isolates the glass switch-free hop (~100 ns intra /
~500 ns inter) vs the NVL72 switched hop (1000 ns).

Uses the bug-immune tpot_gt_ms metric. Output: outputs/panel_dse/plots/.

Usage:
  python scripts/plot_latency_radix.py --csv outputs/panel_dse/dse_latency_controlled.csv
"""
import argparse, csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(REPO, "outputs", "panel_dse", "plots")


def _gt(r):
    v = r.get("tpot_gt_ms")
    if v in (None, "", "None"):
        raise SystemExit("tpot_gt_ms missing — re-run the sweep (bug-immune metric).")
    return float(v)


def load(path):
    glass, nvl = [], []
    axis = "intra"
    for r in csv.DictReader(open(path)):
        if r.get("status") != "ok":
            continue
        axis = r.get("lat_axis", axis)
        row = (float(r["link_lat"]), _gt(r), float(r.get("ttft_ms") or 0))
        (nvl if r["fabric"] == "nvl72" else glass).append(row)
    glass.sort()
    return glass, nvl, axis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--name", default="L1_latency_radix")
    args = ap.parse_args()

    glass, nvl, axis = load(args.csv)
    if not glass:
        raise SystemExit("no glass rows in CSV")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, idx, ylab, title in [
        (axes[0], 1, "decode TPOT [ms] (tpot_gt; lower=better)", "(a) Decode TPOT vs link latency"),
        (axes[1], 2, "prefill TTFT [ms] (lower=better)", "(b) Prefill TTFT vs link latency")]:
        xs = [g[0] for g in glass]
        ys = [g[idx] for g in glass]
        ax.plot(xs, ys, marker="o", color="#2ca02c", lw=2.2, label="Glass FB (optical hop)")
        # glass operating points
        for lab, lat in [("intra ~100ns", 100), ("inter ~500ns", 500)]:
            if any(abs(x - lat) < 1 for x in xs):
                yy = ys[[abs(x - lat) < 1 for x in xs].index(True)]
                ax.annotate(lab, xy=(lat, yy), xytext=(4, 6), textcoords="offset points",
                            fontsize=8, color="#2ca02c")
        # NVL72 reference (its switched hop = 1000ns)
        if nvl:
            ny = nvl[0][idx]
            ax.axhline(ny, color="#d62728", ls="--", lw=1.8,
                       label=f"NVL72 ({nvl[0][0]:.0f}ns switch hop)")
            ax.scatter([nvl[0][0]], [ny], color="#d62728", zorder=5, s=40)
        ax.set_xlabel(f"optical {axis}-panel link latency [ns]")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle(f"Latency-radix ({axis}): glass switch-free hop vs NVL72 1000ns switch hop",
                 fontsize=11)
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUT, args.name + ".png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    print(f"{'lat[ns]':>8} {'TPOT[ms]':>10} {'TTFT[ms]':>10}")
    for x, t, ft in glass:
        print(f"{x:8.0f} {t:10.2f} {ft:10.1f}")
    if nvl:
        print(f"NVL72 @{nvl[0][0]:.0f}ns: TPOT={nvl[0][1]:.2f}  TTFT={nvl[0][2]:.1f}")


if __name__ == "__main__":
    main()
