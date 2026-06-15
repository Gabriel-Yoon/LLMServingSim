"""
Plot the batch x EP network-exposure sweep (sweep_panel_dse.py --sweep batch_x_ep).

Answers "when does the decode all-to-all leave the compute shadow?" via:
  (a) exposed_frac vs per-device batch, one line per EP, per fabric, with the
      crossover batch B*(EP) where exposure passes a threshold;
  (b) the P6 exposure heatmap (x=batch, y=EP, color=exposed_frac) per fabric;
  (c) a mechanism panel: all_to_all_us (raw comm) vs weight_load_us (the
      compute that hides it) — analytical companions to the logged exposed.

Outputs go next to the CSV under outputs/panel_dse/batch_exposure/plots/ plus a
bstar_table.csv. Reads the logged exposed_frac (ground truth from ASTRA), not the
analytical estimate.

Usage (inside Docker, from /app/LLMServingSim):
  python scripts/plot_batch_exposure.py \
    --csv outputs/panel_dse/batch_exposure/dse_batch_x_ep.csv
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FABRIC_COLOR = {"glass_fb": "#2ca02c", "nvl72": "#d62728"}
FABRIC_LABEL = {"glass_fb": "glass FB (optical)", "nvl72": "NVL72 (IB)"}


def load(csv_path):
    rows = []
    for r in csv.DictReader(open(csv_path)):
        if r.get("status") != "ok":
            continue
        try:
            rows.append({
                "fabric": r["fabric"], "ep": int(r["ep"]),
                "batch": int(r["per_device_batch"]),
                "exposed_frac": float(r["exposed_frac"] or 0),
                # tpot_gt_ms (MODE of ASTRA per-iteration decode cycles) is the
                # DP+EP-safe metric; tpot_steady_ms/tpot_avg_ms are distorted by
                # the add_done dummy-completion bug. (exposed_frac above is read
                # straight from ASTRA and is already bug-immune.)
                "tpot": float(r.get("tpot_gt_ms") or 0),
                "ttft": float(r["ttft_ms"] or 0),
                "a2a_us": float(r.get("all_to_all_us") or 0),
                "wl_us": float(r.get("weight_load_us") or 0),
            })
        except (KeyError, ValueError):
            continue
    return rows


def bstar_table(rows, threshold):
    """B*(EP, fabric) = smallest batch whose exposed_frac >= threshold.
    'already' = exposed already over threshold at the smallest profiled batch;
    'never'  = never crosses within the swept range."""
    by = defaultdict(list)
    for r in rows:
        by[(r["fabric"], r["ep"])].append((r["batch"], r["exposed_frac"]))
    out = []
    for (fab, ep), pts in sorted(by.items()):
        pts.sort()
        bstar, over_at_min = None, pts[0][1] >= threshold
        for b, e in pts:
            if e >= threshold:
                bstar = b
                break
        out.append({"fabric": fab, "ep": ep, "threshold": threshold,
                    "bstar": ("already" if over_at_min else (bstar if bstar else "never")),
                    "min_batch": pts[0][0], "exposed_at_min": round(pts[0][1], 4),
                    "exposed_at_max": round(pts[-1][1], 4)})
    return out


def plot_exposed_vs_batch(rows, out_path, threshold):
    eps = sorted({r["ep"] for r in rows})
    fabrics = sorted({r["fabric"] for r in rows})
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(eps)))
    fig, axes = plt.subplots(1, len(fabrics), figsize=(6 * len(fabrics), 4.6), squeeze=False)
    for ax, fab in zip(axes[0], fabrics):
        for ci, ep in enumerate(eps):
            pts = sorted((r["batch"], r["exposed_frac"]) for r in rows
                         if r["fabric"] == fab and r["ep"] == ep)
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, [y * 100 for y in ys], "o-", color=cmap[ci], label=f"EP={ep}")
        ax.axhline(threshold * 100, ls="--", c="gray", lw=1,
                   label=f"B* threshold {threshold*100:.0f}%")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("per-device batch")
        ax.set_ylabel("exposed communication / total (%)")
        ax.set_title(FABRIC_LABEL.get(fab, fab))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Network exposure vs decode batch (logged ASTRA exposed comm)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_heatmap(rows, out_path):
    eps = sorted({r["ep"] for r in rows})
    batches = sorted({r["batch"] for r in rows})
    fabrics = sorted({r["fabric"] for r in rows})
    fig, axes = plt.subplots(1, len(fabrics), figsize=(5.5 * len(fabrics), 4.6), squeeze=False)
    vmax = max((r["exposed_frac"] for r in rows), default=1) * 100
    for ax, fab in zip(axes[0], fabrics):
        grid = np.full((len(eps), len(batches)), np.nan)
        for r in rows:
            if r["fabric"] != fab:
                continue
            grid[eps.index(r["ep"]), batches.index(r["batch"])] = r["exposed_frac"] * 100
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="magma",
                       vmin=0, vmax=vmax)
        ax.set_xticks(range(len(batches)), batches)
        ax.set_yticks(range(len(eps)), eps)
        ax.set_xlabel("per-device batch")
        ax.set_ylabel("EP")
        ax.set_title(f"{FABRIC_LABEL.get(fab, fab)}")
        for i in range(len(eps)):
            for j in range(len(batches)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center",
                            color="white" if grid[i, j] < vmax * 0.6 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, label="exposed comm (%)")
    fig.suptitle("P6 — network exposure map: when does the all-to-all surface?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_mechanism(rows, out_path):
    """Analytical raw all-to-all vs expert weight-load (the compute that hides
    it) per EP — explains WHY exposure rises with batch and EP."""
    eps = sorted({r["ep"] for r in rows})
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(eps)))
    for ci, ep in enumerate(eps):
        pts = sorted((r["batch"], r["a2a_us"], r["wl_us"]) for r in rows
                     if r["ep"] == ep and r["fabric"] == "nvl72")
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ax.plot(xs, [p[1] / 1000 for p in pts], "o-", color=cmap[ci], label=f"a2a EP={ep}")
        ax.plot(xs, [p[2] / 1000 for p in pts], "s--", color=cmap[ci], alpha=0.5,
                label=f"wload EP={ep}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("per-device batch")
    ax.set_ylabel("per-iteration time (ms, analytical)")
    ax.set_title("Mechanism: raw all-to-all (—) vs expert weight-load (- -), NVL72 IB")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--threshold", type=float, default=0.15,
                    help="exposed_frac threshold defining B* (default 0.15 = 15%)")
    args = ap.parse_args()

    rows = load(args.csv)
    if not rows:
        print("no ok rows in", args.csv)
        return
    out_dir = os.path.join(os.path.dirname(args.csv), "plots")
    os.makedirs(out_dir, exist_ok=True)

    plot_exposed_vs_batch(rows, os.path.join(out_dir, "exposed_vs_batch.png"), args.threshold)
    plot_heatmap(rows, os.path.join(out_dir, "p6_exposure_heatmap.png"))
    plot_mechanism(rows, os.path.join(out_dir, "mechanism_a2a_vs_weightload.png"))

    tbl = bstar_table(rows, args.threshold)
    bstar_csv = os.path.join(os.path.dirname(args.csv), "bstar_table.csv")
    with open(bstar_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tbl[0].keys()))
        w.writeheader()
        w.writerows(tbl)

    print(f"plots -> {out_dir}")
    print(f"B* table -> {bstar_csv}\n")
    print(f"{'fabric':10s} {'EP':>4s} {'B*(>=%.0f%%)' % (args.threshold*100):>12s} "
          f"{'exp@min':>8s} {'exp@max':>8s}")
    for r in tbl:
        print(f"{r['fabric']:10s} {r['ep']:>4d} {str(r['bstar']):>12s} "
              f"{r['exposed_at_min']*100:>7.1f}% {r['exposed_at_max']*100:>7.1f}%")


if __name__ == "__main__":
    main()
