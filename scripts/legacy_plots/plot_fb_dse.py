"""
Flattened Butterfly Glass Panel DSE — Visualization.

Reads outputs/dse_results.csv and saves high-resolution figures to outputs/dse_plots/.

Figures produced:
  01_tpot_vs_ep.png          — TPOT vs EP: panel comparison + flat baselines
  02_ttft_vs_ep.png          — TTFT vs EP: same grouping
  03_lat_vs_ep.png           — End-to-end latency vs EP
  04_tpot_heatmap_4x4.png    — TPOT heatmap (elec_bw × inter_bw) for 4×4 panel
  05_tpot_heatmap_6x6_4c.png — Same for 6×6-4c panel
  06_advantage_4x4.png       — TPOT improvement % over flat_900 baseline (4×4 panel)
  07_advantage_6x6_4c.png    — Same for 6×6-4c panel
  08_bw_sensitivity.png      — Sensitivity: TPOT vs each BW axis independently
  09_panel_comparison.png    — 4×4 vs 6×6-4c head-to-head at matched EP
  10_tpot_vs_inter_bw.png    — TPOT vs inter_bw, fixed elec/intra
  11_tpot_vs_elec_bw.png     — TPOT vs elec_bw, fixed intra/inter
  12_tpot_vs_intra_bw.png    — TPOT vs intra_opt_bw, fixed elec/inter
  13_crossover_map.png       — Crossover: which configs beat flat_900 per EP

Usage:
  python scripts/plot_fb_dse.py [--results outputs/dse_results.csv]
                                [--out-dir outputs/dse_plots]
"""

import argparse
import csv
import os
import sys
import warnings
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

DPI = 300
PANEL_LABELS = {"4x4": "4×4 (16-GPU)", "6x6_4c": "6×6−4c (32-GPU)"}
METRIC_LABELS = {"tpot_ms": "TPOT (ms/token)", "ttft_ms": "TTFT (ms)", "lat_ms": "E2E Latency (ms)"}
FLAT_COLORS   = {"flat_900": "#333333", "flat_1800": "#888888"}

# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

def _f(v):
    try:
        return float(v) if v not in ("", "None", None) else None
    except (ValueError, TypeError):
        return None

def _i(v):
    try:
        return int(float(v)) if v not in ("", "None", None) else None
    except (ValueError, TypeError):
        return None


def load(csv_path):
    fb, flat = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            tpot = _f(row.get("tpot_ms"))
            ttft = _f(row.get("ttft_ms"))
            lat  = _f(row.get("lat_ms"))
            if tpot is None:
                continue
            base = {
                "label":        row.get("label", ""),
                "ep":           _i(row.get("ep")),
                "tpot_ms":      tpot,
                "ttft_ms":      ttft,
                "lat_ms":       lat,
                "tpot_p50_ms":  _f(row.get("tpot_p50_ms")),
                "ttft_p50_ms":  _f(row.get("ttft_p50_ms")),
            }
            topo = row.get("topology", "")
            if topo == "fb":
                fb.append({**base,
                    "panel":        row.get("panel", ""),
                    "elec_bw":      _i(row.get("elec_bw")),
                    "intra_opt_bw": _i(row.get("intra_opt_bw")),
                    "inter_bw":     _i(row.get("inter_bw")),
                })
            elif topo == "flat":
                flat.append({**base,
                    "flat_bw": _i(row.get("elec_bw")),
                })
    return fb, flat


def flat_map(flat_rows):
    """(flat_bw, ep) → tpot_ms."""
    return {(r["flat_bw"], r["ep"]): r for r in flat_rows}


# ─────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────

def _ep_colors(eps):
    cmap = plt.cm.plasma
    return {ep: cmap(i / max(len(eps) - 1, 1)) for i, ep in enumerate(sorted(eps))}


def _bw_colors(bws):
    cmap = plt.cm.viridis
    bws = sorted(set(bws))
    return {b: cmap(i / max(len(bws) - 1, 1)) for i, b in enumerate(bws)}


# ─────────────────────────────────────────────────────────────
# Figure helpers
# ─────────────────────────────────────────────────────────────

def _save(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def _add_flat_hlines(ax, fmap, ep, metric="tpot_ms"):
    for bw, color in [(900, FLAT_COLORS["flat_900"]), (1800, FLAT_COLORS["flat_1800"])]:
        row = fmap.get((bw, ep))
        if row:
            ax.axhline(row[metric], ls="--", color=color, lw=1.2, alpha=0.7,
                       label=f"flat {bw} GB/s")


# ─────────────────────────────────────────────────────────────
# Plot 01/02/03 — metric vs EP
# ─────────────────────────────────────────────────────────────

def plot_metric_vs_ep(fb, flat, fmap, metric, fname, out_dir, ref_elec=1800, ref_intra=400, ref_inter=200):
    """Line plot: metric vs EP for each panel type at reference BW."""
    eps = sorted({r["ep"] for r in fb if r["ep"]})
    if not eps:
        return

    panels = sorted({r["panel"] for r in fb})
    panel_colors = {"4x4": "#1f77b4", "6x6_4c": "#ff7f0e"}
    flat_bws = sorted({r["flat_bw"] for r in flat})

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for panel in panels:
        pts = []
        for ep in eps:
            match = [r for r in fb
                     if r["panel"] == panel and r["ep"] == ep
                     and r["elec_bw"] == ref_elec
                     and r["intra_opt_bw"] == ref_intra
                     and r["inter_bw"] == ref_inter]
            if match:
                pts.append((ep, match[0][metric]))
        if pts:
            xs, ys = zip(*pts)
            lbl = PANEL_LABELS.get(panel, panel)
            ax.plot(xs, ys, marker="o", color=panel_colors.get(panel, "C0"),
                    lw=2, ms=7, label=f"FB {lbl}\n(el={ref_elec}, in={ref_intra}, ex={ref_inter} GB/s)")

    for bw, color in [(900, FLAT_COLORS["flat_900"]), (1800, FLAT_COLORS["flat_1800"])]:
        pts = sorted([(r["ep"], r[metric]) for r in flat if r["flat_bw"] == bw], key=lambda x: x[0])
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, ls="--", marker="s", color=color, lw=1.8, ms=6,
                    label=f"Flat FC {bw} GB/s (H100 NVLink×{bw//900})")

    ax.set_xlabel("Expert Parallelism (EP)", fontsize=12)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=12)
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} vs EP\nFB glass panel vs flat H100 baseline", fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xticks(eps)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, fname))


# ─────────────────────────────────────────────────────────────
# Plot 04/05 — TPOT heatmap (elec_bw × inter_bw)
# ─────────────────────────────────────────────────────────────

def plot_tpot_heatmap(fb, panel_name, ref_intra, ref_ep, metric, fname, out_dir):
    subset = [r for r in fb
              if r["panel"] == panel_name
              and r["ep"] == ref_ep
              and r["intra_opt_bw"] == ref_intra]
    if not subset:
        print(f"  [skip] no data for heatmap panel={panel_name} ep={ref_ep} intra={ref_intra}")
        return

    elec_bws  = sorted({r["elec_bw"]  for r in subset})
    inter_bws = sorted({r["inter_bw"] for r in subset})
    Z = np.full((len(inter_bws), len(elec_bws)), np.nan)
    for r in subset:
        xi = elec_bws.index(r["elec_bw"])
        yi = inter_bws.index(r["inter_bw"])
        Z[yi, xi] = r[metric]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(Z, origin="lower", aspect="auto",
                   cmap="RdYlGn_r", interpolation="nearest")
    ax.set_xticks(range(len(elec_bws)))
    ax.set_xticklabels([str(b) for b in elec_bws])
    ax.set_yticks(range(len(inter_bws)))
    ax.set_yticklabels([str(b) for b in inter_bws])
    ax.set_xlabel("Electrical BW (GB/s)", fontsize=11)
    ax.set_ylabel("Inter-panel BW (GB/s)", fontsize=11)
    ax.set_title(
        f"{METRIC_LABELS.get(metric, metric)} — {PANEL_LABELS.get(panel_name, panel_name)}\n"
        f"EP={ref_ep}, intra_opt={ref_intra} GB/s",
        fontsize=11
    )
    for yi in range(len(inter_bws)):
        for xi in range(len(elec_bws)):
            if not np.isnan(Z[yi, xi]):
                ax.text(xi, yi, f"{Z[yi, xi]:.2f}", ha="center", va="center",
                        fontsize=8, color="black")
    plt.colorbar(im, ax=ax, label=METRIC_LABELS.get(metric, metric))
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, fname))


# ─────────────────────────────────────────────────────────────
# Plot 06/07 — Advantage map (improvement % over flat_900)
# ─────────────────────────────────────────────────────────────

def plot_advantage_map(fb, fmap, panel_name, ref_intra, metric, fname, out_dir):
    subset = [r for r in fb if r["panel"] == panel_name and r["intra_opt_bw"] == ref_intra]
    if not subset:
        return

    eps       = sorted({r["ep"] for r in subset})
    inter_bws = sorted({r["inter_bw"] for r in subset})
    elec_bws  = sorted({r["elec_bw"]  for r in subset})
    ncols = len(elec_bws)
    nrows = len(eps)

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(3.5 * ncols, 2.8 * nrows), squeeze=False)
    colors_inter = _bw_colors(inter_bws)

    for ri, ep in enumerate(eps):
        baseline_row = fmap.get((900, ep))
        baseline = baseline_row[metric] if baseline_row else None
        for ci, eb in enumerate(elec_bws):
            ax = axes[ri][ci]
            pts = [(r["inter_bw"], r[metric])
                   for r in subset if r["ep"] == ep and r["elec_bw"] == eb]
            pts.sort(key=lambda x: x[0])
            if not pts or baseline is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            xs = [p[0] for p in pts]
            impr = [100 * (baseline - p[1]) / baseline for p in pts]
            bars = ax.bar(range(len(xs)), impr,
                          color=["#2ecc71" if v > 0 else "#e74c3c" for v in impr])
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels([str(x) for x in xs], fontsize=7)
            ax.set_xlabel("inter_bw (GB/s)", fontsize=7)
            ax.set_ylabel("Improvement %", fontsize=7)
            ax.set_title(f"EP={ep}, el={eb} GB/s", fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)

    panel_lbl = PANEL_LABELS.get(panel_name, panel_name)
    fig.suptitle(
        f"TPOT improvement over flat_900 — {panel_lbl}\n"
        f"intra_opt={ref_intra} GB/s  (green = FB wins)",
        fontsize=11
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, os.path.join(out_dir, fname))


# ─────────────────────────────────────────────────────────────
# Plot 08 — BW sensitivity (one axis at a time)
# ─────────────────────────────────────────────────────────────

def plot_bw_sensitivity(fb, fmap, metric, fname, out_dir,
                        ref_ep=64, ref_elec=1800, ref_intra=400, ref_inter=200):
    panels = sorted({r["panel"] for r in fb})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    sweep_axes = [
        ("elec_bw",      "Electrical BW (GB/s)",       {"intra_opt_bw": ref_intra, "inter_bw": ref_inter}),
        ("intra_opt_bw", "Optical Intra-panel BW (GB/s)", {"elec_bw": ref_elec,  "inter_bw": ref_inter}),
        ("inter_bw",     "Inter-panel BW (GB/s)",       {"elec_bw": ref_elec,  "intra_opt_bw": ref_intra}),
    ]
    panel_colors = {"4x4": "#1f77b4", "6x6_4c": "#ff7f0e"}

    for ax, (sweep_key, xlabel, fixed) in zip(axes, sweep_axes):
        for panel in panels:
            pts = []
            for r in fb:
                if r["panel"] != panel or r["ep"] != ref_ep:
                    continue
                if all(r[k] == v for k, v in fixed.items()):
                    pts.append((r[sweep_key], r[metric]))
            pts.sort(key=lambda x: x[0])
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", lw=2, ms=7,
                        color=panel_colors.get(panel, "C0"),
                        label=PANEL_LABELS.get(panel, panel))

        # flat baselines
        for bw, color in [(900, FLAT_COLORS["flat_900"]), (1800, FLAT_COLORS["flat_1800"])]:
            row = fmap.get((bw, ref_ep))
            if row:
                ax.axhline(row[metric], ls="--", color=color, lw=1.5,
                           label=f"flat {bw} GB/s")

        fixed_str = ", ".join(f"{k.split('_')[0]}={v}" for k, v in fixed.items())
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=10)
        ax.set_title(f"EP={ref_ep} | fixed: {fixed_str}", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"BW Sensitivity — {METRIC_LABELS.get(metric, metric)}", fontsize=13)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, fname))


# ─────────────────────────────────────────────────────────────
# Plot 09 — Panel comparison (4x4 vs 6x6_4c)
# ─────────────────────────────────────────────────────────────

def plot_panel_comparison(fb, fmap, metric, fname, out_dir,
                          ref_elec=1800, ref_intra=400, ref_inter=200):
    eps = sorted({r["ep"] for r in fb if r["ep"]})
    panels = ["4x4", "6x6_4c"]
    colors = {"4x4": "#1f77b4", "6x6_4c": "#ff7f0e"}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.3
    xs = np.arange(len(eps))

    for di, panel in enumerate(panels):
        ys = []
        for ep in eps:
            match = [r for r in fb
                     if r["panel"] == panel and r["ep"] == ep
                     and r["elec_bw"] == ref_elec
                     and r["intra_opt_bw"] == ref_intra
                     and r["inter_bw"] == ref_inter]
            ys.append(match[0][metric] if match else np.nan)
        offset = (di - 0.5) * width
        ax.bar(xs + offset, ys, width, label=PANEL_LABELS.get(panel, panel),
               color=colors.get(panel, f"C{di}"), alpha=0.85, edgecolor="white")

    for bw, color in [(900, FLAT_COLORS["flat_900"]), (1800, FLAT_COLORS["flat_1800"])]:
        flat_ys = [fmap.get((bw, ep), {}).get(metric) for ep in eps]
        if any(y is not None for y in flat_ys):
            flat_ys = [y if y is not None else np.nan for y in flat_ys]
            ax.plot(xs, flat_ys, ls="--", marker="D", color=color, lw=1.8, ms=7,
                    label=f"Flat FC {bw} GB/s", zorder=5)

    ax.set_xlabel("EP", fontsize=12)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=12)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(e) for e in eps])
    ax.set_title(
        f"4×4 vs 6×6−4c Panel Comparison\n"
        f"el={ref_elec}, in={ref_intra}, ex={ref_inter} GB/s",
        fontsize=11
    )
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, fname))


# ─────────────────────────────────────────────────────────────
# Plot 10/11/12 — TPOT vs single BW axis (all combos)
# ─────────────────────────────────────────────────────────────

def plot_tpot_vs_bw_axis(fb, fmap, sweep_key, xlabel, fixed_keys,
                          ref_ep, metric, fname, out_dir):
    panels = sorted({r["panel"] for r in fb})
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.5),
                              squeeze=False)

    for ci, panel in enumerate(panels):
        ax = axes[0][ci]
        subset = [r for r in fb if r["panel"] == panel and r["ep"] == ref_ep]
        if not subset:
            continue

        # Group by fixed axis values
        groups = defaultdict(list)
        for r in subset:
            key = tuple(r[k] for k in fixed_keys)
            groups[key].append(r)

        cmap = plt.cm.tab20
        for gi, (key_val, rows) in enumerate(sorted(groups.items())):
            pts = sorted([(r[sweep_key], r[metric]) for r in rows], key=lambda x: x[0])
            if not pts:
                continue
            xs, ys = zip(*pts)
            lbl = ", ".join(f"{k.split('_')[0]}={v}" for k, v in zip(fixed_keys, key_val))
            ax.plot(xs, ys, marker="o", color=cmap(gi / max(len(groups) - 1, 1)),
                    lw=1.5, ms=5, label=lbl)

        _add_flat_hlines(ax, fmap, ref_ep, metric)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=10)
        ax.set_title(f"{PANEL_LABELS.get(panel, panel)}  EP={ref_ep}", fontsize=10)
        ax.legend(fontsize=5, ncol=2, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{METRIC_LABELS.get(metric, metric)} vs {xlabel}", fontsize=12)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, fname))


# ─────────────────────────────────────────────────────────────
# Plot 13 — Crossover map (which configs beat flat_900?)
# ─────────────────────────────────────────────────────────────

def plot_crossover_map(fb, fmap, metric, fname, out_dir):
    panels = sorted({r["panel"] for r in fb})
    eps    = sorted({r["ep"]    for r in fb if r["ep"]})

    fig, axes = plt.subplots(len(eps), len(panels),
                              figsize=(5 * len(panels), 3.5 * len(eps)), squeeze=False)

    for ri, ep in enumerate(eps):
        baseline_val = fmap.get((900, ep), {}).get(metric)
        for ci, panel in enumerate(panels):
            ax = axes[ri][ci]
            subset = [r for r in fb if r["panel"] == panel and r["ep"] == ep]
            inter_bws = sorted({r["inter_bw"] for r in subset})
            elec_bws  = sorted({r["elec_bw"]  for r in subset})

            if not subset or baseline_val is None:
                ax.set_axis_off()
                continue

            # collapse intra_opt_bw by taking best (min) TPOT
            Z = np.full((len(inter_bws), len(elec_bws)), np.nan)
            for r in subset:
                xi = elec_bws.index(r["elec_bw"])
                yi = inter_bws.index(r["inter_bw"])
                cur = Z[yi, xi]
                Z[yi, xi] = r[metric] if np.isnan(cur) else min(cur, r[metric])

            # improvement over flat_900
            Zimpr = 100 * (baseline_val - Z) / baseline_val  # positive = win

            vmax = max(abs(np.nanmin(Zimpr)), abs(np.nanmax(Zimpr))) + 0.1
            im = ax.imshow(Zimpr, origin="lower", aspect="auto",
                           cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                           interpolation="nearest")
            ax.set_xticks(range(len(elec_bws)))
            ax.set_xticklabels([str(b) for b in elec_bws], fontsize=7)
            ax.set_yticks(range(len(inter_bws)))
            ax.set_yticklabels([str(b) for b in inter_bws], fontsize=7)
            for yi in range(len(inter_bws)):
                for xi in range(len(elec_bws)):
                    if not np.isnan(Zimpr[yi, xi]):
                        ax.text(xi, yi, f"{Zimpr[yi, xi]:+.1f}%",
                                ha="center", va="center", fontsize=7,
                                color="black")
            ax.set_xlabel("elec_bw (GB/s)", fontsize=8)
            ax.set_ylabel("inter_bw (GB/s)", fontsize=8)
            ax.set_title(
                f"{PANEL_LABELS.get(panel, panel)}  EP={ep}\n(best over intra_opt_bw)",
                fontsize=8
            )
            plt.colorbar(im, ax=ax, label="Improvement vs flat_900 (%)", shrink=0.8)

    fig.suptitle(
        "Crossover Map — FB improvement over flat_900 baseline\n"
        "(green = FB wins, red = flat wins)",
        fontsize=12
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, os.path.join(out_dir, fname))


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",  default="outputs/dse_results.csv")
    parser.add_argument("--out-dir",  default="outputs/dse_plots")
    parser.add_argument("--ref-ep",   type=int, default=64,
                        help="Reference EP for single-EP plots")
    parser.add_argument("--ref-elec", type=int, default=1800)
    parser.add_argument("--ref-intra",type=int, default=400)
    parser.add_argument("--ref-inter",type=int, default=200)
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Results not found: {args.results}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Loading {args.results} ...")
    fb, flat = load(args.results)
    fmap = flat_map(flat)
    print(f"  {len(fb)} FB rows, {len(flat)} flat rows (status=ok)")

    if not fb and not flat:
        print("No data to plot."); return

    ref = dict(ref_ep=args.ref_ep, ref_elec=args.ref_elec,
               ref_intra=args.ref_intra, ref_inter=args.ref_inter)
    metric = "tpot_ms"

    print("Generating figures ...")

    # 01 TPOT vs EP
    plot_metric_vs_ep(fb, flat, fmap, "tpot_ms", "01_tpot_vs_ep.png",
                      args.out_dir, ref["ref_elec"], ref["ref_intra"], ref["ref_inter"])

    # 02 TTFT vs EP
    if any(r.get("ttft_ms") for r in fb):
        plot_metric_vs_ep(fb, flat, fmap, "ttft_ms", "02_ttft_vs_ep.png",
                          args.out_dir, ref["ref_elec"], ref["ref_intra"], ref["ref_inter"])

    # 03 E2E latency vs EP
    if any(r.get("lat_ms") for r in fb):
        plot_metric_vs_ep(fb, flat, fmap, "lat_ms", "03_lat_vs_ep.png",
                          args.out_dir, ref["ref_elec"], ref["ref_intra"], ref["ref_inter"])

    # 04/05 Heatmap per panel
    for panel, fname_num in [("4x4", "04"), ("6x6_4c", "05")]:
        plot_tpot_heatmap(fb, panel, ref["ref_intra"], ref["ref_ep"], "tpot_ms",
                          f"{fname_num}_tpot_heatmap_{panel}.png", args.out_dir)

    # 06/07 Advantage map per panel
    for panel, fname_num in [("4x4", "06"), ("6x6_4c", "07")]:
        plot_advantage_map(fb, fmap, panel, ref["ref_intra"], "tpot_ms",
                           f"{fname_num}_advantage_{panel}.png", args.out_dir)

    # 08 BW sensitivity
    plot_bw_sensitivity(fb, fmap, "tpot_ms", "08_bw_sensitivity.png",
                        args.out_dir, ref["ref_ep"],
                        ref["ref_elec"], ref["ref_intra"], ref["ref_inter"])

    # 09 Panel comparison
    plot_panel_comparison(fb, fmap, "tpot_ms", "09_panel_comparison.png",
                          args.out_dir,
                          ref["ref_elec"], ref["ref_intra"], ref["ref_inter"])

    # 10 TPOT vs inter_bw (fixed elec, intra)
    plot_tpot_vs_bw_axis(
        fb, fmap, "inter_bw", "Inter-panel BW (GB/s)",
        ["elec_bw", "intra_opt_bw"], ref["ref_ep"], "tpot_ms",
        "10_tpot_vs_inter_bw.png", args.out_dir,
    )

    # 11 TPOT vs elec_bw (fixed intra, inter)
    plot_tpot_vs_bw_axis(
        fb, fmap, "elec_bw", "Electrical BW (GB/s)",
        ["intra_opt_bw", "inter_bw"], ref["ref_ep"], "tpot_ms",
        "11_tpot_vs_elec_bw.png", args.out_dir,
    )

    # 12 TPOT vs intra_opt_bw (fixed elec, inter)
    plot_tpot_vs_bw_axis(
        fb, fmap, "intra_opt_bw", "Optical Intra-panel BW (GB/s)",
        ["elec_bw", "inter_bw"], ref["ref_ep"], "tpot_ms",
        "12_tpot_vs_intra_bw.png", args.out_dir,
    )

    # 13 Crossover map
    plot_crossover_map(fb, fmap, "tpot_ms", "13_crossover_map.png", args.out_dir)

    print(f"\nAll figures saved to {args.out_dir}/  (dpi={DPI})")


if __name__ == "__main__":
    main()
