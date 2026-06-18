#!/usr/bin/env python3
"""
F1 — collective (all-to-all) latency vs scale: the PRIMARY metric.

Decode TPOT hides the fabric (comm is ~13% of a compute-bound step, F2), so the
discriminating metric is the collective communication itself. TWO figures, TWO metrics
(this one script, two invocations):

  Fig A -- TOPOLOGY PARETO: `--metric exposed` on a `--sweep topo_compare` CSV.
    Measured exposed%% (ASTRA models hops/diameter) separates the glass topologies at
    equal WG budget: FB < Torus ~ Mesh < Ring (FB near fully-connected, Ring pays its
    diameter). e.g. EP32: FB 17.8 / Torus 25.1 / Mesh 25.6 / Ring 43.0 %.
  Fig B -- TECHNOLOGY GAP / IB CLIFF: `--metric a2a` on a `--sweep epscale`/reach CSV.
    Analytical all-to-all latency (msg / link-bw) shows glass-FB vs NVL72: ~10x at the
    EP where NVL72 crosses IB (e.g. EP128: NVL72 298 us vs FB 29 us).

NOTE: `a2a` is a fixed-bandwidth ANALYTICAL companion -- it does NOT differentiate
topology (all glass topologies coincide), so use it only for Fig B (FB vs NVL72). Use
`exposed` (measured) for the topology comparison (Fig A). Rows keyed by `fabric`
(fb/mesh/torus/ring/dragonfly/glass_fb/nvl72).

Run:
  python scripts/plot_collective.py --metric exposed outputs/panel_dse/topo_compare_*.csv   # Fig A
  python scripts/plot_collective.py --metric a2a outputs/panel_dse/reach_deepseek_v3*.csv     # Fig B
"""
import argparse, csv, glob, os
from collections import defaultdict

# fabric key -> (legend label, color, marker, is_baseline)
STYLE = {
    "fb":        ("FB (glass)",        "tab:green",  "o", False),
    "glass_fb":  ("FB (glass)",        "tab:green",  "o", False),
    "dragonfly": ("Dragonfly (glass)", "tab:purple", "D", False),
    "torus":     ("Torus (glass)",     "tab:blue",   "s", False),
    "mesh":      ("Mesh (glass)",      "tab:cyan",   "^", False),
    "ring":      ("Ring (glass)",      "tab:orange", "v", False),
    "nvl72":     ("NVL72 (NVLink+IB)", "tab:red",    "x", True),
}
METRICS = {
    "a2a":     ("all_to_all_us",  "collective all-to-all latency (us)", True),   # log-y
    "exposed": ("exposed_frac",   "decode exposed communication (%)",   False),
    "tpot":    ("tpot_gt_ms",     "decode TPOT (ms)",                    False),
    # PREFILL phase (phase-aware parser) — diameter is most exposed in prefill
    "prefill_exposed": ("prefill_exposed_frac", "prefill exposed communication (%)", False),
    "prefill_tpot":    ("prefill_step_ms",       "prefill step time (ms)",            False),
}
_PCT = {"exposed", "prefill_exposed"}   # metrics reported as a percentage


def load(paths, ycol):
    # fabric -> {ep -> [values]} (dedupe by mean; e.g. multiple batches at same ep)
    data = defaultdict(lambda: defaultdict(list))
    for pat in paths:
        for p in sorted(glob.glob(pat)):
            for r in csv.DictReader(open(p)):
                if r.get("status") != "ok":
                    continue
                fab = (r.get("fabric") or "").strip()
                if fab not in STYLE:
                    continue
                try:
                    ep = int(r["ep"]); y = float(r.get(ycol) or 0)
                except (ValueError, TypeError):
                    continue
                if y <= 0 and ycol not in ("exposed_frac", "prefill_exposed_frac"):
                    continue
                data[fab][ep].append(y)
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--metric", choices=list(METRICS), default="a2a")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ycol, ylabel, logy = METRICS[args.metric]
    scale = 100.0 if args.metric in _PCT else 1.0

    data = load(args.csvs, ycol)
    if not data:
        raise SystemExit(f"no rows with column '{ycol}' and a known fabric in {args.csvs}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5))
    fb_curve, nvl_curve = {}, {}
    for fab in sorted(data, key=lambda f: STYLE[f][3]):   # baselines last (on top)
        label, color, marker, is_base = STYLE[fab]
        eps = sorted(data[fab])
        ys = [sum(data[fab][e]) / len(data[fab][e]) * scale for e in eps]
        ax.plot(eps, ys, marker=marker, color=color, lw=2.2 if not is_base else 2.0,
                ls="--" if is_base else "-", ms=7,
                label=label + (" — baseline" if is_base else ""))
        if fab in ("fb", "glass_fb"):
            fb_curve = dict(zip(eps, ys))
        if fab == "nvl72":
            nvl_curve = dict(zip(eps, ys))

    # annotate FB vs NVL72 gap at the largest shared EP
    shared = sorted(set(fb_curve) & set(nvl_curve))
    if shared and args.metric not in _PCT:
        e = shared[-1]; r = nvl_curve[e] / fb_curve[e] if fb_curve[e] else 0
        if r > 1:
            ax.annotate(f"{r:.0f}x  (NVL72 / FB)", xy=(e, nvl_curve[e]),
                        xytext=(e * 0.55, nvl_curve[e]), fontsize=10, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="black"))

    ax.set_xscale("log", base=2)
    eps_all = sorted({e for f in data for e in data[f]})
    ax.set_xticks(eps_all); ax.set_xticklabels([str(e) for e in eps_all])
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("GPUs (EP)"); ax.set_ylabel(ylabel)
    default_title = ("Topology Pareto — exposed communication vs scale\n"
                     "(FB near fully-connected; Ring pays its diameter)" if args.metric in _PCT
                     else "Collective all-to-all latency vs scale — glass-FB vs NVL72\n"
                          "(optical vs IB: the inter-domain bandwidth gap)")
    ax.set_title(args.title or default_title)
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                   "outputs", "panel_dse", f"F1_collective_{args.metric}.png")
    out = os.path.abspath(out); os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")
    # text summary
    for fab in sorted(data, key=lambda f: STYLE[f][3]):
        eps = sorted(data[fab])
        print(f"  {STYLE[fab][0]:<22}", " ".join(f"EP{e}={sum(data[fab][e])/len(data[fab][e])*scale:.1f}" for e in eps))
    if shared and args.metric not in _PCT:
        e = shared[-1]
        print(f"  FB vs NVL72 @EP{e}: {nvl_curve[e]/fb_curve[e]:.1f}x")


if __name__ == "__main__":
    main()
