#!/usr/bin/env python3
"""
Topology HEADLINE figure (ASP-DAC 2027) — 2 panels, one story:

  LEFT  (ANALYTICAL): traffic-weighted average hop count vs scale (N=16..256).
        FB is diameter-optimal (->~1.9) and scale-FLAT; Torus/Mesh grow ~sqrt(N),
        Ring ~N/4. This is the differentiator the congestion-unaware simulator
        cannot resolve for the low-diameter cluster (direct point-to-point sends
        make their hop-count an O(ns) latency term, dwarfed by the bandwidth term).
  RIGHT (MEASURED): iso-per-link prefill exposed communication vs EP. The sim
        cleanly resolves the ENDPOINT — Ring's explicit store-and-forward collective
        explodes (->90%) while the low-diameter topologies cluster — VALIDATING the
        ordering's tail and leaving the middle to the analytical panel.

Together: FB sits at the diameter-optimal, glass-realizable corner; Ring is the
cautionary low-radix opposite. Run:
  python scripts/plot_topo_headline.py \
    --measured outputs/panel_dse/topo_prefill_deepseek_v3_0324_ep*_isobw_512.csv \
    --out outputs/paper_figures/fig_topo_headline.png
"""
import argparse, csv, glob, importlib.util, os
from collections import defaultdict

# import avg_hops() from topo_project (single source of truth for the hop model)
_spec = importlib.util.spec_from_file_location(
    "tp", os.path.join(os.path.dirname(__file__), "topo_project.py"))
_tp = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_tp)

# fabric key -> (label, color, marker)
STYLE = {
    "fb":        ("FB (glass)",        "tab:green",  "o"),
    "dragonfly": ("Dragonfly (glass)", "tab:purple", "D"),
    "torus":     ("Torus (glass)",     "tab:blue",   "s"),
    "mesh":      ("Mesh (glass)",      "tab:cyan",   "^"),
    "ring":      ("Ring (glass)",      "tab:orange", "v"),
}
ORDER = ["fb", "dragonfly", "torus", "mesh", "ring"]


def load_measured(paths, ycol="prefill_exposed_frac"):
    data = defaultdict(dict)   # fabric -> {ep -> value}
    for pat in paths:
        for p in sorted(glob.glob(pat)):
            for r in csv.DictReader(open(p)):
                if r.get("status") != "ok":
                    continue
                fab = (r.get("fabric") or "").strip()
                if fab not in STYLE:
                    continue
                try:
                    data[fab][int(r["ep"])] = float(r[ycol]) * 100.0
                except (ValueError, TypeError, KeyError):
                    continue
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measured", nargs="+", required=True,
                    help="iso-per-link prefill CSVs (topo_prefill_*_isobw_*.csv)")
    ap.add_argument("--n-list", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    ap.add_argument("--model-label", default="DeepSeek-V3")
    ap.add_argument("--out", default="outputs/paper_figures/fig_topo_headline.png")
    args = ap.parse_args()

    measured = load_measured(args.measured)
    if not measured:
        raise SystemExit(f"no measured rows in {args.measured}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    # LEFT — analytical avg hops vs N
    for fab in ORDER:
        label, color, marker = STYLE[fab]
        ys = [_tp.avg_hops(fab, n) for n in args.n_list]
        axL.plot(args.n_list, ys, marker=marker, color=color, lw=2.2, ms=7, label=label)
    axL.set_xscale("log", base=2); axL.set_yscale("log")
    axL.set_xticks(args.n_list); axL.set_xticklabels([str(n) for n in args.n_list])
    axL.set_xlabel("GPUs (EP)"); axL.set_ylabel("average hop count")
    axL.set_title("(a) Analytical avg-hop vs scale\n(FB diameter-optimal, scale-flat; Ring ~N/4)")
    axL.grid(True, which="both", alpha=0.3); axL.legend(fontsize=9)
    fb256 = _tp.avg_hops("fb", args.n_list[-1]); rg256 = _tp.avg_hops("ring", args.n_list[-1])
    axL.annotate(f"{rg256/fb256:.0f}x  (Ring / FB)", xy=(args.n_list[-1], rg256),
                 xytext=(args.n_list[-1] * 0.4, rg256), fontsize=10, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="black"))

    # RIGHT — measured iso-per-link prefill exposed vs EP
    for fab in ORDER:
        if fab not in measured:
            continue
        label, color, marker = STYLE[fab]
        eps = sorted(measured[fab]); ys = [measured[fab][e] for e in eps]
        axR.plot(eps, ys, marker=marker, color=color, lw=2.2, ms=7, label=label)
    axR.set_xscale("log", base=2)
    eps_all = sorted({e for f in measured for e in measured[f]})
    axR.set_xticks(eps_all); axR.set_xticklabels([str(e) for e in eps_all])
    axR.set_xlabel("GPUs (EP)"); axR.set_ylabel("prefill exposed communication (%)")
    axR.set_title("(b) Measured iso-per-link prefill exposed\n(Ring's store-and-forward explodes; rest cluster)")
    axR.grid(True, which="both", alpha=0.3); axR.legend(fontsize=9)

    fig.suptitle(f"Glass-FB topology: diameter-optimal and glass-realizable ({args.model_label})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.abspath(args.out); os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140); print(f"wrote {out}")
    # summary
    print("  analytical avg-hops @N=256:", " ".join(f"{f}={_tp.avg_hops(f, args.n_list[-1]):.2f}" for f in ORDER))
    for fab in ORDER:
        if fab in measured:
            print(f"  measured {fab:<10}", " ".join(f"EP{e}={measured[fab][e]:.1f}%" for e in sorted(measured[fab])))


if __name__ == "__main__":
    main()
