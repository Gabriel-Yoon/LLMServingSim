#!/usr/bin/env python3
"""
Step-time breakdown (the conventional iteration-time figure): each bar is one
simulated step split into COMPUTE (attention + GEMM + expert) and COMMUNICATION
(the exposed MoE all-to-all). This is the standard way "exposed communication"
appears in ML-systems papers -- as a segment of the step, not a bare %% curve.

For a step of tpot_gt_ms with exposed fraction ef:
    comm    = tpot_gt_ms * ef
    compute = tpot_gt_ms * (1 - ef)

Group the bars by --x (ep, per_device_batch, or label), optionally splitting glass
vs NVL72 side by side. Feed decode and/or prefill sweep CSVs; the honest reading is
that DECODE is compute-bound (thin comm segment) while PREFILL / large batch grows
the comm segment.

Run:
  python scripts/plot_step_breakdown.py --x per_device_batch outputs/panel_dse/expB_deepseek_v3_decode_ep128_kvfix.csv
  python scripts/plot_step_breakdown.py --x ep outputs/panel_dse/reach_deepseek_h100consistent.csv
"""
import argparse, csv, glob, os
from collections import defaultdict


def load(paths, xkey, phase):
    # phase-aware columns: decode -> (tpot_gt_ms, exposed_frac);
    #                      prefill -> (prefill_step_ms, prefill_exposed_frac)
    tcol, ecol = (("prefill_step_ms", "prefill_exposed_frac") if phase == "prefill"
                  else ("tpot_gt_ms", "exposed_frac"))
    rows = {}
    xs = set()
    for pat in paths:
        for p in sorted(glob.glob(pat)):
            for r in csv.DictReader(open(p)):
                if r.get("status") != "ok":
                    continue
                fab = "glass" if (r.get("fabric") or "").startswith(("glass", "fb")) or r.get("fabric") in (
                    "fb", "mesh", "torus", "ring", "dragonfly") else "nvl72"
                try:
                    xv = int(float(r[xkey])); tp = float(r.get(tcol) or 0)
                    ef = float(r.get(ecol) or 0)
                except (ValueError, TypeError, KeyError):
                    continue
                if tp <= 0:
                    continue
                rows[(fab, xv)] = (tp, max(0.0, min(1.0, ef)))
                xs.add(xv)
    return rows, sorted(xs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--x", default="per_device_batch", help="x-axis column (ep / per_device_batch / max_tokens)")
    ap.add_argument("--phase", choices=["decode", "prefill"], default="decode",
                    help="decode uses tpot_gt_ms/exposed_frac; prefill uses prefill_step_ms/prefill_exposed_frac")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows, xs = load(args.csvs, args.x, args.phase)
    if not xs:
        raise SystemExit(f"no ok rows with column {args.x!r} in {args.csvs}")
    fabrics = [f for f in ("glass", "nvl72") if any((f, x) in rows for x in xs)]

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(8, 5))
    n = len(fabrics); w = 0.8 / max(n, 1)
    xi = np.arange(len(xs))
    COMPUTE = "tab:blue"; COMM = "tab:orange"
    for k, fab in enumerate(fabrics):
        comp = [rows.get((fab, x), (0, 0))[0] * (1 - rows.get((fab, x), (0, 0))[1]) for x in xs]
        comm = [rows.get((fab, x), (0, 0))[0] * rows.get((fab, x), (0, 0))[1] for x in xs]
        off = (k - (n - 1) / 2) * w
        ax.bar(xi + off, comp, w, color=COMPUTE, label="compute" if k == 0 else None)
        ax.bar(xi + off, comm, w, bottom=comp, color=COMM, label="communication (exposed a2a)" if k == 0 else None)
        for i, x in enumerate(xs):
            tot = comp[i] + comm[i]
            if tot > 0:
                ax.text(i + off, tot, fab, ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(xi); ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel(args.x); ax.set_ylabel(f"{args.phase} step time (ms)")
    ax.set_title(args.title or f"{args.phase.capitalize()} step-time breakdown (compute vs exposed communication) vs {args.x}")
    ax.legend(loc="upper left"); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs",
                                   "paper_figures", f"fig_step_breakdown_{args.phase}_{args.x}.png")
    out = os.path.abspath(out); os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140); print(f"wrote {out}")
    print(f"  {'x':>8}{'fab':>7}{'compute':>9}{'comm':>8}{'comm%':>7}")
    for x in xs:
        for fab in fabrics:
            if (fab, x) in rows:
                tp, ef = rows[(fab, x)]
                print(f"  {x:>8}{fab:>7}{tp*(1-ef):>9.1f}{tp*ef:>8.1f}{ef*100:>6.1f}%")


if __name__ == "__main__":
    main()
