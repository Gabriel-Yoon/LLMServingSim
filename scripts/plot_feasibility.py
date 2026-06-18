#!/usr/bin/env python3
"""
F5 — in-package feasibility: the operating waveguide count sits within the micro-bump
budget for every FB grid. Pure physical-design figure (no simulation): reuses the
single source of truth in wg_budget.py.

Per GPU the EIC<->PIC interface has a micro-bump budget -> wg_max total WG/GPU. An FB
grid of degree d = (rows-1)+(cols-1) splits that over d neighbour pairs, and each WG is
unidirectional so per-direction N_WG = (wg_max/d)/2. That per-direction cap is the
physical ceiling; we operate at the even feasible FLOOR below it (TX/RX symmetry), so
each link is realizable while still delivering its per-pair bandwidth.

Run: python scripts/plot_feasibility.py
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wg_budget import compute_budget, grid_degree, WG_BW_GBPS  # single source of truth

GRIDS = [(4, 4), (4, 8), (6, 6), (8, 8)]


def even_floor(cap):
    """Operating point: largest EVEN per-direction N_WG <= cap (TX/RX symmetry)."""
    f = int(math.floor(cap))
    return f if f % 2 == 0 else f - 1


def main():
    b = compute_budget()
    wg_max = b["wg_max"]                      # total WG/GPU (TX+RX)
    rows = []
    for r, c in GRIDS:
        d = grid_degree(r, c)
        cap_dir = (wg_max / d) / 2.0          # per-direction micro-bump cap
        op = max(2, even_floor(cap_dir))      # operating even floor (>=2: TX+RX)
        rows.append({"grid": f"{r}x{c}", "deg": d, "cap": cap_dir, "op": op,
                     "cap_bw": cap_dir * WG_BW_GBPS, "op_bw": op * WG_BW_GBPS})

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = range(len(rows)); labels = [f"{r['grid']}\n(deg {r['deg']})" for r in rows]

    # feasible region = 0..cap (shaded), cap line, operating markers
    ax.bar(x, [r["cap"] for r in rows], width=0.6, color="tab:green", alpha=0.22,
           label="micro-bump cap (feasible ceiling)")
    ax.plot(x, [r["cap"] for r in rows], "_", color="tab:green", ms=34, mew=2.5)
    ax.plot(x, [r["op"] for r in rows], "o", color="tab:blue", ms=12,
            label="operating N_WG (even floor)", zorder=5)
    for i, r in enumerate(rows):
        ax.annotate(f"{r['op']} WG/dir\n{r['op_bw']:.0f} GB/s", (i, r["op"]),
                    textcoords="offset points", xytext=(0, -34), ha="center", fontsize=8.5)
        ax.annotate(f"cap {r['cap']:.1f}", (i, r["cap"]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5,
                    color="tab:green")

    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("waveguides per direction per link")
    ax.set_ylim(0, max(r["cap"] for r in rows) * 1.35)
    ax.set_title(f"In-package feasibility — operating N_WG within micro-bump budget\n"
                 f"(wg_max={wg_max:.0f} WG/GPU, {b['per_gpu_egress']:.2f} TB/s egress = "
                 f"{b['ratio']:.1f}x H100 NVLink)")
    ax.legend(loc="upper right", fontsize=9); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "outputs", "paper_figures",
                       "fig_feasibility_wg_budget.png")
    out = os.path.abspath(out); os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")
    print(f"  {'grid':6}{'deg':>5}{'cap/dir':>9}{'op/dir':>8}{'op BW(GB/s)':>13}")
    for r in rows:
        print(f"  {r['grid']:6}{r['deg']:>5}{r['cap']:>9.1f}{r['op']:>8}{r['op_bw']:>13.0f}")


if __name__ == "__main__":
    main()
