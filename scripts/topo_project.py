#!/usr/bin/env python3
"""
topo_project.py — analytical scale projection of the FB-vs-Torus/Mesh/Ring
all-to-all collective latency (ASP-DAC 2027).

WHY: a single glass panel physically caps DIRECT measurement at 16 (4x4) / 32
(4x8) GPUs. To argue about 64-256 we project with a transparent alpha-beta model
and VALIDATE it against the 16/32 ASTRA-measured points (same technique ViBE uses
for rack-scale projection). The contribution is feasibility + Pareto positioning,
not a brute-forced scaling record.

MODEL (iso-budget = same per-GPU waveguide budget, the honest cost-normalized
comparison — see notes/topology_eval_design.md):

  aggregate per-GPU BW = degree * link_bw = (wg_budget/2) * WG_BW   [SAME for all
      topologies under iso-budget — the bandwidth term is a COMMON offset, so it
      cancels in the topology comparison]
  collective all-to-all latency:
      T(topo, N) = L_hops(topo, N) * t_hop  +  M / aggregate_BW
  where the only topology-dependent differentiator is the LATENCY-hop count:
      FB              : diameter 2                      (rich row+col links)
      Mesh   (direct) : diameter (r-1)+(c-1)            (Manhattan)
      Torus  (direct) : diameter floor(r/2)+floor(c/2)  (wrap-around)
      Ring   (ring algo): N-1 nearest-neighbour steps

This is exactly the structure ASTRA's congestion-unaware send() implements
(hops*latency + chunk/bw); the projection just evaluates it where the LLM .et
graph would OOM. Measured 16/32 exposed%/TPOT (from topo_compare CSVs) confirm the
ordering and calibrate the per-hop term.

Run:
  python scripts/topo_project.py                      # table + plot, defaults
  python scripts/topo_project.py --n-list 16 32 64 128 256
  python scripts/topo_project.py --measured outputs/panel_dse/topo_compare_ep16_b128.csv \
                                            outputs/panel_dse/topo_compare_ep32_b128.csv
"""

import argparse
import csv
import math
import os

WG_BW = 128.0          # GB/s per waveguide group (unidirectional)
T_HOP_NS = 100.0       # per-hop optical CPO latency (INTRA_OPT_LAT)
WG_BUDGET = 60.0       # per-GPU WG budget (both directions), from wg_budget.py


def grid_for_n(n):
    """Most-square (rows, cols) with rows <= cols for the 2-D grids."""
    r = int(math.isqrt(n))
    while r > 1 and n % r != 0:
        r -= 1
    return r, n // r


def degree(topo, rows, cols):
    if topo == "fb":
        return (rows - 1) + (cols - 1)
    if topo in ("mesh", "torus"):
        return 4
    if topo == "ring":
        return 2
    raise ValueError(topo)


def latency_hops(topo, rows, cols):
    """Hop count that sets the collective's latency term."""
    n = rows * cols
    if topo == "fb":
        return 2                                   # <=2 hops for any pair
    if topo == "mesh":
        return (rows - 1) + (cols - 1)             # direct all-to-all diameter
    if topo == "torus":
        return rows // 2 + cols // 2               # wrap-around diameter
    if topo == "ring":
        return n - 1                               # ring algorithm: N-1 steps
    raise ValueError(topo)


def collective_latency_ns(topo, n, msg_bytes, wg_budget=WG_BUDGET, t_hop=T_HOP_NS):
    """alpha-beta all-to-all latency. aggregate_BW is iso-budget (common to all
    topologies), so only the hop term differentiates them."""
    rows, cols = (1, n) if topo == "ring" else grid_for_n(n)
    agg_bw_Bpns = (wg_budget / 2.0) * WG_BW         # GB/s == B/ns
    lat_term = latency_hops(topo, rows, cols) * t_hop
    bw_term = (msg_bytes / agg_bw_Bpns) if agg_bw_Bpns > 0 else 0.0
    return lat_term + bw_term, latency_hops(topo, rows, cols), degree(topo, rows, cols)


TOPOS = ["fb", "torus", "mesh", "ring"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-list", nargs="+", type=int, default=[16, 32, 64, 128, 256])
    ap.add_argument("--msg-kb", type=float, default=64.0,
                    help="per-GPU all-to-all message size (KB). Small = latency-bound "
                         "(topology matters most); large = bandwidth-bound (common offset grows).")
    ap.add_argument("--wg-budget", type=float, default=WG_BUDGET)
    ap.add_argument("--t-hop", type=float, default=T_HOP_NS)
    ap.add_argument("--measured", nargs="*", default=[],
                    help="topo_compare CSVs to overlay (measured exposed%% / tpot at 16/32)")
    ap.add_argument("--out", default=None, help="plot path (default outputs/panel_dse/topo_projection.png)")
    args = ap.parse_args()

    msg_bytes = args.msg_kb * 1024.0

    # ---- analytical table -----------------------------------------------------
    print(f"\nAnalytical all-to-all latency (alpha-beta, iso-budget {args.wg_budget} WG/GPU, "
          f"msg {args.msg_kb} KB, t_hop {args.t_hop} ns)")
    print(f"{'N':>5} " + "".join(f"{t:>16}" for t in TOPOS))
    print(f"{'':>5} " + "".join(f"{'(hops|ns)':>16}" for _ in TOPOS))
    curves = {t: [] for t in TOPOS}
    for n in args.n_list:
        cells = []
        for t in TOPOS:
            ns, hops, deg = collective_latency_ns(t, n, msg_bytes, args.wg_budget, args.t_hop)
            curves[t].append(ns)
            cells.append(f"{hops:>4}|{ns:>8.0f}")
        print(f"{n:>5} " + "".join(f"{c:>16}" for c in cells))

    # FB advantage at the largest N
    nmax = args.n_list[-1]
    fb_ns = collective_latency_ns("fb", nmax, msg_bytes, args.wg_budget, args.t_hop)[0]
    print(f"\nAt N={nmax}: FB collective latency {fb_ns:.0f} ns; "
          + ", ".join(f"{t} {collective_latency_ns(t, nmax, msg_bytes, args.wg_budget, args.t_hop)[0]/fb_ns:.1f}x"
                      for t in TOPOS if t != "fb"))

    # ---- measured overlay (validation) ---------------------------------------
    measured = {}  # n -> {fabric: exposed%}
    for path in args.measured:
        if not os.path.exists(path):
            print(f"[warn] measured CSV not found: {path}")
            continue
        for x in csv.DictReader(open(path)):
            if x.get("status") != "ok":
                continue
            n = int(x["ep"])
            measured.setdefault(n, {})[x["fabric"]] = float(x["exposed_frac"]) * 100.0
    if measured:
        print("\nMeasured exposed%% (ASTRA, for validation of the ordering):")
        print(f"{'N':>5} " + "".join(f"{t:>8}" for t in TOPOS))
        for n in sorted(measured):
            print(f"{n:>5} " + "".join(f"{measured[n].get(t, float('nan')):>8.1f}" for t in TOPOS))

    # ---- plot -----------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "outputs", "panel_dse", "topo_projection.png")
        out = os.path.abspath(out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        color = {"fb": "tab:green", "torus": "tab:blue", "mesh": "tab:orange", "ring": "tab:red"}
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for t in TOPOS:
            ax.plot(args.n_list, curves[t], "o-", color=color[t], label=t.upper())
        # mark the measured (physically realizable) region
        ax.axvspan(args.n_list[0], 32, alpha=0.08, color="green")
        ax.text(min(32, args.n_list[-1]), ax.get_ylim()[1] * 0.95, " measured (<=32)",
                fontsize=8, va="top", color="green")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")          # 100x range across topologies -> log shows all curves
        ax.set_xticks(args.n_list)
        ax.set_xticklabels([str(n) for n in args.n_list])
        ax.set_xlabel("GPUs (EP)")
        ax.set_ylabel("all-to-all collective latency (ns, log)")
        ax.set_title(f"Topology scale projection (iso-budget {args.wg_budget:.0f} WG/GPU, "
                     f"{args.msg_kb:.0f} KB msg)\nFB stays flat (diameter 2); mesh/torus/ring diverge")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out, dpi=130)
        print(f"\nplot -> {out}")
    except ImportError:
        print("\n[matplotlib not available; table only]")


if __name__ == "__main__":
    main()
