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


def dragonfly_params(n):
    """Balanced Dragonfly (Kim et al. 2008), 1 GPU/router: target group size
    a ~ (2N)^(1/3); pick the DIVISOR of N closest to that target (so groups are
    equal-sized and not degenerate, e.g. a=4 not a=2 for N=16). groups g = N/a,
    global links/router h = ceil((g-1)/a). Returns (a, g, h)."""
    target = (2 * n) ** (1.0 / 3.0)
    divisors = [d for d in range(2, n + 1) if n % d == 0]
    a = min(divisors, key=lambda d: abs(d - target)) if divisors else max(2, round(target))
    g = max(1, n // a)
    h = 0 if g <= 1 else -(-(g - 1) // a)          # ceil((g-1)/a)
    return a, g, h


def degree(topo, rows, cols):
    n = rows * cols
    if topo == "fb":
        return (rows - 1) + (cols - 1)
    if topo in ("mesh", "torus"):
        return 4
    if topo == "ring":
        return 2
    if topo == "dragonfly":
        a, g, h = dragonfly_params(n)
        return (a - 1) + h                         # intra (full group) + global links
    raise ValueError(topo)


def _pair_hops(topo, n, i, j, rows, cols, group_size=None):
    """Minimal-route hop count between distinct nodes i, j for each topology."""
    if topo == "ring":
        d = abs(i - j)
        return min(d, n - d)
    if topo == "dragonfly":
        return 1 if (i // group_size) == (j // group_size) else 3   # local | local-global-local
    ri, ci = divmod(i, cols)
    rj, cj = divmod(j, cols)
    dr, dc = abs(ri - rj), abs(ci - cj)
    if topo == "fb":
        return 1 if (ri == rj or ci == cj) else 2  # same row/col 1 hop, else 2
    if topo == "mesh":
        return dr + dc                              # Manhattan
    if topo == "torus":
        return min(dr, rows - dr) + min(dc, cols - dc)
    raise ValueError(topo)


def avg_hops(topo, n):
    """Traffic-weighted AVERAGE hop count over all ordered pairs of an all-to-all.

    This (not the diameter) is what sets the congestion-unaware collective cost:
    every pair exchanges data, so the mean path length drives total movement. It is
    why measured Dragonfly (groups-of-2 at small N -> almost all pairs at 3 hops)
    loses to Torus despite a smaller diameter, yet wins at large N where Torus's
    mean grows ~sqrt(N) while Dragonfly stays ~3."""
    if topo == "ring":
        rows, cols = 1, n
    else:
        rows, cols = grid_for_n(n)
    gs = dragonfly_params(n)[0] if topo == "dragonfly" else None
    total, cnt = 0, 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            total += _pair_hops(topo, n, i, j, rows, cols, gs)
            cnt += 1
    return total / cnt if cnt else 0.0


def collective_latency_ns(topo, n, msg_bytes, wg_budget=WG_BUDGET, t_hop=T_HOP_NS,
                          equal_link_bw=None):
    """alpha-beta all-to-all latency, faithful to ASTRA's congestion-unaware
    "direct" send (cost = worst pair = diameter*t_hop + per-pair_chunk/link_bw):

        T = hops * t_hop  +  (msg_bytes / n) / link_bw

    where the per-pair chunk is msg/n (each GPU's egress split across its peers)
    over ONE link at the per-link BW. Under iso-budget link_bw = (wg_budget/degree/2)
    *WG_BW, so a higher-degree topology has THINNER links — its bw term scales with
    degree. So iso-budget trades off BOTH diameter (hops) and degree (link width);
    that is why measured Dragonfly (degree>torus) can lose to Torus at large msg
    despite a smaller diameter. iso-bandwidth ablation (equal_link_bw) fixes the
    per-link BW for every topology, so the bw term is common and ONLY diameter
    differentiates them (pure-structure view)."""
    rows, cols = (1, n) if topo == "ring" else grid_for_n(n)
    deg = degree(topo, rows, cols)
    if equal_link_bw is not None:
        link_bw_Bpns = float(equal_link_bw)             # GB/s == B/ns, common to all
    else:
        link_bw_Bpns = max(1.0, (wg_budget / deg / 2.0) * WG_BW)
    hops = avg_hops(topo, n)                             # traffic-weighted mean path
    bw_term = (msg_bytes / n) / link_bw_Bpns if link_bw_Bpns > 0 else 0.0
    return hops * t_hop + bw_term, hops, deg


TOPOS = ["fb", "dragonfly", "torus", "mesh", "ring"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-list", nargs="+", type=int, default=[16, 32, 64, 128, 256])
    ap.add_argument("--msg-kb", type=float, default=64.0,
                    help="per-GPU all-to-all message size (KB). Small = latency-bound "
                         "(topology matters most); large = bandwidth-bound (common offset grows).")
    ap.add_argument("--wg-budget", type=float, default=WG_BUDGET)
    ap.add_argument("--t-hop", type=float, default=T_HOP_NS)
    ap.add_argument("--equal-link-bw", type=float, default=None,
                    help="iso-bandwidth ablation: fixed per-link BW (GB/s) for every topology "
                         "instead of the budget/degree split (isolates pure structure)")
    ap.add_argument("--measured", nargs="*", default=[],
                    help="topo_compare CSVs to overlay (measured exposed%% / tpot at 16/32)")
    ap.add_argument("--out", default=None, help="plot path (default outputs/panel_dse/topo_projection.png)")
    args = ap.parse_args()

    msg_bytes = args.msg_kb * 1024.0
    eqbw = args.equal_link_bw
    mode_str = (f"iso-bandwidth {eqbw} GB/s/link" if eqbw is not None
                else f"iso-budget {args.wg_budget} WG/GPU")

    def cl(topo, n):
        return collective_latency_ns(topo, n, msg_bytes, args.wg_budget, args.t_hop, eqbw)

    # ---- analytical table -----------------------------------------------------
    print(f"\nAnalytical all-to-all latency (alpha-beta, {mode_str}, "
          f"msg {args.msg_kb} KB, t_hop {args.t_hop} ns)")
    print(f"{'N':>5} " + "".join(f"{t:>16}" for t in TOPOS))
    print(f"{'':>5} " + "".join(f"{'(hops|ns)':>16}" for _ in TOPOS))
    curves = {t: [] for t in TOPOS}
    for n in args.n_list:
        cells = []
        for t in TOPOS:
            ns, hops, deg = cl(t, n)
            curves[t].append(ns)
            cells.append(f"{hops:>5.2f}|{ns:>8.0f}")
        print(f"{n:>5} " + "".join(f"{c:>16}" for c in cells))

    # FB advantage at the largest N
    nmax = args.n_list[-1]
    fb_ns = cl("fb", nmax)[0]
    print(f"\nAt N={nmax}: FB collective latency {fb_ns:.0f} ns; "
          + ", ".join(f"{t} {cl(t, nmax)[0]/fb_ns:.1f}x" for t in TOPOS if t != "fb"))

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
        _suffix = "_isobw" if eqbw is not None else ""
        out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "outputs", "panel_dse", f"topo_projection{_suffix}.png")
        out = os.path.abspath(out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        color = {"fb": "tab:green", "dragonfly": "tab:purple", "torus": "tab:blue",
                 "mesh": "tab:orange", "ring": "tab:red"}
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
        ax.set_title(f"Topology scale projection ({mode_str}, {args.msg_kb:.0f} KB msg)\n"
                     f"FB (diam 2) & Dragonfly (diam 3) stay flat; mesh/torus/ring diverge")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out, dpi=130)
        print(f"\nplot -> {out}")
    except ImportError:
        print("\n[matplotlib not available; table only]")


if __name__ == "__main__":
    main()
