#!/usr/bin/env python3
"""
Contention-aware all-to-all completion-time model (ASP-DAC 2027) — the PERFORMANCE
headline. Replaces the contention-BLIND avg-hop model (which was bandwidth-dwarfed
and could not separate FB from Mesh/Torus).

  T_a2a(topo, N, D) = diameter(topo,N) * t_hop                 # latency floor
                    + (N*D/4) / bisection_BW(topo, N)          # bandwidth bottleneck
  bisection_BW = (#links crossing a min bisection) * per-link BW
  per-link BW  = iso-budget: WG_budget/degree * 128 GB/s  | iso-per-link: fixed
  N*D/4        = all-to-all traffic crossing the bisection (half-to-half), D = per-GPU egress

WHY THIS FIXES EVERYTHING (one model):
  - FB bisection ~ N^1.5/4 links -> bisection_BW ~ N*B/8 -> bisection-term = 2D/B
    CONSTANT in N -> FB a2a time is FLAT. Mesh/Torus ~ sqrt(N), Ring ~ N -> they
    diverge. THIS separates FB from Mesh (avg-hop could not: diameter was BW-dwarfed).
  - Ring bisection = 2 links -> term ~ N*D/4B -> EXPLODES -> predicts the measured
    "Ring explodes" (F2 sim) -> the sim endpoint VALIDATES the model.
  - NVL72: in-domain = FC (huge bisection, NVLink); beyond the NVLink domain the
    bisection_BW collapses to IB -> bisection-term jumps = the cliff. Same model.

Defensible at a design venue: contention-aware analytical model, validated against
the congestion-unaware sim's endpoints (Ring, in-domain). Run:
  python scripts/collective_model.py --out outputs/paper_figures/F1_collective_model.png
"""
import argparse, math, os

WG_BW = 128.0          # GB/s per waveguide (one direction)
T_HOP_NS = 100.0       # per-hop optical latency

STYLE = {  # topo -> (label, color, marker)
    "fb":        ("FB (glass)",        "tab:green",  "o"),
    "dragonfly": ("Dragonfly (glass)", "tab:purple", "D"),
    "torus":     ("Torus (glass)",     "tab:blue",   "s"),
    "mesh":      ("Mesh (glass)",      "tab:cyan",   "^"),
    "ring":      ("Ring (glass)",      "tab:orange", "v"),
    "nvl72":     ("NVL72 (NVLink+IB)", "tab:red",    "x"),
}


def grid(n):
    r = int(round(math.sqrt(n)))
    while r > 1 and n % r:
        r -= 1
    return r, n // r


def degree(topo, r, c):
    if topo == "fb":   return (r - 1) + (c - 1)
    if topo in ("mesh", "torus"): return 4
    if topo == "ring": return 2
    if topo == "dragonfly": return 4          # ~ (a-1)+h for small groups
    return 0


def diameter(topo, r, c):
    n = r * c
    if topo == "fb":   return 2
    if topo == "mesh": return (r - 1) + (c - 1)
    if topo == "torus": return r // 2 + c // 2
    if topo == "ring": return n // 2
    if topo == "dragonfly": return 3
    return 1


def bisection_links(topo, r, c):
    """#links crossing a minimum bisection (square grid)."""
    n = r * c
    if topo == "fb":   return r * (c // 2) * (c - c // 2)      # ~ N^1.5/4: row links across a column cut
    if topo == "mesh": return min(r, c)                        # ~ sqrt(N)
    if topo == "torus": return 2 * min(r, c)
    if topo == "ring": return 2
    if topo == "dragonfly": return max(1, n // 4)              # global links ~ N/4 (approx)
    return max(1, (n // 2) * (n - n // 2) // n)                # FC-ish fallback


def t_a2a_us(topo, n, D_bytes, wg_budget, eq_link_bw=None,
             nvl_bw=450.0, nvl_domain=8, ib_bw=50.0):
    """All-to-all completion time (us). D_bytes = per-GPU a2a egress per step."""
    r, c = grid(n)
    if topo == "nvl72":
        # within-domain non-blocking NVLink; beyond -> IB bisection.
        if n <= nvl_domain:
            bis_bw = (n / 2) * nvl_bw                          # GB/s
            diam = 1
        else:
            bis_bw = (n / 2) * ib_bw
            diam = 2                                           # 1 switch hop in + 1 IB hop
    else:
        deg = degree(topo, r, c)
        link_bw = eq_link_bw if eq_link_bw is not None else (wg_budget / max(deg, 1)) * WG_BW / 2.0
        bis_bw = bisection_links(topo, r, c) * link_bw         # GB/s
        diam = diameter(topo, r, c)
    bis_traffic = n * D_bytes / 4.0                            # bytes crossing bisection
    lat = diam * T_HOP_NS                                      # ns
    bw_ns = bis_traffic / max(bis_bw, 1e-9)                    # bytes / (GB/s=B/ns) = ns
    return (lat + bw_ns) / 1000.0                              # us


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-list", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    ap.add_argument("--msg-kb", type=float, default=512.0, help="per-GPU a2a egress per step (KB)")
    ap.add_argument("--wg-budget", type=float, default=60.0, help="total WG/GPU (iso-budget)")
    ap.add_argument("--equal-link-bw", type=float, default=None, help="iso-per-link GB/s instead of iso-budget")
    ap.add_argument("--nvl-bw", type=float, default=450.0); ap.add_argument("--nvl-domain", type=int, default=8)
    ap.add_argument("--ib-bw", type=float, default=50.0)
    ap.add_argument("--topos", nargs="+", default=["fb", "dragonfly", "torus", "mesh", "ring", "nvl72"])
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default="outputs/paper_figures/F1_collective_model.png")
    args = ap.parse_args()
    D = args.msg_kb * 1024.0
    regime = f"iso-per-link {args.equal_link_bw} GB/s" if args.equal_link_bw else f"iso-budget {args.wg_budget:.0f} WG/GPU"

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    rows = {}
    for topo in args.topos:
        ys = [t_a2a_us(topo, n, D, args.wg_budget, args.equal_link_bw,
                       args.nvl_bw, args.nvl_domain, args.ib_bw) for n in args.n_list]
        rows[topo] = ys
        lab, col, mk = STYLE[topo]
        ax.plot(args.n_list, ys, marker=mk, color=col, lw=2.3, ms=7,
                ls="--" if topo == "nvl72" else "-", label=lab)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(args.n_list); ax.set_xticklabels([str(n) for n in args.n_list])
    ax.set_xlabel("GPUs (EP)"); ax.set_ylabel("all-to-all completion time (us)")
    ax.set_title(args.title or f"Contention-aware all-to-all completion time vs scale\n"
                 f"({regime}; FB bisection ~N^1.5 -> flat; Mesh/Torus ~sqrt(N); Ring ~N; NVL72 IB cliff)")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.abspath(args.out); os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140); print(f"wrote {out}")
    print(f"  regime: {regime}, msg {args.msg_kb} KB/GPU")
    for topo in args.topos:
        print(f"  {STYLE[topo][0]:<22}", " ".join(f"N{n}={y:.1f}" for n, y in zip(args.n_list, rows[topo])))
    # scaling check (normalize to FB)
    fb = rows.get("fb")
    if fb:
        print("  --- ratio to FB @largest N ---", {STYLE[t][0]: round(rows[t][-1]/fb[-1], 1) for t in args.topos if t != "fb"})


if __name__ == "__main__":
    main()
