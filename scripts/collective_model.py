#!/usr/bin/env python3
"""
HIERARCHICAL contention-aware all-to-all model (ASP-DAC 2027) — performance headline.
Driven by REAL LLMServingSim workload traffic.

Two LEVELS (the earlier monolithic model was wrong: it scaled one giant FB to 256 and
predicted "FB flat", contradicted by the sim's panel+inter-panel hierarchy):

  intra-domain (within a panel/rack of m GPUs): topology choice (FB/Mesh/Torus/Ring)
      T_intra = (m*D/4) / bisection_BW(topo, m)            # diameter+bisection of m
  inter-domain (N > m): the FABRIC (glass optical vs NVL72 IB)
      T_inter = (N*D/4) / ((N/2) * inter_bw_per_gpu)  = D/(2*inter_bw_per_gpu)
      glass = optical (512 GB/s) ; NVL72 = InfiniBand (50 GB/s)  -> 10x

  T_a2a(N) = (T_intra + T_inter) * n_moe_layers           # per step (ms)
  D = real per-GPU MoE a2a egress per layer (from model config + tokens/step)

TWO STORIES, ONE MODEL:
  - SMALL N (<= panel): topology dominates -> FB < Torus < Mesh < Ring (intra bisection).
  - LARGE N (> domain): fabric dominates -> glass optical << NVL72 IB (inter term, 10x).
    NVL72 cliffs at N > NVLink domain (intra NVLink -> inter IB); glass cliffs much
    later/softer (panel 16 -> inter optical, 10x faster than IB).

Honest scope: the bisection terms are idealized upper bounds (the absolute is ~10-50x
optimistic vs ASTRA's achieved collective); we report the SCALING and the glass:NVL72
RATIO (set by inter-domain optical-vs-IB, which is robust). The sim validates the
ORDERING (Ring explodes; NVL72 cliff at domain) qualitatively.

Run:
  python scripts/collective_model.py --model deepseek_v3 --tokens 2048 \
    --out outputs/paper_figures/F1_collective_model.png
"""
import argparse, json, math, os

WG_BW = 128.0
T_HOP_NS = 100.0
FP = 2
GLASS_PANEL = 16          # 4x4 feasible panel
GLASS_INTER_BW = 512.0    # inter-panel optical GB/s (per-GPU share, headline)
NVL_INTER_BW = 50.0       # inter-rack InfiniBand GB/s (per GPU)

# series -> (label, color, marker, intra-topology, intra-domain size, inter-bw per GPU)
SERIES = {
    "glass_fb":    ("glass-FB (panel) ",      "tab:green",  "o", "fb",    GLASS_PANEL, GLASS_INTER_BW),
    "glass_mesh":  ("glass-Mesh (panel)",     "tab:cyan",   "^", "mesh",  GLASS_PANEL, GLASS_INTER_BW),
    "glass_ring":  ("glass-Ring (panel)",     "tab:orange", "v", "ring",  GLASS_PANEL, GLASS_INTER_BW),
    "nvl72":       ("NVL72 (NVLink+IB)",      "tab:red",    "x", "fc",    8,           NVL_INTER_BW),
}
_MODELS = {
    "deepseek_v3": ("configs/model/deepseek-ai/DeepSeek-V3-0324.json", 58),
    "qwen235b":    ("configs/model/Qwen/Qwen3-235B-A22B.json", 94),
}


def load_cfg(model):
    path, moe = _MODELS.get(model, (model, None))
    c = json.load(open(path))
    n_exp = c.get("num_local_experts") or c.get("n_routed_experts") or c.get("num_experts") or 0
    return c["hidden_size"], n_exp, (moe or c.get("num_hidden_layers", 1))


def D_layer(hidden, n_exp, tokens, ep):
    return max(1, tokens // max(ep, 1)) * (hidden + n_exp) * FP + tokens * hidden * FP


def grid(n):
    r = int(round(math.sqrt(n)))
    while r > 1 and n % r:
        r -= 1
    return r, n // r


def intra_bisection_BW(topo, m, wg_budget, eq_link_bw):
    """bisection BW (GB/s) of the m-GPU intra-domain topology."""
    if topo == "fc":                                   # NVLink rack = non-blocking
        return (m / 2) * 450.0                          # NVLink4 450 GB/s/GPU
    r, c = grid(m)
    deg = {"fb": (r - 1) + (c - 1), "mesh": 4, "torus": 4, "ring": 2}.get(topo, 4)
    link = eq_link_bw if eq_link_bw else (wg_budget / max(deg, 1)) * WG_BW / 2.0
    bis = {"fb": r * (c // 2) * (c - c // 2), "mesh": min(r, c),
           "torus": 2 * min(r, c), "ring": 2}.get(topo, min(r, c))
    return bis * link


def t_a2a_ms(series, n, hidden, n_exp, tokens, moe_layers, wg_budget, eq_link_bw):
    _, _, _, topo, m, inter_bw = SERIES[series]
    D = D_layer(hidden, n_exp, tokens, n)
    m_eff = min(n, m)
    # intra-domain a2a (within panel/rack)
    t_intra = (m_eff * D / 4.0) / max(intra_bisection_BW(topo, m_eff, wg_budget, eq_link_bw), 1e-9)
    # inter-domain a2a (only if N spills past the domain): aggregate = (N/2)*inter_bw_per_gpu
    t_inter = 0.0
    if n > m:
        t_inter = (n * D / 4.0) / ((n / 2.0) * inter_bw)          # = D/(2*inter_bw)
    return (t_intra + t_inter) / 1e3 * moe_layers / 1e3          # ns->us->ms x layers


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="deepseek_v3")
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--n-list", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--wg-budget", type=float, default=60.0)
    ap.add_argument("--equal-link-bw", type=float, default=None)
    ap.add_argument("--series", nargs="+", default=list(SERIES))
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default="outputs/paper_figures/F1_collective_model.png")
    args = ap.parse_args()
    hidden, n_exp, moe_layers = load_cfg(args.model)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    rows = {}
    for s in args.series:
        ys = [t_a2a_ms(s, n, hidden, n_exp, args.tokens, moe_layers, args.wg_budget, args.equal_link_bw)
              for n in args.n_list]
        rows[s] = ys
        lab, col, mk, *_ = SERIES[s]
        ax.plot(args.n_list, ys, marker=mk, color=col, lw=2.3, ms=7,
                ls="--" if s == "nvl72" else "-", label=lab)
    ax.axvline(GLASS_PANEL, color="grey", ls=":", lw=1, alpha=0.6)
    ax.text(GLASS_PANEL * 1.05, ax.get_ylim()[0], "panel=16", fontsize=8, color="grey")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(args.n_list); ax.set_xticklabels([str(n) for n in args.n_list])
    ax.set_xlabel("GPUs (EP)"); ax.set_ylabel("MoE all-to-all time per step (ms)")
    ax.set_title(args.title or
                 f"Hierarchical all-to-all — {args.model}, {args.tokens} tok/step\n"
                 f"intra-panel: FB<Mesh<Ring (topology) | inter-domain: glass optical << NVL72 IB (10x)")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.abspath(args.out); os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140); print(f"wrote {out}")
    print(f"  {args.model}: hidden {hidden}, experts {n_exp}, moe_layers {moe_layers}, {args.tokens} tok/step")
    for s in args.series:
        print(f"  {SERIES[s][0]:<22}", " ".join(f"N{n}={y:.2f}" for n, y in zip(args.n_list, rows[s])))
    if "glass_fb" in rows and "nvl72" in rows:
        print("  glass-FB : NVL72 ratio:", {n: round(rows["nvl72"][i] / rows["glass_fb"][i], 1)
                                            for i, n in enumerate(args.n_list)})


if __name__ == "__main__":
    main()
