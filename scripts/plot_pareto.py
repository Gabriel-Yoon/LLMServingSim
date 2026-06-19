#!/usr/bin/env python3
"""
Throughput-vs-interactivity Pareto, axes aligned to NVIDIA AIConfigurator (arXiv
2601.06288, Fig.1) and the AFD / Dynamo serving convention:

  x = generation speed (tokens/s/user) = 1000 / TPOT_ms
  y = system throughput (tokens/s/GPU) = 1000/(TTFT_ms + (OSL-1)*TPOT_ms)
                                          * BatchSize * OSL / TotalGPUs

Each point is one batch from a fixed-EP batch sweep; sweeping batch traces the Pareto
frontier (bigger batch -> higher throughput, lower per-user speed). Configs with
TTFT_ms > --ttft-budget are dropped (SLA filter, AIConfigurator uses 1000 ms). glass and
NVL72 (or topologies) are drawn on the same axes.

  BatchSize = per_device_batch * EP (total concurrent requests); TotalGPUs = EP * TP,
  so per-GPU throughput = 1000/(TTFT+(OSL-1)*TPOT) * per_device_batch * OSL / TP.

Run:
  python scripts/plot_pareto.py --osl 24 --tp 1 outputs/panel_dse/expB_deepseek_v3_decode_ep128_kvfix.csv
"""
import argparse, csv, glob, os
from collections import defaultdict

_TOPO = {"fb": "FB (glass)", "glass_fb": "glass-FB", "glass": "Glass-FB", "nvl72": "NVL72",
         "mesh": "Mesh", "torus": "Torus", "ring": "Ring", "dragonfly": "Dragonfly"}
_COLOR = {"fb": "tab:green", "glass_fb": "tab:green", "glass": "tab:green", "nvl72": "tab:red",
          "mesh": "tab:cyan", "torus": "tab:blue", "ring": "tab:orange", "dragonfly": "tab:purple"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--osl", type=int, required=True, help="output sequence length used in the run")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel degree (TotalGPUs = EP*TP)")
    ap.add_argument("--ttft-budget", type=float, default=float("inf"),
                    help="drop configs with TTFT_ms above this (SLA; AIConfigurator uses 1000). "
                         "Leave at inf for controlled-burst runs whose TTFT is queueing-inflated.")
    ap.add_argument("--decode-only", action="store_true",
                    help="steady-decode throughput: req latency = OSL*TPOT (ignore TTFT). Use for "
                         "controlled-burst decode sweeps (TTFT is a t=0 queueing artifact, not online).")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # fabric -> list of (x=tok/s/user, y=tok/s/GPU, batch)
    pts = defaultdict(list)
    for pat in args.csvs:
        for p in sorted(glob.glob(pat)):
            for r in csv.DictReader(open(p)):
                if r.get("status") != "ok":
                    continue
                fab = (r.get("fabric") or "").strip()
                if fab not in _TOPO:
                    continue
                try:
                    tpot = float(r["tpot_gt_ms"]); ttft = float(r.get("ttft_ms") or 0)
                    batch = int(float(r["per_device_batch"]))
                except (ValueError, TypeError, KeyError):
                    continue
                if tpot <= 0 or ttft > args.ttft_budget:
                    continue
                # end-to-end request latency (ms): prefill (TTFT) + (OSL-1) decode steps.
                # --decode-only drops TTFT (steady-decode throughput) for burst runs.
                req_lat = (args.osl * tpot) if args.decode_only else (ttft + (args.osl - 1) * tpot)
                x = 1000.0 / tpot                               # tokens/s/user
                y = 1000.0 / req_lat * batch * args.osl / args.tp   # tokens/s/GPU
                pts[fab].append((x, y, batch))

    if not pts:
        raise SystemExit(f"no ok rows within TTFT<= {args.ttft_budget} ms in {args.csvs}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for fab, pl in sorted(pts.items(), key=lambda kv: kv[0] == "nvl72"):
        pl.sort(key=lambda t: t[0])                              # by interactivity
        xs = [a for a, _, _ in pl]; ys = [b for _, b, _ in pl]
        ax.plot(xs, ys, "o-" if fab != "nvl72" else "s--",
                color=_COLOR.get(fab, "gray"), lw=2, ms=7, label=_TOPO[fab])
        for x, y, b in pl:
            ax.annotate(f"b{b}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("generation speed  (tokens/s/user = 1000/TPOT)")
    ax.set_ylabel("system throughput  (tokens/s/GPU)")
    if args.decode_only:
        _sub = f"AIConfigurator axes; decode-steady throughput (TTFT excluded), OSL={args.osl}"
    else:
        _sub = f"AIConfigurator axes; TTFT<={args.ttft_budget:.0f} ms, OSL={args.osl}"
    ax.set_title(args.title or f"Throughput vs interactivity Pareto\n({_sub})")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs",
                                   "paper_figures", "fig_throughput_interactivity_pareto.png")
    out = os.path.abspath(out); os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140); print(f"wrote {out}")
    for fab, pl in pts.items():
        print(f"  {_TOPO[fab]:<12}", " ".join(f"b{b}:({x:.0f} tok/s/u, {y:.0f} tok/s/gpu)" for x, y, b in sorted(pl, key=lambda t: t[2])))


if __name__ == "__main__":
    main()
