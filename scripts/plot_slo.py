#!/usr/bin/env python3
"""Plot slo_eval.py output: ViBE-style SLO figures, glass-FB vs NVL72.

Two modes over a slo_eval CSV (fixed model/dataset/pd/ep, glass+nvl72 rows
across a QPS sweep):

  --mode qps     H3 / ViBE Fig8 + Fig9/14: goodput vs QPS (with the 90% target)
                 and TTFT/TPOT p50/p90/p99 vs QPS (with the SLO lines).
  --mode pareto  H1 / AIC++: goodput-interactivity Pareto — SLO-compliant
                 throughput (QPS x goodput) vs per-user interactivity (TPOT).

glass = green, NVL72 = red (matching plot_reach). Output: outputs/slo_eval/plots/.

Usage:
  python scripts/plot_slo.py --csv outputs/slo_eval/deepseek_v3_sonnet_decode_ep128.csv --mode qps
  python scripts/plot_slo.py --csv outputs/slo_eval/deepseek_v3_sonnet_decode_ep128.csv --mode pareto
"""
import argparse, csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs", "slo_eval", "plots")
GOODPUT_TARGET = 0.90
COLOR = {"glass": "#2ca02c", "nvl72": "#d62728"}
LABEL = {"glass": "Glass-FB (optical)", "nvl72": "NVL72 (NVLink+IB)"}


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load(path):
    """fabric -> list of rows (sorted by qps), plus dataset/pd/ep/slo context."""
    by_fab, ctx = {}, {}
    for r in csv.DictReader(open(path)):
        if r.get("status") != "ok":
            continue
        fab = "glass" if r["fabric"].startswith("glass") else "nvl72"
        by_fab.setdefault(fab, []).append(r)
        ctx.setdefault("dataset", r.get("dataset"))
        ctx.setdefault("pd_mode", r.get("pd_mode"))
        ctx.setdefault("ep", r.get("ep"))
        ctx.setdefault("ttft_slo", _f(r.get("ttft_slo")))
        ctx.setdefault("tpot_slo", _f(r.get("tpot_slo")))
    for fab in by_fab:
        by_fab[fab].sort(key=lambda r: _f(r["qps"]) or 0)
    return by_fab, ctx


def _has(by_fab, col):
    return any(_f(r.get(col)) is not None for rows in by_fab.values() for r in rows)


def _max_qps_at_target(rows):
    ok = [_f(r["qps"]) for r in rows if (_f(r.get("goodput")) or 0) >= GOODPUT_TARGET]
    return max(ok) if ok else None


def plot_qps(by_fab, ctx, path, title):
    pd_mode = ctx.get("pd_mode")
    has_ttft = _has(by_fab, "ttft_p50") and pd_mode != "decode"
    has_tpot = _has(by_fab, "tpot_p50") and pd_mode != "prefill"
    panels = ["goodput"] + (["ttft"] if has_ttft else []) + (["tpot"] if has_tpot else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.4))
    if len(panels) == 1:
        axes = [axes]

    for ax, kind in zip(axes, panels):
        for fab, rows in by_fab.items():
            xs = [_f(r["qps"]) for r in rows]
            c = COLOR[fab]
            if kind == "goodput":
                ys = [(_f(r.get("goodput")) or 0) * 100 for r in rows]
                ax.plot(xs, ys, marker="o", color=c, lw=2.2, label=LABEL[fab])
            else:
                for pct, ls, mk in [("p50", "-", "o"), ("p90", "--", "s"), ("p99", ":", "^")]:
                    ys = [_f(r.get(f"{kind}_{pct}")) for r in rows]
                    if any(v is not None for v in ys):
                        ax.plot(xs, ys, ls=ls, marker=mk, color=c, lw=1.8, ms=4,
                                label=f"{LABEL[fab].split()[0]} {pct}")
        if kind == "goodput":
            ax.axhline(GOODPUT_TARGET * 100, color="grey", ls="--", lw=1.2)
            ax.set_ylabel("SLO goodput [%]")
            ax.set_title("(a) SLO attainment vs QPS")
            ax.set_ylim(0, 105)
            for fab, rows in by_fab.items():
                q = _max_qps_at_target(rows)
                if q is not None:
                    ax.axvline(q, color=COLOR[fab], ls=":", lw=1.0, alpha=0.6)
        else:
            slo = ctx.get(f"{kind}_slo")
            if slo:
                ax.axhline(slo, color="black", ls="-.", lw=1.2, label=f"{kind.upper()} SLO {slo:.0f}ms")
            ax.set_ylabel(f"{kind.upper()} [ms] (lower=better)")
            ax.set_title(f"({'b' if kind=='ttft' else 'c'}) {kind.upper()} percentiles vs QPS")
            ax.set_ylim(bottom=0)
        ax.set_xlabel("QPS per instance")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=12)
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {os.path.relpath(path, REPO)}")
    for fab, rows in by_fab.items():
        q = _max_qps_at_target(rows)
        print(f"  {fab:<6} max QPS at >={GOODPUT_TARGET:.0%} goodput = {q if q is not None else 'none'}")


def plot_pareto(by_fab, ctx, path, title):
    """SLO-compliant throughput (qps x goodput) vs interactivity (median TPOT)."""
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for fab, rows in by_fab.items():
        pts = []
        for r in rows:
            q, g, t = _f(r["qps"]), _f(r.get("goodput")), _f(r.get("tpot_p50"))
            if q is None or g is None or t is None:
                continue
            pts.append((t, q * g))            # (interactivity x=TPOT, y=good throughput)
        if not pts:
            continue
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", color=COLOR[fab], lw=2.2, label=LABEL[fab])
    slo = ctx.get("tpot_slo")
    if slo:
        ax.axvline(slo, color="black", ls="-.", lw=1.2, label=f"TPOT SLO {slo:.0f}ms")
    ax.set_xlabel("interactivity: median TPOT [ms] (lower=better →)")
    ax.set_ylabel("SLO-compliant goodput [req/s]  (higher=better)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {os.path.relpath(path, REPO)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--mode", choices=["qps", "pareto"], default="qps")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    by_fab, ctx = load(args.csv)
    if not by_fab:
        raise SystemExit(f"no ok rows in {args.csv}")
    base = os.path.splitext(os.path.basename(args.csv))[0]
    title = (f"{base}  (ds={ctx.get('dataset')}, pd={ctx.get('pd_mode')}, "
             f"ep={ctx.get('ep')})  glass-FB vs NVL72")
    name = args.name or f"slo_{args.mode}_{base}"
    path = os.path.join(OUT, name + ".png")
    if args.mode == "qps":
        plot_qps(by_fab, ctx, path, title)
    else:
        plot_pareto(by_fab, ctx, path, title)


if __name__ == "__main__":
    main()
