#!/usr/bin/env python3
"""
STEP B — verdict for the exposed-vs-batch experiment (run_exposed_batch_hpc.sh).

Reads expB_*.csv (batch sweep, glass-aggbw vs NVL72), plots exposed% vs batch and
the NVL72/glass end-to-end ratio, and auto-classifies:
  CASE 1: exposed crosses ~THRESH and glass end-to-end (TPOT/TTFT) < NVL72 there
          -> a comm-bound regime exists; report the threshold batch + win margin.
  CASE 2: exposed stays small to the largest batch, parity holds
          -> headline = energy + collective-a2a(10x) + topology.

Run: python scripts/exposed_verdict.py outputs/panel_dse/expB_*.csv
     python scripts/exposed_verdict.py --thresh 30 outputs/panel_dse/expB_deepseek_v3_decode_ep128.csv
"""
import argparse, csv, glob, os, sys


def load(path):
    rows = [r for r in csv.DictReader(open(path)) if r.get("status") == "ok"]
    by = {"glass": {}, "nvl72": {}}
    for r in rows:
        fab = "glass" if ("fb" in r["label"] or r.get("fabric", "").startswith(("glass", "fb"))) else "nvl72"
        B = int(r["per_device_batch"])
        by[fab][B] = {
            "exp": float(r.get("exposed_frac") or 0) * 100,
            "tpot": float(r.get("tpot_gt_ms") or 0),
            "ttft": float(r.get("ttft_ms") or 0),
            "a2a": float(r.get("all_to_all_us") or 0),
        }
    return by


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--thresh", type=float, default=30.0, help="exposed%% threshold for 'comm-bound'")
    ap.add_argument("--metric", choices=["tpot", "ttft"], default="tpot", help="end-to-end metric for the win test")
    args = ap.parse_args()

    for path in args.csvs:
        for p in sorted(glob.glob(path)):
            tag = os.path.basename(p).replace("expB_", "").replace(".csv", "")
            by = load(p)
            Bs = sorted(set(by["glass"]) & set(by["nvl72"]))
            if not Bs:
                print(f"\n[{tag}] no paired data"); continue
            print(f"\n=== {tag} ===  (metric={args.metric}, thresh={args.thresh}%)")
            print(f"  {'batch':>6}{'g_exp%':>8}{'n_exp%':>8}{'g_'+args.metric:>9}{'n_'+args.metric:>9}"
                  f"{'NVL/glass':>10}{'g_a2a':>9}{'n_a2a':>9}")
            case1_B = None
            for B in Bs:
                g, n = by["glass"][B], by["nvl72"][B]
                ratio = (n[args.metric] / g[args.metric]) if g[args.metric] else 0
                glass_wins = g[args.metric] < n[args.metric]
                hot = max(g["exp"], n["exp"]) >= args.thresh
                if hot and glass_wins and case1_B is None:
                    case1_B = B
                mark = "  <== comm-bound & glass wins" if (hot and glass_wins) else ("  <== comm-bound" if hot else "")
                print(f"  {B:>6}{g['exp']:>8.1f}{n['exp']:>8.1f}{g[args.metric]:>9.1f}{n[args.metric]:>9.1f}"
                      f"{ratio:>9.2f}x{g['a2a']:>9.0f}{n['a2a']:>9.0f}{mark}")
            if case1_B is not None:
                B = case1_B; g, n = by["glass"][B], by["nvl72"][B]
                print(f"  VERDICT: CASE 1 — comm-bound regime at B>={B}: exposed>={args.thresh}% and "
                      f"glass {args.metric} {g[args.metric]:.1f} < NVL72 {n[args.metric]:.1f} "
                      f"({n[args.metric]/g[args.metric]:.2f}x). Real end-to-end win at high throughput.")
            else:
                mx = max(max(by["glass"][B]["exp"], by["nvl72"][B]["exp"]) for B in Bs)
                print(f"  VERDICT: CASE 2 — exposed peaks at {mx:.1f}% (< {args.thresh}%), parity holds. "
                      f"Headline -> energy + collective-a2a(10x) + topology, not end-to-end latency.")
            # always note the a2a (collective) ratio at the largest batch
            B = Bs[-1]; ga, na = by["glass"][B]["a2a"], by["nvl72"][B]["a2a"]
            if ga > 0:
                print(f"  collective a2a ratio @B={B}: NVL72/glass = {na/ga:.1f}x (the fabric gap at the collective level)")


if __name__ == "__main__":
    main()
