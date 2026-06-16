#!/usr/bin/env python3
"""Auto-verdict for a reach-cliff CSV (sweep_panel_dse --sweep epscale).

Prints the per-EP glass-vs-NVL72 table and a PASS/FAIL verdict on the two
physical criteria the EP>rack cliff must satisfy:

  (1) NVL72 exposed-comm RISES across the rack boundary (EP>rack), i.e. the
      inter-rack IB makes the all-to-all more exposed — NOT less. (The old
      inter-only-a2a bug made it FALL, e.g. 49% -> 9.7%.)
  (2) At the largest EP (cliff), glass TPOT < NVL72 TPOT (glass optical beats
      NVL72 IB), and glass exposed stays well below NVL72.

Usage:
  python scripts/check_reach.py --csv outputs/panel_dse/reach_qwen235b.csv --rack 64
"""
import argparse, csv, os


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--rack", type=int, default=64)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.csv)) if r.get("status") == "ok"]
    g = {int(r["ep"]): r for r in rows if r["fabric"].startswith("glass")}
    n = {int(r["ep"]): r for r in rows if r["fabric"] == "nvl72"}
    eps = sorted(set(g) & set(n))
    if not eps:
        raise SystemExit("no paired glass/nvl72 ok rows")

    print(f"{'EP':>5} {'glass_TPOT':>11} {'nvl72_TPOT':>11} {'NVL/glass':>10} "
          f"{'glass_exp':>10} {'nvl72_exp':>10} {'regime':>10}")
    for ep in eps:
        gt, nt = _f(g[ep].get("tpot_gt_ms")), _f(n[ep].get("tpot_gt_ms"))
        ge, ne = _f(g[ep].get("exposed_frac")), _f(n[ep].get("exposed_frac"))
        ratio = (nt / gt) if (gt and nt) else None
        regime = "cliff" if ep > args.rack else "in-domain"
        print(f"{ep:>5} {gt if gt is not None else '-':>11} {nt if nt is not None else '-':>11} "
              f"{(f'{ratio:.2f}x' if ratio else '-'):>10} "
              f"{(f'{ge*100:.1f}%' if ge is not None else '-'):>10} "
              f"{(f'{ne*100:.1f}%' if ne is not None else '-'):>10} {regime:>10}")

    # ---- verdict ----
    cliff_eps = [e for e in eps if e > args.rack]
    boundary = max([e for e in eps if e <= args.rack], default=None)
    print("\n=== VERDICT ===")
    if not cliff_eps:
        print("  (no EP > rack in this CSV — cliff not yet reached; need EP > "
              f"{args.rack})")
        return
    top = max(cliff_eps)
    ok = True

    # (1) NVL72 exposed rises across the boundary
    ne_top = _f(n[top].get("exposed_frac"))
    ne_b = _f(n[boundary].get("exposed_frac")) if boundary else None
    if ne_top is not None and ne_b is not None:
        c1 = ne_top > ne_b
        ok &= c1
        print(f"  (1) NVL72 exposed rises EP{boundary}->EP{top}: "
              f"{ne_b*100:.1f}% -> {ne_top*100:.1f}%  [{'PASS' if c1 else 'FAIL (cliff inverted!)'}]")

    # (2) glass beats NVL72 at the cliff
    gt, nt = _f(g[top].get("tpot_gt_ms")), _f(n[top].get("tpot_gt_ms"))
    if gt and nt:
        c2 = gt < nt
        ok &= c2
        print(f"  (2) glass TPOT < NVL72 TPOT @EP{top}: {gt:.1f} < {nt:.1f}  "
              f"({nt/gt:.2f}x)  [{'PASS' if c2 else 'FAIL (glass slower!)'}]")
    ge, ne = _f(g[top].get("exposed_frac")), _f(n[top].get("exposed_frac"))
    if ge is not None and ne is not None:
        c3 = ge < ne
        ok &= c3
        print(f"  (3) glass exposed < NVL72 exposed @EP{top}: "
              f"{ge*100:.1f}% < {ne*100:.1f}%  [{'PASS' if c3 else 'FAIL'}]")

    print(f"\n  >>> CLIFF {'CONFIRMED — headline holds' if ok else 'NOT confirmed — investigate'} <<<")


if __name__ == "__main__":
    main()
