#!/usr/bin/env python3
"""Regression test for parse_steady_decode's phase split (sweep_panel_dse._phase_split).

This split flipped phase across fabrics three times (median-guard, mode-on-gap,
mode-on-spread). Lock in the robust behaviour: decode step = MEDIAN of the small
cluster (stable to within-cluster spread / dp-sync stalls), prefill = the large
cluster, exposed aggregated over the core band. Run:  python scripts/test_phase_split.py
"""
import importlib.util, os, random, sys

_spec = importlib.util.spec_from_file_location(
    "sw", os.path.join(os.path.dirname(__file__), "sweep_panel_dse.py"))
_sw = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_sw)
_phase_split = _sw._phase_split
ms = lambda x: int(x * 1e6)   # ms -> cycles (1 cycle ~= 1 ns)


def _mixed(seed, dec_lo=37, dec_hi=45):
    """decode steps (spread) + dp-sync stalls + prefill chunks — the b128 pattern."""
    r = random.Random(seed)
    decode = [(ms(r.uniform(dec_lo, dec_hi)), ms(r.uniform(2, 6))) for _ in range(100)]
    stalls = [(ms(r.uniform(90, 110)), ms(0.5)) for _ in range(5)]
    prefill = [(ms(r.uniform(190, 210)), ms(r.uniform(80, 100))) for _ in range(30)]
    return decode + stalls + prefill


def main():
    fails = []

    # 1) STABILITY: decode_ms must be ~identical across seeds (no fabric-flip)
    d = [_phase_split(_mixed(s))[0] for s in range(5)]
    if max(d) - min(d) > 2.0:
        fails.append(f"decode_ms unstable across seeds: {[round(x,1) for x in d]} (spread>2ms)")
    if not all(38 <= x <= 44 for x in d):
        fails.append(f"decode_ms out of expected ~41ms band: {[round(x,1) for x in d]}")

    # 2) PHASE SEPARATION + exposed not diluted by stalls
    dm, de, pm, pe = _phase_split(_mixed(0))
    if not (190 <= pm <= 212):
        fails.append(f"prefill_ms wrong: {pm}")
    if not (0.06 <= de <= 0.14):                 # ~10%, stalls excluded
        fails.append(f"decode exposed off (stalls leaking?): {de}")
    if not (0.40 <= pe <= 0.50):
        fails.append(f"prefill exposed off: {pe}")

    # 3) PURE DECODE: prefill cluster empty
    r = random.Random(0)
    dec = [(ms(r.uniform(4.8, 5.2)), ms(0.2)) for _ in range(200)]
    dm2, _, pm2, _ = _phase_split(dec)
    if pm2 is not None:
        fails.append(f"pure-decode should give prefill=None, got {pm2}")
    if not (4.8 <= dm2 <= 5.2):
        fails.append(f"pure-decode decode_ms off: {dm2}")

    # 4) EMPTY input
    if _phase_split([]) != (None, None, None, None):
        fails.append("empty input should return all None")

    if fails:
        print("FAIL:"); [print("  -", f) for f in fails]; sys.exit(1)
    print("PASS: phase split is stable (median), separates prefill, excludes stalls.")


if __name__ == "__main__":
    main()
