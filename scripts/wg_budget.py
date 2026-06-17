#!/usr/bin/env python3
"""
wg_budget.py — reproducible max-waveguide (WG) budget calculator for the
glass-photonic FlattenedButterfly interconnect (ASP-DAC 2027).

Derives, from a single tile geometry, the per-GPU micro-bump budget ->
microring (MRR) count -> waveguide-group (WG) count -> aggregate optical
bandwidth, and the per-topology WG-per-pair bundle each grid can afford.
This is the single source of truth for the "60 WG / 7.68 TB/s bidir / 4.3x NVLink"
headline numbers; everything downstream (panel DSE caps, bundle labels) should
trace back here.

DERIVATION CHAIN (with sources)
-------------------------------
  tile_area = tile_w * tile_h
  bump_max  = tile_area * bump_density   # micro-bumps on the EIC<->PIC interface
  mrr       = bump_max                   # 1 microring resonator per lambda, and
                                         # 1 data micro-bump per lambda (single-ended)
                                         # => MRR count == bump count
  wg_max    = bump_max / lambda_per_wg   # 1 WG groups 32 lambda (32 MRR / 32 bumps)
  per_gpu_bw = wg_max * wg_bw_gbps       # aggregate optical IO per GPU
  per-topology, degree d = (rows-1)+(cols-1):
      wg_per_pair = wg_max / d           # WGs available for each FB neighbour pair
      per_pair_bw = wg_per_pair * wg_bw_gbps
  ratio = per_gpu_bw / nvlink_h100       # vs H100 NVLink4 per-GPU spec

PHYSICAL ASSUMPTIONS (the part reviewers will probe)
----------------------------------------------------
  - PIC = the FULL tile area. We assume a 3D EIC-PIC stack (glass-embedded PIC
    under the GPU+HBM tile), so the photonic IC spans the whole tile, not just a
    shoreline. TX optics sit at the periphery, RX photodiodes on the bottom face.
  - The GPU die -> EIC bond is fine-pitch hybrid bonding, which avoids funnelling
    all I/O through the 814 mm^2 die shoreline. The PIC -> glass coupling uses
    femtosecond laser direct-write (FLDW) waveguides at 250 um pitch: the
    52x34 tile periphery is 2*(52+34)=172 mm -> 172000/250 = 688 WG launch slots,
    so the 60-WG budget is comfortably within the coupling limit (~9% utilisation).
  - bump_density 1.083 bumps/mm^2 is the PanelScale measurement on a COMPUTE-shared
    XPU (832 bumps / 768 mm^2). For a DEDICATED PIC interface this is a conservative
    LOWER bound -- a PIC-only interposer can pack denser.

CONVENTION NOTE
---------------
  wg_max / per_gpu_bw count the GPU's TOTAL waveguides (TX+RX aggregate). 1 WG =
  128 GB/s is unidirectional, so per_gpu_bw is the aggregate optical IO. The
  ratio vs nvlink_h100 (0.9 TB/s/GPU, NVLink4 per-direction spec) follows the
  user-specified convention. The per-direction N_WG (= wg_per_pair/2) is the knob
  the simulator consumes (see config_builder); it must be EVEN for TX/RX symmetry
  -- this script reports the physical CAP; the even feasible-floor is an
  operational choice applied on top (4x4 -> wg4, see feedback_wg_even / AGENTS.md).

Run:
  python scripts/wg_budget.py                      # 52x34 full-tile PIC (headline)
  python scripts/wg_budget.py --tile-area 814      # die-only conservative scenario
  python scripts/wg_budget.py --grids 4x4 4x8 6x6 8x8
  python scripts/wg_budget.py --check              # assert headline + die-only cases
"""

import argparse
import sys

# ---- default input constants (sources in module docstring) ------------------
TILE_W = 52.0          # mm  GH100 die (814 mm^2) + HBM3/3e 6-stack footprint
TILE_H = 34.0          # mm
BUMP_DENSITY = 1.083   # bumps/mm^2  PanelScale 832/768 mm^2 (XPU, conservative)
LAMBDA_PER_WG = 32     # 1 WG = 32 wavelengths
WG_BW_GBPS = 128.0     # 32 lambda x 32 Gb/s = 1.024 Tb/s = 128 GB/s (unidirectional)
NVLINK_H100 = 0.9      # TB/s/GPU  H100 NVLink4 baseline (per-direction spec)
FLDW_PITCH_UM = 250.0  # um  PIC->glass femtosecond-laser-direct-write pitch


def grid_degree(rows, cols):
    """FlattenedButterfly degree = (rows-1) row neighbours + (cols-1) col neighbours."""
    return (rows - 1) + (cols - 1)


def compute_budget(tile_w=TILE_W, tile_h=TILE_H, tile_area=None,
                   bump_density=BUMP_DENSITY, lambda_per_wg=LAMBDA_PER_WG,
                   wg_bw_gbps=WG_BW_GBPS, nvlink_h100=NVLINK_H100):
    area = float(tile_area) if tile_area is not None else tile_w * tile_h
    bump_max = area * bump_density
    mrr = bump_max                              # 1 MRR per lambda == 1 bump per lambda
    wg_max = bump_max / lambda_per_wg           # TOTAL WG/GPU (TX + RX)
    per_gpu_bw = wg_max * wg_bw_gbps / 1000.0   # TB/s, BIDIRECTIONAL aggregate IO (all WG)
    # CONVENTION-CONSISTENT ratio: a WG is unidirectional, so half the WG are RX;
    # the per-direction EGRESS = (wg_max/2) x 128. Compare that to NVLink's
    # per-direction spec (nvlink_h100). Comparing the bidir-aggregate 7.68 against
    # the unidir 0.9 was a mixed-convention error (gave a spurious 8.5x); like-for-
    # like (unidir egress vs unidir NVLink) is 4.3x — matching the bidir/bidir ratio
    # (7.68 / 1.8) too.
    per_gpu_egress = per_gpu_bw / 2.0           # TB/s, per-direction (unidirectional)
    ratio = per_gpu_egress / nvlink_h100
    # periphery-coupling headroom (only meaningful for w/h geometry)
    periphery_mm = 2.0 * (tile_w + tile_h) if tile_area is None else None
    fldw_slots = int(periphery_mm * 1000.0 / FLDW_PITCH_UM) if periphery_mm else None
    return {
        "area": area, "bump_max": bump_max, "mrr": mrr, "wg_max": wg_max,
        "per_gpu_bw": per_gpu_bw, "per_gpu_egress": per_gpu_egress, "ratio": ratio,
        "periphery_mm": periphery_mm, "fldw_slots": fldw_slots,
        "wg_bw_gbps": wg_bw_gbps,
    }


def per_topology(wg_max, degree, wg_bw_gbps=WG_BW_GBPS):
    wg_per_pair = wg_max / degree
    per_pair_bw = wg_per_pair * wg_bw_gbps / 1000.0   # TB/s
    n_wg_dir = wg_per_pair / 2.0                       # per-direction (simulator knob)
    return wg_per_pair, per_pair_bw, n_wg_dir


def parse_grid(g):
    r, c = g.lower().split("x")
    return int(r), int(c)


def report(b, grids):
    g = b
    print(f"  tile area        : {g['area']:.0f} mm^2")
    print(f"  bump_max         : {g['bump_max']:.0f}   (= area x {BUMP_DENSITY} bumps/mm^2)")
    print(f"  MRR count        : {g['mrr']:.0f}   (1 microring per lambda)")
    print(f"  wg_max / GPU     : {g['wg_max']:.1f}   (= bump_max / {LAMBDA_PER_WG} lambda)")
    print(f"  per-GPU optical  : {g['per_gpu_bw']:.2f} TB/s bidir aggregate "
          f"({g['per_gpu_egress']:.2f} TB/s per-direction egress)")
    print(f"  vs H100 NVLink   : {g['ratio']:.2f}x   (unidir egress vs {NVLINK_H100} TB/s/GPU unidir)")
    if g["fldw_slots"] is not None:
        util = 100.0 * g["wg_max"] / g["fldw_slots"]
        print(f"  FLDW coupling    : periphery {g['periphery_mm']:.0f} mm @ {FLDW_PITCH_UM:.0f} um "
              f"= {g['fldw_slots']} WG slots ({util:.0f}% used)")
    print(f"  {'grid':6} {'degree':>6} {'wg/pair':>8} {'bundle':>7} {'pair BW(TB/s)':>13} {'N_WG/dir':>9}")
    for grid in grids:
        rows, cols = parse_grid(grid)
        d = grid_degree(rows, cols)
        wpp, pbw, ndir = per_topology(g["wg_max"], d, g["wg_bw_gbps"])
        print(f"  {grid:6} {d:>6} {wpp:>8.1f} {'x'+str(round(wpp)):>7} {pbw:>13.2f} {ndir:>9.1f}")


def run_check():
    ok = True
    # headline: 52x34 full-tile PIC
    h = compute_budget()
    checks = [
        ("bump_max~1900", abs(h["bump_max"] - 1900) < 60),
        ("wg_max~60",     abs(h["wg_max"] - 60) < 2),
        ("per_gpu_bw~7.68(bidir)", abs(h["per_gpu_bw"] - 7.68) < 0.15),
        ("ratio~4.3",     abs(h["ratio"] - 4.3) < 0.3),
    ]
    wpp44 = per_topology(h["wg_max"], grid_degree(4, 4))[0]
    wpp48 = per_topology(h["wg_max"], grid_degree(4, 8))[0]
    checks += [("4x4 wg/pair~10", abs(wpp44 - 10) < 1),
               ("4x8 wg/pair~6",  abs(wpp48 - 6) < 1)]
    # die-only conservative: 814 mm^2 -> bump~880, wg~27
    d = compute_budget(tile_area=814)
    checks += [("814 bump~880", abs(d["bump_max"] - 880) < 20),
               ("814 wg~27",    abs(d["wg_max"] - 27) < 2)]
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    return ok


def main():
    ap = argparse.ArgumentParser(description="Glass-FB max-waveguide budget calculator")
    ap.add_argument("--tile-w", type=float, default=TILE_W, help="tile width mm (default 52)")
    ap.add_argument("--tile-h", type=float, default=TILE_H, help="tile height mm (default 34)")
    ap.add_argument("--tile-area", type=float, default=None,
                    help="override area mm^2 (e.g. 814 for die-only conservative scenario; "
                         "disables periphery/FLDW geometry)")
    ap.add_argument("--bump-density", type=float, default=BUMP_DENSITY)
    ap.add_argument("--grids", nargs="+", default=["4x4", "4x8"],
                    help="grids to size (default 4x4 4x8)")
    ap.add_argument("--check", action="store_true",
                    help="assert headline (52x34->wg60) and die-only (814->wg27) cases")
    args = ap.parse_args()

    if args.check:
        print("wg_budget self-check:")
        sys.exit(0 if run_check() else 1)

    scenario = ("die-only (conservative)" if args.tile_area is not None
                else f"{args.tile_w:.0f}x{args.tile_h:.0f} full-tile PIC")
    print(f"=== Glass-FB waveguide budget — {scenario} ===")
    b = compute_budget(tile_w=args.tile_w, tile_h=args.tile_h, tile_area=args.tile_area,
                       bump_density=args.bump_density)
    report(b, args.grids)


if __name__ == "__main__":
    main()
