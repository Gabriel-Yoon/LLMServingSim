#!/usr/bin/env python3
"""
fb_physical_design.py — GLASS-FB physical-design budget calculator (ASP-DAC 2027).

Chain: tile bump count -> position-wise electrical/optical split -> WG/pair ->
bandwidth -> feasibility gates -> thermal. Parameterised; all constants cited.
This is the RICH companion to wg_budget.py (which gives only the per-GPU WG cap):
here the electrical/optical split is POSITION-dependent in the FB 4x4 grid (a
corner drives 4 optical + 2 electrical links; a center 2 optical + 4 electrical),
because reach (not free choice) decides the medium.

CONVENTION: W = optical WG per PAIR (TX+RX), even. A WG is unidirectional
(128 GB/s). Per optical link: bidir BW = W*128, per-direction = (W/2)*128.

Run:
  python scripts/fb_physical_design.py                 # full report (a)-(d)
  python scripts/fb_physical_design.py --tile-area 814 # die-only conservative
  python scripts/fb_physical_design.py --w-compare 4 8 # headline scenarios
"""
import argparse

# ── [1] input constants (sources in comments) ─────────────────────────────────
TILE_W, TILE_H = 52.0, 34.0      # mm; GH100 die 814 + HBM3/3e 6-stack (H200 same)
RHO_BUMP       = 1.083           # bump/mm^2  PanelScale (832 bumps / 768 mm^2, sparse data-bump)
LAMBDA_PER_WG  = 32              # WDM: 1 WG = 32 lambda
R_LAMBDA       = 32.0            # Gb/s per lambda
BUMP_PER_WG    = 32              # 1 lambda = 1 bump = 1 MRR
R_ELEC         = 112.0           # Gb/s  high-speed short-reach electrical RDL lane
ELEC_RDL_BW    = 1800.0          # GB/s  electrical RDL link bandwidth, FIXED (inherited from
                                 # the sweep's ELEC_BW; dist-1 adjacent links). Unidirectional.
FLDW_PITCH     = 0.25            # mm    glass waveguide lateral pitch
GLASS_LAYERS   = 3               # L1 50um / L2 300um / L3 550um
NVLINK_H100    = 900.0           # GB/s  in-domain baseline
IB_BW          = 50.0            # GB/s  EP>72 scale-out
N_GLASS        = 1.5             # glass refractive index (latency)
C_MM_PER_NS    = 299.792458      # speed of light, mm/ns
# thermal per-ring tuning power (W/ring), two technologies:
P_TUNE_THERMO  = 0.75e-3         # thermo-optic 0.75 mW/pi (Yang)
P_TUNE_IIIV    = 352e-15         # III-V MOS 352 fW (Akazawa)
# loss reference (dB), reported for the link model:
LOSS_IOX, LOSS_GLASS_DB_CM, LOSS_FLDW = 0.38, 0.067, 1.25   # 0.034-0.1 dB/cm -> mid 0.067

ROWS = COLS = 4                  # FB 4x4
DEGREE = (ROWS - 1) + (COLS - 1) # 6, vertex-transitive


# ── [2] derived ────────────────────────────────────────────────────────────────
def derived(tile_area=None):
    area = tile_area if tile_area is not None else TILE_W * TILE_H
    n_bump = area * RHO_BUMP
    bw_wg = LAMBDA_PER_WG * R_LAMBDA / 8.0            # 128 GB/s (one WG, unidirectional)
    return {
        "area": area, "n_bump": n_bump, "bw_wg": bw_wg,
        "wg_ceil": n_bump / BUMP_PER_WG,             # bump ceiling on total WG/GPU
        "bw_per_opt_bump": R_LAMBDA / 8.0,           # 4 GB/s/bump
        "bw_per_elec_bump": R_ELEC / 8.0,            # 14 GB/s/bump
        "eff_ratio": (R_ELEC / 8.0) / (R_LAMBDA / 8.0),   # 3.5x
    }


# ── [3]/[4] topology + reach-based E/O split ────────────────────────────────────
def node_eo(r, c):
    """electrical (dist==1) vs optical (dist>=2) link counts for node (r,c).
    Each axis has 3 mates; dist-1 mates -> electrical, the rest -> optical."""
    e_row = (1 if c - 1 >= 0 else 0) + (1 if c + 1 < COLS else 0)
    e_col = (1 if r - 1 >= 0 else 0) + (1 if r + 1 < ROWS else 0)
    E = e_row + e_col
    return E, DEGREE - E                              # (electrical, optical)


POSITIONS = {"corner": (0, 0), "edge": (0, 1), "center": (1, 1)}


# ── [5] per-tile budget at design var W (optical WG/pair) ───────────────────────
def tile_budget(E, O, W, d):
    bw_wg = d["bw_wg"]
    opt_bump  = O * BUMP_PER_WG * W                  # 32W per optical link (both directions; W=TX+RX)
    # Electrical RDL is FIXED at ELEC_RDL_BW (1800 GB/s unidir, inherited), NOT matched
    # to optical. Bumps (both directions) = 2 * 1800 / 14 ~= 257 per elec link, W-INDEPENDENT.
    elec_bump_per_link = 2.0 * ELEC_RDL_BW / d["bw_per_elec_bump"]
    elec_bump = E * elec_bump_per_link
    # per-direction aggregate egress: electrical links @ 1800, optical @ (W/2)*128.
    agg_dir = E * ELEC_RDL_BW + O * (W / 2.0) * bw_wg
    return {
        "E": E, "O": O,
        "opt_bump": opt_bump, "elec_bump": elec_bump, "bump": opt_bump + elec_bump,
        "mrr": O * W * BUMP_PER_WG,                  # 1 MRR per optical bump
        "opt_wg": O * W, "elec_link_bw": ELEC_RDL_BW,
        "opt_link_bw_dir": (W / 2.0) * bw_wg,        # optical per-direction (W=8 -> 512)
        "bw_agg_dir": agg_dir,                       # per-direction egress (electrical fatter)
        "bw_agg": 2.0 * agg_dir,                     # bidirectional aggregate
    }


def corner_bump(W, d):
    E, O = node_eo(*POSITIONS["corner"])
    return tile_budget(E, O, W, d)["bump"]


# ── [6] feasibility gates (worst case = corner) ────────────────────────────────
def gates(W, d):
    E, O = node_eo(*POSITIONS["corner"])
    b = tile_budget(E, O, W, d)
    edge_short_slots = TILE_H / FLDW_PITCH            # 34mm/0.25 = 136
    g = {}
    g["G1_bump"]   = (b["bump"] <= d["n_bump"], f"{b['bump']:.0f}/{d['n_bump']:.0f} bump")
    # G2 TX on the shorter tile edge: 2 optical links * (W/2) TX WG = W WG <= slots
    g["G2_tx_edge"] = (W <= edge_short_slots, f"{W} <= {edge_short_slots:.0f} FLDW slots")
    # G3 routing: ~ (3 optical links * W) / GLASS_LAYERS per layer <= channel slots
    g["G3_routing"] = ((3 * W) / GLASS_LAYERS <= edge_short_slots,
                       f"{(3*W)/GLASS_LAYERS:.0f}/layer <= {edge_short_slots:.0f}")
    # G4 MRR area (ring pitch ~50um): corner rings tiny vs tile
    mrr_area = b["mrr"] * (0.05 ** 2)                 # mm^2 @ 50um pitch
    g["G4_mrr_area"] = (mrr_area <= d["area"], f"{mrr_area:.2f} <= {d['area']:.0f} mm^2")
    g["G5_rx_bottom"] = (b["mrr"] * (0.05 ** 2) <= d["area"], "trivial (full-tile bottom)")
    return g


def binding_W(d):
    W = 2
    while corner_bump(W + 2, d) <= d["n_bump"]:
        W += 2
    # also report the exact integer ceiling (any W, not just even)
    w_int = 0
    while corner_bump(w_int + 1, d) <= d["n_bump"]:
        w_int += 1
    return W, w_int


# ── [9] thermal / loss / latency ───────────────────────────────────────────────
def thermal(mrr):
    return {"thermo_optic_W": mrr * P_TUNE_THERMO, "iii_v_mos_W": mrr * P_TUNE_IIIV}


def link_latency_ns(dist_tiles):
    """propagation latency for a `dist_tiles`-hop glass link (tile pitch ~ TILE_W)."""
    length_mm = dist_tiles * TILE_W
    return length_mm / (C_MM_PER_NS / N_GLASS)


# ── output ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tile-area", type=float, default=None,
                    help="override tile area mm^2 (e.g. 814 for die-only conservative)")
    ap.add_argument("--w-list", nargs="+", type=int, default=[2, 4, 6, 8, 10, 12, 14],
                    help="W (optical WG/pair, even) sweep for the bump-utilisation table")
    ap.add_argument("--w-compare", nargs=2, type=int, default=[4, 8],
                    help="the two headline W scenarios for the design-space table")
    args = ap.parse_args()
    d = derived(args.tile_area)

    scen = "die-only (conservative)" if args.tile_area else f"{TILE_W:.0f}x{TILE_H:.0f} full-tile"
    print(f"=== GLASS-FB physical design budget — {scen} ===")
    print(f"[2] A_tile={d['area']:.0f} mm^2  N_bump={d['n_bump']:.0f}  BW_WG={d['bw_wg']:.0f} GB/s  "
          f"WG_ceil={d['wg_ceil']:.0f}  opt {d['bw_per_opt_bump']:.0f} / elec {d['bw_per_elec_bump']:.0f} "
          f"GB/s/bump (eff {d['eff_ratio']:.1f}x)")

    # (a) per-position E/O / WG / bump / MRR / aggregate, at the first compare-W
    W0 = args.w_compare[0]
    print(f"\n(a) per-position links @ W={W0} (FB 4x4, degree {DEGREE}):")
    print(f"  {'pos':7}{'E':>3}{'O':>3}{'opt_wg':>8}{'opt_bump':>10}{'elec_bump':>11}"
          f"{'bump':>8}{'MRR':>7}{'agg(TB/s)':>11}")
    for name, (r, c) in POSITIONS.items():
        E, O = node_eo(r, c)
        b = tile_budget(E, O, W0, d)
        print(f"  {name:7}{E:>3}{O:>3}{b['opt_wg']:>8}{b['opt_bump']:>10.0f}{b['elec_bump']:>11.0f}"
              f"{b['bump']:>8.0f}{b['mrr']:>7}{b['bw_agg']/1000:>11.2f}")

    # (b) W sweep: corner bump utilisation + binding gate
    Weven, Wint = binding_W(d)
    print(f"\n(b) W sweep — corner bump utilisation (binding gate = G1 bump; "
          f"feasible W <= {Wint} any / {Weven} even):")
    print(f"  {'W':>3}{'corner_bump':>13}{'util%':>7}  gate")
    for W in args.w_list:
        cb = corner_bump(W, d)
        util = 100 * cb / d["n_bump"]
        gate = "OK" if cb <= d["n_bump"] else "FAIL(G1 bump)"
        print(f"  {W:>3}{cb:>13.0f}{util:>7.0f}  {gate}")

    # (c) headline W scenarios + thermal
    print(f"\n(c) headline scenarios (corner = worst case):")
    print(f"  {'metric':22}" + "".join(f"{'W='+str(W):>12}" for W in args.w_compare))
    Ec, Oc = node_eo(*POSITIONS["corner"])
    rows = []
    bs = [tile_budget(Ec, Oc, W, d) for W in args.w_compare]
    rows.append(("optical link /dir (GB/s)", [b["opt_link_bw_dir"] for b in bs], "{:.0f}"))
    rows.append(("electrical link (GB/s)",   [b["elec_link_bw"] for b in bs], "{:.0f}"))
    rows.append(("optical bump/link",        [W * BUMP_PER_WG for W in args.w_compare], "{:.0f}"))
    rows.append(("elec bump/link (fixed)",   [b["elec_bump"] / max(b["E"], 1) for b in bs], "{:.0f}"))
    rows.append(("corner total bump",        [b["bump"] for b in bs], "{:.0f}"))
    rows.append(("corner util %",            [100 * b["bump"] / d["n_bump"] for b in bs], "{:.0f}%"))
    rows.append(("corner MRR",               [b["mrr"] for b in bs], "{:.0f}"))
    rows.append(("corner agg /dir (TB/s)",   [b["bw_agg_dir"] / 1000 for b in bs], "{:.2f}"))
    rows.append(("corner vs NVLink 900*",    [b["bw_agg_dir"] / NVLINK_H100 for b in bs], "{:.1f}x"))
    rows.append(("therm-optic tune (W)",     [thermal(b["mrr"])["thermo_optic_W"] for b in bs], "{:.2f}"))
    rows.append(("III-V MOS tune (W)",       [thermal(b["mrr"])["iii_v_mos_W"] * 1e9 for b in bs], "{:.1e} nW"))
    for label, vals, fmt in rows:
        print(f"  {label:22}" + "".join(f"{fmt.format(v):>12}" for v in vals))
    print("  * per-DIRECTION egress (E x 1800 elec + O x opt/dir) vs NVLink 900 unidir (like-for-like).")
    print("    Electrical RDL is now FIXED at 1800 GB/s (> optical for W<14), so links are UNBALANCED;")
    print("    the MoE all-to-all is bottlenecked by the slower OPTICAL links, and aggregate is")
    print("    position-dependent (electrical-heavy center has MORE egress than corner).")

    # (d) feasibility verdict
    print(f"\n(d) feasibility gates @ W={args.w_compare[-1]} (worst case = corner):")
    for name, (ok, detail) in gates(args.w_compare[-1], d).items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:13} {detail}")

    # loss / latency reference
    print(f"\n[9] link model: IOX {LOSS_IOX} dB + FLDW {LOSS_FLDW} dB + glass {LOSS_GLASS_DB_CM} dB/cm; "
          f"latency {link_latency_ns(2):.2f} ns (dist-2) .. {link_latency_ns(3):.2f} ns (dist-3), n={N_GLASS}")
    print(f"[8] inter-panel: free panel-edge sides per tile -> corner 2, edge 1, center 0 "
          f"(uplink WG x {d['bw_wg']:.0f} GB/s; place off-panel links on free edges).")


if __name__ == "__main__":
    main()
