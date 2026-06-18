#!/usr/bin/env python3
"""
STEP C — energy model for glass-FB vs NVL72 (post-hoc, no re-sim).

Computes E/token, pJ/bit breakdown, and EDP from: (i) an ANALYTICAL per-token
MoE all-to-all byte count split by link class, (ii) per-class dynamic energy
(pJ/bit), (iii) per-class STATIC power x time (laser + microring tuning for glass,
NVSwitch + SerDes for NVL72 -- the part ASTRA's energy hook misses). Parameterised
so the link-energy constants can be swept without re-simulating.

Link classes: glass_intra (within-panel optical), glass_inter (cross-panel optical),
              nvlink (within-rack), ib (cross-rack InfiniBand).

bytes split: an all-to-all over EP GPUs in D domains of size S=EP/D moves, per GPU
per token per layer, k_top*hidden*fp (dispatch) + hidden*fp (combine). The fraction
crossing domains = (EP-S)/(EP-1); the rest is intra-domain.

T (for static energy / EDP) comes from a reach CSV's TPOT (per-token) if given, else
--tpot-ms. NOTE: end-to-end TPOT is compute-bound (~parity glass vs NVL72), so the
ENERGY difference is dominated by (a) static power (NVSwitch 540W/rack vs glass
passive) and (b) the cross-domain dynamic (IB vs optical). EDP turns the latency
parity into an asset.

Run:
  python scripts/energy_model.py --ep-list 8 16 32 64 128 --model deepseek_v3
  python scripts/energy_model.py --ep-list 128 --reach-csv outputs/panel_dse/reach_deepseek_v3_wg4_aggbw.csv
"""
import argparse, csv, math, os

# ---- model configs (subset; hidden, experts, k_top, layers) ------------------
MODELS = {
    "deepseek_v3": dict(hidden=7168, experts=256, k_top=8, layers=61),
    "qwen235b":    dict(hidden=4096, experts=128, k_top=8, layers=94),
    "qwen30b":     dict(hidden=2048, experts=128, k_top=8, layers=48),
}
FP_BYTES = 2  # bf16

# ---- glass optical link DYNAMIC energy (pJ/bit) — SOURCED ---------------------
#   TX (driver+mod, laser folded in): 1.66 pJ/bit  [QD MLL 8λ×100G, RG 378058284]
#       aggressive 0.437, conservative 2.81 (AWGR NoC)
#   RX (PD+TIA): 0.96 pJ/bit  [Raj ISSCC'23 / Li ESSCIRC'18]
#   link = TX + RX. Laser's amortised pJ/bit (~0.04 @100G full util) is negligible in
#   the dynamic term; the always-on laser is accounted as STATIC below (correct for
#   the low-utilisation decode regime that ASTRA's energy hook misses).
GLASS_EDYN = {"default": 1.66 + 0.96, "aggressive": 0.437 + 0.96, "conservative": 2.81 + 0.96}
# NVLink electrical SerDes ~2 pJ/bit; IB NDR NIC+optics ~5 pJ/bit (placeholders, cite).
E_DYN = {"glass_intra": GLASS_EDYN["default"], "glass_inter": GLASS_EDYN["default"],
         "nvlink": 2.0, "ib": 5.0}

# ---- glass per-GPU STATIC power (W) — SOURCED, computed from WG count ----------
#   P_laser = N_lambda * 1 mW/lambda / WPE   [1 mW/λ DFB/QD comb; WPE 17–30%]
#   P_tune  = N_rings  * per-ring            [thermo-optic 0.75 mW/π (Yang IEEE Access'25)
#                                             | III-V MOS ~hundreds of fW (Akazawa) ~0]
#   N_lambda(TX) = wg_per_gpu * 32 (32 λ/WG); N_rings ≈ 2*N_lambda (TX MRM + RX filter).
WG_LAMBDAS = 32
LASER_MW_PER_LAMBDA = 1.0
TUNE_W_PER_RING = {"thermo": 0.75e-3, "mos": 0.0}
P_STATIC_GLASS = 1.0     # recomputed in main() from --wg-per-gpu/--wpe/--tune
P_STATIC_NVLINK_SERDES = 3.0
P_STATIC_IB = 4.0        # per-GPU IB share, only when cross-rack

PANEL = 16   # glass panel size (4x4)

# Baseline NVLink domain + NVSwitch static power per GPU. Default H100-CONSISTENT
# (compute is profiled on H100): NVLink4, 8-GPU HGX island, 4 NVSwitch3/node. GB200
# NVL72 (NVLink5, 64-GPU domain, 540 W/rack) is a forward-looking sensitivity --
# use --baseline gb200 (but note the compute is still H100).
BASELINES = {
    "h100":  dict(rack=8,  p_nvswitch=4 * 25.0 / 8.0),   # 4 NVSwitch3 ~25 W /8 GPU = 12.5 W/GPU
    "gb200": dict(rack=64, p_nvswitch=540.0 / 64.0),     # NVL72 540 W/rack /64 = 8.44 W/GPU
}
RACK = 8                       # set from --baseline in main()
P_STATIC_NVSWITCH = 12.5       # set from --baseline in main()


def per_token_bytes(cfg):
    """Per GPU, per token, total MoE a2a bytes (dispatch + combine, all layers)."""
    disp = cfg["k_top"] * cfg["hidden"] * FP_BYTES
    comb = cfg["hidden"] * FP_BYTES * cfg["k_top"]
    return (disp + comb) * cfg["layers"]


def split_fracs(ep, domain):
    """intra/cross fraction of the all-to-all for EP GPUs in domains of size `domain`."""
    if ep <= domain:
        return 1.0, 0.0
    s = domain
    cross = (ep - s) / (ep - 1)
    return 1.0 - cross, cross


def energy_per_token(fabric, ep, cfg, tpot_s):
    B = per_token_bytes(cfg)            # bytes/token (both directions accounted in disp+comb)
    if fabric == "glass":
        intra_f, cross_f = split_fracs(ep, PANEL)
        e_dyn = (B * intra_f * E_DYN["glass_intra"] + B * cross_f * E_DYN["glass_inter"]) * 8 * 1e-12
        p_static = P_STATIC_GLASS
    else:  # nvl72
        intra_f, cross_f = split_fracs(ep, RACK)
        e_dyn = (B * intra_f * E_DYN["nvlink"] + B * cross_f * E_DYN["ib"]) * 8 * 1e-12
        p_static = P_STATIC_NVSWITCH + P_STATIC_NVLINK_SERDES + (P_STATIC_IB if cross_f > 0 else 0.0)
    e_static = p_static * tpot_s        # J/token (static power over the token's time)
    return {"dyn": e_dyn, "static": e_static, "total": e_dyn + e_static,
            "intra_f": intra_f, "cross_f": cross_f, "p_static": p_static, "bytes": B}


def load_tpot(reach_csv):
    """Map EP -> {glass_tpot_s, nvl72_tpot_s} from a reach CSV (tpot_gt_ms)."""
    out = {}
    for r in csv.DictReader(open(reach_csv)):
        if r.get("status") != "ok":
            continue
        ep = int(r["ep"]); fab = "glass" if ("fb" in r["label"] or r.get("fabric","").startswith(("glass","fb"))) else "nvl72"
        out.setdefault(ep, {})[fab] = float(r.get("tpot_gt_ms") or 0) / 1000.0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ep-list", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    ap.add_argument("--model", default="deepseek_v3", choices=list(MODELS))
    ap.add_argument("--tpot-ms", type=float, default=40.0, help="per-token time for static energy (if no --reach-csv)")
    ap.add_argument("--reach-csv", default=None, help="reach CSV to read per-EP TPOT (overrides --tpot-ms)")
    ap.add_argument("--baseline", choices=list(BASELINES), default="h100",
                    help="NVLink baseline: h100 (NVLink4, 8-GPU HGX island; consistent with the "
                         "H100 compute profile) or gb200 (NVL72, 64-GPU domain; forward-looking).")
    ap.add_argument("--wg-per-gpu", type=int, default=4,
                    help="per-direction WG/GPU driving the laser+ring static (4x4 feasible floor=4; "
                         "headline 52x34 PIC ~30). N_lambda = wg*32.")
    ap.add_argument("--wpe", type=float, default=0.25, help="laser wall-plug efficiency (0.17-0.30, QD comb)")
    ap.add_argument("--tune", choices=list(TUNE_W_PER_RING), default="thermo",
                    help="microring tuning: thermo (0.75 mW/ring) or mos (III-V ~0)")
    ap.add_argument("--glass-edyn", choices=list(GLASS_EDYN), default="default",
                    help="glass link dynamic energy variant (default 2.62 / aggressive 1.40 / conservative 3.77 pJ/bit)")
    args = ap.parse_args()
    cfg = MODELS[args.model]
    global RACK, P_STATIC_NVSWITCH, P_STATIC_GLASS
    RACK = BASELINES[args.baseline]["rack"]
    P_STATIC_NVSWITCH = BASELINES[args.baseline]["p_nvswitch"]
    E_DYN["glass_intra"] = E_DYN["glass_inter"] = GLASS_EDYN[args.glass_edyn]
    n_lambda = args.wg_per_gpu * WG_LAMBDAS
    p_laser = n_lambda * LASER_MW_PER_LAMBDA * 1e-3 / args.wpe          # W
    p_tune = (2 * n_lambda) * TUNE_W_PER_RING[args.tune]               # W (TX MRM + RX filter)
    P_STATIC_GLASS = p_laser + p_tune
    print(f"[glass static] wg/gpu={args.wg_per_gpu} -> {n_lambda} λ; "
          f"P_laser={p_laser:.2f} W (WPE {args.wpe}), P_tune={p_tune*1e3:.1f} mW ({args.tune}) "
          f"-> P_static_glass={P_STATIC_GLASS:.2f} W; e_dyn_glass={E_DYN['glass_intra']:.2f} pJ/bit")
    tpot_map = load_tpot(args.reach_csv) if args.reach_csv else {}

    bname = args.baseline.upper()
    print(f"=== energy model — {args.model} (hidden {cfg['hidden']}, {cfg['experts']} exp, {cfg['layers']} layers) "
          f"vs {bname} (NVLink domain {RACK}) ===")
    print(f"e_dyn pJ/bit: {E_DYN}  | P_static/GPU: glass {P_STATIC_GLASS}W, {bname} "
          f"{P_STATIC_NVSWITCH+P_STATIC_NVLINK_SERDES:.1f}W(+IB {P_STATIC_IB} cross)")
    print(f"  {'EP':>5}{'fab':>7}{'cross%':>8}{'E_dyn(mJ)':>11}{'E_stat(mJ)':>11}{'E/tok(mJ)':>11}"
          f"{'pJ/bit':>9}{'EDP':>10}")
    rows = {}
    for ep in args.ep_list:
        for fab in ("glass", "nvl72"):
            tp = (tpot_map.get(ep, {}).get(fab) or args.tpot_ms / 1000.0)
            e = energy_per_token(fab, ep, cfg, tp)
            pj_bit = e["total"] / (e["bytes"] * 8) * 1e12
            edp = e["total"] * tp        # J*s per token
            rows[(ep, fab)] = (e, tp, pj_bit, edp)
            print(f"  {ep:>5}{fab:>7}{e['cross_f']*100:>8.0f}{e['dyn']*1e3:>11.3f}{e['static']*1e3:>11.3f}"
                  f"{e['total']*1e3:>11.3f}{pj_bit:>9.2f}{edp*1e3:>10.3f}")
    # glass vs NVL72 ratio
    print(f"\n  {'EP':>5}{'E/tok glass/NVL':>18}{'EDP glass/NVL':>16}")
    for ep in args.ep_list:
        if (ep, "glass") in rows and (ep, "nvl72") in rows:
            g = rows[(ep, "glass")][0]["total"]; n = rows[(ep, "nvl72")][0]["total"]
            ge = rows[(ep, "glass")][3]; ne = rows[(ep, "nvl72")][3]
            print(f"  {ep:>5}{g/n:>17.2f}x{ge/ne:>15.2f}x")
    print("\n  NOTE: this is INTERCONNECT-only energy. Glass passive optical (laser+tune,"
          " no switch) vs NVSwitch dominates the static term -> the big ratio. At SYSTEM level"
          " the GPU compute (~hundreds W) dwarfs interconnect, so total-energy savings are a few %"
          " -- report the interconnect breakdown AND the system fraction.")
    print("  Glass e_dyn (TX 1.66 + RX 0.96 pJ/bit) and static (laser 1mW/λ÷WPE, ring tuning) are"
          " SOURCED; NVLink/IB e_dyn and NVSwitch static are still placeholders to cite. Bytes are"
          " ANALYTICAL (model config + domain split) -- swap in sim-logged per-link-class bytes for fidelity.")

    # plot
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "panel_dse", f"energy_{args.model}.png")
        out = os.path.abspath(out); os.makedirs(os.path.dirname(out), exist_ok=True)
        eps = args.ep_list
        g = [rows[(e, "glass")][0]["total"]*1e3 for e in eps]
        n = [rows[(e, "nvl72")][0]["total"]*1e3 for e in eps]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(eps, g, "o-", color="tab:green", label="glass-FB")
        ax.plot(eps, n, "s-", color="tab:red", label=bname)
        ax.axvline(RACK, ls="--", color="gray", alpha=0.6); ax.text(RACK, max(n)*0.9, f" EP>{RACK} (IB)", fontsize=8)
        ax.set_xscale("log", base=2); ax.set_xticks(eps); ax.set_xticklabels([str(e) for e in eps])
        ax.set_xlabel("GPUs (EP)"); ax.set_ylabel("energy / token (mJ)")
        ax.set_title(f"Energy/token vs EP — {args.model}\nglass optical vs {bname} (NVLink domain {RACK})")
        ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout(); fig.savefig(out, dpi=130)
        print(f"  plot -> {out}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
