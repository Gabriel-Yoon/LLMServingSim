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

# ---- link DYNAMIC energy (pJ/bit) — PLACEHOLDERS, cite & TUNE ----------------
#   glass optical CPO: driver+mod+det+TIA+EIC ~ 1.15 pJ/bit (PanelScale Tbl1 / Ayar)
#   NVLink5 electrical SerDes ~ 2 pJ/bit;  IB NDR NIC+optics ~ 5 pJ/bit
E_DYN = {"glass_intra": 1.15, "glass_inter": 1.15, "nvlink": 2.0, "ib": 5.0}

# ---- per-GPU STATIC power (W) — PLACEHOLDERS, cite & TUNE --------------------
#   glass: comb laser (wall-plug for the WDM lambdas) + microring tuning.
#     P_laser ~ N_WG*32 lambda * per-lambda wall-plug; here a lumped ~15 W/GPU.
#     P_tune: III-V MOS 352 fW/ring x ~1000 rings ~ negligible (thermo-optic would be ~W).
#   NVL72: NVSwitch 540 W/rack / 64 = 8.4 W/GPU  + NVLink SerDes static ~ a few W.
#   IB: NDR switch+NIC static share ~ a few W/GPU (only when EP crosses racks).
P_STATIC_GLASS = 15.0    # laser + tuning, always-on
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
    args = ap.parse_args()
    cfg = MODELS[args.model]
    global RACK, P_STATIC_NVSWITCH
    RACK = BASELINES[args.baseline]["rack"]
    P_STATIC_NVSWITCH = BASELINES[args.baseline]["p_nvswitch"]
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
    print("\n  NOTE: bytes are ANALYTICAL (model config + domain split); swap in sim-logged"
          " per-link-class bytes for higher fidelity. Static power (laser+tuning vs NVSwitch)"
          " is the main glass energy lever; EDP capitalises the latency parity.")
    print("  All e_dyn / P_static are CITED PLACEHOLDERS — tune before the paper.")

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
