#!/usr/bin/env python3
"""
F5 — Interconnect power / efficiency: glass-panel FB vs NVL72.

Analytical (no simulation) for the per-GPU bandwidth/power panels; the optional
energy-per-token panel reads a reach/epscale CSV for token throughput.

Model (transparent, fair):
  - 1 WG = 1.024 Tb/s (128 GB/s) unidirectional; bit-transport energy e_bit
    = 1.15 pJ/bit applied to BOTH fabrics (PanelScale Table 1; optical CPO).
  - glass per-GPU usable BW (unidir) = degree x N_WG x 128 GB/s; WG/GPU (TX+RX)
    = degree x N_WG x 2 -> I/O power = WG/GPU x 1.024e12 x 1.15e-12 W.
  - glass fabric is PASSIVE (no switch) -> switch power = 0.
  - NVL72 per-GPU BW = 900 GB/s unidir (1.8 TB/s bidir); I/O power = BW x e_bit;
    PLUS NVSwitch 540 W/rack / 72 GPU = 7.5 W/GPU (active switch).

Usage:
  python scripts/plot_power.py                                   # analytical only
  python scripts/plot_power.py --reach-csv outputs/panel_dse/epscale_full_b128.csv
"""
import argparse, csv, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_panel_dse import (PANELS, compute_nwg_cap, WG_BW, NVL72_ELEC_BW,
                             NVL72_SWITCH_POWER)  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(REPO, "outputs", "panel_dse", "plots")
E_BIT = 1.15e-12           # J/bit (1.15 pJ/bit), both fabrics
WG_BPS = WG_BW * 1e9 * 8   # bits/s per WG per direction (128 GB/s -> 1.024 Tb/s)
NVL72_SWITCH_PER_GPU = NVL72_SWITCH_POWER / 72.0   # 540/72 = 7.5 W/GPU

# Operating N_WG per direction for each panel = its micro-bump cap (feasible).
def panel_point(name):
    cap = compute_nwg_cap(*PANELS[name])
    nwg = max(1, int(round(cap["nwg_per_dir"])))        # feasible per-dir N_WG
    deg = cap["degree"]
    bw_unidir = deg * nwg * WG_BW                        # GB/s per GPU
    wg_per_gpu = deg * nwg * 2                           # TX+RX
    io_power = wg_per_gpu * WG_BPS * E_BIT               # W (optical I/O)
    return {"name": name, "nwg": nwg, "bw": bw_unidir,
            "io_power": io_power, "switch": 0.0, "power": io_power}

def nvl72_point():
    bw = NVL72_ELEC_BW                                   # 900 GB/s unidir
    io = bw * 1e9 * 8 * E_BIT
    return {"name": "NVL72", "nwg": 0, "bw": bw,
            "io_power": io, "switch": NVL72_SWITCH_PER_GPU,
            "power": io + NVL72_SWITCH_PER_GPU}

def load_decode(csv_path):
    """fabric -> {ep -> (tpot_gt_s, batch_per_device)} from a reach CSV."""
    d = {}
    for r in csv.DictReader(open(csv_path)):
        if r.get("status") != "ok":
            continue
        gt = r.get("tpot_gt_ms")
        if gt in (None, "", "None"):
            continue
        try:
            batch = int(r.get("per_device_batch") or 1)
            d.setdefault(r["fabric"], {})[int(r["ep"])] = (float(gt) / 1e3, batch)
        except (ValueError, KeyError):
            pass
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reach-csv", default=None,
                    help="reach/epscale CSV for the energy-per-token panel")
    ap.add_argument("--gpu-tdp", type=float, default=1000.0,
                    help="per-GPU compute TDP in W (dominant energy term; default 1 kW)")
    args = ap.parse_args()

    pts = [panel_point(p) for p in PANELS] + [nvl72_point()]
    labels = [p["name"] for p in pts]
    colors = ["#d62728", "#2ca02c", "#9467bd", "#1f77b4"][:len(pts)]

    n_panels = 3 if args.reach_csv else 3
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (1) Aggregate BW per GPU
    ax = axes[0]
    ax.bar(labels, [p["bw"] / 1000 for p in pts], color=colors)
    ax.set_ylabel("per-GPU usable BW [TB/s, unidir]")
    ax.set_title("(a) Interconnect bandwidth per GPU")
    for i, p in enumerate(pts):
        ax.text(i, p["bw"]/1000, f"{p['bw']/1000:.2f}", ha="center", va="bottom", fontsize=8)

    # (2) Interconnect power per GPU (I/O + switch breakdown)
    ax = axes[1]
    io = [p["io_power"] for p in pts]
    sw = [p["switch"] for p in pts]
    ax.bar(labels, io, color=colors, label="optical/electrical I/O")
    ax.bar(labels, sw, bottom=io, color="black", alpha=0.5, label="active switch")
    ax.set_ylabel("per-GPU interconnect power [W]")
    ax.set_title("(b) Power (glass = passive, switch=0)")
    ax.legend(fontsize=8)
    for i, p in enumerate(pts):
        ax.text(i, p["power"], f"{p['power']:.0f}W", ha="center", va="bottom", fontsize=8)

    # (3) Bandwidth-per-watt  (and energy/token if reach CSV given)
    ax = axes[2]
    bw_per_w = [(p["bw"]/1000) / p["power"] for p in pts]
    ax.bar(labels, bw_per_w, color=colors)
    ax.set_ylabel("BW efficiency [TB/s per W]")
    ax.set_title("(c) Bandwidth per watt  (higher = better)")
    for i, v in enumerate(bw_per_w):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Interconnect power & efficiency: glass-panel FB vs NVL72 "
                 f"(e_bit={E_BIT*1e12:.2f} pJ/bit, NVSwitch {NVL72_SWITCH_POWER:.0f} W/rack)",
                 fontsize=11)
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUT, "f5_power_efficiency.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    for p in pts:
        print(f"  {p['name']:8s}  BW={p['bw']/1000:.2f} TB/s  "
              f"power={p['power']:.1f}W (I/O {p['io_power']:.1f} + switch {p['switch']:.1f})  "
              f"BW/W={(p['bw']/1000)/p['power']:.3f}")

    # (4) TOTAL energy-per-token (GPU TDP-dominated) — the headline power story.
    # At large EP, NVL72 wastes GPU time waiting on inter-rack IB (exposed comm
    # explodes), so its per-token decode latency climbs; glass's BW keeps it
    # flat. Energy/token = (TDP + interconnect_power) x tpot_gt / batch. The
    # interconnect difference (71 vs 16 W) is a small correction to the ~1 kW
    # GPU TDP — glass wins by FINISHING FASTER, not by lower link power.
    if args.reach_csv and os.path.exists(args.reach_csv):
        dec = load_decode(args.reach_csv)
        if dec.get("glass_fb") and dec.get("nvl72"):
            glass_pw = panel_point("4x4")["power"]
            nvl_pw = nvl72_point()["power"]
            fig2, ax2 = plt.subplots(figsize=(7, 4.5))
            for fab, pw, c, lab in [("glass_fb", glass_pw, "#2ca02c", "Glass FB (4x4)"),
                                    ("nvl72", nvl_pw, "#d62728", "NVL72")]:
                eps = sorted(dec[fab])
                ys = []
                for e in eps:
                    tpot_s, batch = dec[fab][e]
                    # J/token = (TDP + interconnect)[W] * tpot[s] / batch[tokens]
                    ys.append((args.gpu_tdp + pw) * tpot_s / batch * 1e3)  # mJ/token
                ax2.plot(eps, ys, marker="o", color=c, lw=2, label=lab)
            ax2.set_xscale("log", base=2)
            ax2.set_xlabel("Expert-parallel degree EP")
            ax2.set_ylabel("energy per token [mJ/token]  (GPU TDP + interconnect)")
            ax2.set_title(f"(d) Total energy per token, GPU TDP={args.gpu_tdp:.0f}W "
                          "(lower = better)")
            ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, which="both")
            p2 = os.path.join(OUT, "f5_energy_per_token.png")
            fig2.tight_layout(); fig2.savefig(p2, dpi=150, bbox_inches="tight")
            print(f"Saved: {p2}")
            for fab in ("glass_fb", "nvl72"):
                for e in sorted(dec[fab]):
                    tpot_s, batch = dec[fab][e]
                    pw = glass_pw if fab == "glass_fb" else nvl_pw
                    print(f"  {fab:8s} EP={e:3d}  E/tok={ (args.gpu_tdp+pw)*tpot_s/batch*1e3:.3f} mJ")
        else:
            print("(energy/token skipped: reach CSV lacks glass_fb+nvl72 tpot_gt)")

if __name__ == "__main__":
    main()
