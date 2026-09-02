#!/usr/bin/env python3
"""Fd: iso-budget vs iso-per-link contrast (the 4-quadrant story), composed from Fb+Fc.
Two fairness lenses (columns) x {exposed_frac ordering, absolute prefill_step_ms} (rows).
"""
import csv, glob, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PANEL_DSE = os.path.join(ROOT, "outputs", "panel_dse")
OUT = os.path.join(ROOT, "outputs", "paper_figures", "fig_Fd_isobudget_vs_isoperlink_deepseek.png")

TOPO_ORDER = ["fb_2d", "dragonfly", "torus_2d", "mesh_2d", "ring_1d"]
TOPO_LABEL = {"fb_2d": "FB", "dragonfly": "Dragonfly", "torus_2d": "Torus", "mesh_2d": "Mesh", "ring_1d": "Ring"}
COLOR = {"fb_2d": "#1b7837", "dragonfly": "#5aae61", "torus_2d": "#9970ab", "mesh_2d": "#e08214", "ring_1d": "#b2182b"}
EPS = [16, 32, 64]

def load(tag):
    data = {t: {} for t in TOPO_ORDER}
    for ep in EPS:
        path = os.path.join(PANEL_DSE, f"topo_prefill_deepseek_v3_0324_ep{ep}_{tag}.csv")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                if row.get("status") != "ok":
                    continue
                t = row.get("topology")
                if t not in data:
                    continue
                data[t][ep] = {
                    "exposed": float(row["prefill_exposed_frac"]) * 100.0,
                    "step_ms": float(row["prefill_step_ms"]),
                }
    return data

isobudget = load("isobudget")
isoperlink = load("isobw_512")

fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex="col")

def plot_panel(ax, data, ycol, ylabel, title):
    for t in TOPO_ORDER:
        xs = [ep for ep in EPS if ep in data.get(t, {})]
        ys = [data[t][ep][ycol] for ep in xs]
        if xs:
            ax.plot(xs, ys, marker="o", label=TOPO_LABEL[t], color=COLOR[t], linewidth=2)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(EPS)
    ax.grid(alpha=0.3)

plot_panel(axes[0, 0], isobudget, "exposed", "Prefill exposed comm (%)",
           "Iso-budget (equal WG/GPU split by degree)")
plot_panel(axes[0, 1], isoperlink, "exposed", "Prefill exposed comm (%)",
           "Iso-per-link (EQBW=512 GB/s, all topologies)")
plot_panel(axes[1, 0], isobudget, "step_ms", "Prefill step (ms)", "")
plot_panel(axes[1, 1], isoperlink, "step_ms", "Prefill step (ms)", "")

for ax in axes[1, :]:
    ax.set_xlabel("EP degree", fontsize=9)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.03), fontsize=9, frameon=False)
fig.suptitle("Fd: Topology ordering under two fairness lenses (DeepSeek-V3, prefill)", y=1.08, fontsize=11)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote", OUT)

# Print the story in text form too: does ranking hold across the two lenses?
print()
print("EP64 ranking (exposed %%), iso-budget:  ", sorted(((isobudget[t].get(64,{}).get('exposed'), TOPO_LABEL[t]) for t in TOPO_ORDER if 64 in isobudget.get(t,{}))))
print("EP64 ranking (exposed %%), iso-per-link:", sorted(((isoperlink[t].get(64,{}).get('exposed'), TOPO_LABEL[t]) for t in TOPO_ORDER if 64 in isoperlink.get(t,{}))))
