#!/usr/bin/env python3
"""Per-topology medium ablation: glass (iso-budget, WGB=60) vs copper-feasible
(iso-budget, WGB=12 and WGB=6 -- iso-power-derived, see plan doc 2026-09-02/03).
Shows which topology benefits MOST from glass -- the degree-dependent budget
dilution (correctly captured by iso-budget, unlike iso-per-link) is the point."""
import csv, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PANEL_DSE = os.path.join(ROOT, "outputs", "panel_dse")
OUT = os.path.join(ROOT, "outputs", "paper_figures", "fig_medium_ablation_per_topo.png")

TOPO_ORDER = ["fb_2d", "dragonfly", "torus_2d", "mesh_2d", "ring_1d"]
TOPO_LABEL = {"fb_2d": "FB", "dragonfly": "Dragonfly", "torus_2d": "Torus", "mesh_2d": "Mesh", "ring_1d": "Ring"}
COLOR = {"fb_2d": "#1b7837", "dragonfly": "#5aae61", "torus_2d": "#9970ab", "mesh_2d": "#e08214", "ring_1d": "#b2182b"}
EPS = [16, 32, 64]

def load(pattern):
    rows = {t: {} for t in TOPO_ORDER}
    for f in sorted(glob.glob(pattern)):
        ep = int(f.split("_ep")[1].split("_")[0])
        with open(f) as fh:
            for row in csv.DictReader(fh):
                if row.get("status") != "ok":
                    continue
                t = row.get("topology")
                if t in rows:
                    rows[t][ep] = float(row["prefill_exposed_frac"])
    return rows

glass = load(os.path.join(PANEL_DSE, "topo_prefill_deepseek_v3_0324_ep*_isobudget.csv"))
copper12 = load(os.path.join(PANEL_DSE, "topo_prefill_deepseek_v3_0324_ep*_isobudget_wgb12_copper.csv"))
copper6 = load(os.path.join(PANEL_DSE, "topo_prefill_deepseek_v3_0324_ep*_isobudget_wgb6_copper.csv"))

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

# Left: ratio (copper/glass) per topology, both copper points, vs EP
ax = axes[0]
for t in TOPO_ORDER:
    xs = [ep for ep in EPS if ep in glass[t] and ep in copper12[t]]
    ys12 = [copper12[t][ep] / glass[t][ep] for ep in xs]
    ax.plot(xs, ys12, marker="o", color=COLOR[t], label=TOPO_LABEL[t], linewidth=2)
ax.set_xlabel("EP degree")
ax.set_ylabel("Copper / glass exposed-comm ratio")
ax.set_title("Iso-power copper (WGB=12) vs glass\n(higher = benefits more from glass)")
ax.set_xticks(EPS)
ax.grid(alpha=0.3)
ax.legend(fontsize=8, ncol=2)

# Right: bar chart at EP16, both copper points, ranked
ax2 = axes[1]
ep = 16
order = sorted(TOPO_ORDER, key=lambda t: -(copper12[t][ep] / glass[t][ep]))
x = range(len(order))
r12 = [copper12[t][ep] / glass[t][ep] for t in order]
r6 = [copper6[t][ep] / glass[t][ep] for t in order]
w = 0.35
ax2.bar([i - w/2 for i in x], r12, width=w, label="WGB=12 (~100 GB/s-eq.)", color="#4393c3")
ax2.bar([i + w/2 for i in x], r6, width=w, label="WGB=6 (~50 GB/s-eq.)", color="#d6604d")
ax2.set_xticks(list(x))
ax2.set_xticklabels([TOPO_LABEL[t] for t in order])
ax2.set_ylabel("Copper / glass exposed-comm ratio")
ax2.set_title(f"Ranked at EP{ep}")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3, axis="y")

fig.suptitle("Which topology benefits most from glass? (DeepSeek-V3, iso-budget)", y=1.03)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote", OUT)
