"""
DSE topology comparison: 4 candidate 128-GPU interconnect topologies.

  1. NVL72          — hierarchical_fb (NVLink tile + IB inter)
  2. 4×4 FB panel   — FlattenedButterfly 4×4, 16 GPUs/panel
  3. 6×6-4c FB panel — FlattenedButterfly 4×8, 32 GPUs/panel
  4. 6×6 full panel  — FlattenedButterfly 6×6, 36 GPUs/panel

Generates:
  outputs/topology_plots/10_dse_topology_overview.png   — 4-panel grid layouts + hop stats
  outputs/topology_plots/11_dse_ep_compatibility.png    — EP compatibility matrix
  outputs/topology_plots/12_dse_hop_bw_table.png        — per-topology BW/latency/hop summary

Usage:
  python scripts/plot_dse_topology.py
"""

import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

OUT = "outputs/topology_plots"
DPI = 300

# ── colour palette ─────────────────────────────────────────────────────────────
C_NVL    = "#D62728"   # NVL72 red
C_4X4    = "#2CA02C"   # 4×4 green
C_6X6_4C = "#1F77B4"   # 6×6-4c blue
C_6X6    = "#9467BD"   # 6×6 purple

TOPO_COLORS = {"NVL72": C_NVL, "4x4": C_4X4, "6x6_4c": C_6X6_4C, "6x6": C_6X6}
TOPO_LABELS = {
    "NVL72":  "NVL72\n(hierarchical FB)",
    "4x4":    "4×4 FB panel\n(16 GPUs/panel)",
    "6x6_4c": "6×6-4c FB panel\n(32 GPUs/panel)",
    "6x6":    "6×6 FB panel\n(36 GPUs/panel)",
}

NODE_R = 0.28


def savefig(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")


# ── topology helpers ────────────────────────────────────────────────────────────
def fb_hop_count(u, v, cols):
    """FlattenedButterfly hop count between u and v."""
    ru, cu = divmod(u, cols)
    rv, cv = divmod(v, cols)
    return 1 if (ru == rv or cu == cv) else 2


def fb_hop_stats(n, rows, cols):
    """Returns (one_hop_count, two_hop_count, pct_one_hop) for FlattenedButterfly(rows,cols)."""
    one_hop = 0
    total_pairs = n * (n - 1) // 2
    for u in range(n):
        for v in range(u + 1, n):
            if fb_hop_count(u, v, cols) == 1:
                one_hop += 1
    two_hop = total_pairs - one_hop
    return one_hop, two_hop, 100.0 * one_hop / total_pairs if total_pairs else 0.0


def fb_edges(n, rows, cols):
    """Returns (row_edges, col_edges) for FlattenedButterfly(rows,cols)."""
    row_edges, col_edges = [], []
    for u in range(n):
        for v in range(u + 1, n):
            ru, cu = divmod(u, cols)
            rv, cv = divmod(v, cols)
            if ru == rv:
                row_edges.append((u, v))
            elif cu == cv:
                col_edges.append((u, v))
    return row_edges, col_edges


def grid_pos(nid, cols, step=1.3):
    r, c = divmod(nid, cols)
    return c * step, -r * step


def draw_node(ax, x, y, label, color="#4C72B0", r=NODE_R, fs=6.5):
    circ = plt.Circle((x, y), r, color=color, zorder=4, linewidth=0.8, ec="white")
    ax.add_patch(circ)
    ax.text(x, y, str(label), ha="center", va="center", fontsize=fs,
            color="white", fontweight="bold", zorder=5)


def draw_edge(ax, x0, y0, x1, y1, color, lw=1.2, alpha=0.5):
    ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, alpha=alpha, zorder=2)


# ── layout helpers ──────────────────────────────────────────────────────────────
def draw_fb_panel(ax, rows, cols, color, title, show_bw=None, step=1.3):
    """Draw a FlattenedButterfly panel on ax."""
    n = rows * cols
    pos = [grid_pos(i, cols, step) for i in range(n)]
    re, ce = fb_edges(n, rows, cols)
    C_ROW = "#5B9BD5"
    C_COL = "#70AD47"

    for u, v in re:
        x0, y0 = pos[u]; x1, y1 = pos[v]
        draw_edge(ax, x0, y0, x1, y1, C_ROW, lw=0.9, alpha=0.3)
    for u, v in ce:
        x0, y0 = pos[u]; x1, y1 = pos[v]
        draw_edge(ax, x0, y0, x1, y1, C_COL, lw=0.9, alpha=0.3)
    for i in range(n):
        x, y = pos[i]
        draw_node(ax, x, y, i, color=color, r=NODE_R * 0.85, fs=5.5 if n > 16 else 6.0)

    one, two, pct1 = fb_hop_stats(n, rows, cols)
    total = one + two
    bw_note = f"\nIntra BW: {show_bw}" if show_bw else ""
    ax.set_title(
        f"{title}\n"
        f"{rows}×{cols}={n} GPUs — 1-hop: {pct1:.0f}%  2-hop: {100-pct1:.0f}%{bw_note}",
        fontsize=8.0, color=color, fontweight="bold", pad=3)
    ax.set_aspect("equal")
    ax.axis("off")
    pad = 0.7
    ax.set_xlim(-pad, (cols - 1) * step + pad)
    ax.set_ylim(-(rows - 1) * step - pad, pad + 0.5)


def draw_nvl72_tile(ax, tile_size=8, color=C_NVL):
    """Draw NVL72 as a FullyConnected tile (all-to-all)."""
    n = tile_size
    radius = 2.2
    pos = []
    for i in range(n):
        angle = 2 * math.pi * i / n - math.pi / 2
        pos.append((radius * math.cos(angle), radius * math.sin(angle)))

    # Draw all-to-all edges
    for u in range(n):
        for v in range(u + 1, n):
            x0, y0 = pos[u]; x1, y1 = pos[v]
            draw_edge(ax, x0, y0, x1, y1, color, lw=0.6, alpha=0.12)

    for i in range(n):
        x, y = pos[i]
        draw_node(ax, x, y, i, color=color, r=NODE_R * 0.9, fs=5.5)

    ax.set_title(
        "NVL72 (hierarchical FB)\n"
        "8-GPU schematic — all-to-all within NVLink tile\n"
        "Max hops: 1 (FC within tile) | BW: 1800 GB/s intra, 50 GB/s IB inter",
        fontsize=8.0, color=color, fontweight="bold", pad=3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-radius - 0.7, radius + 0.7)
    ax.set_ylim(-radius - 0.7, radius + 0.9)

    ax.text(0.5, 0.01, "★ All pairs: 1 hop (FullyConnected model)\n"
            "  Actual NVLink fabric: 64-GPU NVLink Switch rack",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.25", fc="#FFF0F0", ec=color, alpha=0.9))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 10 — 4-topology overview
# ─────────────────────────────────────────────────────────────────────────────
def fig10_topology_overview():
    fig = plt.figure(figsize=(18, 9))
    fig.suptitle(
        "128-GPU DSE: 4 Candidate Interconnect Topologies\n"
        "Qwen3-30B-A3B (128 experts) · EP=4–64 · H100 profiling",
        fontsize=13, fontweight="bold", y=1.00)

    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)

    # ── NVL72 ────────────────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    draw_nvl72_tile(ax0, tile_size=10, color=C_NVL)

    # ── 4×4 FB panel ─────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    draw_fb_panel(ax1, 4, 4, C_4X4,
                  "4×4 FB Panel",
                  show_bw="N_WG×128 GB/s / 300 ns")

    # ── 6×6-4c FB panel ──────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    draw_fb_panel(ax2, 4, 8, C_6X6_4C,
                  "6×6-4c FB Panel",
                  show_bw="N_WG×128 GB/s / 300 ns",
                  step=1.1)

    # ── 6×6 full panel ───────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[3])
    draw_fb_panel(ax3, 6, 6, C_6X6,
                  "6×6 Full Panel",
                  show_bw="N_WG×128 GB/s / 300 ns",
                  step=1.1)

    # shared legend
    leg = [
        mpatches.Patch(color="#5B9BD5", label="Same-row link (1 hop)"),
        mpatches.Patch(color="#70AD47", label="Same-col link (1 hop)"),
        mpatches.Patch(color="#AAAAAA", label="2-hop (diff row & col)"),
    ]
    fig.legend(handles=leg, loc="lower center", ncol=3, fontsize=9,
               framealpha=0.9, edgecolor="#BBBBBB", bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    savefig(fig, "10_dse_topology_overview.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 11 — EP compatibility matrix
# ─────────────────────────────────────────────────────────────────────────────
def _4x4_ep_grid(ep):
    """4×4 panel EP layout: (dims_str, fb_layout)"""
    panel = 16
    if ep <= panel:
        if ep <= 4:    r, c = 1, ep
        elif ep == 8:  r, c = 2, 4
        elif ep == 16: r, c = 4, 4
        else:          return None
        return f"[{ep}]", f"{r}×{c} within 4×4"
    elif ep % panel == 0:
        return f"[{panel},{ep//panel}]", f"4×4 + Ring×{ep//panel}"
    return None


def _6x6_4c_ep_grid(ep):
    """6×6-4c panel (4×8) EP layout."""
    panel = 32
    if ep <= panel:
        if ep <= 8:    r, c = 1, ep
        elif ep == 16: r, c = 2, 8
        elif ep == 32: r, c = 4, 8
        else:
            # general fallback
            for rr in range(1, 5):
                if ep % rr == 0 and ep // rr <= 8:
                    r, c = rr, ep // rr
                    break
            else:
                return None
        return f"[{ep}]", f"{r}×{c} within 4×8"
    elif ep % panel == 0:
        return f"[{panel},{ep//panel}]", f"4×8 + Ring×{ep//panel}"
    return None


def _6x6_ep_grid(ep):
    """6×6 full panel EP layout."""
    panel = 36
    if ep <= panel:
        # Try full-column first (ep % 6 == 0)
        if ep <= 6:
            r, c = 1, ep
        elif ep % 6 == 0 and ep // 6 <= 6:
            r, c = ep // 6, 6
        else:
            best = None
            for rr in range(2, 7):
                if ep % rr == 0 and ep // rr <= 6:
                    cc = ep // rr
                    if best is None or abs(rr - cc) < abs(best[0] - best[1]):
                        best = (rr, cc)
            if best is None:
                return None
            r, c = best
        return f"[{ep}]", f"{r}×{c} within 6×6"
    elif ep % panel == 0:
        return f"[{panel},{ep//panel}]", f"6×6 + Ring×{ep//panel}"
    return None


def fig11_ep_compatibility():
    ep_sizes = [4, 8, 16, 32, 64, 128]
    topos = ["NVL72", "4x4", "6x6_4c", "6x6"]
    colors = [C_NVL, C_4X4, C_6X6_4C, C_6X6]

    fig, axes = plt.subplots(1, 1, figsize=(14, 6))
    ax = axes
    ax.set_title("EP Compatibility & Grid Layout for 128-GPU DSE\n"
                 "Model: Qwen3-30B-A3B (128 experts, MoE)",
                 fontsize=12, fontweight="bold")

    # Build table data
    table_data = []
    cell_colors = []
    col_labels = [f"EP={ep}" for ep in ep_sizes]
    row_labels = []

    for topo, color in zip(topos, colors):
        row = []
        row_c = []
        row_labels.append(TOPO_LABELS[topo].replace("\n", " "))
        for ep in ep_sizes:
            if topo == "NVL72":
                if ep <= 64:
                    txt = f"[{ep}]\n1800G flat"
                    c = "#FFE0E0"
                elif ep == 128:
                    txt = "[64,2]\n1800G+50G"
                    c = "#FFCCCC"
                else:
                    txt = "N/A"
                    c = "#F5F5F5"
            elif topo == "4x4":
                result = _4x4_ep_grid(ep)
                if result is None:
                    txt, c = "N/A", "#F5F5F5"
                elif ep <= 16:
                    txt = f"{result[0]}\n{result[1]}"
                    c = "#E0FFE0"
                else:
                    txt = f"{result[0]}\n{result[1]}"
                    c = "#CCFFCC"
            elif topo == "6x6_4c":
                result = _6x6_4c_ep_grid(ep)
                if result is None:
                    txt, c = "N/A", "#F5F5F5"
                elif ep <= 32:
                    txt = f"{result[0]}\n{result[1]}"
                    c = "#E0EEFF"
                else:
                    txt = f"{result[0]}\n{result[1]}"
                    c = "#C0DDFF"
            else:  # 6x6
                result = _6x6_ep_grid(ep)
                if result is None:
                    txt = "×\nIncompat."
                    c = "#EEEEEE"
                elif ep <= 36:
                    txt = f"{result[0]}\n{result[1]}"
                    c = "#F0E0FF"
                else:
                    txt = f"{result[0]}\n{result[1]}"
                    c = "#E0CCFF"
            row.append(txt)
            row_c.append(c)
        table_data.append(row)
        cell_colors.append(row_c)

    table = ax.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1.55, 2.8)

    # Color row labels
    for i, color in enumerate(colors):
        table[i + 1, -1].set_text_props(color=color, fontweight="bold")
        table[i + 1, -1].set_facecolor("#FAFAFA")

    ax.axis("off")
    ax.text(0.5, 0.02,
            "Green=single-panel | Darker=multi-panel | Grey×=incompatible EP size\n"
            "4×4 panel: 16 GPUs/panel (all EPs ≤128 compatible)  |  "
            "6×6-4c panel: 32 GPUs/panel (all EPs ≤128 compatible)  |  "
            "6×6 panel: 36 GPUs/panel (EP=32,64,128 incompatible — 32 doesn't divide 36)",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    savefig(fig, "11_dse_ep_compatibility.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 12 — BW/latency/hop summary table
# ─────────────────────────────────────────────────────────────────────────────
def fig12_bw_hop_summary():
    WG_BW = 128  # GB/s per waveguide group

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("128-GPU DSE: Topology BW / Latency / Hop Statistics",
                 fontsize=12, fontweight="bold")

    # ── Left: hop distribution bar chart ─────────────────────────────────────
    ax = axes[0]
    ax.set_title("Intra-panel hop distribution (FlattenedButterfly)", fontsize=10)

    panels = [
        ("4×4 (16 GPUs)", 4, 4, C_4X4),
        ("6×6-4c (32 GPUs)", 4, 8, C_6X6_4C),
        ("6×6 (36 GPUs)", 6, 6, C_6X6),
    ]

    x_pos = np.arange(len(panels))
    pct1_vals = []
    pct2_vals = []

    for label, r, c, color in panels:
        n = r * c
        one, two, pct1 = fb_hop_stats(n, r, c)
        pct1_vals.append(pct1)
        pct2_vals.append(100 - pct1)

    bars1 = ax.bar(x_pos, pct1_vals, color=[p[3] for p in panels],
                   alpha=0.85, label="1-hop pairs", width=0.5)
    bars2 = ax.bar(x_pos, pct2_vals, bottom=pct1_vals,
                   color=[p[3] for p in panels], alpha=0.4,
                   label="2-hop pairs", width=0.5, hatch="///")

    for i, (p1, p2) in enumerate(zip(pct1_vals, pct2_vals)):
        ax.text(i, p1 / 2, f"{p1:.0f}%\n1-hop", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")
        ax.text(i, p1 + p2 / 2, f"{p2:.0f}%\n2-hop", ha="center", va="center",
                fontsize=9, fontweight="bold", color="#333333")

    ax.text(-0.35, 50, "NVL72:\n100% 1-hop\n(FullyConn.)",
            ha="center", va="center", fontsize=8, color=C_NVL, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFE0E0", ec=C_NVL, alpha=0.9))

    ax.set_xticks(x_pos)
    ax.set_xticklabels([p[0] for p in panels], fontsize=9)
    ax.set_ylabel("Percentage of GPU pairs (%)", fontsize=10)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # ── Right: BW/latency table ───────────────────────────────────────────────
    ax2 = axes[1]
    ax2.axis("off")
    ax2.set_title("Network parameters (128-GPU EP sweep)", fontsize=10)

    wg_options = [2, 4, 6, 8]
    topo_rows = [
        ("NVL72", "FullyConnected\n(NVLink)", "64 (1 tile)",
         "1800 GB/s", "1000 ns", "—", "—",
         "1 (intra)\n50G/1000ns (IB)"),
        ("4×4 FB", "FlattenedButterfly\n4×4", "16",
         f"{wg_options[1]*WG_BW}–{wg_options[-1]*WG_BW} GB/s\n(N_WG×128)",
         "300 ns", "512 GB/s", "5000 ns",
         "Ring (inter-panel)"),
        ("6×6-4c FB", "FlattenedButterfly\n4×8", "32",
         f"{wg_options[1]*WG_BW}–{wg_options[-1]*WG_BW} GB/s\n(N_WG×128)",
         "300 ns", "512 GB/s", "5000 ns",
         "Ring (inter-panel)"),
        ("6×6 FB", "FlattenedButterfly\n6×6", "36",
         f"{wg_options[1]*WG_BW}–{wg_options[-1]*WG_BW} GB/s\n(N_WG×128)",
         "300 ns", "512 GB/s", "5000 ns",
         "Ring (inter-panel)"),
    ]

    col_labels2 = ["Topology", "ASTRA-Sim model", "GPUs/panel",
                   "Intra BW", "Intra lat", "Inter BW", "Inter lat", "Notes"]
    cell_data = [[r[i] for i in range(8)] for r in topo_rows]
    cell_c = [
        ["#FFE0E0"] * 8,
        ["#E0FFE0"] * 8,
        ["#E0EEFF"] * 8,
        ["#F0E0FF"] * 8,
    ]

    tbl = ax2.table(
        cellText=cell_data,
        colLabels=col_labels2,
        cellColours=cell_c,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.0)
    tbl.scale(1.0, 2.4)
    for j in range(8):
        tbl[0, j].set_facecolor("#DDDDDD")
        tbl[0, j].set_text_props(fontweight="bold")

    ax2.text(0.5, 0.01,
             f"Glass FB panel sweep: N_WG ∈ {{{', '.join(str(w) for w in wg_options)}}} "
             f"→ intra BW ∈ {{{', '.join(str(w*WG_BW) for w in wg_options)}}} GB/s\n"
             "Model: Qwen3-30B-A3B (128 MoE experts) · H100 profiling",
             transform=ax2.transAxes, ha="center", va="bottom", fontsize=8,
             style="italic",
             bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    plt.tight_layout()
    savefig(fig, "12_dse_bw_hop_summary.png")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating DSE topology comparison plots...")
    fig10_topology_overview()
    fig11_ep_compatibility()
    fig12_bw_hop_summary()
    print("Done.")
