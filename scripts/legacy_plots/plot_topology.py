"""
Glass panel GPU topology visualization.

Draws physical layout and ASTRA-Sim model for each system candidate:
  NVL72     — NVLink rack (72 GPUs), IB inter-rack for EP=128
  6×6-4c    — 6×6 glass panel (4 corners inactive = 32 GPUs), [4×8] ASTRA-Sim model
  4×4       — 4×4 glass panel (16 GPUs), [4×4] ASTRA-Sim model
  Multi-panel — how multiple panels connect for EP=32, 64, 128

Saves to: outputs/topology_plots/ (dpi=300)
"""

import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

OUT = "outputs/topology_plots"
DPI = 300
WG_BW = 128  # GB/s per WG
N_WG_REF = 6  # reference WG count for label


def savefig(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")


# ─────────────────────────────────────────────────────────────
# Color / style constants
# ─────────────────────────────────────────────────────────────
C_ACTIVE   = "#4C72B0"   # active GPU
C_INACTIVE = "#CCCCCC"   # corner dummy (6×6-4c)
C_EDGE_ELEC  = "#E84B3A" # electrical RDL edge
C_EDGE_OPT   = "#2CA02C" # optical WG edge
C_EDGE_INTER = "#FF7F0E" # inter-panel
C_NVLINK     = "#9467BD" # NVLink
C_IB         = "#8C564B" # InfiniBand
GPU_R = 0.38


def _gpu_patch(ax, x, y, color=C_ACTIVE, label=None, fontsize=6.5, idx=None):
    circ = plt.Circle((x, y), GPU_R, color=color, zorder=3, linewidth=0.8,
                       edgecolor="white")
    ax.add_patch(circ)
    if label:
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
                color="white" if color != C_INACTIVE else "#888", zorder=4,
                fontweight="bold")
    elif idx is not None:
        ax.text(x, y, str(idx), ha="center", va="center", fontsize=5.5,
                color="white", zorder=4)


def _edge(ax, x0, y0, x1, y1, color, lw=1.5, ls="-", alpha=0.7, zorder=2):
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, ls=ls, alpha=alpha,
            zorder=zorder, solid_capstyle="round")


def _bw_label(ax, x, y, text, color, fontsize=7):
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=color, style="italic",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color,
                      alpha=0.85, lw=0.8), zorder=5)


# ─────────────────────────────────────────────────────────────
# Figure 1: Physical layouts side by side
# ─────────────────────────────────────────────────────────────
def fig1_physical_layouts():
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle("Glass Panel GPU Physical Layouts", fontsize=14, fontweight="bold")

    _draw_nvl72_physical(axes[0])
    _draw_6x6_4c_physical(axes[1])
    _draw_4x4_physical(axes[2])

    fig.tight_layout()
    savefig(fig, "01_physical_layouts.png")


def _draw_nvl72_physical(ax):
    ax.set_title("NVL72 (H100 NVLink rack)", fontweight="bold", fontsize=11)
    ax.set_aspect("equal")
    ax.axis("off")

    # 72 GPUs in a rack — draw as 8×9 grid
    cols, rows = 8, 9
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx >= 72:
                break
            x, y = j * 1.1, (rows - 1 - i) * 1.1
            _gpu_patch(ax, x, y, color=C_NVLINK, idx=idx, fontsize=5.5)

    # Draw rack border
    ax.add_patch(plt.Rectangle((-0.6, -0.6), cols * 1.1 + 0.1, rows * 1.1 + 0.1,
                                fill=False, edgecolor=C_NVLINK, lw=2.5,
                                linestyle="-", zorder=1))
    ax.text(cols * 1.1 / 2 - 0.5, rows * 1.1 + 0.3, "NVLink Rack (72 GPUs)",
            ha="center", fontsize=9, color=C_NVLINK, fontweight="bold")

    # BW annotation
    ax.text(cols * 1.1 / 2 - 0.5, -1.0,
            "Intra-rack: NVLink 1800 GB/s / 1000 ns\n"
            "EP≤64: 1D flat ring | EP=128: [64,2] + IB 50 GB/s",
            ha="center", fontsize=8, color=C_NVLINK,
            bbox=dict(boxstyle="round", fc="lavender", alpha=0.8))

    ax.set_xlim(-1, cols * 1.1 + 0.5)
    ax.set_ylim(-1.8, rows * 1.1 + 0.8)


def _draw_6x6_4c_physical(ax):
    ax.set_title("6×6-4c Glass Panel (32 active GPUs)", fontweight="bold", fontsize=11)
    ax.set_aspect("equal")
    ax.axis("off")

    CORNERS = {(0, 0), (0, 5), (5, 0), (5, 5)}
    S = 1.1  # grid spacing
    active_idx = 0

    pos = {}
    for r in range(6):
        for c in range(6):
            x, y = c * S, (5 - r) * S
            if (r, c) in CORNERS:
                _gpu_patch(ax, x, y, color=C_INACTIVE, label="✕", fontsize=9)
            else:
                _gpu_patch(ax, x, y, color=C_ACTIVE, idx=active_idx, fontsize=5.5)
                pos[(r, c)] = (x, y)
                active_idx += 1

    # Draw edges: adjacent = electrical, non-adjacent same row/col = optical WG
    drawn = set()
    for r in range(6):
        for c in range(6):
            if (r, c) in CORNERS:
                continue
            x0, y0 = c * S, (5 - r) * S
            # Row neighbors
            for dc in range(1, 6):
                c2 = c + dc
                if c2 >= 6:
                    break
                if (r, c2) in CORNERS:
                    continue
                x2, y2 = c2 * S, (5 - r) * S
                key = tuple(sorted([(r, c), (r, c2)]))
                if key not in drawn:
                    color = C_EDGE_ELEC if dc == 1 else C_EDGE_OPT
                    lw    = 1.8 if dc == 1 else 0.9
                    ls    = "-" if dc == 1 else "--"
                    _edge(ax, x0, y0, x2, y2, color, lw=lw, ls=ls, alpha=0.6)
                    drawn.add(key)
            # Col neighbors
            for dr in range(1, 6):
                r2 = r + dr
                if r2 >= 6:
                    break
                if (r2, c) in CORNERS:
                    continue
                x2, y2 = c * S, (5 - r2) * S
                key = tuple(sorted([(r, c), (r2, c)]))
                if key not in drawn:
                    color = C_EDGE_ELEC if dr == 1 else C_EDGE_OPT
                    lw    = 1.8 if dr == 1 else 0.9
                    ls    = "-" if dr == 1 else "--"
                    _edge(ax, x0, y0, x2, y2, color, lw=lw, ls=ls, alpha=0.6)
                    drawn.add(key)

    # Legend
    legend_handles = [
        mpatches.Patch(color=C_ACTIVE,   label="Active GPU (32)"),
        mpatches.Patch(color=C_INACTIVE, label="Corner (inactive, 4)"),
        plt.Line2D([0],[0], color=C_EDGE_ELEC, lw=2.5, label="Electrical RDL  400 GB/s / 10 ns"),
        plt.Line2D([0],[0], color=C_EDGE_OPT,  lw=1.5, ls="--",
                   label=f"Optical WG  {N_WG_REF}×128={N_WG_REF*WG_BW} GB/s / 300 ns"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", fontsize=7.5,
              bbox_to_anchor=(0.5, -0.22), ncol=2,
              framealpha=0.9, edgecolor="gray")

    ax.text(5 * S / 2, -0.8,
            "ASTRA-Sim model: [4×8]=32  (rectangular approx)\n"
            "Both dims → optical BW (conservative)",
            ha="center", fontsize=8, color="#555",
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))

    ax.set_xlim(-0.7, 6 * S)
    ax.set_ylim(-1.6, 6 * S + 0.3)


def _draw_4x4_physical(ax):
    ax.set_title("4×4 Glass Panel (16 active GPUs)", fontweight="bold", fontsize=11)
    ax.set_aspect("equal")
    ax.axis("off")

    S = 1.4
    for r in range(4):
        for c in range(4):
            idx = r * 4 + c
            x, y = c * S, (3 - r) * S
            _gpu_patch(ax, x, y, color="#2CA02C", idx=idx, fontsize=7)

    # Draw edges
    drawn = set()
    for r in range(4):
        for c in range(4):
            x0, y0 = c * S, (3 - r) * S
            for dc in range(1, 4):
                c2 = c + dc
                if c2 >= 4:
                    break
                x2, y2 = c2 * S, (3 - r) * S
                key = tuple(sorted([(r, c), (r, c2)]))
                if key not in drawn:
                    color = C_EDGE_ELEC if dc == 1 else C_EDGE_OPT
                    lw    = 2.0 if dc == 1 else 1.0
                    ls    = "-" if dc == 1 else "--"
                    _edge(ax, x0, y0, x2, y2, color, lw=lw, ls=ls, alpha=0.65)
                    drawn.add(key)
            for dr in range(1, 4):
                r2 = r + dr
                if r2 >= 4:
                    break
                x2, y2 = c * S, (3 - r2) * S
                key = tuple(sorted([(r, c), (r2, c)]))
                if key not in drawn:
                    color = C_EDGE_ELEC if dr == 1 else C_EDGE_OPT
                    lw    = 2.0 if dr == 1 else 1.0
                    ls    = "-" if dr == 1 else "--"
                    _edge(ax, x0, y0, x2, y2, color, lw=lw, ls=ls, alpha=0.65)
                    drawn.add(key)

    legend_handles = [
        mpatches.Patch(color="#2CA02C", label="Active GPU (16)"),
        plt.Line2D([0],[0], color=C_EDGE_ELEC, lw=2.5, label="Electrical RDL  400 GB/s / 10 ns"),
        plt.Line2D([0],[0], color=C_EDGE_OPT,  lw=1.5, ls="--",
                   label=f"Optical WG  {N_WG_REF}×128={N_WG_REF*WG_BW} GB/s / 300 ns"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", fontsize=7.5,
              bbox_to_anchor=(0.5, -0.18), ncol=2,
              framealpha=0.9, edgecolor="gray")

    ax.text(3 * S / 2, -0.85,
            "ASTRA-Sim model: [4×4]=16  (exact panel shape)\n"
            "Both dims → optical BW (conservative)",
            ha="center", fontsize=8, color="#555",
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))

    ax.set_xlim(-0.8, 4 * S + 0.2)
    ax.set_ylim(-1.5, 4 * S + 0.3)


# ─────────────────────────────────────────────────────────────
# Figure 2: ASTRA-Sim topology models (ring dims)
# ─────────────────────────────────────────────────────────────
def fig2_astra_sim_models():
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle("ASTRA-Sim Topology Models (EP=32 single-panel example)",
                 fontsize=13, fontweight="bold")

    _draw_nvl72_model(axes[0])
    _draw_6x6_model(axes[1])
    _draw_4x4_model(axes[2])

    fig.tight_layout()
    savefig(fig, "02_astra_sim_models.png")


def _ring_coords(n, cx, cy, radius):
    angles = [math.pi / 2 + 2 * math.pi * i / n for i in range(n)]
    return [(cx + radius * math.cos(a), cy + radius * math.sin(a)) for a in angles]


def _draw_ring(ax, coords, color, lw=2.0, ls="-"):
    n = len(coords)
    for i in range(n):
        x0, y0 = coords[i]
        x1, y1 = coords[(i + 1) % n]
        _edge(ax, x0, y0, x1, y1, color, lw=lw, ls=ls, alpha=0.6)


def _draw_nvl72_model(ax):
    ax.set_title("NVL72 — EP=32: 1D flat ring [32]", fontweight="bold", fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")

    n = 32
    coords = _ring_coords(n, 0, 0, 3.5)
    _draw_ring(ax, coords, C_NVLINK, lw=2.5)
    for i, (x, y) in enumerate(coords):
        _gpu_patch(ax, x, y, color=C_NVLINK, idx=i, fontsize=5.5)

    _bw_label(ax, 0, 0, f"Ring: 1800 GB/s\n1000 ns", C_NVLINK, fontsize=8.5)
    ax.text(0, -5.0, "EP ≤ 64: single ring  [EP]\nAll GPUs in 1 NVLink rack",
            ha="center", fontsize=8, color=C_NVLINK,
            bbox=dict(boxstyle="round", fc="lavender", alpha=0.8))
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.8, 5.5)


def _draw_6x6_model(ax):
    ax.set_title("6×6-4c — EP=32: [4×8] 2D torus", fontweight="bold", fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")

    ROWS, COLS = 4, 8
    S = 1.1
    xoff = -(COLS - 1) * S / 2
    yoff = -(ROWS - 1) * S / 2
    idx = 0
    pos = {}
    for r in range(ROWS):
        for c in range(COLS):
            x, y = xoff + c * S, yoff + (ROWS - 1 - r) * S
            pos[(r, c)] = (x, y)
            _gpu_patch(ax, x, y, color=C_ACTIVE, idx=idx, fontsize=5)
            idx += 1

    # dim0 edges (row direction, optical)
    for r in range(ROWS):
        for c in range(COLS):
            c2 = (c + 1) % COLS
            x0, y0 = pos[(r, c)]
            x1, y1 = pos[(r, c2)]
            if c2 != 0:  # non-wrap
                _edge(ax, x0, y0, x1, y1, C_EDGE_OPT, lw=1.8, alpha=0.7)
            else:  # wrap-around shown as arrow
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="->", color=C_EDGE_OPT,
                                            lw=1.2, connectionstyle="arc3,rad=0.5"),
                            zorder=2)

    # dim1 edges (col direction, optical)
    for r in range(ROWS):
        r2 = (r + 1) % ROWS
        for c in range(COLS):
            x0, y0 = pos[(r, c)]
            x1, y1 = pos[(r2, c)]
            if r2 != 0:
                _edge(ax, x0, y0, x1, y1, C_EDGE_OPT, lw=1.2, ls="--", alpha=0.6)
            else:
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="->", color=C_EDGE_OPT,
                                            lw=0.9, ls="dashed",
                                            connectionstyle="arc3,rad=-0.5"),
                            zorder=2)

    ax.text(0, yoff - 1.1,
            f"dim0 (size=8, row ring):  optical {N_WG_REF}×128={N_WG_REF*WG_BW} GB/s / 300 ns\n"
            f"dim1 (size=4, col ring):  optical {N_WG_REF}×128={N_WG_REF*WG_BW} GB/s / 300 ns",
            ha="center", fontsize=8, color="#333",
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))
    ax.set_xlim(xoff - 1, xoff + COLS * S + 0.5)
    ax.set_ylim(yoff - 2.0, yoff + ROWS * S + 0.5)


def _draw_4x4_model(ax):
    ax.set_title("4×4 — EP=16: [4×4] 2D torus", fontweight="bold", fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")

    ROWS, COLS = 4, 4
    S = 1.4
    xoff = -(COLS - 1) * S / 2
    yoff = -(ROWS - 1) * S / 2
    idx = 0
    pos = {}
    for r in range(ROWS):
        for c in range(COLS):
            x, y = xoff + c * S, yoff + (ROWS - 1 - r) * S
            pos[(r, c)] = (x, y)
            _gpu_patch(ax, x, y, color="#2CA02C", idx=idx, fontsize=6)
            idx += 1

    for r in range(ROWS):
        for c in range(COLS):
            c2 = (c + 1) % COLS
            x0, y0 = pos[(r, c)]
            x1, y1 = pos[(r, c2)]
            if c2 != 0:
                _edge(ax, x0, y0, x1, y1, C_EDGE_OPT, lw=2.0, alpha=0.7)
            else:
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="->", color=C_EDGE_OPT,
                                            lw=1.3, connectionstyle="arc3,rad=0.5"),
                            zorder=2)

    for r in range(ROWS):
        r2 = (r + 1) % ROWS
        for c in range(COLS):
            x0, y0 = pos[(r, c)]
            x1, y1 = pos[(r2, c)]
            if r2 != 0:
                _edge(ax, x0, y0, x1, y1, C_EDGE_OPT, lw=1.3, ls="--", alpha=0.65)
            else:
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="->", color=C_EDGE_OPT,
                                            lw=1.0, ls="dashed",
                                            connectionstyle="arc3,rad=-0.5"),
                            zorder=2)

    ax.text(0, yoff - 1.3,
            f"dim0 (size=4, row ring):  optical {N_WG_REF}×128={N_WG_REF*WG_BW} GB/s / 300 ns\n"
            f"dim1 (size=4, col ring):  optical {N_WG_REF}×128={N_WG_REF*WG_BW} GB/s / 300 ns",
            ha="center", fontsize=8, color="#333",
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))
    ax.set_xlim(xoff - 1.3, xoff + COLS * S + 0.7)
    ax.set_ylim(yoff - 2.2, yoff + ROWS * S + 0.5)


# ─────────────────────────────────────────────────────────────
# Figure 3: Multi-panel EP comparison
# ─────────────────────────────────────────────────────────────
def fig3_multi_panel():
    """Show how panels stack for EP=32, 64 (NVL72 vs 6×6-4c vs 4×4)."""
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle("Multi-panel Topology: EP=16 / EP=32 / EP=64\n"
                 "(each box = one glass panel or NVLink rack segment)",
                 fontsize=13, fontweight="bold")

    ep_vals = [16, 32, 64]
    topo_names = ["NVL72", "6×6-4c", "4×4"]

    for col, ep in enumerate(ep_vals):
        axes[0][col].set_title(f"EP = {ep}", fontsize=12, fontweight="bold",
                               color="darkblue")

    for row, topo in enumerate(topo_names):
        axes[row][0].set_ylabel(topo, fontsize=12, fontweight="bold",
                                rotation=90, labelpad=10)

    for col, ep in enumerate(ep_vals):
        _draw_multi_nvl72(axes[0][col], ep)
        _draw_multi_6x6(axes[1][col], ep)
        _draw_multi_4x4(axes[2][col], ep)

    for ax in axes.flat:
        ax.axis("off")

    fig.tight_layout()
    savefig(fig, "03_multi_panel_ep_comparison.png")


def _panel_box(ax, x, y, w, h, color, label, n_gpus, fontsize=8):
    rect = plt.Rectangle((x, y), w, h, facecolor=color, alpha=0.25,
                          edgecolor=color, lw=2.0)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2 + 0.15, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=color)
    ax.text(x + w / 2, y + h / 2 - 0.25, f"{n_gpus} GPUs", ha="center",
            va="center", fontsize=fontsize - 1, color=color)


def _inter_link(ax, x0, x1, y, color=C_EDGE_INTER, bw_label=None):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="<->", color=color, lw=2.0))
    if bw_label:
        xm = (x0 + x1) / 2
        ax.text(xm, y + 0.15, bw_label, ha="center", fontsize=7,
                color=color, style="italic")


def _draw_multi_nvl72(ax, ep):
    ax.set_aspect("equal")
    if ep <= 64:
        _panel_box(ax, 0.2, 0.3, 3.5, 1.5, C_NVLINK, "NVLink Rack", ep)
        ax.text(2.0, 0.0, f"1D ring [{ep}]\n1800 GB/s / 1000 ns",
                ha="center", fontsize=8, color=C_NVLINK)
    else:  # EP=128
        _panel_box(ax, 0.1, 0.3, 2.2, 1.5, C_NVLINK, "Rack-0", 64)
        _panel_box(ax, 2.6, 0.3, 2.2, 1.5, C_NVLINK, "Rack-1", 64)
        _inter_link(ax, 2.3, 2.6, 1.05, C_IB, "IB 50 GB/s")
        ax.text(2.5, 0.0, "[64, 2]  dim0=1800G / dim1=50G",
                ha="center", fontsize=7.5, color=C_NVLINK)
    ax.set_xlim(0, 5.2)
    ax.set_ylim(-0.3, 2.5)


def _draw_multi_6x6(ax, ep):
    ax.set_aspect("equal")
    panel_w, panel_h = 1.8, 1.4
    gap = 0.5

    def _dims(ep_):
        if ep_ <= 4:
            return f"[{ep_}]"
        elif ep_ <= 8:
            return "[2×4]"
        elif ep_ <= 16:
            return "[4×4]"
        elif ep_ <= 32:
            return "[4×8]"
        elif ep_ <= 64:
            return "[4×8×2]"
        else:
            return "[4×8×4]"

    n_panels = max(1, ep // 32)
    total_w = n_panels * panel_w + (n_panels - 1) * gap
    xstart = (5.5 - total_w) / 2

    for i in range(n_panels):
        x = xstart + i * (panel_w + gap)
        _panel_box(ax, x, 0.4, panel_w, panel_h, C_ACTIVE,
                   f"Panel {i}", min(32, ep - i * 32) if i < n_panels - 1 else ep - i * 32)
        if i < n_panels - 1:
            _inter_link(ax, x + panel_w, x + panel_w + gap, 1.1,
                        C_EDGE_INTER, "512 GB/s / 5μs")

    ax.text(5.5 / 2, 0.05,
            f"{_dims(ep)}  optical {N_WG_REF}×128={N_WG_REF*WG_BW} GB/s / 300 ns",
            ha="center", fontsize=8, color=C_ACTIVE)
    ax.set_xlim(0, 5.5)
    ax.set_ylim(-0.2, 2.4)


def _draw_multi_4x4(ax, ep):
    ax.set_aspect("equal")
    panel_w, panel_h = 1.5, 1.3
    gap = 0.4

    def _dims(ep_):
        if ep_ <= 4:
            return f"[{ep_}]"
        elif ep_ <= 8:
            return "[2×4]"
        elif ep_ <= 16:
            return "[4×4]"
        elif ep_ <= 32:
            return "[4×4×2]"
        elif ep_ <= 64:
            return "[4×4×4]"
        else:
            return "[4×4×8]"

    n_panels = max(1, ep // 16)
    total_w = n_panels * panel_w + (n_panels - 1) * gap
    xstart = (5.5 - total_w) / 2

    for i in range(n_panels):
        x = xstart + i * (panel_w + gap)
        gpus_this = min(16, ep - i * 16) if i < n_panels - 1 else ep - i * 16
        _panel_box(ax, x, 0.4, panel_w, panel_h, "#2CA02C",
                   f"P{i}", gpus_this, fontsize=7.5)
        if i < n_panels - 1:
            _inter_link(ax, x + panel_w, x + panel_w + gap, 1.05,
                        C_EDGE_INTER, "512 G" if n_panels <= 5 else "")

    ax.text(5.5 / 2, 0.05,
            f"{_dims(ep)}  optical {N_WG_REF}×128={N_WG_REF*WG_BW} GB/s / 300 ns",
            ha="center", fontsize=8, color="#2CA02C")
    ax.set_xlim(0, 5.5)
    ax.set_ylim(-0.2, 2.4)


# ─────────────────────────────────────────────────────────────
# Figure 4: BW/latency summary table + topology comparison
# ─────────────────────────────────────────────────────────────
def fig4_bw_summary():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Topology Parameter Summary", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.axis("off")
    ax.set_title("Connectivity Parameters", fontweight="bold", fontsize=11)

    headers = ["System", "Tier", "BW (GB/s)", "Latency (ns)", "ASTRA-Sim Dim"]
    rows = [
        ["NVL72",    "Intra-rack (NVLink)", "1800", "1000", "dim0 (ring)"],
        ["NVL72",    "Inter-rack (IB)",     "50",   "1000", "dim1 (EP>64)"],
        ["6×6-4c",   "Intra-panel optical", f"{N_WG_REF}×128={N_WG_REF*WG_BW}", "300",  "dim0, dim1"],
        ["6×6-4c",   "Adjacent electrical", "400",  "10",   "—  (not sep. modeled)"],
        ["6×6-4c",   "Inter-panel fiber",   "512",  "5000", "dim2"],
        ["4×4",      "Intra-panel optical", f"{N_WG_REF}×128={N_WG_REF*WG_BW}", "300",  "dim0, dim1"],
        ["4×4",      "Adjacent electrical", "400",  "10",   "—  (not sep. modeled)"],
        ["4×4",      "Inter-panel fiber",   "512",  "5000", "dim2"],
    ]
    row_colors = [
        ["lavender", "lavender", "lavender", "lavender", "lavender"],
        ["lavender", "lavender", "lavender", "lavender", "lavender"],
        ["lightcyan", "lightcyan", "lightcyan", "lightcyan", "lightcyan"],
        ["#f0fff0",  "#f0fff0",  "#f0fff0",  "#f0fff0",  "#f0fff0"],
        ["lightcyan", "lightcyan", "lightcyan", "lightcyan", "lightcyan"],
        ["#e8ffe8", "#e8ffe8", "#e8ffe8", "#e8ffe8", "#e8ffe8"],
        ["#f0fff0",  "#f0fff0",  "#f0fff0",  "#f0fff0",  "#f0fff0"],
        ["#e8ffe8", "#e8ffe8", "#e8ffe8", "#e8ffe8", "#e8ffe8"],
    ]
    tbl = ax.table(cellText=rows, colLabels=headers,
                   cellColours=row_colors, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.2, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight="bold")

    ax2 = axes[1]
    ax2.axis("off")
    ax2.set_title("ASTRA-Sim Topology Dimensions per EP", fontweight="bold", fontsize=11)

    headers2 = ["EP", "NVL72", "6×6-4c [4×8]", "4×4 [4×4]"]
    rows2 = [
        ["4",   "[4]",    "[4]",      "[4]"],
        ["8",   "[8]",    "[2×4]",    "[2×4]"],
        ["16",  "[16]",   "[4×4]",    "[4×4]"],
        ["32",  "[32]",   "[4×8]  ★", "[4×4×2]"],
        ["64",  "[64]",   "[4×8×2]",  "[4×4×4]"],
        ["128", "[64×2]†","[4×8×4]",  "[4×4×8]"],
    ]
    row_colors2 = [["white"]*4]*4 + [["#fffbe6"]*4]*2
    for i in [3, 4, 5]:
        row_colors2[i] = ["#fffbe6"] * 4

    tbl2 = ax2.table(cellText=rows2, colLabels=headers2,
                     cellColours=row_colors2, loc="center", cellLoc="center")
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(10)
    tbl2.scale(1.3, 1.7)
    for (r, c), cell in tbl2.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight="bold")

    ax2.text(0.5, 0.02,
             "★ Key difference: 6×6-4c EP=32 stays on 1 panel [4×8]\n"
             "   vs 4×4 EP=32 spans 2 panels [4×4×2]\n"
             "† NVL72 EP=128: IB inter-rack (50 GB/s)",
             transform=ax2.transAxes, ha="center", fontsize=8.5, va="bottom",
             bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))

    fig.tight_layout()
    savefig(fig, "04_bw_summary_table.png")


# ─────────────────────────────────────────────────────────────
# Figure 5: 6×6-4c physical + model side by side (detail)
# ─────────────────────────────────────────────────────────────
def fig5_6x6_detail():
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("6×6-4c Panel: Physical vs ASTRA-Sim Model", fontsize=13,
                 fontweight="bold")

    axes[0].set_title("Physical Layout\n(6×6 grid, 4 corners inactive = 32 GPUs)",
                      fontweight="bold", fontsize=11)
    _draw_6x6_4c_physical(axes[0])

    axes[1].set_title(f"ASTRA-Sim Model\n([4×8] ring torus, uniform optical BW = {N_WG_REF}×128={N_WG_REF*WG_BW} GB/s)",
                      fontweight="bold", fontsize=11)
    _draw_6x6_model(axes[1])

    # Annotation arrow between subplots
    fig.text(0.5, 0.5, "→", ha="center", va="center", fontsize=28,
             color="gray", fontweight="bold")
    fig.text(0.5, 0.44, "rectangular\napproximation\n(ASTRA-Sim\nrequires rect.)",
             ha="center", va="center", fontsize=8.5, color="gray",
             bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

    fig.tight_layout()
    savefig(fig, "05_6x6_physical_vs_model.png")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT, exist_ok=True)
    print("Generating topology plots...")
    fig1_physical_layouts()
    fig2_astra_sim_models()
    fig3_multi_panel()
    fig4_bw_summary()
    fig5_6x6_detail()
    print(f"\nAll topology figures → {OUT}/")


if __name__ == "__main__":
    main()
