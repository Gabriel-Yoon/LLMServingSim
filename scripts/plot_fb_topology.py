"""
FlattenedButterfly topology visualization.

Verifies the ASTRA-Sim FlattenedButterfly implementation:
  - Node grid layout (rows x cols)
  - Same-row direct links (1 hop, blue)
  - Same-col direct links (1 hop, green)
  - 2-hop paths via intermediate (src_row*cols + dst_col)
  - Comparison with Ring topology on same N nodes
  - Hop-count statistics across panel sizes

Saves:
  outputs/topology_plots/06_fb_topology_main.png
  outputs/topology_plots/07_fb_panel_variants.png
  outputs/topology_plots/08_fb_hop_stats.png

Usage (from repo root):
  python scripts/plot_fb_topology.py
"""

import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.patches import FancyArrowPatch

OUT = "outputs/topology_plots"
DPI = 300

# ── colour palette ────────────────────────────────────────────────────────────
C_NODE      = "#4C72B0"   # GPU node
C_ROW_LINK  = "#5B9BD5"   # same-row direct link
C_COL_LINK  = "#70AD47"   # same-col direct link
C_HOP2      = "#FF7F0E"   # 2-hop example path
C_INTER     = "#D9534F"   # intermediate node for 2-hop
C_RING_LINK = "#888888"   # Ring links
C_RING_HOP  = "#FF7F0E"   # Ring path
NODE_R      = 0.32


def savefig(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def auto_factorize(n):
    """Largest factor of n that is <= sqrt(n) → (rows, cols) with rows <= cols."""
    rows = 1
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            rows = i
    return rows, n // rows


def grid_pos(nid, cols, step=1.4):
    """Return (x, y) for node nid in a rows×cols grid."""
    r = nid // cols
    c = nid % cols
    return c * step, -r * step


def ring_pos(nid, n, radius=None):
    """Return (x, y) for node nid on a Ring of n nodes."""
    if radius is None:
        radius = max(1.8, n * 0.22)
    angle = 2 * math.pi * nid / n - math.pi / 2
    return radius * math.cos(angle), radius * math.sin(angle)


def draw_node(ax, x, y, label, color=C_NODE, r=NODE_R, fs=7.5, highlight=False):
    ec = "#D9534F" if highlight else "white"
    lw = 2.0 if highlight else 0.8
    circ = plt.Circle((x, y), r, color=color, zorder=4,
                       linewidth=lw, edgecolor=ec)
    ax.add_patch(circ)
    ax.text(x, y, str(label), ha="center", va="center",
            fontsize=fs, color="white", zorder=5, fontweight="bold")


def draw_edge(ax, x0, y0, x1, y1, color, lw=1.2, ls="-", alpha=0.6, zorder=2):
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, linestyle=ls,
            alpha=alpha, zorder=zorder, solid_capstyle="round")


def draw_arrow(ax, x0, y0, x1, y1, color, lw=2.0, ls="-",
               arrowsize=12, zorder=6):
    """Draw a directed arrow from (x0,y0) to (x1,y1)."""
    dx, dy = x1 - x0, y1 - y0
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=f"->,head_width=0.18,head_length=0.14",
                                color=color, lw=lw, linestyle=ls),
                zorder=zorder)


def label_box(ax, x, y, text, color, fs=6.5):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color=color,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=color,
                      alpha=0.9, lw=0.9), zorder=7)


# ─────────────────────────────────────────────────────────────────────────────
# FB topology helpers
# ─────────────────────────────────────────────────────────────────────────────
def fb_edges(n, rows, cols):
    """Return (row_edges, col_edges) — all undirected links in the FB topology."""
    row_edges, col_edges = [], []
    for u in range(n):
        for v in range(u + 1, n):
            ur, uc = u // cols, u % cols
            vr, vc = v // cols, v % cols
            if ur == vr:
                row_edges.append((u, v))
            elif uc == vc:
                col_edges.append((u, v))
    return row_edges, col_edges


def fb_hop_count(src, dst, cols):
    if src == dst:
        return 0
    sr, sc = src // cols, src % cols
    dr, dc = dst // cols, dst % cols
    if sr == dr or sc == dc:
        return 1
    return 2


# ─────────────────────────────────────────────────────────────────────────────
# Figure 06 — main overview: FB 4×4  vs  Ring 16
# ─────────────────────────────────────────────────────────────────────────────
def fig06_main():
    rows, cols, n = 4, 4, 16
    step = 1.4

    # pick a 2-hop example: src=(0,0)=0, dst=(3,3)=15
    src, dst = 0, 15
    interm = (src // cols) * cols + (dst % cols)   # (0)*4 + 3 = 3

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("FlattenedButterfly (4×4, N=16) vs Ring (N=16)",
                 fontsize=13, fontweight="bold", y=1.01)

    # ── Left: FlattenedButterfly 4×4 ─────────────────────────────────────────
    ax = axes[0]
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("FlattenedButterfly (4 rows × 4 cols)\n"
                 "Blue = same-row link (1 hop)   Green = same-col link (1 hop)\n"
                 "Orange arrows = 2-hop example: node 0 → node 15",
                 fontsize=8.5, loc="left")

    pos = [grid_pos(i, cols, step) for i in range(n)]

    re, ce = fb_edges(n, rows, cols)
    # draw row edges (light blue)
    for u, v in re:
        x0, y0 = pos[u]; x1, y1 = pos[v]
        draw_edge(ax, x0, y0, x1, y1, C_ROW_LINK, lw=1.5, alpha=0.45)
    # draw col edges (light green)
    for u, v in ce:
        x0, y0 = pos[u]; x1, y1 = pos[v]
        draw_edge(ax, x0, y0, x1, y1, C_COL_LINK, lw=1.5, alpha=0.45)

    # draw 2-hop path: src → interm → dst (with thicker coloured arrows)
    xi, yi = pos[interm]
    xs, ys = pos[src]
    xd, yd = pos[dst]

    # highlight the two hops
    ax.annotate("", xy=(xi, yi), xytext=(xs, ys),
                arrowprops=dict(arrowstyle="->,head_width=0.22,head_length=0.16",
                                color=C_HOP2, lw=2.4, connectionstyle="arc3,rad=0.0"),
                zorder=8)
    ax.annotate("", xy=(xd, yd), xytext=(xi, yi),
                arrowprops=dict(arrowstyle="->,head_width=0.22,head_length=0.16",
                                color=C_HOP2, lw=2.4, connectionstyle="arc3,rad=0.0"),
                zorder=8)

    # label hops
    ax.text((xs + xi) / 2 + 0.05, (ys + yi) / 2 - 0.25,
            "hop 1\n(same row)", ha="center", fontsize=7, color=C_HOP2,
            fontweight="bold")
    ax.text((xi + xd) / 2 + 0.35, (yi + yd) / 2,
            "hop 2\n(same col)", ha="center", fontsize=7, color=C_HOP2,
            fontweight="bold")

    # intermediate node annotation
    label_box(ax, xi - 0.55, yi + 0.05,
              f"intermediate\n(same row as src,\nsame col as dst)", C_INTER, fs=6.5)
    ax.plot([xi - 0.30, xi - NODE_R], [yi, yi], color=C_INTER, lw=0.9, ls="--")

    # draw nodes
    for i in range(n):
        x, y = pos[i]
        color = C_INTER if i == interm else C_NODE
        highlight = (i == src or i == dst)
        draw_node(ax, x, y, i, color=color, highlight=highlight)

    # annotate src / dst
    xs2, ys2 = pos[src]
    xd2, yd2 = pos[dst]
    ax.text(xs2 - 0.45, ys2, "src\n(0,0)", ha="center", fontsize=7,
            color=C_HOP2, fontweight="bold")
    ax.text(xd2 + 0.52, yd2, "dst\n(3,3)", ha="center", fontsize=7,
            color=C_HOP2, fontweight="bold")

    # row/col grid labels
    for r in range(rows):
        x, y = pos[r * cols]
        ax.text(x - 0.65, y, f"row {r}", ha="right", va="center",
                fontsize=7, color="#555")
    for c in range(cols):
        x, y = pos[c]
        ax.text(x, y + 0.62, f"col {c}", ha="center", va="bottom",
                fontsize=7, color="#555")

    # formula note
    ax.text(0.02, 0.03,
            "Routing rule:\n"
            "  same row OR same col  →  1 hop (direct)\n"
            "  diff row AND diff col →  2 hops via (src_row × cols + dst_col)",
            transform=ax.transAxes, fontsize=7.5, va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", fc="#F9F9F9", ec="#CCCCCC",
                      lw=0.8, alpha=0.9))

    # legend
    legend_items = [
        mpatches.Patch(color=C_ROW_LINK, label="Same-row link  (1 hop, direct)"),
        mpatches.Patch(color=C_COL_LINK, label="Same-col link  (1 hop, direct)"),
        mpatches.Patch(color=C_HOP2,     label="2-hop path example  (0 → 3 → 15)"),
        mpatches.Patch(color=C_INTER,    label="Intermediate node (node 3)"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=7.5,
              framealpha=0.9, edgecolor="#BBBBBB")

    margin = 1.0
    ax.set_xlim(-margin, (cols - 1) * step + margin)
    ax.set_ylim(-(rows - 1) * step - margin, margin + 0.8)

    # ── Right: Ring 16 ────────────────────────────────────────────────────────
    ax = axes[1]
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Ring (N=16) — shown for comparison\n"
                 "Gray = ring links   Orange = shortest path: node 0 → node 15\n"
                 "(path length = min(|dst-src|, N-|dst-src|) = 1 hop)",
                 fontsize=8.5, loc="left")

    rpos = [ring_pos(i, n) for i in range(n)]

    # draw ring edges
    for i in range(n):
        x0, y0 = rpos[i]
        x1, y1 = rpos[(i + 1) % n]
        draw_edge(ax, x0, y0, x1, y1, C_RING_LINK, lw=1.5, alpha=0.5)

    # shortest path: 0 → 15 is 1 hop (they're adjacent in Ring)
    # Actually in a ring of 16, 0→15 = 1 hop (going backward)
    ring_path_0_to_15 = [0, 15]  # 1-hop going backward
    for a, b in zip(ring_path_0_to_15, ring_path_0_to_15[1:]):
        x0, y0 = rpos[a]; x1, y1 = rpos[b]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->,head_width=0.22,head_length=0.16",
                                    color=C_RING_HOP, lw=2.4),
                    zorder=8)

    # also show a "hard" pair: 0 → 8 = 8 hops (worst case)
    hard_src, hard_dst = 0, 8
    hard_path = list(range(9))  # 0,1,2,...,8
    for a, b in zip(hard_path, hard_path[1:]):
        x0, y0 = rpos[a]; x1, y1 = rpos[b]
        ax.plot([x0, x1], [y0, y1], color="#AA4499", lw=2.0,
                alpha=0.7, zorder=6, linestyle="--")

    # draw Ring nodes
    for i in range(n):
        x, y = rpos[i]
        highlight = (i == 0 or i == 15)
        col = C_NODE if i not in (hard_src, hard_dst) else C_NODE
        draw_node(ax, x, y, i, color=col, highlight=highlight)

    # annotations for the hard pair
    x0, y0 = rpos[hard_src]; x8, y8 = rpos[hard_dst]
    ax.text(x0 + 0.1, y0 - 0.55, "node 0", ha="center", fontsize=7, color="#AA4499")
    ax.text(x8 - 0.1, y8 + 0.55, "node 8\n(8 hops away!)", ha="center",
            fontsize=7, color="#AA4499")

    # hop comparison callout box
    ax.text(0.50, 0.04,
            "Hop count comparison (N=16):\n"
            "  FlattenedButterfly  max = 2  (always ≤ 2)\n"
            "  Ring                max = N/2 = 8",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFF9E6", ec="#DDAA00",
                      lw=1.0, alpha=0.95))

    legend_r = [
        mpatches.Patch(color=C_RING_LINK, label="Ring link (1-hop adjacent)"),
        mpatches.Patch(color=C_RING_HOP,  label="0 → 15: 1 hop (adjacent)"),
        mpatches.Patch(color="#AA4499",    label="0 → 8: 8 hops (worst case)"),
    ]
    ax.legend(handles=legend_r, loc="lower right", fontsize=7.5,
              framealpha=0.9, edgecolor="#BBBBBB")

    rad = max(1.8, n * 0.22)
    ax.set_xlim(-rad - 0.8, rad + 0.8)
    ax.set_ylim(-rad - 0.8, rad + 1.2)

    plt.tight_layout()
    savefig(fig, "06_fb_topology_main.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 07 — panel size variants (auto-factorize)
# ─────────────────────────────────────────────────────────────────────────────
def fig07_panel_variants():
    configs = [
        (4,  "EP=4",  *auto_factorize(4)),
        (8,  "EP=8",  *auto_factorize(8)),
        (16, "EP=16", *auto_factorize(16)),
        (32, "EP=32", *auto_factorize(32)),
        (6,  "EP=6 (6×6-4c\ninner 32→6 wg)",  2, 3),
        (12, "EP=12", *auto_factorize(12)),
    ]
    # configs: (n, title, rows, cols)
    ncols_fig = 3
    nrows_fig = 2
    step = 1.2

    fig, axes = plt.subplots(nrows_fig, ncols_fig, figsize=(15, 9))
    fig.suptitle("FlattenedButterfly Panel Variants — Auto-factorized Grid Layouts",
                 fontsize=12, fontweight="bold")

    for ax_idx, (n, title, rows, cols) in enumerate(configs):
        ax = axes[ax_idx // ncols_fig][ax_idx % ncols_fig]
        ax.set_aspect("equal")
        ax.axis("off")

        # compute 1-hop and 2-hop pair counts
        total_pairs = n * (n - 1) // 2
        one_hop = 0
        for u in range(n):
            for v in range(u + 1, n):
                if fb_hop_count(u, v, cols) == 1:
                    one_hop += 1
        two_hop = total_pairs - one_hop
        pct1 = 100 * one_hop / total_pairs
        pct2 = 100 * two_hop / total_pairs

        ax.set_title(
            f"{title}  ({rows}×{cols} grid)\n"
            f"1-hop: {one_hop}/{total_pairs} pairs ({pct1:.0f}%)  "
            f"2-hop: {two_hop}/{total_pairs} pairs ({pct2:.0f}%)",
            fontsize=8.0, pad=4)

        pos = [grid_pos(i, cols, step) for i in range(n)]
        re, ce = fb_edges(n, rows, cols)

        for u, v in re:
            x0, y0 = pos[u]; x1, y1 = pos[v]
            draw_edge(ax, x0, y0, x1, y1, C_ROW_LINK, lw=1.2, alpha=0.4)
        for u, v in ce:
            x0, y0 = pos[u]; x1, y1 = pos[v]
            draw_edge(ax, x0, y0, x1, y1, C_COL_LINK, lw=1.2, alpha=0.4)

        for i in range(n):
            x, y = pos[i]
            draw_node(ax, x, y, i, r=NODE_R * 0.85, fs=6.5)

        # row/col labels
        for r in range(rows):
            x, y = pos[r * cols]
            ax.text(x - 0.48, y, f"r{r}", ha="right", va="center",
                    fontsize=6, color="#666")
        for c in range(cols):
            x, y = pos[c]
            ax.text(x, y + 0.46, f"c{c}", ha="center", va="bottom",
                    fontsize=6, color="#666")

        pad = 0.6
        ax.set_xlim(-pad, (cols - 1) * step + pad)
        ax.set_ylim(-(rows - 1) * step - pad, pad + 0.5)

    # shared legend
    legend_items = [
        mpatches.Patch(color=C_ROW_LINK, label="Same-row link (1 hop)"),
        mpatches.Patch(color=C_COL_LINK, label="Same-col link (1 hop)"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=2,
               fontsize=9, framealpha=0.9, edgecolor="#BBBBBB",
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    savefig(fig, "07_fb_panel_variants.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 08 — hop-count statistics vs Ring
# ─────────────────────────────────────────────────────────────────────────────
def fig08_hop_stats():
    ep_sizes = [4, 6, 8, 12, 16, 32, 64, 128]

    fb_max_hop, fb_avg_hop = [], []
    ring_max_hop, ring_avg_hop = [], []
    fb_pct_1hop = []

    for n in ep_sizes:
        rows, cols = auto_factorize(n)

        # FlattenedButterfly
        hops = []
        for u in range(n):
            for v in range(u + 1, n):
                hops.append(fb_hop_count(u, v, cols))
        fb_max_hop.append(max(hops))
        fb_avg_hop.append(sum(hops) / len(hops))
        fb_pct_1hop.append(100 * hops.count(1) / len(hops))

        # Ring: hops between u and v = min(|u-v|, n-|u-v|)
        rhops = []
        for u in range(n):
            for v in range(u + 1, n):
                d = abs(u - v)
                rhops.append(min(d, n - d))
        ring_max_hop.append(max(rhops))
        ring_avg_hop.append(sum(rhops) / len(rhops))

    x = np.arange(len(ep_sizes))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("FlattenedButterfly vs Ring — Hop Count Statistics",
                 fontsize=12, fontweight="bold")

    # ── subplot 1: max hops ───────────────────────────────────────────────────
    ax = axes[0]
    bars_fb   = ax.bar(x - width / 2, fb_max_hop,   width, color=C_ROW_LINK,
                       label="FlattenedButterfly", edgecolor="white", linewidth=0.5)
    bars_ring = ax.bar(x + width / 2, ring_max_hop, width, color=C_RING_LINK,
                       label="Ring", edgecolor="white", linewidth=0.5)
    ax.set_title("Max Hops (worst-case pair)", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([f"N={n}" for n in ep_sizes], fontsize=8)
    ax.set_ylabel("Max hops"); ax.set_ylim(0, max(ring_max_hop) * 1.25)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(2, color=C_ROW_LINK, ls="--", lw=1.0, alpha=0.6)
    ax.text(len(ep_sizes) - 0.5, 2.2, "FB ceiling = 2", color=C_ROW_LINK,
            fontsize=7.5, ha="right")
    for bar in bars_fb:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=7)
    for bar in bars_ring:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=7)

    # ── subplot 2: average hops ───────────────────────────────────────────────
    ax = axes[1]
    ax.bar(x - width / 2, fb_avg_hop,   width, color=C_ROW_LINK,
           label="FlattenedButterfly", edgecolor="white", linewidth=0.5)
    ax.bar(x + width / 2, ring_avg_hop, width, color=C_RING_LINK,
           label="Ring", edgecolor="white", linewidth=0.5)
    ax.set_title("Average Hops (all pairs)", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([f"N={n}" for n in ep_sizes], fontsize=8)
    ax.set_ylabel("Avg hops"); ax.set_ylim(0, max(ring_avg_hop) * 1.25)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for i, (fb, rg) in enumerate(zip(fb_avg_hop, ring_avg_hop)):
        ax.text(i - width / 2, fb + 0.05, f"{fb:.2f}", ha="center",
                va="bottom", fontsize=6.5, color=C_ROW_LINK)
        ax.text(i + width / 2, rg + 0.05, f"{rg:.2f}", ha="center",
                va="bottom", fontsize=6.5, color=C_RING_LINK)

    # ── subplot 3: % 1-hop pairs for FlattenedButterfly ───────────────────────
    ax = axes[2]
    ax.bar(x, fb_pct_1hop, width * 1.5, color=C_COL_LINK,
           edgecolor="white", linewidth=0.5, label="% pairs reachable in 1 hop")
    ax.set_title("FlattenedButterfly: Fraction of 1-hop Pairs", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([f"N={n}" for n in ep_sizes], fontsize=8)
    ax.set_ylabel("% of all node-pairs"); ax.set_ylim(0, 110)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for i, p in enumerate(fb_pct_1hop):
        ax.text(i, p + 1.5, f"{p:.0f}%", ha="center", va="bottom", fontsize=7.5)

    # bottom annotation
    fig.text(0.5, -0.04,
             "auto_factorize: rows = largest factor ≤ √N  (same rule as FlattenedButterfly C++ constructor with num_rows=0)",
             ha="center", fontsize=8, color="#555", style="italic")

    plt.tight_layout()
    savefig(fig, "08_fb_hop_stats.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 09 — 2-hop routing detail: 4×8 grid (EP=32)
# ─────────────────────────────────────────────────────────────────────────────
def fig09_2hop_routing():
    """Show all 2-hop intermediate paths for a subset of pairs in a 4×8 grid."""
    rows, cols, n = 4, 8, 32
    step = 1.3

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "FlattenedButterfly 4×8 (EP=32) — 2-hop routing via intermediate node\n"
        "src(row r, col c) → intermediate(row r, col c') → dst(row r', col c')",
        fontsize=9.5, pad=8)

    pos = [grid_pos(i, cols, step) for i in range(n)]
    re, ce = fb_edges(n, rows, cols)

    # draw all edges faintly
    for u, v in re:
        x0, y0 = pos[u]; x1, y1 = pos[v]
        draw_edge(ax, x0, y0, x1, y1, C_ROW_LINK, lw=0.9, alpha=0.18)
    for u, v in ce:
        x0, y0 = pos[u]; x1, y1 = pos[v]
        draw_edge(ax, x0, y0, x1, y1, C_COL_LINK, lw=0.9, alpha=0.18)

    # highlight 4 representative 2-hop examples
    example_pairs = [
        (0,  27, "#E84B3A", "0→27\n(r0c0→r3c3)"),   # top-left → bottom-right near
        (1,  30, "#9467BD", "1→30\n(r0c1→r3c6)"),
        (4,  23, "#2CA02C", "4→23\n(r0c4→r2c7)"),
        (9,  22, "#FF7F0E", "9→22\n(r1c1→r2c6)"),
    ]

    for src, dst, color, label_text in example_pairs:
        interm = (src // cols) * cols + (dst % cols)
        xs, ys = pos[src]
        xi, yi = pos[interm]
        xd, yd = pos[dst]

        # hop 1: src → intermediate (thicker)
        ax.annotate("", xy=(xi, yi), xytext=(xs, ys),
                    arrowprops=dict(
                        arrowstyle="->,head_width=0.18,head_length=0.14",
                        color=color, lw=2.2,
                        connectionstyle="arc3,rad=0.08"),
                    zorder=8)
        # hop 2: intermediate → dst
        ax.annotate("", xy=(xd, yd), xytext=(xi, yi),
                    arrowprops=dict(
                        arrowstyle="->,head_width=0.18,head_length=0.14",
                        color=color, lw=2.2,
                        connectionstyle="arc3,rad=-0.08"),
                    zorder=8)

        # label the src
        mx = (xs + xi + xd) / 3
        my = min(ys, yi, yd) - 0.52
        ax.text(mx, my, label_text, ha="center", va="top",
                fontsize=6.5, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec=color, alpha=0.88, lw=0.8))

    # draw all nodes
    highlighted = set()
    for src, dst, color, _ in example_pairs:
        interm = (src // cols) * cols + (dst % cols)
        highlighted.update([src, dst, interm])

    for i in range(n):
        x, y = pos[i]
        r_i, c_i = i // cols, i % cols
        is_hi = i in highlighted
        draw_node(ax, x, y, i, color=C_NODE, r=NODE_R * 0.78,
                  fs=5.8, highlight=is_hi)

    # row / col labels
    for r in range(rows):
        x, y = pos[r * cols]
        ax.text(x - 0.52, y, f"row {r}", ha="right", va="center",
                fontsize=7, color="#555")
    for c in range(cols):
        x, y = pos[c]
        ax.text(x, y + 0.5, f"col {c}", ha="center", va="bottom",
                fontsize=7, color="#555")

    # formula box
    ax.text(0.01, 0.05,
            "Intermediate node for 2-hop:\n"
            "  intermediate = src_row × cols + dst_col\n"
            "  (same row as src, same col as dst)\n\n"
            "Implementation: FlattenedButterfly.cpp\n"
            "  compute_hops_count(src, dst):\n"
            "    if src_row == dst_row → 1 hop\n"
            "    if src_col == dst_col → 1 hop\n"
            "    else                  → 2 hops",
            transform=ax.transAxes, fontsize=7.8, va="bottom",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="#F5F5F5", ec="#AAAAAA",
                      lw=0.9, alpha=0.92))

    margin = 0.7
    ax.set_xlim(-margin - 0.5, (cols - 1) * step + margin)
    ax.set_ylim(-(rows - 1) * step - 1.2, margin + 0.7)

    # legend
    legend_items = [
        mpatches.Patch(color=C_ROW_LINK, label="Same-row link (1 hop, all pairs)"),
        mpatches.Patch(color=C_COL_LINK, label="Same-col link (1 hop, all pairs)"),
    ] + [
        mpatches.Patch(color=color, label=lbl)
        for _, _, color, lbl in example_pairs
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=7.5,
              framealpha=0.9, edgecolor="#BBBBBB")

    plt.tight_layout()
    savefig(fig, "09_fb_2hop_routing_detail.png")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating FlattenedButterfly topology plots...")
    fig06_main()
    fig07_panel_variants()
    fig08_hop_stats()
    fig09_2hop_routing()
    print("Done.")
