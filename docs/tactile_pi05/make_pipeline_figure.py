#!/usr/bin/env python3
"""VTLA data-collection pipeline figure (paper-ready, easy to edit).

Edit the CONFIG blocks below (NODES, EDGES, GROUPS, COLORS) to change text,
position, or styling — no need to touch the drawing code at the bottom.

Run:  python3 docs/tactile_pi05/make_pipeline_figure.py
Out :  vtla_pipeline.{pdf,svg,png}  (SVG/PDF text stays editable in Illustrator/Inkscape)
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

# ============================ STYLE ========================================
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "svg.fonttype": "none",  # keep SVG text as editable text
        "pdf.fonttype": 42,
    }
)

COLORS = {
    "obs": ("#EAF1FB", "#3B6FB0"),  # Robot & visual observations  (fill, edge)
    "tact": ("#FBE9EF", "#C24C6B"),  # Bilateral tactile stream (highlight)
    "hub": ("#EFEAF7", "#7A57AC"),  # Synchronization hub
    "data": ("#E6F4F1", "#2E9187"),  # Dataset & downstream
}
EDGE_COLOR = "#4a4a4a"
TITLE = "Multimodal VTLA Data-Collection Pipeline"

# ============================ GROUPS (dashed containers) ===================
# name, x, y, w, h, color-key
GROUPS = [
    ("Robot & visual observations", 3, 42, 43, 34, "obs"),
    ("Bilateral tactile stream", 3, 4, 82, 30, "tact"),
    ("Synchronization hub", 52, 42, 30, 34, "hub"),
    ("Dataset & downstream", 90, 4, 47, 72, "data"),
]

# ============================ NODES ========================================
# key: (letter, x, y, w, h, title, subtitle, color-key)
NODES = {
    "A": ("A", 6, 65, 37, 8, "Leader arm", "teleoperation input", "obs"),
    "B": ("B", 6, 55, 37, 8, "SO-101 follower arm", "joint states", "obs"),
    "C": ("C", 6, 45, 37, 8, "3 RGB cameras", "top / wrist / BEV", "obs"),
    "D": ("D", 6, 14, 22, 10, "2$\\times$ Paxini sensors", "154 taxels", "tact"),
    "E": ("E", 34, 14, 22, 10, "High-speed board", "UART, ~55 Hz", "tact"),
    "F": ("F", 62, 14, 21, 10, "Async tactile reader", "latest sample", "tact"),
    "G": (
        "G",
        55,
        50,
        24,
        18,
        "Extended LeRobot\nrecord loop",
        "sync state · image ·\ntactile · action",
        "hub",
    ),
    "H": (
        "H",
        94,
        58,
        40,
        12,
        "Multimodal episode dataset",
        "obs.tactile.primary  (B$\\times$2$\\times$9$\\times$9)",
        "data",
    ),
    "I": ("I", 94, 40, 40, 10, "Replay & visualization", "tactile heatmap validation", "data"),
    "J": ("J", 94, 22, 40, 12, "pi0.5 VTLA training", "A0 / A1.5 / A2 variants", "data"),
}

# ============================ EDGES ========================================
# (from, to, label, rad[, from_side, to_side])  rad = curvature; sides optional
# sides = one of "l r t b" to force where the arrow attaches (else auto).
EDGES = [
    ("A", "G", "operator action", 0.0),
    ("B", "G", "robot state", 0.0),
    ("C", "G", "multi-view RGB", 0.0),
    ("D", "E", "distributed force", 0.0),
    ("E", "F", "sample", 0.0),
    ("F", "G", "154-dim vector", 0.0),
    ("G", "H", "synchronized episode", 0.0),
    ("H", "I", "data loading", 0.0),
    ("H", "J", "tactile tokens + masking", -0.55, "l", "l"),
]

# ===========================================================================
# ============================ DRAWING (rarely edit) ========================
# ===========================================================================
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 140)
ax.set_ylim(0, 80)
ax.axis("off")
ax.text(70, 78.5, TITLE, ha="center", va="center", fontsize=16, fontweight="bold", color="#111")


def anchor(key, side):
    _, x, y, w, h = NODES[key][:5]
    return {
        "r": (x + w, y + h / 2),
        "l": (x, y + h / 2),
        "t": (x + w / 2, y + h),
        "b": (x + w / 2, y),
        "c": (x + w / 2, y + h / 2),
    }[side]


for name, gx, gy, gw, gh, ckey in GROUPS:
    ax.add_patch(
        FancyBboxPatch(
            (gx, gy),
            gw,
            gh,
            boxstyle="round,pad=0.02,rounding_size=0.9",
            linewidth=1.3,
            edgecolor=COLORS[ckey][1],
            facecolor="none",
            linestyle=(0, (6, 4)),
            zorder=0,
        )
    )
    ax.text(
        gx + 1.4,
        gy + gh - 1.6,
        name,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        style="italic",
        color=COLORS[ckey][1],
    )

for _key, (letter, x, y, w, h, title, sub, ckey) in NODES.items():
    fc, ec = COLORS[ckey]
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.5",
            linewidth=1.6,
            edgecolor=ec,
            facecolor=fc,
            zorder=2,
        )
    )
    ax.text(
        x + w / 2,
        y + h * 0.62,
        title,
        ha="center",
        va="center",
        fontsize=10.2,
        fontweight="bold",
        color="#1a1a1a",
        zorder=3,
        linespacing=1.15,
    )
    if sub:
        ax.text(
            x + w / 2,
            y + h * 0.26,
            sub,
            ha="center",
            va="center",
            fontsize=8.6,
            color="#555",
            zorder=3,
            linespacing=1.1,
        )
    # letter badge (top-left corner)
    bx, by = x + 0.2, y + h - 0.2
    ax.add_patch(Circle((bx, by), 1.35, facecolor=ec, edgecolor="white", linewidth=1.2, zorder=4))
    ax.text(bx, by, letter, ha="center", va="center", fontsize=9, fontweight="bold", color="white", zorder=5)


def side_pair(a, b):
    ax0, ay0, aw, ah = NODES[a][1:5]
    bx0, by0, bw, bh = NODES[b][1:5]
    acy, bcy = ay0 + ah / 2, by0 + bh / 2
    if bx0 >= ax0 + aw - 0.5:  # b clearly to the right
        return "r", "l"
    if ax0 >= bx0 + bw - 0.5:  # b to the left
        return "l", "r"
    return ("t", "b") if bcy > acy else ("b", "t")


for edge in EDGES:
    a, b, label, rad = str(edge[0]), str(edge[1]), str(edge[2]), float(edge[3])
    if len(edge) >= 6:
        s_from, s_to = str(edge[4]), str(edge[5])
    else:
        s_from, s_to = side_pair(a, b)
    p1, p2 = anchor(a, s_from), anchor(b, s_to)
    col = COLORS["tact"][1] if (NODES[a][7] == "tact" or NODES[b][7] == "tact") else EDGE_COLOR
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.7,
            color=col,
            zorder=1,
            connectionstyle=f"arc3,rad={rad}",
        )
    )
    mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    # nudge label off the chord for curved edges so it sits on the arc
    lx = mx - rad * (p2[1] - p1[1]) * 0.5
    ly = my + rad * (p2[0] - p1[0]) * 0.5 + 1.2
    ax.text(
        lx,
        ly,
        label,
        ha="center",
        va="center",
        fontsize=8.2,
        color=col,
        style="italic",
        zorder=6,
        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85},
    )

out = Path(__file__).resolve().parent
fig.tight_layout(pad=0.4)
for ext in ("pdf", "svg", "png"):
    f = out / f"vtla_pipeline.{ext}"
    fig.savefig(f, dpi=300, bbox_inches="tight")
    print("saved:", f)
