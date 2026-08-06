#!/usr/bin/env python3
"""Generate the VTLA architecture figure for the paper.

Outputs vector PDF + high-res PNG to docs/tactile_pi05/.
Run:  python3 docs/tactile_pi05/make_architecture_figure.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,  # editable text in vector output
        "ps.fonttype": 42,
    }
)

# ---- palette (colorblind-friendly, muted for print) -----------------------
C_HW = "#E8EEF7"  # hardware
C_HW_E = "#4C78A8"
C_ROBOT = "#EAF3EA"  # robot wrapper
C_ROBOT_E = "#59A14F"
C_PRE = "#FBF0E1"  # preprocessing
C_PRE_E = "#E1A354"
C_TACT = "#F7E7EC"  # tactile path (highlight = contribution)
C_TACT_E = "#C44E6B"
C_POLICY = "#ECE7F5"  # policy backbone
C_POLICY_E = "#8464B0"
C_OUT = "#E5F2F2"  # output
C_OUT_E = "#3E9C9C"
GREY = "#555555"

fig, ax = plt.subplots(figsize=(13.5, 7.4))
ax.set_xlim(0, 135)
ax.set_ylim(0, 74)
ax.axis("off")


def box(x, y, w, h, text, fc, ec, fs=10, bold=False, rounding=0.6, lw=1.4, tc="#1a1a1a"):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight="bold" if bold else "normal",
        color=tc,
        zorder=3,
        linespacing=1.25,
    )
    return (x, y, w, h)


def arrow(p1, p2, color=GREY, lw=1.8, style="-|>", ls="-", rad=0.0):
    a = FancyArrowPatch(
        p1,
        p2,
        arrowstyle=style,
        mutation_scale=16,
        linewidth=lw,
        color=color,
        zorder=1,
        connectionstyle=f"arc3,rad={rad}",
        linestyle=ls,
    )
    ax.add_patch(a)


def band(x, y, w, h, label, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.8",
            linewidth=1.2,
            edgecolor=color,
            facecolor="none",
            linestyle=(0, (6, 4)),
            zorder=0,
        )
    )
    ax.text(
        x + 1.2,
        y + h - 1.6,
        label,
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color=color,
        style="italic",
    )


# ============================ TITLE ========================================
ax.text(
    67.5,
    72.2,
    "VTLA: Vision–Tactile–Language–Action Architecture",
    ha="center",
    va="center",
    fontsize=15,
    fontweight="bold",
    color="#111",
)

# ============================ 1. INPUTS ====================================
band(2, 8, 22, 56, "Inputs", C_HW_E)
box(4, 52, 18, 8, "Wrist / scene\ncameras", C_HW, C_HW_E)
box(4, 40, 18, 8, "Paxini tactile\n154 taxels (2$\\times$77)", C_TACT, C_TACT_E, bold=True)
box(4, 28, 18, 8, "Language\ninstruction", C_HW, C_HW_E)
box(4, 16, 18, 8, "Robot proprio.\nstate", C_HW, C_HW_E)

# ============================ 2. ENCODERS / TOKENIZERS =====================
band(28, 8, 26, 56, "Robot wrapper + Encoders", C_ROBOT_E)
box(30, 52, 22, 8, "SigLIP\nvision encoder", C_ROBOT, C_ROBOT_E)

# tactile rendering + encoder (the contribution)
box(
    30,
    38.5,
    22,
    11,
    "Tactile render\n(2,9,9) / heatmap\n$\\rightarrow$ TactileTokenEncoder",
    C_TACT,
    C_TACT_E,
    fs=9.2,
    bold=True,
)

box(30, 28, 22, 8, "Text\ntokenizer", C_ROBOT, C_ROBOT_E)
box(30, 16, 22, 8, "State\nprojection", C_ROBOT, C_ROBOT_E)

# ============================ 3. BACKBONE ==================================
band(58, 8, 40, 56, "pi0.5 backbone", C_POLICY_E)
box(
    60,
    30,
    24,
    26,
    "PaliGemma\n(vision + language)\n\nprefix attention\nover multimodal\ntokens",
    C_POLICY,
    C_POLICY_E,
    fs=9.6,
    bold=True,
)
box(60, 12, 24, 12, "Gemma\naction expert\n(flow matching)", C_POLICY, C_POLICY_E, fs=9.6, bold=True)

# small "AdaRMS / conditioning" hint between backbone and expert
arrow((72, 30), (72, 24), color=C_POLICY_E, lw=1.6)

# ============================ 4. OUTPUT ====================================
band(101, 8, 32, 56, "Output", C_OUT_E)
box(103, 30, 28, 12, "Action chunk\n$a_{t:t+H}$   (H = 50)", C_OUT, C_OUT_E, bold=True)
box(103, 14, 28, 10, "SO-101\nexecution", C_OUT, C_OUT_E)

# ============================ MAIN ARROWS ==================================
# inputs -> encoders
arrow((22, 56), (30, 56))
arrow((22, 44), (30, 44), color=C_TACT_E)
arrow((22, 32), (30, 32))
arrow((22, 20), (30, 20))

# encoders -> backbone
arrow((52, 56), (60, 50))  # vision -> paligemma
arrow((52, 32), (60, 46))  # text -> paligemma
arrow((52, 20), (60, 18))  # state -> action expert

# tactile encoder -> BOTH backbone entry points (two injection targets)
arrow((52, 44), (60, 48), color=C_TACT_E, lw=2.0)  # -> prefix (paligemma)
arrow((52, 42), (60, 20), color=C_TACT_E, lw=2.0, rad=-0.25, ls=(0, (4, 3)))  # -> action expert gate

# backbone internal + to output
arrow((84, 18), (103, 20))  # action expert -> action chunk (routes low)
arrow((84, 43), (95, 43), color=C_POLICY_E, lw=1.4)
ax.text(89.5, 45.4, "denoise", ha="center", fontsize=8, color=C_POLICY_E, style="italic")
arrow((95, 43), (95, 24), color=C_POLICY_E, lw=1.4, rad=0.0)
arrow((95, 24), (84, 22), color=C_POLICY_E, lw=1.4, rad=-0.2)
arrow((117, 30), (117, 24), color=C_OUT_E)  # action chunk -> execution

# ============================ INJECTION-MODE INSET =========================
inset_y = 0.5
ax.add_patch(
    FancyBboxPatch(
        (2, inset_y),
        131,
        6.4,
        boxstyle="round,pad=0.02,rounding_size=0.6",
        linewidth=1.2,
        edgecolor=C_TACT_E,
        facecolor="#FCF6F8",
        zorder=0,
    )
)
ax.text(
    4,
    inset_y + 4.7,
    "Tactile injection modes",
    ha="left",
    va="center",
    fontsize=10,
    fontweight="bold",
    color=C_TACT_E,
)
ax.text(
    4,
    inset_y + 2.0,
    "A0  none  —  no tactile (baseline)        "
    "A1.5  prefix  —  TactileTokenEncoder $\\rightarrow$ prefix tokens into PaliGemma       "
    "A2  action_gated  —  ActionSideGatedTactileEncoder $\\rightarrow$ gates the action expert",
    ha="left",
    va="center",
    fontsize=9.2,
    color="#3a2a30",
)

# subtle legend for tactile path
ax.plot([], [], color=C_TACT_E, lw=2.2, label="tactile path")
ax.text(133, 70.5, "", ha="right")

out_dir = Path(__file__).resolve().parent
fig.tight_layout(pad=0.4)
pdf = out_dir / "vtla_architecture.pdf"
png = out_dir / "vtla_architecture.png"
fig.savefig(pdf, bbox_inches="tight")
fig.savefig(png, dpi=300, bbox_inches="tight")
print(f"saved:\n  {pdf}\n  {png}")
