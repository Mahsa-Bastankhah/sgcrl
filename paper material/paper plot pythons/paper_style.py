"""Paper figure template.

Visual source of truth: kl_dirac_and_preimage_paper.pdf
(draw large, include at ~0.9\\textwidth; fonts shrink to ~11–12pt).

Recipe
------
    import paper_style as ps
    ps.apply()

    # one panel
    fig, ax = ps.figure("single")
    ax.plot(x, y, color=ps.C["blue"], **ps.HOLLOW)
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel("Goal-Reaching Optimality")
    ax.set_title("Optimality")
    ps.style_axes(ax)
    ps.savefig(fig, "out_paper")          # writes .pdf only

    # two panels side by side (the Dirac + preimage layout)
    fig, (ax_l, ax_r) = ps.figure("sidebyside")

Do not set fontsizes / linewidths / marker sizes in the plot script unless
that figure is a genuine exception. Change tokens here instead.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

# ---------------------------------------------------------------------------
# Colorblind-safe Okabe–Ito palette
# ---------------------------------------------------------------------------
C = {
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "green":      "#009E73",
    "orange":     "#E69F00",
    "sky":        "#56B4E9",
    "purple":     "#CC79A7",
    "yellow":     "#F0E442",
    "black":      "#000000",
    "gray":       "#666666",
}
SERIES = [C["blue"], C["vermillion"], C["green"], C["orange"], C["sky"], C["purple"]]
MARKERS = ["o", "s", "^", "D", "v", "P"]

# ---------------------------------------------------------------------------
# Visual tokens (from kl_dirac_and_preimage_paper)
# ---------------------------------------------------------------------------
LW = 4.8                 # line width
MS = 13                  # marker size
MEW = 2.4                # marker edge width
DASH = (0, (7, 3.5))     # dashed steps / dual-axis left spine
SPINE = 1.35
SPINE_EMPH = 1.8         # dual-axis spines

FS_TICK = 28
FS_LABEL = 28
FS_LABEL_MATH = 36       # short math xlabel such as $\varepsilon$
FS_TITLE = 26
FS_LEGEND = 20
TITLE_PAD = 18

SAVE_PAD = 0.18
LAYOUT_PADS = dict(w_pad=0.18, h_pad=0.22, hspace=0.05, wspace=0.05)
LAYOUT_PADS_SIDEBYSIDE = dict(w_pad=0.06, h_pad=0.18, hspace=0.04, wspace=0.02)

# Drawn at ~2× paper size so shrinking to NeurIPS/ICML text width (~6.75")
# leaves ~11–12pt labels. Do not use FIGSIZE_SINGLE for these fonts.
FIGSIZE_SINGLE = (8.6, 6.2)       # one panel, full-width include
FIGSIZE_SIDEBYSIDE = (15.4, 6.2)  # 1×2, the Dirac + preimage layout
FIGSIZE_STACKED = (7.8, 8.2)      # 2×1
FIGSIZE_DOUBLE = FIGSIZE_SINGLE   # alias used by older scripts
FIGSIZE_COLUMN = (3.6, 2.9)       # true 1-column; too small for these fonts

# Unpack onto ax.plot / ax.step so scripts stay ordinary matplotlib.
HOLLOW = dict(
    linewidth=LW,
    marker="o",
    markersize=MS,
    markerfacecolor="white",
    markeredgewidth=MEW,
    zorder=3,
    clip_on=False,
)
STEP = dict(
    linewidth=LW,
    linestyle=DASH,
    where="post",
)
SOLID_MARKED = dict(
    linewidth=LW,
    linestyle="-",
    marker="o",
    markersize=MS,
    markerfacecolor="white",
    markeredgewidth=MEW,
    zorder=4,
    clip_on=False,
)

_LAYOUTS = {
    "single":     dict(figsize=FIGSIZE_SINGLE,     nrows=1, ncols=1, pads=LAYOUT_PADS),
    "sidebyside": dict(figsize=FIGSIZE_SIDEBYSIDE, nrows=1, ncols=2, pads=LAYOUT_PADS_SIDEBYSIDE),
    "stacked":    dict(figsize=FIGSIZE_STACKED,    nrows=2, ncols=1, pads=LAYOUT_PADS),
}


def apply(*, serif: bool = True) -> None:
    """Set rcParams. Call once before creating figures.

    Default is Times + Computer Modern math (LaTeX / NeurIPS look).
    Pass serif=False for matplotlib's default DejaVu Sans.
    """
    extra = {}
    if serif:
        extra.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "cm",
        })
    plt.rcParams.update({
        **extra,
        "font.size": 24,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_TITLE,
        "axes.titlepad": TITLE_PAD,
        "axes.labelpad": 12,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "xtick.major.width": 1.3,
        "ytick.major.width": 1.3,
        "xtick.minor.size": 3.5,
        "ytick.minor.size": 3.5,
        "xtick.minor.width": 1.0,
        "ytick.minor.width": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": FS_LEGEND,
        "legend.title_fontsize": FS_LEGEND,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.framealpha": 0.96,
        "legend.edgecolor": "#B0B0B0",
        "legend.facecolor": "white",
        "legend.handlelength": 2.2,
        "legend.handletextpad": 0.45,
        "legend.borderpad": 0.3,
        "legend.labelspacing": 0.28,
        "legend.columnspacing": 1.6,
        "legend.markerscale": 0.9,
        "axes.linewidth": SPINE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.linestyle": ":",
        "grid.linewidth": 0.9,
        "grid.alpha": 0.45,
        "grid.color": "#888888",
        "lines.linewidth": LW,
        "lines.markersize": MS,
        "lines.markeredgewidth": MEW,
        "axes.prop_cycle": plt.cycler(color=SERIES),
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": SAVE_PAD,
        "figure.dpi": 120,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    })


def figure(layout="single"):
    """Constrained-layout figure in one of the paper sizes.

    layout: "single" | "sidebyside" | "stacked"
    Returns (fig, ax) or (fig, (ax1, ax2)).
    """
    spec = _LAYOUTS[layout]
    fig, axes = plt.subplots(
        spec["nrows"], spec["ncols"], figsize=spec["figsize"],
        layout="constrained",
    )
    fig.set_constrained_layout_pads(**spec["pads"])
    return fig, axes


def style_axes(ax, *, grid: bool = True, which: str = "both") -> None:
    """Spines, ticks, optional dotted grid. Call after plotting."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(SPINE)
    ax.spines["bottom"].set_linewidth(SPINE)
    ax.tick_params(axis="both", which="major", length=7, width=1.3,
                   labelsize=FS_TICK)
    ax.tick_params(axis="both", which="minor", length=3.5, width=1.0)
    if grid:
        ax.grid(True, which=which, linestyle=":", alpha=0.45, color="#888888",
                linewidth=0.9)
        ax.set_axisbelow(True)


def style_twin(ax_left, ax_right, *, left_dashed=True) -> None:
    """Right-hand y-axis as in the smoothness panel (solid right spine)."""
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["left"].set_linewidth(SPINE_EMPH)
    if left_dashed:
        ax_left.spines["left"].set_linestyle(DASH)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.spines["bottom"].set_visible(False)
    ax_right.spines["right"].set_visible(True)
    ax_right.spines["right"].set_linewidth(SPINE_EMPH)
    ax_right.spines["right"].set_linestyle("solid")
    ax_right.tick_params(axis="y", labelsize=FS_TICK, length=7, width=1.3)
    ax_right.grid(False)


def log_x(ax) -> None:
    """Log x-axis without cluttering minor-tick labels."""
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(NullFormatter())


def nice_legend(ax, loc="best", ncol=1, title=None, *, overlay=False, **kwargs):
    """In-axes legend. overlay=True → translucent frame (lines may pass under)."""
    kwargs.setdefault("frameon", True)
    kwargs.setdefault("fancybox", False)
    kwargs.setdefault("framealpha", 0.35 if overlay else 0.96)
    kwargs.setdefault("facecolor", "white")
    kwargs.setdefault("edgecolor", "#B0B0B0")
    kwargs.setdefault("handlelength", 2.2)
    kwargs.setdefault("markerscale", 0.9)
    kwargs.setdefault("borderpad", 0.3)
    kwargs.setdefault("labelspacing", 0.28)
    kwargs.setdefault("handletextpad", 0.45)
    kwargs.setdefault("fontsize", FS_LEGEND)
    leg = ax.legend(loc=loc, ncol=ncol, title=title, **kwargs)
    if title is not None:
        leg.get_title().set_fontweight("medium")
    leg.get_frame().set_linewidth(1.1)
    try:
        leg.get_frame().set_boxstyle("square")
    except Exception:
        pass
    if overlay:
        leg.set_zorder(6)
    return leg


def fig_legend(fig, handles=None, labels=None, *, ncol=2, loc="outside upper center",
               **kwargs):
    """
    Shared figure legend sitting *outside* the axes (needs matplotlib >= 3.6
    and a figure created with layout='constrained').
    """
    kwargs.setdefault("frameon", True)
    kwargs.setdefault("fancybox", False)
    kwargs.setdefault("framealpha", 0.96)
    kwargs.setdefault("edgecolor", "#B0B0B0")
    kwargs.setdefault("handlelength", 2.2)
    kwargs.setdefault("markerscale", 1.25)
    kwargs.setdefault("borderpad", 0.5)
    kwargs.setdefault("labelspacing", 0.45)
    kwargs.setdefault("columnspacing", 1.8)
    kwargs.setdefault("handletextpad", 0.6)
    kwargs.setdefault("fontsize", FS_LEGEND)
    if handles is None:
        leg = fig.legend(loc=loc, ncol=ncol, **kwargs)
    else:
        leg = fig.legend(handles, labels, loc=loc, ncol=ncol, **kwargs)
    leg.get_frame().set_linewidth(1.1)
    try:
        leg.get_frame().set_boxstyle("square")
    except Exception:
        pass
    return leg


def proxy(color, *, ls="-", marker=None, label="", **kwargs) -> Line2D:
    """Legend handle that does not depend on a particular Axes line."""
    return Line2D([0], [0], color=color, ls=ls, marker=marker, label=label, **kwargs)


def savefig(fig, path, *, pdf=True, png=False, dpi=300, pad_inches=SAVE_PAD,
            bbox_inches="tight"):
    """Write PDF only (paper figures). Pass png=True to also write a PNG."""
    path = Path(path)
    written = []
    kw = dict(facecolor="white", bbox_inches=bbox_inches)
    if bbox_inches == "tight":
        kw["pad_inches"] = pad_inches
    if png:
        p = path.with_suffix(".png")
        fig.savefig(p, dpi=dpi, **kw)
        written.append(p)
    if pdf:
        p = path.with_suffix(".pdf")
        fig.savefig(p, **kw)
        written.append(p)
    for p in written:
        print("Saved:", p)
    return written[0] if written else None
