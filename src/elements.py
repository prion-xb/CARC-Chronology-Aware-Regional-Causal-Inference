"""Shared style and save helpers for the standalone figure *elements*.

The submitted figures were composites: one canvas carrying six regional panels,
or a map plus four bar blocks. At page size the inner labels collided, and the
whole thing could only be redesigned by rerunning the generating script.

An *element* is the opposite: one plot, one message, one file, sized so it is
already legible at single-column width without shrinking. Elements carry no
figure title and no panel letter, because the layout they end up in is chosen
later in Illustrator; the filename and `elements/MANIFEST.md` say what each one
is. Text is exported as real text objects in the SVG so it stays editable.
"""
from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.text  # noqa: F401  (findobj filter below needs the class)

# Single-column width for most Elsevier journals is ~90 mm (3.54 in); a
# half-height plot at that width is the default element footprint.
ONE_COL = (3.45, 2.55)
ONE_COL_TALL = (3.45, 3.20)
TWO_COL = (7.10, 3.60)
SQUARE = (3.45, 3.30)

INK = "#1a1a1a"
INK_SEC = "#4a4a4a"
INK_MUTED = "#7a7a7a"
GRID = "#d9d9d4"
SURFACE = "#ffffff"
MEMBER = "#b9b9b2"

REGION_COLORS = {
    "North_Atlantic": "#c0392b",
    "Antarctic": "#2874a6",
    "Tropical": "#d68910",
    "East_Asian_Monsoon": "#1e8449",
    "South_America": "#7d3c98",
    "Other_Lakes": "#5d6d7e",
}

TIER_COLORS = {1: "#c0392b", 2: "#d68910", 3: "#c9a227"}

ELEMENT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Figures", "elements"
)


def use_element_style():
    """Apply the element rcParams. Call once per script, before plotting."""
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "axes.edgecolor": INK_SEC,
        "axes.linewidth": 0.7,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK_SEC,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        # keep text as text in vector output so Illustrator can edit it
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Arial has no subscript digits (U+2082-2084), so CO2 / CaCO3 / CH4 in the
        # record names came out as empty boxes. `safe_label` rewrites them as
        # mathtext, and this fontset keeps that mathtext in Arial rather than
        # switching to matplotlib's default math font mid-label.
        "mathtext.fontset": "custom",
        "mathtext.default": "regular",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    })


_SUBSCRIPTS = {chr(0x2080 + d): str(d) for d in range(10)}


def safe_label(s):
    """Rewrite Unicode subscript digits as mathtext so Arial can render them.

    ``'ODP 663 CaCO₃'`` -> ``'ODP 663 CaCO$_3$'``. Superscripts are left
    alone: Arial does carry those, so δ¹⁸O needs no help.
    """
    out, run = [], []
    for ch in str(s):
        if ch in _SUBSCRIPTS:
            run.append(_SUBSCRIPTS[ch])
            continue
        if run:
            out.append("$_{%s}$" % "".join(run))
            run = []
        out.append(ch)
    if run:
        out.append("$_{%s}$" % "".join(run))
    return "".join(out)


def tidy(ax, grid_axis="both"):
    """Despine and apply the light element grid."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid_axis != "none":
        ax.grid(True, axis=grid_axis, alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
    return ax


def text_collisions(fig, pad=0.5):
    """Return pairs of visible Text artists whose rendered boxes overlap.

    This is the check the composites never had: at page size their inner
    labels ran into each other, and nothing caught it before the co-author
    did. Every element is screened here before it is written.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    items = []
    for txt in fig.findobj(matplotlib.text.Text):
        if not txt.get_visible() or not txt.get_text().strip():
            continue
        try:
            bb = txt.get_window_extent(renderer=renderer)
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        items.append((txt, bb.expanded(1.0, 1.0).padded(-pad)))

    clashes = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (t1, b1), (t2, b2) = items[i], items[j]
            if b1.overlaps(b2):
                clashes.append((t1.get_text(), t2.get_text()))
    return clashes


_saved: list[str] = []
_warnings: list[str] = []


def save_element(fig, name, subdir="", tight=True, dpi=300, check=True):
    """Write one element as PNG (300 dpi), PDF and editable SVG."""
    directory = os.path.join(ELEMENT_DIR, subdir) if subdir else ELEMENT_DIR
    os.makedirs(directory, exist_ok=True)
    if tight:
        fig.tight_layout(pad=0.35)

    label = os.path.join(subdir, name) if subdir else name
    note = ""
    if check:
        clashes = text_collisions(fig)
        if clashes:
            note = f"  <-- {len(clashes)} TEXT OVERLAP(S): {clashes[:3]}"
            _warnings.append(f"{label}: {clashes}")

    for fmt in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(directory, f"{name}.{fmt}"), format=fmt, dpi=dpi)
    _saved.append(label)
    plt.close(fig)
    print(f"  element: {label}{note}")


def report():
    """Print the collision summary; returns the number of flagged elements."""
    print(f"\n{len(_saved)} elements written to {ELEMENT_DIR}")
    if _warnings:
        print(f"{len(_warnings)} element(s) with overlapping text:")
        for w in _warnings:
            print(f"   {w}")
    else:
        print("no overlapping text detected")
    return len(_warnings)


def saved_elements():
    return list(_saved)
