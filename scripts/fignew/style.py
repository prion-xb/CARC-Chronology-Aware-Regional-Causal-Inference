# -*- coding: utf-8 -*-
"""Shared figure style for the rebuilt figure suite.

Palette from the validated CVD-safe reference palette (light surface).
Roles are named by job, never by series number:
  INK      - text / axes / line marks
  BAND     - uncertainty envelope fill
  GRID     - recessive gridlines
  CAT1..8  - categorical slots in fixed validated order (blue, orange, aqua,
             yellow, magenta, green, violet, red)
  SEQ      - blue sequential ramp (light -> dark)
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.figure import Figure

# ---------------------------------------------------------------- palette ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
GRID = "#d9d8d4"
BAND = "#b7d3f6"          # sequential blue step 150, for uncertainty fills
BAND_EDGE = "#5598e7"     # sequential blue step 350

# honesty ladder (95/90/85/80/75/70): neutral greys so data colours stay clean
THR_95 = "#57534e"        # 95% rung: bold dashed reference line
THR_LADDER = "#b5b2ad"    # 90-70% rungs: light dashed
THR_BAND = "#ececea"      # 70-95% sub-threshold band fill

CAT1 = "#2a78d6"   # blue
CAT2 = "#eb6834"   # orange
CAT3 = "#1baf7a"   # aqua
CAT4 = "#eda100"   # yellow
CAT5 = "#e87ba4"   # magenta
CAT6 = "#008300"   # green
CAT7 = "#4a3aa7"   # violet
CAT8 = "#e34948"   # red

# ----------------------------------------------------- regional colours -----
# Fixed region colour slots (cross-figure symbol, README_规划.md): the six
# paper regions take CAT1..CAT6 in the canonical PAPER_REGIONS order used by
# scripts/08/09/17/18, so every adjacent pair in region order is a validated
# CVD pair.  Colour follows the entity, never the rank - do NOT re-order
# regions between figures, and do not reuse CAT4/CAT5 slots for non-region
# entities in figures that also show regions.
REGION_ORDER = [
    "North_Atlantic", "Antarctic", "Tropical",
    "East_Asian_Monsoon", "South_America", "Other_Lakes",
]
REGION_CATS = {
    "North_Atlantic": CAT1,      # blue
    "Antarctic": CAT2,           # orange
    "Tropical": CAT3,            # aqua
    "East_Asian_Monsoon": CAT4,  # yellow
    "South_America": CAT5,       # magenta
    "Other_Lakes": CAT6,         # green
}

# ------------------------------------------------------------- constants ----
# journal single-column width, mm
SINGLE_COLUMN_MM = 89.0
# age axis direction: paleoclimate convention puts older ages on the left
AGE_AXIS_INVERTED = True

# ------------------------------------------------------------- fonts / AI ---
# Arial first (Windows + Illustrator native); mathtext glyphs (beta, sigma,
# delta, ...) also Arial so AI never reports missing glyphs. SVG text stays
# editable text objects. Applied once at import time for every fignew script.
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.cal": "Arial:italic",
    "mathtext.default": "regular",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

# ------------------------------------------------------------- SVG layers ---
# Adobe Illustrator editing layers (see AI_EDITING.md). Data artists are
# tagged by their palette colour so the SVG gets semantic layers:
#   band = uncertainty envelopes (BAND/BAND_EDGE)
#   thr  = honesty ladder (THR_95/THR_LADDER/THR_BAND)
#   cat  = coloured series (CAT1-8)
#   ink  = black data lines (INK/SECONDARY_INK)
#   grid = gridlines
# Untagged artists fall back to the generic class layers of src.svg_export
# (frame, background, data, markers, text, annotations, legend).
# Paint order, bottom -> top.
SVG_LAYER_ORDER = ["frame", "background", "grid", "band", "thr",
                   "cat", "ink", "data", "markers", "text",
                   "annotations", "legend"]

ROLE_COLORS = {
    "ink": (INK, SECONDARY_INK),
    "grid": (GRID,),
    "band": (BAND, BAND_EDGE),
    "thr": (THR_95, THR_LADDER, THR_BAND),
    "cat": (CAT1, CAT2, CAT3, CAT4, CAT5, CAT6, CAT7, CAT8),
}
ROLE_LOOKUP = {
    mpl.colors.to_hex(mpl.colors.to_rgba(c), keep_alpha=False).upper(): role
    for role, cols in ROLE_COLORS.items() for c in cols
}


def _role_gid(artist):
    """Palette colour -> role name, or None (generic class layer applies)."""
    from matplotlib.collections import Collection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    candidates = []
    if isinstance(artist, Line2D):
        candidates.append(artist.get_color())
        if artist.get_marker() not in (None, "", "None"):
            candidates.append(artist.get_markerfacecolor())
    if isinstance(artist, (Collection, Patch)):
        candidates.extend([artist.get_facecolor(), artist.get_edgecolor()])
    for c in candidates:
        if c is None or (isinstance(c, str) and c.lower() == "none"):
            continue
        try:
            rgba = mpl.colors.to_rgba(c)
        except (ValueError, TypeError):
            continue
        if rgba[3] == 0:          # transparent (e.g. hollow markers' face)
            continue
        role = ROLE_LOOKUP.get(mpl.colors.to_hex(rgba, keep_alpha=False).upper())
        if role:
            return role
    return None


def _all_axes(fig):
    """Every axes of *fig*, recursing into inset/child axes.

    ``Axes.inset_axes`` registers the panel as a child of the parent axes;
    such axes are NOT listed in ``fig.axes``, so a flat iteration silently
    skips their artists (first hit by 04's card insets).
    """
    out = []

    def rec(ax):
        out.append(ax)
        for c in getattr(ax, "child_axes", []):
            rec(c)

    for ax in fig.axes:
        rec(ax)
    return out


def tag_role_gids(fig):
    """Tag every data artist of *fig* whose colour matches the palette.

    Idempotent: artists that already carry a ``layer-`` gid are left alone.
    Spines are skipped (they belong to the frame layer) and scatter points
    too (they belong to the markers layer); both are classed by
    src.svg_export. On matplotlib >= 3.10 ``ax.get_children()`` includes
    the four spines, so the exclusion is required.

    Gridlines are deliberately NOT tagged: they live inside the axis tick
    groups (axes > matplotlib.axis_N > xtick_N > line), so a per-gridline
    gid would be wrapped by src.svg_export into a useless micro-layer
    nested in each tick group. They stay in layer-frame instead (per
    需求总结.md section 4: ``layer-grid`` only covers gridlines that are
    direct children of the axes).
    """
    from matplotlib.collections import Collection, PathCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.spines import Spine

    artists = list(fig.get_children())
    for ax in _all_axes(fig):
        artists += list(ax.get_children())
    n = 0
    for a in artists:
        gid = a.get_gid()
        if gid and gid.startswith("layer-"):
            continue
        if isinstance(a, (Spine, PathCollection)):
            continue
        if not isinstance(a, (Line2D, Collection, Patch)):
            continue
        role = _role_gid(a)
        if role:
            a.set_gid(f"layer-{role}-{n:03d}")
            n += 1


# --------------------------------------------------------------- ladder -----
# honesty ladder (README_规划.md cross-figure symbol): 95/90/85/80/75/70
HONESTY_RUNGS = (0.95, 0.90, 0.85, 0.80, 0.75, 0.70)
HONESTY_BAND = (0.70, 0.95)


def honesty_ladder(ax, rungs=HONESTY_RUNGS, band=HONESTY_BAND, axis="y"):
    """Draw the honesty ladder as panel background.

    The 95% rung is the bold dashed honesty threshold (THR_95); the
    90-70% rungs fade progressively away from it (THR_LADDER); the
    70-95% interval is the light sub-threshold band (THR_BAND). Every
    artist carries a palette colour, so tag_role_gids groups the whole
    ladder into the SVG layer-thr.
    """
    lo, hi = band
    band_fn = ax.axhspan if axis == "y" else ax.axvspan
    line_fn = ax.axhline if axis == "y" else ax.axvline
    band_fn(lo, hi, color=THR_BAND, zorder=0)
    top = line_fn(rungs[0], color=THR_95, lw=1.2, ls=(0, (4, 2)), zorder=0.5)
    rest = [line_fn(r, color=THR_LADDER, lw=0.8, ls=(0, (3, 3)),
                    alpha=0.72 - 0.13 * i, zorder=0.5)
            for i, r in enumerate(rungs[1:])]
    return [top] + rest


# -------------------------------------------------------------- styling ----
def apply_paleo_style(ax):
    """Recessive grid, no box, thin ticks: the common look for all panels."""
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(SECONDARY_INK)
        ax.spines[spine].set_linewidth(0.6)
    ax.tick_params(colors=INK, labelsize=9, length=2.5, width=0.6)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)


def setup_axes(figsize_mm=(SINGLE_COLUMN_MM, 60.0)):
    """New figure + axes at print size, white background, no interpolation."""
    w_in = figsize_mm[0] / 25.4
    h_in = figsize_mm[1] / 25.4
    fig = Figure(figsize=(w_in, h_in), facecolor="white")
    ax = fig.add_subplot(111)
    apply_paleo_style(ax)
    return fig, ax


def save(fig, out_stem, dpi=300):
    """Write PNG (final render) and a layered, text-editable SVG (AI source).

    The SVG is grouped per axes into semantic layers — frame / background /
    grid / band / thr / cat / ink / data / markers / text / annotations /
    legend (see AI_EDITING.md) — so Illustrator's Layers panel shows one
    group per element class, text stays editable, and palette-role layers
    (band / thr / cat / ink / grid) allow batch restyling.
    """
    import os as _os
    import sys as _sys

    _root = _os.path.dirname(_os.path.dirname(
        _os.path.dirname(_os.path.abspath(__file__))))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    from src.svg_export import save_layered_svg  # noqa: E402

    tag_role_gids(fig)
    fig._svg_layer_order = SVG_LAYER_ORDER
    fig.savefig(f"{out_stem}.png", dpi=dpi, facecolor="white")
    save_layered_svg(fig, f"{out_stem}.svg")
    print(f"wrote {out_stem}.png / .svg (layered)")
