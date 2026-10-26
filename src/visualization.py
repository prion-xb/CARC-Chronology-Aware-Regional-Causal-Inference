"""
Shared visualization style and helper functions for CARC figures.

Provides a consistent, publication-quality appearance across all scripts:
  - Larger, readable typography
  - Fine axis ticks (major + minor)
  - High-contrast scientific colour palettes
  - Subtle background grids
  - Convenient annotation & legend helpers
  - Geographic map helpers (Cartopy)

Usage (at the top of any plotting script):
    from src.visualization import apply_style, CARC_COLORS, add_panel_label
    apply_style()
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import os

# ---------------------------------------------------------------------------
# Global matplotlib rcParams – call once per script
# ---------------------------------------------------------------------------

def apply_style(context: str = "paper", font_scale: float = 1.0):
    """
    Apply CARC publication style to all matplotlib figures.

    Parameters
    ----------
    context : str
        "paper"   – tight, publication-ready (smaller margins)
        "poster"  – larger elements for presentations
        "talk"    – intermediate
    font_scale : float
        Multiply all font sizes by this factor.
    """
    # Print-size typography (2026-09-21): paper figures are rendered at the
    # final column width (178 mm = 7.0 in), so the base font is 8 pt — the
    # Elsevier minimum is 7 pt. Larger canvases are a layout error, not a
    # license for larger fonts (see CARC_final2 figure-audit notes).
    base_font_size = {"paper": 8, "talk": 14, "poster": 18}.get(context, 8)
    base_font_size = int(base_font_size * font_scale)

    plt.rcParams.update({
        # --- Typography ---
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        # mathtext in Arial too: DejaVu math glyphs trigger "missing glyph"
        # warnings in Adobe Illustrator (no DejaVu on Windows)
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.cal": "Arial:italic",
        "font.size": base_font_size,
        "axes.titlesize": base_font_size + 1,
        "axes.labelsize": base_font_size,
        "xtick.labelsize": base_font_size - 1,
        "ytick.labelsize": base_font_size - 1,
        "legend.fontsize": base_font_size - 1,
        "legend.title_fontsize": base_font_size,

        # --- Figure ---
        "figure.dpi": 150,
        "savefig.dpi": 300,
        # NOTE: fixed canvas (no tight bbox). Tight cropping makes margins
        # figure-dependent and shifts panels relative to each other; layout
        # QA (scripts/figure_tools/qa_figures.py) relies on the fixed canvas. Panels that
        # stick out are reported as text_outside instead of silently
        # re-cropping the figure.
        "savefig.bbox": None,
        "savefig.pad_inches": 0.05,
        # TrueType subset embedding: Elsevier (and most publishers) reject
        # matplotlib's default Type3 outlines — the delivered Fig1 PDF must
        # carry /FontFile entries for every glyph it uses
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # --- Axes ---
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "both",
        "grid.alpha": 0.25,
        "grid.linestyle": "dotted",
        "grid.linewidth": 0.4,

        # --- Ticks ---
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "xtick.minor.width": 0.5,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,

        # --- Lines & markers ---
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "lines.markeredgewidth": 0.5,
        "patch.linewidth": 0.6,

        # --- Legend ---
        "legend.frameon": True,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "#cccccc",
        "legend.fancybox": False,
        "legend.loc": "best",

        # --- Colour maps ---
        "image.cmap": "viridis",
    })


def apply_style_example(font_scale: float = 1.0):
    """
    Apply publication-quality style matching top-journal figure aesthetics:
      - Pure white background, ~90% white space
      - Thin solid gridlines (recessive, one shade off background)
      - Clean outward ticks, no minor ticks by default
      - Content in black/grayscale; color reserved for scientific accents
      - Thin marks, generous padding between panels

    Use this for figures that should read like Nature/Science multi-panel
    compositions: single-column vertical narrative flow, one scientific
    question answered per figure.

    Parameters
    ----------
    font_scale : float
        Multiply all font sizes by this factor (1.0 = ~9pt base).
    """
    base_fs = int(9 * font_scale)

    plt.rcParams.update({
        # --- Typography ---
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        # mathtext in Arial too (see apply_style)
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.cal": "Arial:italic",
        "font.size": base_fs,
        "axes.titlesize": base_fs + 1,
        "axes.labelsize": base_fs,
        "xtick.labelsize": base_fs - 1,
        "ytick.labelsize": base_fs - 1,
        "legend.fontsize": base_fs - 1,
        "legend.title_fontsize": base_fs,

        # --- Figure ---
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": None,          # fixed canvas; see note in apply_style
        "savefig.pad_inches": 0.08,
        "figure.facecolor": "white",

        # --- Axes ---
        "axes.facecolor": "white",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,          # off by default; panels opt in
        "axes.axisbelow": True,

        # --- Grid (when enabled per-panel) ---
        "grid.alpha": 0.30,
        "grid.linestyle": "-",       # solid hairline — never dashed
        "grid.linewidth": 0.4,
        "grid.color": "#D8D8D8",

        # --- Ticks ---
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.size": 0,
        "ytick.minor.size": 0,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "xtick.color": "#333333",
        "ytick.color": "#333333",

        # --- Lines & markers ---
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "lines.markeredgewidth": 0.8,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "patch.linewidth": 0.6,
        "patch.edgecolor": "#333333",
        "patch.force_edgecolor": False,

        # --- Legend ---
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#D0D0D0",
        "legend.fancybox": False,
        "legend.loc": "best",
        "legend.borderpad": 0.5,
        "legend.labelspacing": 0.3,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.6,

        # --- Colour maps ---
        "image.cmap": "Greys",
    })


# ---------------------------------------------------------------------------
# Colour palettes – rich, high-contrast, colourblind-friendly
# ---------------------------------------------------------------------------

# Primary categorical palette
CARC_COLORS = {
    "blue":       "#2166AC",   # strong blue
    "red":        "#D73027",   # strong red
    "green":      "#1B7837",   # dark green
    "orange":     "#E66101",   # vivid orange
    "purple":     "#762A83",   # deep purple
    "teal":       "#008080",   # teal
    "brown":      "#8C510A",   # brown
    "pink":       "#C51B7D",   # magenta/pink
    "grey":       "#636363",   # medium grey
    "gold":       "#D4A017",   # gold
    "navy":       "#1A3A5C",   # dark navy
}

# Lighter variants for fills / backgrounds
CARC_COLORS_LIGHT = {
    "blue":       "#D1E5F0",
    "red":        "#F4A582",
    "green":      "#A6DBA0",
    "orange":     "#FDB863",
    "purple":     "#C2A5CF",
    "teal":       "#80CDCD",
    "brown":      "#BF812D",
    "pink":       "#DE77AE",
    "grey":       "#BDBDBD",
    "gold":       "#F0DC82",
    "navy":       "#7BA0C0",
}

# Ordered categorical list – cycle through this
CARC_PALETTE = [
    CARC_COLORS["blue"],
    CARC_COLORS["red"],
    CARC_COLORS["green"],
    CARC_COLORS["orange"],
    CARC_COLORS["purple"],
    CARC_COLORS["teal"],
    CARC_COLORS["brown"],
    CARC_COLORS["pink"],
    CARC_COLORS["gold"],
    CARC_COLORS["navy"],
    CARC_COLORS["grey"],
]

# Diverging pairs for A↔B comparisons
CARC_DIVERGING_PAIRS = [
    (CARC_COLORS["blue"], CARC_COLORS["orange"]),
    (CARC_COLORS["red"],  CARC_COLORS["teal"]),
    (CARC_COLORS["purple"], CARC_COLORS["green"]),
]

# Sequential light-to-dark for progressions
CARC_SEQUENTIAL = {
    "blues":   ["#D1E5F0", "#92C5DE", "#4393C3", "#2166AC", "#053061"],
    "reds":    ["#FDDBC7", "#F4A582", "#D6604D", "#B2182B", "#67001F"],
    "greens":  ["#E5F5E0", "#A6DBA0", "#4DAF4A", "#1B7837", "#00441B"],
    "oranges": ["#FEE0B6", "#FDB863", "#E08214", "#B35806", "#7F3B08"],
    "purples": ["#E7D4E8", "#C2A5CF", "#9970AB", "#762A83", "#40004B"],
}


# ---------------------------------------------------------------------------
# Helper: add subtle minor ticks on both axes
# ---------------------------------------------------------------------------

def add_minor_ticks(ax, x_ticks=None, y_ticks=None):
    """Enable and configure minor ticks on an axes."""
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(4))
    ax.tick_params(which="minor", length=3, width=0.6, color="#999999")


# ---------------------------------------------------------------------------
# Helper: add a bold panel label (a), (b), etc.
# ---------------------------------------------------------------------------

def add_panel_label(ax, label, x=-0.08, y=1.06, **kwargs):
    """
    Add a bold panel label e.g. '(a)' to the top-left of an axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    label : str — e.g. '(a)', '(b)', 'A', 'B'
    x, y : float — position in axes fraction coords
    """
    defaults = dict(
        transform=ax.transAxes,
        fontsize=plt.rcParams["font.size"] + 1,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    defaults.update(kwargs)
    ax.text(x, y, label, **defaults)


# ---------------------------------------------------------------------------
# Helper: statistical annotation box
# ---------------------------------------------------------------------------

def add_stats_box(ax, stats_dict, loc="upper right", **kwargs):
    """
    Add a text box with summary statistics.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    stats_dict : dict — {label: value} pairs
    loc : str — "upper right", "upper left", etc.
    """
    lines = []
    for k, v in stats_dict.items():
        if isinstance(v, float):
            lines.append(f"{k}: {v:.3f}")
        else:
            lines.append(f"{k}: {v}")
    text = "\n".join(lines)

    loc_map = {
        "upper right": (0.97, 0.97),
        "upper left": (0.03, 0.97),
        "lower right": (0.97, 0.03),
        "lower left": (0.03, 0.03),
    }
    x, y = loc_map.get(loc, (0.97, 0.97))
    ha = "right" if "right" in loc else "left"
    va = "top" if "upper" in loc else "bottom"

    defaults = dict(
        transform=ax.transAxes,
        fontsize=plt.rcParams["font.size"] - 1,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                   edgecolor="#cccccc", alpha=0.9),
        va=va, ha=ha,
    )
    defaults.update(kwargs)
    ax.text(x, y, text, **defaults)


# ---------------------------------------------------------------------------
# Helper: annotation arrow + text
# ---------------------------------------------------------------------------

def add_annotation(ax, text, xy, xytext, arrow_color="#333333", **kwargs):
    """
    Add an annotated arrow pointing to a feature.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    text : str — annotation text
    xy : tuple — (x, y) point to annotate (data coords)
    xytext : tuple — (x, y) text position (data coords)
    """
    defaults = dict(
        arrowprops=dict(arrowstyle="->", color=arrow_color,
                        lw=1.2, connectionstyle="arc3,rad=0.2"),
        fontsize=plt.rcParams["font.size"] - 1,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                   edgecolor=arrow_color, alpha=0.85),
        ha="center", va="center",
    )
    defaults.update(kwargs)
    ax.annotate(text, xy=xy, xytext=xytext, **defaults)


# ---------------------------------------------------------------------------
# Helper: highlighted region box on maps
# ---------------------------------------------------------------------------

def add_region_box(ax, lon_min, lon_max, lat_min, lat_max,
                   color="red", linewidth=2, linestyle="--", alpha=0.8,
                   label=None, facecolor=None, facealpha=0.08):
    """
    Draw a highlighted rectangle on a map (e.g. to mark a region of interest).
    """
    from matplotlib.patches import Rectangle
    width = lon_max - lon_min
    height = lat_max - lat_min
    rect = Rectangle((lon_min, lat_min), width, height,
                     linewidth=linewidth, linestyle=linestyle,
                     edgecolor=color, facecolor=facecolor or color,
                     alpha=facealpha, transform=ax.transData,
                     zorder=2)
    ax.add_patch(rect)
    if label:
        ax.text(lon_min + width / 2, lat_max + 1.5, label,
                ha="center", va="bottom", fontsize=plt.rcParams["font.size"] - 1,
                fontweight="bold", color=color, transform=ax.transData)


# ---------------------------------------------------------------------------
# Helper: save with consistent settings
# ---------------------------------------------------------------------------

def save_figure(fig, name, out_dir=None, fmts=("pdf", "png"), **kwargs):
    """
    Save figure in multiple formats with consistent settings.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    name : str — base filename (without extension)
    out_dir : str or None — output directory; defaults to FIGURE_DIR
    fmts : tuple of str — e.g. ("pdf", "png")
    """
    from src.config import FIGURE_DIR

    directory = out_dir or FIGURE_DIR
    os.makedirs(directory, exist_ok=True)

    for fmt in fmts:
        path = os.path.join(directory, f"{name}.{fmt}")
        fig.savefig(path, format=fmt, **kwargs)
        # Journal-ready raster post-processing (2026-09-21): journals reject
        # RGBA PNGs (empty alpha channel) and uncompressed TIFFs (multi-MB
        # uploads fail). Re-save both with Pillow: PNG flattened to RGB,
        # TIFF losslessly LZW-compressed, 300-dpi metadata preserved.
        # The temp file sidesteps Windows file locking (source still open
        # for read while writing).
        if fmt in ("png", "tif"):
            import tempfile as _tf
            from PIL import Image as _Img
            _im = _Img.open(path)
            _im.load()
            # journals reject alpha channels outright: PNG preflight checks
            # for RGB, and TIFF readers vary on RGBA; the canvas below is
            # opaque (figure facecolor), so flattening loses nothing
            if _im.mode == "RGBA":
                _im = _im.convert("RGB")
            # Temp must live on the same drive as `path` — os.replace
            # cannot move a file across volumes on Windows.
            with _tf.NamedTemporaryFile(suffix="." + fmt, delete=False,
                                        dir=os.path.dirname(path)) as _tmp:
                _tmp_path = _tmp.name
            if fmt == "png":
                _im.save(_tmp_path, "PNG", dpi=(300, 300))
            else:
                _im.save(_tmp_path, "TIFF", compression="tiff_lzw",
                         dpi=(300, 300))
            _im.close()
            os.replace(_tmp_path, path)
        print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Helper: add a horizontal or vertical reference band
# ---------------------------------------------------------------------------

def add_threshold_band(ax, value, orientation="horizontal", color="red",
                       linestyle="--", linewidth=1.2, alpha=0.6, label=None):
    """Add a dashed reference line with optional label."""
    if orientation == "horizontal":
        ax.axhline(value, color=color, linestyle=linestyle,
                   linewidth=linewidth, alpha=alpha, label=label)
    else:
        ax.axvline(value, color=color, linestyle=linestyle,
                   linewidth=linewidth, alpha=alpha, label=label)


# ---------------------------------------------------------------------------
# Helper: add a shaded band between two values
# ---------------------------------------------------------------------------

def add_shaded_band(ax, low, high, orientation="vertical", color="grey",
                    alpha=0.15, label=None):
    """Add a shaded band (e.g. climate window) between two values."""
    if orientation == "vertical":
        ax.axvspan(low, high, facecolor=color, alpha=alpha, zorder=0, label=label)
    else:
        ax.axhspan(low, high, facecolor=color, alpha=alpha, zorder=0, label=label)


# ---------------------------------------------------------------------------
# Helper: richer boxplot styling
# ---------------------------------------------------------------------------

def styled_boxplot(ax, data, positions, widths=0.5, colors=None, labels=None):
    """
    Draw boxplots with CARC styling.
    Returns the boxplot dict for further customisation.
    """
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=widths,
        patch_artist=True,
        showfliers=True,
        showcaps=True,
        showmeans=False,
        medianprops={"color": "black", "linewidth": 2},
        whiskerprops={"color": "#333333", "linewidth": 1.2},
        capprops={"color": "#333333", "linewidth": 1.2},
        flierprops={"marker": "o", "markerfacecolor": "#999999",
                    "markersize": 4, "alpha": 0.5},
    )
    if colors:
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor("#333333")
            patch.set_linewidth(1.0)

    return bp


# ---------------------------------------------------------------------------
# Helper: colorbar with consistent styling
# ---------------------------------------------------------------------------

def add_colorbar(fig, mappable, label, ax=None, orientation="vertical",
                 shrink=0.8, pad=0.02, **kwargs):
    """Add a styled colorbar to the figure."""
    cbar = fig.colorbar(mappable, ax=ax, orientation=orientation,
                        shrink=shrink, pad=pad, **kwargs)
    cbar.set_label(label, fontsize=plt.rcParams["axes.labelsize"])
    cbar.ax.tick_params(labelsize=plt.rcParams["xtick.labelsize"])
    return cbar


# ---------------------------------------------------------------------------
# Helper: custom legend with coloured markers
# ---------------------------------------------------------------------------

def add_custom_legend(ax, items, loc="upper left", title=None, **kwargs):
    """
    Add a custom legend from a list of (label, color, marker) tuples.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    items : list of (label, color, marker) — marker can be 'o', 's', '^', etc.
    """
    handles = []
    for label, color, marker in items:
        handles.append(mpatches.Patch(color=color, label=label) if marker is None
                       else plt.Line2D([0], [0], marker=marker, color="w",
                                       markerfacecolor=color, markersize=8,
                                       markeredgewidth=0.5, markeredgecolor="#333333"))
    legend = ax.legend(handles=handles, loc=loc, title=title,
                       framealpha=0.85, edgecolor="#cccccc", **kwargs)
    if title:
        legend.get_title().set_fontweight("bold")
    return legend


# ---------------------------------------------------------------------------
# Map helpers (Cartopy)
# ---------------------------------------------------------------------------

def add_map_gridlines(ax, draw_labels=True, alpha=0.3, linewidth=0.5,
                      linestyle="dotted", **kwargs):
    """Add geographic gridlines to a Cartopy GeoAxes."""
    try:
        import cartopy.crs as ccrs
        gl = ax.gridlines(draw_labels=draw_labels, alpha=alpha,
                          linewidth=linewidth, linestyle=linestyle, **kwargs)
        if draw_labels:
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {"size": plt.rcParams["xtick.labelsize"] - 1}
            gl.ylabel_style = {"size": plt.rcParams["ytick.labelsize"] - 1}
        return gl
    except ImportError:
        pass


def add_map_scale_bar(ax, length_deg, label, lon0=None, lat0=None,
                      color="black", linewidth=2):
    """
    Add a simple scale bar to a geographic map.
    Default position: bottom-left.
    """
    if lon0 is None or lat0 is None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        lon0 = xlim[0] + (xlim[1] - xlim[0]) * 0.05
        lat0 = ylim[0] + (ylim[1] - ylim[0]) * 0.06

    ax.plot([lon0, lon0 + length_deg], [lat0, lat0], color=color,
            linewidth=linewidth, transform=ax.transData, clip_on=False)
    ax.plot([lon0, lon0], [lat0 - 0.3, lat0 + 0.3], color=color,
            linewidth=linewidth, transform=ax.transData, clip_on=False)
    ax.plot([lon0 + length_deg, lon0 + length_deg], [lat0 - 0.3, lat0 + 0.3],
            color=color, linewidth=linewidth, transform=ax.transData, clip_on=False)
    ax.text(lon0 + length_deg / 2, lat0 + 0.5, label, ha="center", va="bottom",
            fontsize=plt.rcParams["font.size"] - 2, color=color,
            transform=ax.transData)


def add_ocean_background(ax, color="#E8F0F8", zorder=0):
    """Fill ocean with a subtle blue tint."""
    try:
        import cartopy.feature as cfeature
        ax.add_feature(cfeature.OCEAN, facecolor=color, zorder=zorder)
    except ImportError:
        pass


def add_land_background(ax, color="#F5F0E8", zorder=0):
    """Fill land with a subtle beige tint."""
    try:
        import cartopy.feature as cfeature
        ax.add_feature(cfeature.LAND, facecolor=color, zorder=zorder)
    except ImportError:
        pass


def add_coastlines(ax, linewidth=0.8, edgecolor="#555555", zorder=1):
    """Add coastline outlines."""
    try:
        import cartopy.feature as cfeature
        ax.add_feature(cfeature.COASTLINE, linewidth=linewidth,
                       edgecolor=edgecolor, zorder=zorder)
    except ImportError:
        pass
