#!/usr/bin/env python3
"""
17_regional_pairwise_matrix.py
------------------------------
Symmetric 6x6 directed robustness matrix for the six regional composites
(paper Section 4.3). Both axes carry the same six region names; cell
(row, col) shows the robustness of row -> col.

Each off-diagonal cell is a mini strip-chart over the six climate windows
in chronological order (windows encoded by position, not color):
  - blue dots + line: per-window directional robustness
  - thick bar: median across windows
  - hairlines: 95% and 80% honesty thresholds
  - star: pair-window marked phase-significant in the paper (Table 5)
  - yellow border: |A->B - B->A| median asymmetry >= 0.3

Inputs:
  outputs/regional_robustness_windows.csv  (script 09)
Outputs:
  figures/FigS_Regional_Pairwise_Matrix.png/.pdf
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from src.config import DATA_DIR, FIGURE_DIR, OUTPUT_DIR
from src.io import load_all_records
from src.stacking import build_region_dict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAPER_REGIONS = ['North_Atlantic', 'Antarctic', 'Tropical',
                 'East_Asian_Monsoon', 'South_America', 'Other_Lakes']
REGION_LABELS = {
    'North_Atlantic': 'North Atlantic',
    'Antarctic': 'Antarctic/SO',
    'Tropical': 'Tropical',
    'East_Asian_Monsoon': 'E. Asian Monsoon',
    'South_America': 'S. Am. Monsoon',
    'Other_Lakes': 'High-Lat Lakes',
}
# Climate windows in chronological order (ka)
WINDOW_ORDER = ['Bølling-Allerød', 'H1 event', 'Last Glacial Maximum',
                'H4 event', 'GI-12 (interstadial)', 'GS-12 (stadial)']
WINDOW_SHORT = {'Bølling-Allerød': 'BA', 'H1 event': 'H1',
                'Last Glacial Maximum': 'LGM', 'H4 event': 'H4',
                'GI-12 (interstadial)': 'GI-12', 'GS-12 (stadial)': 'GS-12'}

# Paper Table 5 (phase-difference sign consistency, scripts/08):
# phase-significant (pair, window) combinations
PHASE_SIGNIFICANT = {
    ('Antarctic', 'South_America', 'H1 event'),
    ('Antarctic', 'South_America', 'Last Glacial Maximum'),
    ('Antarctic', 'North_Atlantic', 'H4 event'),
    ('East_Asian_Monsoon', 'Tropical', 'Bølling-Allerød'),
    ('Tropical', 'Antarctic', 'Bølling-Allerød'),
}

# Palette: project light-mode reference (validated)
INK, INK_SEC, INK_MUTED = '#0b0b0b', '#52514e', '#898781'
SURFACE, GRID = '#fcfcfb', '#c3c2b7'
BLUE, BLUE_HI = '#2a78d6', '#104281'
ACCENT = '#eda100'          # asymmetry highlight (slot-4 yellow)
THRESH_95, THRESH_80 = 0.95, 0.80
ASYM_MIN = 0.30             # median |A->B - B->A| to flag a cell


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_rates():
    """Return {(src, dst, window): rate} and region record counts."""
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'regional_robustness_windows.csv'))
    rates = {}
    for _, row in df.iterrows():
        rates[(row['region_A'], row['region_B'], row['window'])] = row['A_to_B']
        rates[(row['region_B'], row['region_A'], row['window'])] = row['B_to_A']

    records, _ = load_all_records(DATA_DIR, age_min=10.0, age_max=60.0)
    records = [(i, s, 'Antarctic' if r == 'Southern_Ocean' else r, n)
               for i, s, r, n in records]
    region_dict = build_region_dict(records)
    counts = {r: len(region_dict.get(r, [])) for r in PAPER_REGIONS}
    return rates, counts


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def main():
    rates, counts = load_rates()
    print("Region record counts:", counts)

    plt.rcParams.update({'font.size': 9, 'font.family': 'Arial',
                         'axes.edgecolor': GRID, 'text.color': INK})
    fig = plt.figure(figsize=(7.0, 6.337), facecolor=SURFACE)  # 178 mm wide, same aspect
    # left margin reserves the row-label column; top margin holds the figure
    # title + subtitle + the top row's column labels (title 0.965, subtitle
    # 0.928, column labels ~0.89-0.91)
    gs = fig.add_gridspec(6, 6, wspace=0.10, hspace=0.12,
                          left=0.21, right=0.94, bottom=0.075, top=0.86)

    for i, src in enumerate(PAPER_REGIONS):
        for j, dst in enumerate(PAPER_REGIONS):
            ax = fig.add_subplot(gs[i, j])
            ax.set_xlim(-0.5, len(WINDOW_ORDER) - 0.5)
            ax.set_ylim(-0.06, 1.06)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if i == j:  # diagonal: region name + record count
                ax.set_facecolor('none')
                ax.text(0.5, 0.58, REGION_LABELS[src], ha='center', va='center',
                        fontsize=8.5, fontweight='bold', color=INK,
                        transform=ax.transAxes)
                ax.text(0.5, 0.36, f'{counts[src]} records', ha='center',
                        va='center', fontsize=7, color=INK_MUTED,
                        transform=ax.transAxes)
                continue

            # per-window rates, chronological
            vals = np.array([rates.get((src, dst, w), np.nan)
                             for w in WINDOW_ORDER])
            med = np.nanmedian(vals)

            # asymmetry flag (vs the mirror cell)
            rev = np.array([rates.get((dst, src, w), np.nan)
                            for w in WINDOW_ORDER])
            asym = abs(np.nanmedian(vals) - np.nanmedian(rev)) >= ASYM_MIN
            if asym:
                ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                       fill=False, edgecolor=ACCENT,
                                       linewidth=1.2, clip_on=False,
                                       gid='layer-asym-000'))

            # threshold hairlines
            ax.axhline(THRESH_95, color=GRID, linewidth=0.7,
                       linestyle=(0, (5, 3)), zorder=1,
                       gid='layer-thresholds-000')
            ax.axhline(THRESH_80, color=GRID, linewidth=0.5,
                       linestyle=(0, (1, 4)), zorder=1,
                       gid='layer-thresholds-001')

            xs = np.arange(len(WINDOW_ORDER))
            ok = np.isfinite(vals)
            curve = ax.plot(xs[ok], vals[ok], color=BLUE, linewidth=1.2,
                            zorder=2, alpha=0.85)
            curve[0].set_gid('layer-series-000')
            dots = ax.scatter(xs[ok], vals[ok], s=16, c=BLUE, alpha=0.85,
                              edgecolors='none', zorder=3, linewidths=0)
            dots.set_gid('layer-series-001')

            # phase-significant stars (paper Table 5)
            for k, w in enumerate(WINDOW_ORDER):
                if (src, dst, w) in PHASE_SIGNIFICANT and np.isfinite(vals[k]):
                    star = ax.scatter([k], [vals[k]], marker='*', s=45,
                                      c='white', edgecolors=INK,
                                      linewidths=0.8, zorder=4)
                    star.set_gid('layer-stars-000')

            # median bar
            medbar = ax.plot([-0.35, len(WINDOW_ORDER) - 0.65], [med, med],
                             color=BLUE_HI, linewidth=1.4, zorder=5,
                             solid_capstyle='round')
            medbar[0].set_gid('layer-median-000')

            # row & column labels (kept inside the reserved margins so the
            # canvas never needs re-cropping); column headers use short forms
            # so adjacent cells never collide at the 178-mm canvas width
            if j == 0:
                ax.text(-0.60, 0.5, REGION_LABELS[src], ha='right', va='center',
                        fontsize=8.5, color=INK, rotation=0,
                        transform=ax.transAxes)
            if i == 0:
                col_label = {'East_Asian_Monsoon': 'E. Asian',
                             'South_America': 'S. America'}.get(dst,
                             REGION_LABELS[dst])
                ax.text(0.5, 1.14, col_label, ha='center', va='bottom',
                        fontsize=8.5, color=INK, transform=ax.transAxes)

    fig.text(0.5, 0.965,
             'Directional robustness of the six regional composites '
             '(cell = row → col)',
             fontsize=10, fontweight='bold', color=INK, ha='center')
    fig.text(0.5, 0.928,
             'each cell: robustness in the six climate windows '
             '(BA, H1, LGM, H4, GI-12, GS-12) · bar = window median',
             fontsize=9, color=INK_SEC, ha='center')

    # legend
    handles = [
        Line2D([], [], marker='o', linestyle='-', ms=4.5, mfc=BLUE, mec='none',
               color=BLUE, linewidth=1.2, label='per-window robustness\n(chronological order)'),
        Line2D([0], [0], color=BLUE_HI, linewidth=1.4,
               label='median across windows'),
        Line2D([0], [0], color=GRID, linewidth=1, linestyle=(0, (5, 3)),
               label='95% honesty threshold'),
        Line2D([0], [0], color=GRID, linewidth=1, linestyle=(0, (1, 4)),
               label='80% threshold (relaxed)'),
        Line2D([], [], marker='*', linestyle='none', ms=9, mfc='white',
               mec=INK, mew=0.8, label='phase-significant\n(Table 5)'),
        Rectangle((0, 0), 1, 1, fill=False, edgecolor=ACCENT, linewidth=1.2,
                  label='asymmetric pair\n(Δmedian ≥ 0.30)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, 0.012), handlelength=1.4,
               columnspacing=1.2)

    # semantic layer order for the layered SVG (scripts/export_layered_svg.py
    # reads this attribute automatically); row/column/diagonal labels stay in
    # the generic 'text' layer, everything else falls back to the generic
    # class layers (frame/data/markers/...)
    fig._svg_layer_order = ('thresholds', 'series', 'median', 'stars',
                            'asym')

    # route through save_figure: the direct savefig path left the PNG in
    # RGBA, which journals reject on preflight; main-text Fig. 6, so the
    # LZW TIFF is the delivery format
    from src.visualization import save_figure
    save_figure(fig, 'FigS_Regional_Pairwise_Matrix',
                fmts=('pdf', 'png', 'tif'), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
