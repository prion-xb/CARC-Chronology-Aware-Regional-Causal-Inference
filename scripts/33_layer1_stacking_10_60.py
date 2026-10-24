#!/usr/bin/env python3
"""
33_layer1_stacking_10_60.py
----------------------------
Layer 1 evidence figure: the six regional composites on the 10-60 ka
analysis grid, shown as member records (thin) under the median stack
(bold) with the P5-P95 perturbation band. Six climate-state windows
(BA, H1, LGM, H4, GI-12, GS-12) are shaded so the Layer-1 composites can
be read against the same windows used by the downstream causality tests
(main-text Fig. 4 and Figs. S4-S7). Each panel carries the measured
noise-reduction ratio (stack noise / median member noise, from
outputs/stacking_noise_reduction.csv) next to the 1/sqrt(K) reference.

Reuses the stacking machinery of scripts/18_stacking_visualization.py
(median scheme only; same shared perturbation stream, seed 42).

Output: figures/FigS_Layer1_Stacking_10-60.{pdf,png} (+svg via wrapper).
"""

import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.config import (DATA_DIR, OUTPUT_DIR, AGE_GRID_10_60, N_PERTURB,
                        CLIMATE_WINDOWS)
from src.io import load_all_records
from src.stacking import build_region_dict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- reuse stacking machinery (median scheme) from 18_stacking_visualization ---
_spec = importlib.util.spec_from_file_location(
    'stackviz', os.path.join(SCRIPT_DIR, '18_stacking_visualization.py'))
stackviz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stackviz)

PAPER_REGIONS = stackviz.PAPER_REGIONS
REGION_LABELS = stackviz.REGION_LABELS
REGION_COLORS = stackviz.REGION_COLORS
INK, INK_SEC, INK_MUTED = stackviz.INK, stackviz.INK_SEC, stackviz.INK_MUTED
SURFACE, GRID = stackviz.SURFACE, stackviz.GRID
MEMBER = stackviz.MEMBER

# windows shown on the 10-60 ka grid, in chronological order
WINDOW_ORDER = ['GS-12 (stadial)', 'GI-12 (interstadial)', 'H4 event',
                'Last Glacial Maximum', 'H1 event', 'Bølling-Allerød']
WINDOW_ABBR = {'GS-12 (stadial)': 'GS-12', 'GI-12 (interstadial)': 'GI-12',
               'H4 event': 'H4', 'Last Glacial Maximum': 'LGM',
               'H1 event': 'H1', 'Bølling-Allerød': 'B-A'}

# noise-ratio CSV region name -> pipeline region key
NOISE_REGION_MAP = {'Tropical': 'Tropical',
                    'Antarctic/Southern Ocean': 'Antarctic',
                    'North Atlantic': 'North_Atlantic',
                    'East Asian Monsoon': 'East_Asian_Monsoon',
                    'High-latitude Lakes': 'Other_Lakes',
                    'South American Monsoon': 'South_America'}


def load_noise_ratios():
    """{region_key: (achieved_ratio, theoretical_1_sqrt_K)}"""
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'stacking_noise_reduction.csv'))
    return {NOISE_REGION_MAP[row.region]: (row.achieved_ratio,
                                           row.theoretical_1_sqrt_K)
            for _, row in df.iterrows() if row.region in NOISE_REGION_MAP}


def main():
    grid, smooth = AGE_GRID_10_60, 0.5

    print('Loading records (10-60 ka)...')
    records, _ = load_all_records(DATA_DIR, age_min=10.0, age_max=60.0)
    records = [(i, s, 'Antarctic' if r == 'Southern_Ocean' else r, n)
               for i, s, r, n in records]
    region_dict = build_region_dict(records)
    region_dict = {r: v for r, v in region_dict.items() if r in PAPER_REGIONS}
    noise = load_noise_ratios()

    rng = np.random.default_rng(42)
    stacks = {}
    for region, recs in region_dict.items():
        schemes = stackviz.stack_schemes(recs, grid, N_PERTURB, smooth, rng)
        stacks[region] = schemes['median']
        print(f'  {region}: {len(recs)} records')

    # ------------------------------------------------------------------ fig
    plt.rcParams.update({'font.size': 8.5, 'font.family': 'Arial',
                         'axes.edgecolor': GRID, 'text.color': INK})
    fig = plt.figure(figsize=(7.0, 4.991), facecolor=SURFACE)  # 178 mm wide, same aspect
    gs = fig.add_gridspec(2, 3, wspace=0.14, hspace=0.30,
                          left=0.06, right=0.96, bottom=0.09, top=0.90)

    # climate windows as alternating shaded bands (drawn in every panel)
    win_col = ['#ecece8', '#e2e2dc']

    for k, region in enumerate(PAPER_REGIONS):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        ax.set_facecolor(SURFACE)
        color = REGION_COLORS[region]

        # window shading (behind data)
        for j, wname in enumerate(WINDOW_ORDER):
            w0, w1 = CLIMATE_WINDOWS[wname]
            ax.add_patch(Rectangle((w0, -3.6), w1 - w0, 7.2,
                                   facecolor=win_col[j % 2],
                                   edgecolor='none', zorder=0))
        # member records
        for interp, sigma, name in region_dict[region]:
            idx = np.round(np.linspace(0, len(interp) - 1, len(grid))).astype(int)
            ax.plot(grid, interp[idx], color=MEMBER, lw=0.4, alpha=0.7,
                    zorder=1)
        # median stack + P5-P95 band
        med = np.median(stacks[region], axis=0)
        p5, p95 = np.percentile(stacks[region], [5, 95], axis=0)
        ax.fill_between(grid, p5, p95, color=color, alpha=0.22,
                        zorder=2, linewidth=0)
        ax.plot(grid, med, color=color, lw=2.2, zorder=3)

        # noise-reduction annotation (top-right)
        ratio, ref = noise[region]
        ax.text(0.985, 0.93, f'noise ×{ratio:.2f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=7.5, color=INK_SEC,
                fontweight='bold')
        ax.text(0.985, 0.80, f'1/√K = {ref:.2f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=7, color=INK_MUTED)

        ax.set_title(f"{REGION_LABELS[region]}  (n = {len(region_dict[region])})",
                     fontsize=9.5, color=INK, fontweight='bold', pad=4)
        ax.set_xlim(10, 60)
        ax.set_ylim(-3.6, 3.6)
        ax.tick_params(colors=INK_MUTED, labelsize=7)
        ax.grid(True, color='#e1e0d9', lw=0.5, alpha=0.5, zorder=1.5)
        if k // 3 == 1:
            ax.set_xlabel('Age (ka BP)', fontsize=8, color=INK_SEC)
        else:
            ax.tick_params(labelbottom=False)
        if k % 3 == 0:
            ax.set_ylabel('standardized value', fontsize=8, color=INK_SEC)
        else:
            ax.tick_params(labelleft=False)

    # window shading legend strip (bottom): swatch + abbreviation per window
    lax = fig.add_axes([0.06, 0.048, 0.90, 0.024])
    lax.axis('off')
    lax.set_xlim(0, 1); lax.set_ylim(0, 1)
    lax.text(0.0, 0.5, 'climate windows:', fontsize=8, color=INK_SEC,
             ha='left', va='center')
    item_w = 0.115
    x_start = 0.16
    for j, wname in enumerate(WINDOW_ORDER):
        xc = x_start + j * item_w
        lax.add_patch(Rectangle((xc, 0.32), 0.030, 0.40,
                                facecolor=win_col[j % 2],
                                edgecolor=GRID, lw=0.5))
        lax.text(xc + 0.038, 0.5, WINDOW_ABBR[wname], fontsize=7.5,
                 color=INK_MUTED, ha='left', va='center')

    # shorter title so the subtitle (right-aligned on the same row)
    # cannot collide with it at the 178-mm width
    fig.text(0.06, 0.955,
             'Layer 1 — regional stacking (10–60 ka)',
             fontsize=10, fontweight='bold', color=INK)
    # one compact subtitle on the title row (right of the bold title):
    # the panel headers below leave no second free row at 178 mm
    fig.text(0.985, 0.955,
             f'median of the {N_PERTURB}-realization ensemble; '
             f'noise × = stack / median-member noise',
             ha='right', fontsize=8, color=INK_SEC)

    from src.visualization import save_figure
    save_figure(fig, 'FigS_Layer1_Stacking_10-60', fmts=('pdf', 'png', 'tif'),
                dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
