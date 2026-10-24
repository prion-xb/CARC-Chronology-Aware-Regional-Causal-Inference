#!/usr/bin/env python3
"""
23_scenarioA_matrix_final.py
----------------------------
Scenario A 6x6 directed robustness matrix drawn from the CANONICAL final
experiment (scripts/22_scenarioA_final.py, selected configuration
[alpha=0.5, diff='force', test='aic_f', pert='test_joint', maxlag=50]).

Unlike scripts/16 (which re-simulated with the earlier ADF-gated min-p F
pipeline), this script only plots: all numbers come from
outputs/scenarioA_final_table1_full_beta0.25.csv, so the figure is exactly
consistent with Table 1 of the main text.

  Panel (a): robustness without chronology error
             = detection rate over the 20 generator realizations on the
               unperturbed frame.
  Panel (b): robustness with chronology error
             = detection rate over 20 realizations x 200 joint age
               perturbations (sigma = sqrt((s1^2 + s2^2) / 2) in ka).

Cell (row, col) = robustness of row -> col:
  - yellow box: prescribed causal edge (with lag and lag/sigma annotation)
  - dashed gray box: reverse (spurious) direction of a prescribed edge
  - dashed hairline: 95% honesty threshold
  - value text: robustness (percent)

Outputs:
  figures/scenarioA_pairwise_matrix.png / .pdf  (replaces the stale figure)
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from src.config import OUTPUT_DIR, FIGURE_DIR

VAR_ORDER = ['NGRIP', 'EDC_dD', 'CO2', 'CH4', 'Cariaco', 'Cheng']
SIGMA_LABEL = {'NGRIP': 0.65, 'EDC_dD': 0.83, 'CO2': 1.65, 'CH4': 1.66,
               'Cariaco': 0.23, 'Cheng': 0.14}
TRUE_EDGES = [('NGRIP', 'EDC_dD', 0.8), ('CO2', 'CH4', 1.5), ('Cariaco', 'Cheng', 2.3)]
THRESHOLD = 0.95

# Palette (matches scripts/16 and the other pairwise-matrix figures)
INK, INK_SEC, INK_MUTED = '#0b0b0b', '#52514e', '#898781'
SURFACE, GRID = '#fcfcfb', '#c3c2b7'
BLUE_HI = '#104281'
TRUE_EDGE = '#eda100'          # yellow slot: prescribed-edge emphasis


def load_results():
    """Load the canonical per-cell robustness values (beta = 0.25)."""
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'scenarioA_final_table1_full_beta0.25.csv'))
    assert (df['beta'] == 0.25).all()
    out_noerr, out_err = {}, {}
    for row in df.itertuples(index=False):
        out_noerr[(row.source, row.target)] = row.robustness_no_error
        out_err[(row.source, row.target)] = row.robustness_with_error_median
    return out_noerr, out_err


def draw_matrix_panel(fig, gs, values, show_labels):
    """Draw one 6x6 directed robustness matrix into a GridSpec."""
    for i, src in enumerate(VAR_ORDER):
        for j, dst in enumerate(VAR_ORDER):
            ax = fig.add_subplot(gs[i, j])
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.06, 1.06)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if i == j:  # diagonal: name + sigma
                ax.set_facecolor('none')
                ax.text(0.5, 0.58, src, ha='center', va='center',
                        fontsize=9, fontweight='bold', color=INK,
                        transform=ax.transAxes)
                ax.text(0.5, 0.36, f'σ = {SIGMA_LABEL[src]:.2f} ka', ha='center',
                        va='center', fontsize=7, color=INK_MUTED,
                        transform=ax.transAxes)
                continue

            v = values[(src, dst)]
            is_true = any(src == t[0] and dst == t[1] for t in TRUE_EDGES)
            is_reverse = any(src == t[1] and dst == t[0] for t in TRUE_EDGES)

            # cell background / border
            if is_true:
                ax.set_facecolor(TRUE_EDGE)
                ax.patch.set_alpha(0.08)
                ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                       fill=False, edgecolor=TRUE_EDGE,
                                       linewidth=2.2, clip_on=False))
            elif is_reverse:
                ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                       fill=False, edgecolor=GRID,
                                       linewidth=1.0, linestyle=(0, (4, 3)),
                                       clip_on=False))

            # 95% threshold hairline
            ax.axhline(THRESHOLD, color=GRID, linewidth=0.7, linestyle=(0, (5, 3)),
                       zorder=1)

            # robustness: thick bar from 0 to value + dot at the value
            color = BLUE_HI if is_true else INK_SEC
            ax.plot([0.12, 0.88], [v, v], color=color, linewidth=2.4,
                    zorder=4, solid_capstyle='round')
            ax.scatter([0.5], [v], s=26, c='white', edgecolors=color,
                       linewidths=1.2, zorder=5)

            # value text (offset right for high values so it stays clear of
            # the true-edge annotation in the cell's top-left corner and of
            # the column label above the panel)
            if v < 0.9:
                ax.text(0.5, v + 0.055, f'{100 * v:.0f}%', ha='center',
                        va='bottom', fontsize=5.8, color=color, zorder=6,
                        transform=ax.get_xaxis_transform())
            elif i == 0:
                # top row: below the bar, so the label stays inside the panel
                # (above the bar it would collide with the column headers)
                ax.text(0.72, v - 0.055, f'{100 * v:.0f}%', ha='center',
                        va='top', fontsize=5.8, color=color, zorder=6,
                        transform=ax.get_xaxis_transform())
            else:
                ax.text(0.72, v + 0.055, f'{100 * v:.0f}%', ha='center',
                        va='bottom', fontsize=5.8, color=color, zorder=6,
                        transform=ax.get_xaxis_transform())

            # true-edge annotation
            if is_true:
                lag = next(t[2] for t in TRUE_EDGES if t[0] == src and t[1] == dst)
                ratio = lag / np.sqrt((SIGMA_LABEL[src] ** 2 + SIGMA_LABEL[dst] ** 2) / 2.0)
                ax.text(0.04, 0.90, f'lag {lag} ka\nlag/σ ≈ {ratio:.1f}',
                        ha='left', va='top', fontsize=6, color=INK_SEC,
                        transform=ax.transAxes, zorder=5)

            # row & column labels
            if show_labels:
                if j == 0:
                    ax.text(-0.42, 0.5, src, ha='right', va='center',
                            fontsize=8.5, color=INK, rotation=0,
                            transform=ax.transAxes)
                if i == 0:
                    ax.text(0.5, 1.18, dst, ha='center', va='center',
                            fontsize=8, color=INK, transform=ax.transAxes)


def main():
    print("Loading canonical Scenario A robustness values (beta = 0.25)...")
    results_noerr, results_err = load_results()

    # sanity print (matches Table 1 of the main text)
    for t in TRUE_EDGES:
        print(f"  {t[0]:>8} -> {t[1]:<8} no-err {results_noerr[(t[0], t[1])]:.2f} "
              f"with-err {results_err[(t[0], t[1])]:.3f}")

    plt.rcParams.update({'font.size': 9, 'font.family': 'Arial',
                         'axes.edgecolor': GRID, 'text.color': INK})
    fig = plt.figure(figsize=(13.5, 7.4), facecolor=SURFACE)

    gs_outer = fig.add_gridspec(1, 2, wspace=0.10, left=0.13, right=0.965,
                                bottom=0.115, top=0.88)
    gs_a = gs_outer[0].subgridspec(6, 6, wspace=0.07, hspace=0.07)
    gs_b = gs_outer[1].subgridspec(6, 6, wspace=0.07, hspace=0.07)

    draw_matrix_panel(fig, gs_a, results_noerr, True)
    draw_matrix_panel(fig, gs_b, results_err, False)

    fig.text(0.13, 0.948, '(a) Without chronology error\n'
             '(20 generator realizations)',
             fontsize=9.5, fontweight='bold', color=INK, va='top')
    fig.text(0.60, 0.948, '(b) With chronology error\n'
             '(20 × 200 joint age perturbations)',
             fontsize=9.5, fontweight='bold', color=INK, va='top')
    fig.text(0.055, 0.968, 'Scenario A: robustness of every directed pair '
             '(cell = row → col; canonical pipeline of Table 1)',
             fontsize=12, fontweight='bold', color=INK)

    handles = [
        Line2D([0], [0], color=BLUE_HI, linewidth=2.4,
               label='robustness (prescribed edge)'),
        Line2D([0], [0], color=INK_SEC, linewidth=2.4,
               label='robustness (control pair)'),
        Line2D([0], [0], color=GRID, linewidth=1, linestyle=(0, (5, 3)),
               label='95% honesty threshold'),
        Rectangle((0, 0), 1, 1, fill=False, edgecolor=TRUE_EDGE, linewidth=2,
                  label='prescribed causal edge'),
        Rectangle((0, 0), 1, 1, fill=False, edgecolor=GRID, linewidth=1,
                  linestyle=(0, (4, 3)), label='reverse direction of a true edge'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, 0.012), handlelength=1.4,
               columnspacing=1.2)

    os.makedirs(FIGURE_DIR, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig_path = os.path.join(FIGURE_DIR, f'scenarioA_pairwise_matrix.{ext}')
        fig.savefig(fig_path, dpi=300, facecolor=SURFACE)
        print(f"Saved: {fig_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
