#!/usr/bin/env python3
"""
28_eff_sample_size.py
---------------------
Data-sufficiency diagnostic for the six climate windows (paper Section 4.2 /
Table 3): effective sample size of an AR(1) series as a function of window
width, at the fitted autoregression coefficients of the core records.

The two narrowest windows (Bolling-Allerod and GS 12, width 2 ka, ~40
interpolated points at dt = 0.05 ka) show 0% directional robustness in both
directions for every pairing (Table 4). The main text attributes this to
model selection / F-test power collapse on short series. This figure
quantifies the loss: for an AR(1) series the effective sample size is

    n_eff = n * (1 - rho) / (1 + rho)

(matching the variance of the mean of a correlated series), so at the
fitted phi values of the core records (0.71-0.99, scenarioA_final_params.csv)
a 2 ka window retains only a handful of independent observations.

Analytic only -- no simulations. Output: FigS10_Effective_Sample_Size.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import OUTPUT_DIR, FIGURE_DIR
from src.visualization import apply_style, save_figure, CARC_COLORS

DT = 0.05  # ka, analysis grid of the six-record analysis (Section 4.2)

# Paper Table 3 climate windows: (name, width_ka)  (age ranges in Table 3)
WINDOWS = [
    ('H1 event', 4.0),
    ('H4 event', 4.0),
    ('Bolling-Allerod', 2.0),
    ('LGM', 7.0),
    ('GI 12 (interstadial)', 3.0),
    ('GS 12 (stadial)', 2.0),
]

# Fitted marginal AR(1) coefficients of the six core records
# (outputs/scenarioA_final_params.csv, fitted_phi column).
FITTED_PHI = {'NGRIP': 0.7096, 'EDC_dD': 0.9482, 'CO2': 0.9932,
              'CH4': 0.9872, 'Cariaco': 0.9261, 'Cheng': 0.9331}

RHO_SWEEP = [0.7, 0.8, 0.9]          # AR(1) reference curves
RHO_COLORS = [CARC_COLORS['blue'], CARC_COLORS['orange'], CARC_COLORS['red']]


def n_eff(n, rho):
    """Effective sample size of an AR(1) series (variance-of-mean form)."""
    return n * (1.0 - rho) / (1.0 + rho)


def main():
    apply_style('paper')

    width_grid = np.linspace(1.0, 8.0, 400)
    n_grid = width_grid / DT

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    # Reference curves
    for rho, c in zip(RHO_SWEEP, RHO_COLORS):
        ax.plot(width_grid, n_eff(n_grid, rho), color=c, lw=2.2,
                label=r'AR(1) $\rho$ = ' + f'{rho:.1f}')

    # Core-record fitted coefficients (light markers on the rho=0.9 side)
    ax.scatter([], [], s=0)  # keep legend order stable
    for name, phi in FITTED_PHI.items():
        ax.plot([], [], alpha=0)  # noop guard
    for name, phi in sorted(FITTED_PHI.items(), key=lambda kv: -kv[1]):
        ax.plot(width_grid, n_eff(n_grid, phi), lw=0.9, alpha=0.28,
                color=CARC_COLORS['navy'])

    # Climate windows (Table 3)
    y_win = np.array([n_eff(w / DT, 0.85) for _, w in WINDOWS])
    zero_windows = {'Bolling-Allerod', 'GS 12 (stadial)'}
    # label stagger (n_eff units) so windows sharing a width (H1/H4 at
    # 4 ka, Bolling/GS 12 at 2 ka) do not overlap
    STAGGER = {'H1 event': 6.0, 'H4 event': 0.0, 'Bolling-Allerod': 9.0,
               'LGM': 6.0, 'GI 12 (interstadial)': 6.0, 'GS 12 (stadial)': 0.0}
    for (name, w), y in zip(WINDOWS, y_win):
        is_zero = name in zero_windows
        c = CARC_COLORS['red'] if is_zero else CARC_COLORS['grey']
        ax.axvline(w, color=c, lw=1.0, ls='--' if is_zero else ':', alpha=0.75)
        ax.annotate(name.replace(' (interstadial)', '\ninterstadial').replace(
            ' (stadial)', '\nstadial'),
            xy=(w, y), xytext=(w - 0.18, y + 3.2 + STAGGER[name]),
            fontsize=8, color=c, ha='right', va='bottom',
            annotation_clip=False)

    # Under-powered zone
    ax.axhspan(0, 25, color=CARC_COLORS['grey'], alpha=0.10, lw=0)
    ax.text(7.72, 12, 'under-powered\nzone', fontsize=8,
            color=CARC_COLORS['grey'], ha='right', va='center')

    ax.set_xlabel('Climate window width (ka)')
    ax.set_ylabel('Effective sample size  $n_{eff}$')
    ax.set_xlim(1.0, 8.0)
    ax.set_ylim(0, 160)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.legend(loc='upper left', frameon=True, fontsize=9,
              title='dt = 0.05 ka')

    ax.text(1.02, 156,
            'faint navy curves: fitted $\\rho$ of the six core records\n'
            '(0.71-0.99; Table S3 / scenarioA_final_params.csv)\n'
            'red dashes: windows with 0% robustness in Table 4',
            fontsize=7.5, ha='left', va='top', color='0.25')

    fig.tight_layout()
    save_figure(fig, 'FigS10_Effective_Sample_Size')
    plt.close(fig)

    # Also dump the numeric table for the caption / inventory
    rows = [{'window': n, 'width_ka': w, 'n_points': w / DT,
             **{f'n_eff_rho{rho}': round(n_eff(w / DT, rho), 1)
                for rho in RHO_SWEEP}}
            for n, w in WINDOWS]
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, 'eff_sample_size_by_window.csv'), index=False)
    print('Saved outputs/eff_sample_size_by_window.csv')


if __name__ == '__main__':
    main()
