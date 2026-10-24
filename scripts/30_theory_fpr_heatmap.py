#!/usr/bin/env python3
"""
30_theory_fpr_heatmap.py
-----------------------
Theoretical false-positive-rate lower bound (Theorem 3) over the parameter
space of the two-record common-forcing model, linking the abstract bound to
the empirical results (paper Section 2.5, eqs. 4-5).

Model (paper eqs. 1-2):
    X_t = phi_X X_{t-1} + beta_X Z_t + eps_X,t
    Y_t = phi_Y Y_{t-1} + beta_Y Z_t + eps_Y,t
with Z_t ~ N(0, sigma_Z^2), lag-1 autocovariance gamma_Z(1) != 0, and
eps ~ N(0, sigma_eps^2).

Effective signal-to-noise ratio (eq. 4) and FPR lower bound (Theorem 3,
eq. 5; formulas extracted from the OMML equations of the manuscript):

    lambda_eff = n * beta_X^2 beta_Y^2 gamma_Z^2
                 / ((1 - phi_X^2) sigma_epsY^2 (beta_X^2 sigma_Z^2
                                                + sigma_epsX^2))
    FPR >= 1 - Phi(z_{1-alpha} - lambda_eff / 2)

Anchor check (paper Section 5.1): at the stated typical paleoclimate
parameters beta_X = beta_Y = 0.35, sigma_Z^2 = 1, sigma_eps^2 = 0.5,
gamma_Z = 0.6, phi_X = 0.55, n = 100 the bound is ~34.4%, matching the
"approximately 35%" quoted in the manuscript. Scenario B (weaker forcing
and autocorrelation) predicts ~18% (marked as a contour level).

Analytic only -- no simulations. Output: FigS12_Theoretical_FPR_Heatmap.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from src.visualization import apply_style, save_figure, CARC_COLORS

Z_ALPHA = norm.ppf(0.95)          # 1.6449, alpha = 0.05
N_CANON = 100                     # sample size of the manuscript's example

# Manuscript "typical paleoclimate parameters" (Section 5.1)
CANON = dict(beta_X=0.35, beta_Y=0.35, gamma_Z=0.6, phi_X=0.55,
             sigma_Z2=1.0, sigma_epsX2=0.5, sigma_epsY2=0.5, n=N_CANON)

CONTOURS = [0.05, 0.18, 0.35, 0.5, 0.7, 0.9]


def fpr_lb(beta_X, beta_Y, gamma_Z, phi_X, sigma_Z2=1.0,
           sigma_epsX2=0.5, sigma_epsY2=0.5, n=N_CANON):
    """Theorem 3 FPR lower bound (eqs. 4-5 of the manuscript)."""
    lam = (n * beta_X ** 2 * beta_Y ** 2 * gamma_Z ** 2
           / ((1.0 - phi_X ** 2) * sigma_epsY2
              * (beta_X ** 2 * sigma_Z2 + sigma_epsX2)))
    return 1.0 - norm.cdf(Z_ALPHA - lam / 2.0)


def _draw_heatmap(ax, X, Y, Z, title, marker=None, marker_label=None,
                  cbar_label='FPR lower bound (Theorem 3)', clabel_fmt='%.2f'):
    cs = ax.contourf(X, Y, Z, levels=CONTOURS, vmin=0.0, vmax=1.0,
                     cmap='Reds', extend='both')
    cl = ax.contour(X, Y, Z, levels=CONTOURS, colors='0.25',
                    linewidths=0.5)
    ax.clabel(cl, inline=True, fontsize=7, fmt=clabel_fmt)
    cb = ax.figure.colorbar(cs, ax=ax, fraction=0.046, pad=0.04,
                            label=cbar_label)
    cb.set_ticks(CONTOURS)
    cb.ax.tick_params(labelsize=7)
    if marker is not None:
        ax.plot(marker[0], marker[1], 'o', ms=8,
                mfc='none', mec=CARC_COLORS['navy'], mew=2.0)
        ax.annotate(marker_label, xy=marker,
                    xytext=(marker[0] + 0.02, marker[1] + 0.02),
                    fontsize=8, color=CARC_COLORS['navy'],
                    annotation_clip=False)
    ax.set_title(title, fontsize=10)
    ax.grid(False)
    return cs


def main():
    apply_style('paper')

    # --- anchor verification (printed for the record) ---
    anchor = fpr_lb(**{k: CANON[k] for k in
                       ('beta_X', 'beta_Y', 'gamma_Z', 'phi_X',
                        'sigma_Z2', 'sigma_epsX2', 'sigma_epsY2', 'n')})
    print(f'Anchor check: FPR lower bound at canonical parameters '
          f'= {anchor:.3f} (manuscript: ~35%)')

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9),
                             gridspec_kw={'width_ratios': [1, 1, 1.15]})

    # (a) beta_X x beta_Y at canonical gamma_Z / phi_X
    g = np.linspace(0.05, 0.5, 120)
    BX, BY = np.meshgrid(g, g)
    Za = fpr_lb(BX, BY, CANON['gamma_Z'], CANON['phi_X'],
                CANON['sigma_Z2'], CANON['sigma_epsX2'], CANON['sigma_epsY2'])
    _draw_heatmap(axes[0], BX, BY, Za,
                  r'(a) FPR bound vs forcing loadings' + '\n'
                  + r'($\gamma_Z$=0.6, $\phi_X$=0.55, n=100)',
                  marker=(CANON['beta_X'], CANON['beta_Y']),
                  marker_label='typical\nparams')

    # (b) gamma_Z x phi_X at canonical beta
    g2 = np.linspace(0.1, 0.9, 120)
    g3 = np.linspace(0.3, 0.9, 120)
    GZ, PH = np.meshgrid(g2, g3)
    Zb = fpr_lb(CANON['beta_X'], CANON['beta_Y'], GZ, PH,
                CANON['sigma_Z2'], CANON['sigma_epsX2'], CANON['sigma_epsY2'])
    _draw_heatmap(axes[1], GZ, PH, Zb,
                  r'(b) FPR bound vs forcing memory' + '\n'
                  + r'($\beta_X$=$\beta_Y$=0.35, n=100)',
                  marker=(CANON['gamma_Z'], CANON['phi_X']),
                  marker_label='typical\nparams')

    # (c) FPR vs gamma_Z for several loadings
    for beta, c, ls in [(0.2, CARC_COLORS['blue'], ':'),
                        (0.35, CARC_COLORS['navy'], '-'),
                        (0.5, CARC_COLORS['red'], '--')]:
        axes[2].plot(g2, fpr_lb(beta, beta, g2, CANON['phi_X']),
                     color=c, ls=ls, lw=2.0,
                     label=r'$\beta_X$=$\beta_Y$=' + f'{beta:.2f}')
    axes[2].axhline(0.05, color=CARC_COLORS['grey'], lw=1.2, ls='-.')
    axes[2].text(0.865, 0.07, 'nominal $\\alpha$ = 0.05', fontsize=7.5,
                 color=CARC_COLORS['grey'], ha='right')
    axes[2].axhline(0.18, color=CARC_COLORS['purple'], lw=1.2, ls=':')
    axes[2].text(0.865, 0.205, 'Scenario B bound ~18%', fontsize=7.5,
                 color=CARC_COLORS['purple'], ha='right')
    axes[2].axhline(0.35, color=CARC_COLORS['red'], lw=0.8, alpha=0.5)
    axes[2].set_xlabel(r'$\gamma_Z(1)$  (forcing lag-1 autocovariance)')
    axes[2].set_ylabel('FPR lower bound')
    axes[2].set_xlim(0.1, 0.9)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].grid(True, alpha=0.25, lw=0.6)
    axes[2].legend(fontsize=8, loc='upper left', frameon=True)
    axes[2].set_title(r'(c) FPR bound vs $\gamma_Z$' + '\n'
                      + r'($\phi_X$=0.55, n=100)', fontsize=10)

    axes[0].set_xlabel(r'$\beta_X$'); axes[0].set_ylabel(r'$\beta_Y$')
    axes[1].set_xlabel(r'$\gamma_Z(1)$'); axes[1].set_ylabel(r'$\phi_X$')

    fig.tight_layout()
    save_figure(fig, 'FigS12_Theoretical_FPR_Heatmap')
    plt.close(fig)


if __name__ == '__main__':
    main()
