# -*- coding: utf-8 -*-
"""Step 03 - why the naive Granger test cannot self-correct (Section 2).

Two-record common-forcing model (paper eqs. 1-2, Theorems 2-3):

    X_t = phi X_{t-1} + beta Z_t + eps_X,t      (Y likewise; H0: no edge)
    Z: persistent common forcing, lag-1 autocovariance gamma_Z(1)

Analytic only (as in scripts/30_theory_fpr_heatmap.py): both panels are
direct plots of the manuscript equations.

    lambda_eff = n beta^4 gamma_Z^2
                 / ((1 - phi^2) sigma_eps^2 (beta^2 sigma_Z^2 + sigma_eps^2))
    E[F]       = 1 + lambda_eff (n - p - q - 1) / (n q)          (Theorem 2)
    FPR        >= 1 - Phi(z_0.95 - lambda_eff / 2)               (Theorem 3)

Panel (a): expected F under H0 vs forcing loading beta at n = 100 - a
modest coupling inflates the F-statistic several-fold.
Panel (b): the false-positive rate never returns to the nominal 5%:
lambda_eff grows with n, so more data makes the problem worse.

Convergence targets: at the manuscript's typical paleoclimate parameters
(beta = 0.35, gamma_Z = 0.6, phi = 0.55, sigma_Z^2 = 1, sigma_eps^2 = 0.5,
n = 100) the bound is 34.4% (the "~35%" quoted in Section 5.1).

Note: a faithful Monte Carlo of model (1)-(2) yields noncentrality ~20%
below eq. 4 (eq. 4 approximates Cov(X_{t-1}, Z_t) by beta gamma_Z / (1-phi^2)
and the residual variance by sigma_eps^2); simulations therefore stay out of
this figure (see the 攻关题 8 exercise for the numerical check).

Usage:
    python scripts/fignew/03_theorem_fpr.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from matplotlib.figure import Figure
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import (apply_paleo_style, save, INK, SECONDARY_INK,  # noqa: E402
                   CAT1, THR_95)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "figures_new", "03_theorem_fpr")

# manuscript "typical paleoclimate parameters" (Section 5.1, scripts/30)
PHI = 0.55          # AR(1) coefficient of both records
GAMMA_Z = 0.6       # lag-1 autocovariance of Z (sigma_Z^2 = 1)
SIGMA_Z2 = 1.0
SIGMA_EPS2 = 0.5
N_CANON = 100       # sample size of the manuscript's example
P, Q = 1, 1         # lag-1 Granger test (p Y-lags, q X-lags tested)


def lam_eff(n, beta):
    """Effective forcing coupling (paper eq. 4)."""
    return (n * beta ** 4 * GAMMA_Z ** 2
            / ((1.0 - PHI ** 2) * SIGMA_EPS2
               * (beta ** 2 * SIGMA_Z2 + SIGMA_EPS2)))


def e_f(n, beta):
    """Theorem 2: expected F under H0."""
    return 1.0 + lam_eff(n, beta) * (n - P - Q - 1) / (n * Q)


def fpr_bound(n, beta):
    """Theorem 3 lower bound, 1 - Phi(z_0.95 - lambda_eff / 2)."""
    return 1.0 - norm.cdf(norm.ppf(0.95) - lam_eff(n, beta) / 2.0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # anchor check (canonical, printed for the record)
    lam = lam_eff(N_CANON, 0.35)
    bound = fpr_bound(N_CANON, 0.35)
    print(f"anchor: lambda_eff(n=100, beta=0.35) = {lam:.3f}, "
          f"E[F] = {e_f(N_CANON, 0.35):.2f}, "
          f"FPR lower bound = {bound:.3f} (manuscript: ~35%)")

    fig = Figure(figsize=(178.0 / 25.4, 60.0 / 25.4), facecolor="white")
    ax_a = fig.add_subplot(121)
    ax_b = fig.add_subplot(122)
    for ax in (ax_a, ax_b):
        apply_paleo_style(ax)

    # ---- (a) Theorem 2: expected F vs forcing loading ---------------------
    betas = np.linspace(0.05, 0.5, 200)
    ax_a.plot(betas, e_f(N_CANON, betas), "-", color=CAT1, lw=1.2)
    ax_a.axhline(1.0, color=SECONDARY_INK, lw=0.8, ls=(0, (3, 3)))
    ax_a.text(0.05, 1.16, "E[F] = 1", fontsize=9, color=INK)
    ax_a.plot([0.35], [e_f(N_CANON, 0.35)], "o", ms=6.0, mfc="none",
              mec=INK, mew=1.4)
    ax_a.annotate(r"$\beta$ = 0.35 $\rightarrow$ E[F] = 3.4",
                  xy=(0.35, e_f(N_CANON, 0.35)),
                  xytext=(0.055, 7.6), fontsize=9, color=INK)
    ax_a.set_xlabel(r"forcing loading $\beta$ ($\beta_X = \beta_Y$)",
                    fontsize=10)
    ax_a.set_ylabel("expected F under H0", fontsize=10)
    ax_a.set_title("(a) Theorem 2 — F inflation (n = 100)", fontsize=10.5)
    ax_a.set_ylim(0.8, 9.5)

    # ---- (b) Theorem 3: FPR lower bound vs n -------------------------------
    ns = np.linspace(50, 2000, 200)
    ax_b.plot(ns, fpr_bound(ns, 0.35), "-", color=CAT1, lw=1.2,
              label="Theorem 3 lower bound")
    ax_b.axhline(0.05, color=THR_95, lw=1.2, ls=(0, (4, 2)),
                 label="nominal 5%")
    ax_b.plot([N_CANON], [bound], "o", ms=5.5, mfc="none", mec=INK, mew=1.2)
    ax_b.annotate("34% at n = 100",
                  xy=(N_CANON, bound), xytext=(N_CANON + 60, bound + 0.10),
                  fontsize=9, color=INK)
    ax_b.set_xlabel("record length n")
    ax_b.set_ylabel("false-positive rate")
    ax_b.set_ylim(-0.02, 1.05)
    ax_b.set_title("(b) Theorem 3 — FPR never returns to 5%", fontsize=10.5)
    ax_b.legend(frameon=False, fontsize=9, loc="upper left")

    fig.tight_layout(pad=0.6)
    save(fig, os.path.join(OUT_DIR, "theorem_fpr"))


if __name__ == "__main__":
    main()
