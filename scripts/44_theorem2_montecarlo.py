#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
44_theorem2_montecarlo.py
-------------------------
Independent Monte Carlo verification of Theorem 2 (main text Section 2.2,
eq (4)-(5)) for the numerical example (n = 100, phi_X = phi_Y = 0.55,
phi_Z = 0.6, b = 0.35, Var(Z) = 1, Var(eps) = 0.5):

    X_t = phiX X_{t-1} + bX Z_t + eps_X,t ;  Y_t likewise
    Z_t = phiZ Z_{t-1} + eta_t,  eta ~ N(0, 1 - phiZ^2)   (Var(Z) = 1)

Restricted:   Y_t ~ Y_{t-1}
Unrestricted: Y_t ~ Y_{t-1} + X_{t-1}
F = [(RSS_r - RSS_u)/1] / [RSS_u/97]

Checks printed:
  - lambda_eff from eq (4);
  - E[F] ~= 1 + rho^2 (n - p - q - 1) / q, with rho^2 = lambda_eff / n,
    against the MC mean of F;
  - MC FPR against the eq (5) normal approximation and the exact
    noncentral chi-squared tail P(chi2_1(lambda_eff) > 3.8415).

This DGP/seed also produces the 15.93% anchor of Table S7 (see
42_theory_fpr_table.py) and of the Fig. 2(a) curve (45_fig2a_fpr_sweep.py).
Critical value used here: F_{1,n-2,0.95} (as in the archived computation;
F_{1,n-3,0.95} differs by < 0.002 and moves the MC FPR by < 0.05 pp).

CSV output (OUTPUT_DIR): theorem2_mc_verification.csv
"""
import os, sys, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from scipy import stats
from src.config import OUTPUT_DIR


def main():
    rng = np.random.default_rng(42)
    n = 100
    T = n + 200                      # 200 burn-in points
    phiX, phiY, phiZ = 0.55, 0.55, 0.6
    bX = bY = 0.35
    varZ, varE = 1.0, 0.5
    reps = 20000

    crit = stats.f.ppf(0.95, 1, n - 2)          # F_{1,98,0.95}
    Fvals = np.empty(reps)

    for r in range(reps):
        eta = rng.normal(0, np.sqrt(1 - phiZ**2), T)
        Z = np.empty(T); Z[0] = eta[0]
        for t in range(1, T):
            Z[t] = phiZ * Z[t-1] + eta[t]
        eX = rng.normal(0, np.sqrt(varE), T)
        eY = rng.normal(0, np.sqrt(varE), T)
        X = np.empty(T); Y = np.empty(T)
        X[0] = eX[0] / np.sqrt(1 - phiX**2)     # stationary start
        Y[0] = eY[0] / np.sqrt(1 - phiY**2)
        for t in range(1, T):
            X[t] = phiX * X[t-1] + bX * Z[t] + eX[t]
            Y[t] = phiY * Y[t-1] + bY * Z[t] + eY[t]
        x, y = X[-n:], Y[-n:]
        y0, y1, x0 = y[:-1], y[1:], x[:-1]
        Xr = np.column_stack([np.ones(n-1), y0])
        br = np.linalg.lstsq(Xr, y1, rcond=None)[0]
        rss_r = ((y1 - Xr @ br)**2).sum()
        Xu = np.column_stack([np.ones(n-1), y0, x0])
        bu = np.linalg.lstsq(Xu, y1, rcond=None)[0]
        rss_u = ((y1 - Xu @ bu)**2).sum()
        Fvals[r] = ((rss_r - rss_u) / 1) / (rss_u / (n - 3))

    lam_eff = n * (bX**2 * bY**2 * (phiZ * varZ)**2 * (1 - phiX**2)) \
              / ((bY**2 * varZ + varE) * (bX**2 * varZ + varE))
    mean_F_theory = 1 + lam_eff * (n - 3) / n      # 1 + rho^2 (n-p-q-1)/q
    mc_mean_F = Fvals.mean()
    mc_fpr = (Fvals > crit).mean() * 100
    approx_fpr = (1 - stats.norm.cdf(
        (stats.chi2.ppf(0.95, 1) - 1 - lam_eff) / np.sqrt(2 * (1 + 2 * lam_eff)))) * 100
    chi2_tail = stats.ncx2.sf(stats.chi2.ppf(0.95, 1), 1, lam_eff) * 100

    print('lambda_eff (eq.4)              =', round(lam_eff, 4))
    print('E[F] ~= 1 + rho^2 (n-p-q-1)/q  =', round(mean_F_theory, 4))
    print('MC mean F                     =', round(mc_mean_F, 4))
    print('F_{1,98,0.95} critical value   =', round(crit, 4))
    print('MC FPR                        =', round(mc_fpr, 2), '%')
    print('normal approximation FPR      =', round(approx_fpr, 2), '%')
    print('exact P(chi2_1(lambda_eff) > 3.8415) =', round(chi2_tail, 2), '%')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'theorem2_mc_verification.csv'),
              'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerows([
            ['lambda_eff_eq4', lam_eff],
            ['mean_F_theory', mean_F_theory],
            ['mc_mean_F', mc_mean_F],
            ['crit_F_1_98_0.95', crit],
            ['mc_fpr_percent', mc_fpr],
            ['approx_fpr_percent', approx_fpr],
            ['chi2_tail_fpr_percent', chi2_tail],
        ])
    print('\nSaved: theorem2_mc_verification.csv (in %s)' % OUTPUT_DIR)


if __name__ == '__main__':
    main()
