#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
42_theory_fpr_table.py
----------------------
Theoretical false-positive-rate (FPR) profile as a function of the test
level alpha (paper Table S7, Appendix S6.2), plus the alpha x tau binomial
gate matrix quoted in S6.2.

Chain of estimators (complete-logic eq (5), q = 1, lambda_eff = 0.9724):
    FPR ~= 1 - Phi((chi2_{q,1-alpha} - q - lambda_eff) / sqrt(2(q + 2 lambda_eff)))

For each alpha in {0.30 ... 0.05} (confidence 70%-95%) the script reports:
  1. the normal approximation (eq (5)),
  2. the exact noncentral chi-squared tail  P(chi2_1(lambda_eff) > c),
  3. the exact noncentral F tail           P(F_{1,97}(lambda_eff) > c_F),
  4. the Monte Carlo FPR (20,000 reps, seed 42) under the DGP shared by
     44_theorem2_montecarlo.py and 45_fig2a_fpr_sweep.py:

       Z_t = phiZ Z_{t-1} + eta_t,  eta ~ N(0, 1 - phiZ^2), Var(Z) = 1;
       X_t = phiX X_{t-1} + b Z_t + e_t,  Y_t likewise
       (phiX = phiY = 0.55, b = 0.35, Var(e) = 0.5, n = 100,
        200 burn-in points);
       F = [(RSS_r - RSS_u)/1] / [RSS_u/97]  (restricted Y ~ Y_{t-1},
       unrestricted Y ~ Y_{t-1} + X_{t-1}).
     One simulation serves all six critical values (the alpha = 0.05 row
     is the 15.93% anchor reproduced by 44/45).

The binomial gate matrix at the end tabulates, for a pure-noise edge under
independent perturbations (the S6.2 firewall argument),
    P(Bin(200, FPR(alpha)) >= 200*tau)
with FPR(alpha) taken from the exact noncentral-F tail.  Note (as in S6.2):
the independent-binomial model is an idealized upper bound -- real draws
are strongly correlated through shared data (see the stacking-saturation
example).

CSV outputs (OUTPUT_DIR):
    tableS7_fpr_profile.csv    -- the six alpha rows, all four estimators
    tableS7_binomial_gate.csv  -- the alpha x tau gate matrix
"""
import os, sys, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from scipy import stats
from src.config import OUTPUT_DIR

LAM = 0.9724          # eq (4) numerical example (Section 2.2)
Q, N = 1, 100
DF2 = N - 3           # denominator df of the F statistic: n - p - q - 1 = 97
ALPHAS = [0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
TAUS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------------- three analytic estimators ----------------
    print('lambda_eff = %.4f, q = 1, df2 = %d' % (LAM, DF2))
    print('=' * 92)
    print('%-10s %-8s %-10s %-10s %-12s %-12s %-12s' % (
        'confid.', 'alpha', 'chi2 crit', 'F crit',
        'normal appr.', 'chi2 tail', 'F tail'))
    print('-' * 92)
    theo = {}
    for a in ALPHAS:
        c_chi = stats.chi2.ppf(1 - a, Q)
        c_f = stats.f.ppf(1 - a, 1, DF2)
        f_approx = 1 - stats.norm.cdf((c_chi - Q - LAM) / np.sqrt(2 * (Q + 2 * LAM)))
        f_chi = stats.ncx2.sf(c_chi, Q, LAM)
        f_F = stats.ncf.sf(c_f, 1, DF2, LAM)
        theo[a] = (c_chi, c_f, f_approx, f_chi, f_F)
        print('%-10s %-8s %-10.4f %-10.4f %-11.1f%% %-11.1f%% %-11.1f%%' % (
            '%d%%' % round((1 - a) * 100), '%.2f' % a, c_chi, c_f,
            f_approx * 100, f_chi * 100, f_F * 100))

    # ---------------- MC (same DGP/stream as 44/45) ----------------
    rng = np.random.default_rng(42)
    npts, T = N, N + 200
    phiX, phiY, phiZ = 0.55, 0.55, 0.6
    bX = bY = 0.35
    varZ, varE = 1.0, 0.5
    reps = 20000

    Fvals = np.empty(reps)
    for r in range(reps):
        eta = rng.normal(0, np.sqrt(1 - phiZ ** 2), T)
        Z = np.empty(T); Z[0] = eta[0]
        for t in range(1, T):
            Z[t] = phiZ * Z[t - 1] + eta[t]
        eX = rng.normal(0, np.sqrt(varE), T)
        eY = rng.normal(0, np.sqrt(varE), T)
        X = np.empty(T); Y = np.empty(T)
        X[0] = eX[0] / np.sqrt(1 - phiX ** 2)     # stationary start
        Y[0] = eY[0] / np.sqrt(1 - phiY ** 2)
        for t in range(1, T):
            X[t] = phiX * X[t - 1] + bX * Z[t] + eX[t]
            Y[t] = phiY * Y[t - 1] + bY * Z[t] + eY[t]
        x, y = X[-npts:], Y[-npts:]
        y0, y1, x0 = y[:-1], y[1:], x[:-1]
        Xr = np.column_stack([np.ones(npts - 1), y0])
        br = np.linalg.lstsq(Xr, y1, rcond=None)[0]
        rss_r = ((y1 - Xr @ br) ** 2).sum()
        Xu = np.column_stack([np.ones(npts - 1), y0, x0])
        bu = np.linalg.lstsq(Xu, y1, rcond=None)[0]
        rss_u = ((y1 - Xu @ bu) ** 2).sum()
        Fvals[r] = ((rss_r - rss_u) / 1) / (rss_u / (npts - 3))

    print('-' * 92)
    print('MC (20,000 reps, seed 42; one simulation, six critical values):')
    mc_fpr = {}
    for a in ALPHAS:
        crit = stats.f.ppf(1 - a, 1, DF2)
        mc = (Fvals > crit).mean() * 100
        mc_fpr[a] = mc
        print('  %d%% confidence (alpha=%.2f): FPR = %.2f%%' % (
            round((1 - a) * 100), a, mc))
    print('  anchor (alpha=0.05) equals the 15.93% reproduced by 44/45.')

    # ---------------- save Table S7 profile ----------------
    with open(os.path.join(OUTPUT_DIR, 'tableS7_fpr_profile.csv'),
              'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['confidence', 'alpha', 'chi2_crit', 'f_crit',
                    'approx_fpr', 'chi2_tail_fpr', 'f_tail_fpr', 'mc_fpr'])
        for a in ALPHAS:
            c_chi, c_f, f_approx, f_chi, f_F = theo[a]
            w.writerow([round((1 - a) * 100), a, c_chi, c_f,
                        f_approx * 100, f_chi * 100, f_F * 100, mc_fpr[a]])

    # ---------------- alpha x tau binomial gate matrix ----------------
    # For each draw the rejection rate is p = per-draw FPR(alpha) (exact
    # noncentral-F tail); under independent Bernoulli draws across the 200
    # perturbations, P(Bin(200, p) >= 200*tau) is the probability that a
    # pure-noise edge passes the robustness threshold tau.
    print()
    print('=' * 92)
    print('alpha x tau gate: P(Bin(200, FPR(alpha)) >= 200*tau) for a pure-noise edge')
    print('-' * 92)
    header = '%-10s' % 'alpha \\ tau'
    for t in TAUS:
        header += ' %-14s' % ('tau=%d%%' % round(t * 100))
    print(header)
    gate = {}
    for a in ALPHAS:
        p = stats.ncf.sf(stats.f.ppf(1 - a, 1, DF2), 1, DF2, LAM)
        row = 'alpha=%-7.2f ' % a
        gate[a] = {}
        for t in TAUS:
            k = int(np.ceil(200 * t))
            prob = stats.binom.sf(k - 1, 200, p)
            gate[a][t] = prob
            row += ' %-14.2e' % prob
        print(row)
    print()
    print('Note: the independent-binomial model is an idealized upper bound '
          '(real draws are strongly correlated through shared data, see the '
          'stacking-saturation example).  Even at a per-draw FPR of 55% '
          '(alpha=0.30), reaching 140 of 200 draws (tau=70%) has probability '
          '~1e-5; reaching 190 draws (tau=95%) is astronomically small at '
          'any alpha.')

    with open(os.path.join(OUTPUT_DIR, 'tableS7_binomial_gate.csv'),
              'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['alpha'] + ['tau_%d' % round(t * 100) for t in TAUS])
        for a in ALPHAS:
            w.writerow([a] + [gate[a][t] for t in TAUS])

    print('\nSaved: tableS7_fpr_profile.csv, tableS7_binomial_gate.csv '
          '(in %s)' % OUTPUT_DIR)


if __name__ == '__main__':
    main()
