#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
45_fig2a_fpr_sweep.py
---------------------
Vectorized Monte Carlo FPR curve of Fig. 2(a) (main text): FPR of the
lag-1 Granger test against the eq (4) strength parameter lambda_eff,
swept through phi_Z at fixed (n = 100, phi_X = phi_Y = 0.55, b = 0.35,
Var(Z) = 1, Var(eps) = 0.5).

The vectorized estimator solves the restricted/unrestricted lag-1
regressions by 2x2 / 3x3 Cramer-rule normal equations over all reps at
once; it is checked against the loop implementation (identical per-rep
draw order: eta(T), eX(T), eY(T) drawn sequentially per rep -- the
ziggurat stream depends on the call shape, so the calls must not be
merged).

Sweep design (reproduces the archived figure data bit-exactly,
_fig2_mc_sweep_20261019c.npz, 2026-10-19):
    phi_Z in {0.15, 0.25, 0.35, 0.45, 0.55, 0.60, 0.70, 0.80, 0.90},
    seed = grid index per point, EXCEPT the anchor phi_Z = 0.60, which
    uses the canonical seed-42 check (15.93%); the anchor is asserted
    against the archived value.

Critical value: F_{1,n-2,0.95} as in the archived figure (F_{1,n-3,0.95}
differs by < 0.002 and shifts the FPR by < 0.05 pp -- see
44_theorem2_montecarlo.py).

CSV output (OUTPUT_DIR): fig2a_mc_sweep.csv (phi_z, lambda_eff, mc_fpr)
"""
import os, sys, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from scipy import stats
from src.config import OUTPUT_DIR

N = 100
PHI_X = 0.55
B = 0.35
VAR_Z, VAR_E = 1.0, 0.5
CRIT = stats.f.ppf(0.95, 1, N - 2)      # F_{1,98,0.95} (archived figure)

PHI_Z_GRID = [0.15, 0.25, 0.35, 0.45, 0.55, 0.60, 0.70, 0.80, 0.90]
ANCHOR_IDX = 5                            # phi_Z = 0.60
# archived figure data (_fig2_mc_sweep_20261019c.npz, 2026-10-19)
ARCHIVED_LAM = [0.06077433, 0.16881759, 0.33088248, 0.54696899, 0.81707713,
                0.97238932, 1.3235299, 1.72869212, 2.18787596]
ARCHIVED_FPR = [0.05850, 0.07370, 0.08765, 0.11230, 0.14515,
                0.15935, 0.21410, 0.26295, 0.31200]


def lam_eff(phiZ):
    """eq (4): n (b^2 b^2 (phiZ VarZ)^2 (1 - phiX^2)) / ((b^2 VarZ + VarE)^2)"""
    return N * (B**4 * (phiZ * VAR_Z)**2 * (1 - PHI_X**2)) \
           / ((B**2 * VAR_Z + VAR_E)**2)


def mc_fpr_vec(phiZ, reps=20000, seed=0):
    T = N + 200
    rng = np.random.default_rng(seed)
    # identical draw order to the loop version: per rep, eta(T), eX(T), eY(T).
    sZ = np.sqrt(1 - phiZ**2)
    sE = np.sqrt(VAR_E)
    eta = np.empty((reps, T)); eX = np.empty((reps, T)); eY = np.empty((reps, T))
    for r in range(reps):
        eta[r] = rng.normal(0, sZ, T)
        eX[r] = rng.normal(0, sE, T)
        eY[r] = rng.normal(0, sE, T)
    Z = np.empty((reps, T)); Z[:, 0] = eta[:, 0]
    for t in range(1, T):
        Z[:, t] = phiZ * Z[:, t - 1] + eta[:, t]
    X = np.empty((reps, T)); Y = np.empty((reps, T))
    X[:, 0] = eX[:, 0] / np.sqrt(1 - PHI_X**2)
    Y[:, 0] = eY[:, 0] / np.sqrt(1 - PHI_X**2)
    for t in range(1, T):
        X[:, t] = PHI_X * X[:, t - 1] + B * Z[:, t] + eX[:, t]
        Y[:, t] = PHI_X * Y[:, t - 1] + B * Z[:, t] + eY[:, t]
    x, y = X[:, -N:], Y[:, -N:]
    y0, y1, x0 = y[:, :-1], y[:, 1:], x[:, :-1]
    m = N - 1
    S2 = (y1**2).sum(1)
    # restricted regression y1 ~ [1, y0]: 2x2 normal-equation Cramer
    A11 = m
    A12 = y0.sum(1)
    A22 = (y0**2).sum(1)
    c1 = y1.sum(1)
    c2 = (y0 * y1).sum(1)
    detA = A11 * A22 - A12**2
    b1r = (A22 * c1 - A12 * c2) / detA
    b2r = (A11 * c2 - A12 * c1) / detA
    rss_r = S2 - (b1r * c1 + b2r * c2)
    # unrestricted y1 ~ [1, y0, x0]: 3x3 normal-equation Cramer
    B11 = m
    B12 = A12
    B13 = x0.sum(1)
    B22 = A22
    B23 = (y0 * x0).sum(1)
    B33 = (x0**2).sum(1)
    d1 = c1
    d2 = c2
    d3 = (x0 * y1).sum(1)
    det = B11 * (B22 * B33 - B23**2) - B12 * (B12 * B33 - B23 * B13) \
        + B13 * (B12 * B23 - B22 * B13)
    b1u = (d1 * (B22 * B33 - B23**2) - B12 * (d2 * B33 - B23 * d3)
           + B13 * (d2 * B23 - B22 * d3)) / det
    b2u = (B11 * (d2 * B33 - d3 * B23) - d1 * (B12 * B33 - B13 * B23)
           + B13 * (B12 * d3 - d2 * B13)) / det
    b3u = (B11 * (B22 * d3 - B23 * d2) - B12 * (B12 * d3 - B13 * d2)
           + d1 * (B12 * B23 - B22 * B13)) / det
    rss_u = S2 - (b1u * d1 + b2u * d2 + b3u * d3)
    Fv = ((rss_r - rss_u) / 1) / (rss_u / (N - 3))
    return (Fv > CRIT).mean()


def mc_fpr_loop(phiZ, reps=2000, seed=0):
    T = N + 200
    rng = np.random.default_rng(seed)
    Fv = np.empty(reps)
    for r in range(reps):
        eta = rng.normal(0, np.sqrt(1 - phiZ**2), T)
        Z = np.empty(T); Z[0] = eta[0]
        for t in range(1, T):
            Z[t] = phiZ * Z[t - 1] + eta[t]
        eX = rng.normal(0, np.sqrt(VAR_E), T)
        eY = rng.normal(0, np.sqrt(VAR_E), T)
        X = np.empty(T); Y = np.empty(T)
        X[0] = eX[0] / np.sqrt(1 - PHI_X**2)
        Y[0] = eY[0] / np.sqrt(1 - PHI_X**2)
        for t in range(1, T):
            X[t] = PHI_X * X[t - 1] + B * Z[t] + eX[t]
            Y[t] = PHI_X * Y[t - 1] + B * Z[t] + eY[t]
        x, y = X[-N:], Y[-N:]
        y0, y1, x0 = y[:-1], y[1:], x[:-1]
        Xr = np.column_stack([np.ones(N - 1), y0])
        br = np.linalg.lstsq(Xr, y1, rcond=None)[0]
        rss_r = ((y1 - Xr @ br)**2).sum()
        Xu = np.column_stack([np.ones(N - 1), y0, x0])
        bu = np.linalg.lstsq(Xu, y1, rcond=None)[0]
        rss_u = ((y1 - Xu @ bu)**2).sum()
        Fv[r] = ((rss_r - rss_u) / 1) / (rss_u / (N - 3))
    return (Fv > CRIT).mean()


def main():
    # vectorized vs loop cross-check (same seeds, same draw order)
    for seed in (0, 1):
        a = mc_fpr_loop(0.6, reps=2000, seed=seed)
        b = mc_fpr_vec(0.6, reps=2000, seed=seed)
        print('seed %d: loop=%.3f%%  vec=%.3f%%  diff=%.3fpp' % (
            seed, a * 100, b * 100, abs(a - b) * 100))

    # canonical anchor check (20k, seed 42)
    anchor = mc_fpr_vec(0.6, reps=20000, seed=42)
    print('canonical check (20k, seed 42): %.2f%%  (expect 15.93%%)' % (anchor * 100))

    # sweep: seed = grid index, except the anchor point (canonical seed 42)
    print('\n%-8s %-12s %-12s %-10s %-10s' % (
        'phi_Z', 'lambda_eff', 'archived lam', 'mc FPR', 'archived'))
    print('-' * 56)
    rows = []
    for i, phz in enumerate(PHI_Z_GRID):
        seed = 42 if i == ANCHOR_IDX else i
        v = mc_fpr_vec(phz, reps=20000, seed=seed)
        lam = lam_eff(phz)
        rows.append((phz, lam, v))
        # archived lam is stored at 8 dp, archived fpr at 5 dp -> compare
        # at the stored precision (5e-6 = half the last fpr digit).
        ok_lam = abs(lam - ARCHIVED_LAM[i]) < 1e-8
        ok_fpr = abs(v - ARCHIVED_FPR[i]) < 5e-6
        print('%-8.2f %-12.8f %-12.8f %-10.5f %-10.5f  lam:%s fpr:%s' % (
            phz, lam, ARCHIVED_LAM[i], v, ARCHIVED_FPR[i],
            'OK' if ok_lam else 'DIFF', 'OK' if ok_fpr else 'DIFF'))

    assert abs(anchor - ARCHIVED_FPR[ANCHOR_IDX]) < 5e-6, \
        'anchor point drifted from the archived figure value'

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'fig2a_mc_sweep.csv'),
              'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['phi_z', 'lambda_eff', 'mc_fpr'])
        w.writerows(rows)
    print('\nSaved: fig2a_mc_sweep.csv (in %s)' % OUTPUT_DIR)


if __name__ == '__main__':
    main()
