#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
43_pairwise_beta_inversion.py
-----------------------------
Pairwise inversion of the deconfounding strength beta from the six real
proxy records (Appendix S4.1; the beta = 0.25 anchor of Section 4.1 and
of S6.3 (i)(ii)).

Method (as stated in the appendix):
  1. Interpolate each record onto the 10-60 ka grid (dt = 0.05 ka) and
     standardize it over its own finite coverage;
  2. For every record pair, take the maximum absolute cross-correlation
     |rho| over lags +-10 samples (+-0.5 ka);
  3. Invert |rho| through the closed-form model correlation

        rho_beta(b; fx, fy, sx, sy) = b^2 * C / sqrt(Vx * Vy)
        C  = (1 + fy fz/(1 - fx fz) + fx fz/(1 - fy fz)) / (1 - fx fy)
        Vx = (b^2 (1 + fx fz)/(1 - fx fz) + sx^2) / (1 - fx^2)
        Vy = likewise,

     with the per-record Table S1 AR(1) coefficients (phi_gen, sigma_eps,
     as printed in Table S1 and stored in src/paleo_generator.py) and the
     forcing autocorrelation fz = 0.6;
  4. Median over the 12 independent pairs (the 15 pairs minus the 3
     same-source EDC pairs EDC_dD-CO2, EDC_dD-CH4, CO2-CH4) -- the paper
     reports median 0.254 ~ 0.255 (the beta = 0.25 anchor).

The model ceiling rho_max(fx, fy) = rho_beta(b -> inf) is reported per
pair.  Pairs whose |rho| exceeds rho_max (e.g. EDC_dD-CO2, |rho| = 0.944
vs rho_max = 0.909) invert to the search-grid cap b* = 2.5 and are
flagged; the Cheng pairs are anti-correlated over 10-60 ka, so their
inversion runs on |rho| as well.

CSV output (OUTPUT_DIR): beta_inversion_pairs.csv
"""
import os, sys, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pandas as pd
from src.config import DATA_DIR, OUTPUT_DIR

FILES = {
    'NGRIP':   'Erhardt_NGRIP_aerosol_d18O_step1.csv',
    'EDC_dD':  '32_Jouzel-2007-dD.csv',
    'CO2':     '32_Bereiter et al.-2015-GRL-revised CO2 .xlsx',
    'CH4':     '32_CH4 over past 800kyr.csv',
    'Cariaco': '16_MD2622ODP1002-Mo.csv',
    'Cheng':   '14_cheng2016composite-noaa.xlsx',
}
NAMES = list(FILES)
FZ = 0.6                     # forcing AR(1) coefficient (Table S1 / S6.3)
MAX_LAG = 10                 # +-10 samples = +-0.5 ka at dt = 0.05 ka
B_GRID = np.linspace(0, 2.5, 5001)
SAME_SOURCE = {('EDC_dD', 'CO2'), ('EDC_dD', 'CH4'), ('CO2', 'CH4')}


def load_record(fn):
    p = os.path.join(DATA_DIR, fn)
    df = pd.read_excel(p) if fn.endswith('.xlsx') else pd.read_csv(p)
    sub = df.iloc[:, :4].apply(pd.to_numeric, errors='coerce')
    age = sub.iloc[:, 0].values
    val = sub.iloc[:, 1].values
    m = ~(np.isnan(age) | np.isnan(val))
    return age[m], val[m]


def rho_beta(b, fx, fy, sx, sy):
    C = (1 + fy * FZ / (1 - fx * FZ) + fx * FZ / (1 - fy * FZ)) / (1 - fx * fy)
    Vx = (b * b * (1 + fx * FZ) / (1 - fx * FZ) + sx * sx) / (1 - fx * fx)
    Vy = (b * b * (1 + fy * FZ) / (1 - fy * FZ) + sy * sy) / (1 - fy * fy)
    return b * b * C / np.sqrt(Vx * Vy)


def rho_max(fx, fy):
    C = (1 + fy * FZ / (1 - fx * FZ) + fx * FZ / (1 - fy * FZ)) / (1 - fx * fy)
    vx = (1 + fx * FZ) / (1 - fx * FZ) / (1 - fx * fx)
    vy = (1 + fy * FZ) / (1 - fy * FZ) / (1 - fy * fy)
    return C / np.sqrt(vx * vy)


def main():
    # Per-record AR(1) coefficients as PRINTED in Table S1 of the paper
    # (half-up-rounded copies of the higher-precision fits stored in
    # src/paleo_generator.py, e.g. 0.8515 -> 0.852, 0.9355 -> 0.936).
    PHI = {'NGRIP': 0.639, 'EDC_dD': 0.852, 'CO2': 0.936,
           'CH4': 0.910, 'Cariaco': 0.835, 'Cheng': 0.841}
    SE = {'NGRIP': 0.702, 'EDC_dD': 0.299, 'CO2': 0.071,
          'CH4': 0.111, 'Cariaco': 0.377, 'Cheng': 0.358}

    recs = {}
    for n, f in FILES.items():
        a, v = load_record(f)
        recs[n] = (a, v)
        print('%-9s n=%5d age %7.1f-%7.1f ka  |  file: %s' % (
            n, len(a), a.min(), a.max(), f))

    grid = np.arange(10, 60 + 0.024, 0.05)
    series = {}
    for n, (a, v) in recs.items():
        order = np.argsort(a)
        a, v = a[order], v[order]
        u, idx = np.unique(a, return_index=True)
        series[n] = np.interp(grid, u, v[idx], left=np.nan, right=np.nan)

    print('\n10-60 ka coverage:')
    for n, s in series.items():
        print('  %-9s %5.1f%%' % (n, np.isfinite(s).mean() * 100))

    # standardize each record over its own finite coverage, keep grid positions
    A = {}
    for n in NAMES:
        s = series[n]
        std = s[np.isfinite(s)]
        std = (std - std.mean()) / std.std()
        arr = np.full(len(grid), np.nan)
        arr[np.isfinite(s)] = std
        A[n] = arr

    print('\n%-22s %-5s %-8s %-8s %-8s %-8s %s' % (
        'pair', 'lag', 'rho', '|rho|', 'rho_max', 'b*', 'flag'))
    print('-' * 68)
    rows = []
    betas = []
    for i in range(len(NAMES)):
        for j in range(i + 1, len(NAMES)):
            n1, n2 = NAMES[i], NAMES[j]
            a1, a2 = A[n1], A[n2]
            best = None
            for k in range(-MAX_LAG, MAX_LAG + 1):
                i1 = slice(max(0, k), len(grid) + min(0, k))
                i2 = slice(max(0, -k), len(grid) - max(0, k))
                m = np.isfinite(a1[i1]) & np.isfinite(a2[i2])
                if m.sum() < 20:
                    continue
                r = np.corrcoef(a1[i1][m], a2[i2][m])[0, 1]
                if best is None or abs(r) > abs(best[1]):
                    best = (k, r)
            k_hat, rho_hat = best
            rm = rho_max(PHI[n1], PHI[n2])
            bstar = B_GRID[np.argmin(
                np.abs(rho_beta(B_GRID, PHI[n1], PHI[n2],
                                SE[n1], SE[n2]) - abs(rho_hat)))]
            same = (n1, n2) in SAME_SOURCE
            ceil = abs(rho_hat) > rm
            flag = ('same-source' if same else '') \
                + ('+' if same and ceil else '') \
                + ('exceeds rho_max' if ceil else '')
            if not same:
                betas.append(bstar)
            rows.append((n1, n2, k_hat, rho_hat, abs(rho_hat), rm,
                         bstar, same, ceil))
            print('%-22s %-5d %-+8.3f %-8.3f %-8.3f %-8.2f %s' % (
                '%s-%s' % (n1, n2), k_hat, rho_hat, abs(rho_hat), rm,
                bstar, flag))

    med = np.median(betas)
    print('-' * 68)
    print('independent-pair b* median = %.3f (n=%d)   -> paper anchor 0.254 ~ 0.255'
          % (med, len(betas)))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'beta_inversion_pairs.csv'),
              'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['name1', 'name2', 'lag', 'rho', 'abs_rho', 'rho_max',
                    'inverted_beta', 'same_source', 'exceeds_rho_max'])
        w.writerows(rows)
    print('Saved: beta_inversion_pairs.csv (in %s)' % OUTPUT_DIR)


if __name__ == '__main__':
    main()
