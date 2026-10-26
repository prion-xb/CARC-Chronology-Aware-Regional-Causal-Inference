#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
34_scenarioA_submitted.py
-------------------
Scenario A rebuild per the final submitted version of the paper.

Differences from scripts/21_scenarioA_table1.py (old version):
  * span 10-58 ka (submitted supplementary Table S1) instead of 10-60;
  * Z = stationary Gaussian AR(1) (rho_F=0.6) + 25 D-O Gaussian pulses
    (width 0.4 ka, amplitude decay 1.2 ka); the 40-ka orbital sinusoid is
    NOT part of the submitted generator spec (--orbital re-adds it for probes);
  * Cariaco age error = 0.23 ka (measured median of the real record's
    per-point errors; the paper's Table S1 value 0.50 is a typo -- the
    paper's own lambda = lag/error column shows 2.3/0.23 = 10.0);
  * per-point age errors sampled from the real records' empirical error
    distributions (GICC05/AICC2012/SISALv2 files in proxy_errors/);
    for the six Scenario A records these distributions are degenerate
    (constant per point): 0.65 / 0.83 / 1.65 / 1.66 / 0.23 / 0.14 ka.

Pipeline (paper Section 3):
  Layer 2 : ARC deconfounding, alpha = 0.5, proxy = 1-ka Gaussian-smoothed
            NGRIP trend; NGRIP itself kept unchanged (keep mode). The paper
            states that NGRIP-involving pairs use the orbital-scale (>5 ka)
            component of NGRIP as the alternative Z proxy (--deconfound
            altproxy implements that reading for probes).
  Layer 3 : ADF unit-root gate (0.05, differencing if not rejected);
            AIC selects VAR lag from 1 to 5; F-test at 0.05
            (--test minp = min-p across lags 1..L, the statsmodels
            decision-equivalent; --test lagonly = F at lag L only).
  Layer 4 : M = 200 chronology-perturbation realizations; robustness =
            fraction significant; honesty filter at 95%.

Targets (submitted Table 1):
  no-error   100% / 100% / 100%  +  controls 0%
  with-error ~70% / ~60% /  96%  +  controls 0%
  (edges NGRIP->EDC_dD, CO2->CH4, Cariaco->Cheng; lambda ~1.2 / ~0.9 / ~10.0)

Outputs:
  outputs/scenarioA_submitted_table1.csv      per-edge Table 1 rows
  outputs/scenarioA_submitted_table1_full.csv all 30 directed cells

Usage:
  python scripts/34_scenarioA_submitted.py --probe            # 1 realization, clean
                                                        # world, per-pair F/p
                                                        # diagnostics
  python scripts/34_scenarioA_submitted.py --quick            # smoke test (3 x 40)
  python scripts/34_scenarioA_submitted.py                    # full run (20 x 200)
"""

import os, sys, argparse, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR

from src.config import OUTPUT_DIR
from src.arc import arc_deconfound_batch
from src.granger_fast import (_granger_stats, _f_crit, _f_pvalue,
                               granger_decision_block, granger_decision_fast)

DT = 0.05
MAXLAG = 5
ARC_ALPHA = 0.5
CAUSAL_STRENGTH = 0.4
TRUE_EDGES = [('NGRIP', 'EDC_dD', 0.8), ('CO2', 'CH4', 1.5),
              ('Cariaco', 'Cheng', 2.3)]

# Common-forcing Z generator parameters (module constants so Scenario B,
# scripts/38, can override them before calling generate_submitted).
RHO_F = 0.6      # AR(1) persistence of Z (Table S1/S5)
Z_SIGMA = 0.8    # AR(1) innovation sigma of Z (fixed across scenarios)

# Submitted supplementary Table S1 generator parameters.
# Cariaco age error: 0.23 ka measured median (paper's 0.50 is a typo; the
# paper's own lambda = |L|/sigma_driver column gives 2.3/0.23 = 10.0).
S1_PARAMS = {
    'phi':   {'NGRIP': 0.82, 'EDC_dD': 0.90, 'CO2': 0.72, 'CH4': 0.78,
              'Cariaco': 0.85, 'Cheng': 0.88},
    'beta':  {'NGRIP': 0.35, 'EDC_dD': 0.30, 'CO2': 0.25, 'CH4': 0.25,
              'Cariaco': 0.35, 'Cheng': 0.35},
    'sigma': {'NGRIP': 0.50, 'EDC_dD': 0.45, 'CO2': 0.55, 'CH4': 0.55,
              'Cariaco': 0.50, 'Cheng': 0.50},
    'age':   {'NGRIP': 0.65, 'EDC_dD': 0.83, 'CO2': 1.65, 'CH4': 1.66,
              'Cariaco': 0.23, 'Cheng': 0.14},
}
VAR_ORDER = ['NGRIP', 'EDC_dD', 'CO2', 'CH4', 'Cariaco', 'Cheng']

# Real-record per-point age-error files (10-58 ka subset used for the
# empirical error distributions). All six are constant per point, so the
# empirical distribution is degenerate at the value below.
ERR_FILES = {
    'NGRIP':   'proxy_errors/Erhardt_NGRIP_aerosol_d18O_step1_err.csv',
    'EDC_dD':  'proxy_errors/32_Jouzel-2007-dD_err.csv',
    'CO2':     'proxy_errors/32_Bereiter et al.-2015-GRL-revised CO2 _err.csv',
    'CH4':     'proxy_errors/32_CH4 over past 800kyr_err.csv',
    'Cariaco': 'proxy_errors/16_MD2622ODP1002-Mo_err.csv',
    'Cheng':   'proxy_errors/14_cheng2016composite-noaa_err.csv',
}

REPO = os.path.join(os.path.dirname(__file__), '..')


def _load_empirical_errors(span):
    """Per-record array of per-point sigma values (10-58 ka window)."""
    out = {}
    for name, rel in ERR_FILES.items():
        df = pd.read_csv(os.path.join(REPO, rel))
        age, err = df.iloc[:, 0].values, df.iloc[:, 1].values
        m = (age >= 10.0) & (age <= span)
        out[name] = np.asarray(err[m], dtype=float)
    return out


# ---------------------------------------------------------------------------
# Generator (submitted Table S1)
# ---------------------------------------------------------------------------
def generate_submitted(seed, with_gen_errors, span=58.0, orbital=False,
                 err_sigma=None, rng=None):
    """
    Scenario A generator per submitted supplementary Table S1.

    err_sigma: per-record constant sigma, or None to sample each point's
    sigma from the empirical per-point error distribution.
    """
    np.random.seed(seed)
    if rng is None:
        rng = np.random.default_rng(seed * 1000 + 7)
    t_grid = np.arange(10.0, span + DT / 2, DT)
    N = len(t_grid)

    # Common forcing Z: stationary Gaussian AR(1) (RHO_F, Z_SIGMA)
    # + 25 D-O-type Gaussian pulses (FWHM width 0.4 ka, amplitude decay 1.2 ka)
    Z = np.zeros(N)
    Z[0] = np.random.normal(0, Z_SIGMA)
    for t in range(1, N):
        Z[t] = RHO_F * Z[t - 1] + np.random.normal(
            0, Z_SIGMA * np.sqrt(1 - RHO_F ** 2))
    if orbital:
        Z += 0.5 * np.sin(2 * np.pi * (t_grid - 10.0) / 40.0)
    positions = np.random.uniform(10.0 + 2 * 0.4, span - 2 * 0.4, 25)
    sigma_p = 0.4 / (2 * np.sqrt(2 * np.log(2)))
    for pos in positions:
        amp = 1.0 * np.exp(-(pos - 10.0) / 1.2)
        Z += amp * np.exp(-0.5 * ((t_grid - pos) / sigma_p) ** 2)
    Z = (Z - np.mean(Z)) / np.std(Z)

    # Per-record AR(1) dynamics with common-forcing loading
    data = {}
    for name in VAR_ORDER:
        x = np.zeros(N)
        x[0] = np.random.normal(0, 1)
        for t in range(1, N):
            x[t] = (S1_PARAMS['phi'][name] * x[t - 1]
                    + S1_PARAMS['beta'][name] * Z[t]
                    + np.random.normal(0, S1_PARAMS['sigma'][name]))
        data[name] = (x - np.mean(x)) / np.std(x)

    # Prescribed causal lags
    for driver, target, lag_ka in TRUE_EDGES:
        lag_step = int(round(lag_ka / DT))
        causal = np.zeros(N)
        causal[lag_step:] = data[driver][:-lag_step]
        data[target] = data[target] + CAUSAL_STRENGTH * causal
        data[target] = (data[target] - np.mean(data[target])) / np.std(data[target])

    # Pointwise age errors (empirical per-point sigma distributions)
    if with_gen_errors:
        for name in VAR_ORDER:
            if err_sigma is not None:
                sig = np.full(N, err_sigma[name])
            else:
                emp = _load_empirical_errors(span)[name]
                sig = rng.choice(emp, size=N, replace=True)
            pert = t_grid + rng.normal(0.0, sig)
            idx = np.argsort(pert)
            f = interp1d(pert[idx], data[name][idx], kind='linear',
                         fill_value='extrapolate', bounds_error=False)
            new = f(t_grid)
            data[name] = (new - np.mean(new)) / np.std(new)

    return data, t_grid, Z


# ---------------------------------------------------------------------------
# Layers 2-3
# ---------------------------------------------------------------------------
def layer2(data, mode='keep', alpha=ARC_ALPHA, Z=None):
    """
    Layer 2 deconfounding.

    mode 'keep'    : proxy = 1-ka smoothed NGRIP, NGRIP kept unchanged
                     (paper Section 5.4 keep-reading).
    mode 'altproxy': NGRIP-involving pairs use the orbital-scale (>5 ka)
                     low-pass component of NGRIP as the alternative Z proxy;
                     NGRIP itself is deconfounded by that component.
    mode 'ztrue'   : proxy = 1-ka smoothed true forcing Z (synthetic-only
                     probe; the paper calls the proxy choice 'adaptive ...
                     selected according to the timescale and domain of
                     analysis').
    mode 'zraw'    : proxy = raw true forcing Z (synthetic-only probe).
    """
    trend = gaussian_filter1d(data['NGRIP'], sigma=1.0 / DT, mode='nearest')
    if mode == 'keep':
        return arc_deconfound_batch(data, trend, alpha=alpha,
                                    keep_proxy='NGRIP')
    if mode == 'ztrue' and Z is not None:
        ztrend = gaussian_filter1d(Z, sigma=1.0 / DT, mode='nearest')
        return arc_deconfound_batch(data, ztrend, alpha=alpha,
                                    keep_proxy=None)
    if mode == 'zraw' and Z is not None:
        return arc_deconfound_batch(data, Z, alpha=alpha,
                                    keep_proxy=None)
    orbital = gaussian_filter1d(data['NGRIP'], sigma=5.0 / DT, mode='nearest')
    res = {}
    for name, signal in data.items():
        proxy = orbital if name == 'NGRIP' else trend
        res[name] = arc_deconfound_batch({name: signal}, proxy,
                                         alpha=alpha, keep_proxy=None)[name]
    return res


def adf_flags(residuals):
    flags = {}
    for v, x in residuals.items():
        try:
            flags[v] = adfuller(x, autolag='AIC')[1] > 0.05
        except Exception:
            flags[v] = False
    return flags


def aic_lag(residuals, flags, maxlag=MAXLAG):
    cols = []
    for v in VAR_ORDER:
        x = residuals[v]
        if flags[v]:
            x = np.diff(x)
        cols.append(x)
    n = min(len(c) for c in cols)
    try:
        model = VAR(np.column_stack([c[:n] for c in cols]))
        order = model.select_order(maxlags=maxlag)
        L = order.aic
        return int(L) if (L is not None and L >= 1) else maxlag
    except Exception:
        return maxlag


def _maybe_diff(x, flag):
    return np.diff(x) if flag else x


def _pair_significant(Y, X, L, test='minp', alpha=0.05):
    """Paper Layer 3 F-test decision for one (driver X -> target Y) pair."""
    n = min(len(Y), len(X))
    Yn, Xn = Y[:n], X[:n]
    if test == 'minp':
        try:
            return bool(granger_decision_block(Yn, Xn, L, alpha))
        except Exception:
            return bool(granger_decision_fast(Yn, Xn, L, alpha))
    if test == 'lagonly':
        stats = _granger_stats(Yn, Xn, L)
        if not stats:
            return False
        F, p, df2 = stats[L - 1]
        return F > _f_crit(p, df2, alpha)
    raise ValueError(test)


def test_pairs(residuals, flags, L, test='minp'):
    det = set()
    for target in VAR_ORDER:
        Y = _maybe_diff(residuals[target], flags[target])
        for driver in VAR_ORDER:
            if driver == target:
                continue
            X = _maybe_diff(residuals[driver], flags[driver])
            if _pair_significant(Y, X, L, test):
                det.add((driver, target))
    return det


def perturb_draw(residuals, t, rng, sigma_map):
    return {v: residuals[v][np.argsort(t + rng.normal(0, sigma_map[v], len(t)))]
            for v in VAR_ORDER}


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def run_realization(seed, n_perturb, sigma_map, span, orbital, args):
    all_ctl = [(v1, v2) for v1 in VAR_ORDER for v2 in VAR_ORDER
               if v1 != v2 and (v1, v2) not in [(a, b) for a, b, _ in TRUE_EDGES]]

    def flags_for(residuals):
        if getattr(args, 'diff', 'adf') == 'forced':
            return {v: True for v in VAR_ORDER}
        return adf_flags(residuals)

    data, t, Z = generate_submitted(seed, False, span=span, orbital=orbital)
    res = layer2(data, mode=args.deconfound, alpha=args.alpha, Z=Z)
    flags = flags_for(res)
    L = aic_lag(res, flags, args.maxlag)
    # The F-test spans lags 1..maxlag (the VAR order search cap; src/granger
    # min-p engine). With the paper's AIC cap 1-5 the injected causal lags
    # (16/30/46 steps) are outside the test window entirely -- maxlag must
    # cover them for the 96% edge to be testable at all.
    det0 = test_pairs(res, flags, args.maxlag, args.test)
    noerr_det = set(det0)

    data_e, t_e, Z_e = generate_submitted(seed, True, span=span, orbital=orbital)
    res_e = layer2(data_e, mode=args.deconfound, alpha=args.alpha, Z=Z_e)
    flags_e = flags_for(res_e)
    L_e = aic_lag(res_e, flags_e, args.maxlag)
    rng = np.random.default_rng(seed * 1000 + 7)
    counts = {(v1, v2): 0 for v1 in VAR_ORDER for v2 in VAR_ORDER if v1 != v2}
    for _ in range(n_perturb):
        if getattr(args, 'perdraw', False):
            # Paper 3.4: "For each perturbation m, Layers 1 and 2 produce
            # deconfounded indices" -- each draw applies a fresh 1-sigma
            # Gaussian perturbation to the CLEAN series (one sigma per draw,
            # not sqrt(2) from double scrambling), then re-runs Layer 2 and
            # the ADF/AIC gates inside the draw.
            raw = {v: data[v][np.argsort(
                t + rng.normal(0, sigma_map[v], len(t)))]
                for v in VAR_ORDER}
            ps = layer2(raw, mode=args.deconfound, alpha=args.alpha, Z=Z)
            f = flags_for(ps)
            det = test_pairs(ps, f, args.maxlag, args.test)
        else:
            ps = perturb_draw(res_e, t_e, rng, sigma_map)
            det = test_pairs(ps, flags_e, args.maxlag, args.test)
        for pair in det:
            counts[pair] += 1

    err_rob = {p: c / n_perturb for p, c in counts.items()}
    return noerr_det, err_rob, flags, L, flags_e, L_e, (data, res), (data_e, res_e)


# ---------------------------------------------------------------------------
# Probe mode: clean-world + error-scrambled diagnostics for one realization
# ---------------------------------------------------------------------------
def probe(args):
    span = float(args.span)
    data, t, Z = generate_submitted(42, False, span=span, orbital=args.orbital)
    res = layer2(data, mode=args.deconfound, alpha=args.alpha, Z=Z)
    flags = adf_flags(res)
    L = aic_lag(res, flags, args.maxlag)
    print(f"span={span} orbital={args.orbital} deconfound={args.deconfound} "
          f"alpha={args.alpha} test={args.test} maxlag={args.maxlag}")
    print(f"ADF diff flags: {flags}")
    print(f"AIC lag L = {L}\n")

    pairs = [('NGRIP', 'EDC_dD'), ('CO2', 'CH4'), ('Cariaco', 'Cheng'),
             ('NGRIP', 'Cariaco'), ('Cariaco', 'NGRIP'), ('EDC_dD', 'CO2'),
             ('Cheng', 'CO2'), ('CH4', 'Cariaco')]
    print("--- clean world (no errors, perturbation disabled) ---")
    for driver, target in pairs:
        Y = _maybe_diff(res[target], flags[target])
        X = _maybe_diff(res[driver], flags[driver])
        n = min(len(Y), len(X))
        stats = _granger_stats(Y[:n], X[:n], max(L, args.maxlag))
        fs = []
        for i, (F, p, df2) in enumerate(stats[:args.maxlag], 1):
            pv = _f_pvalue(F, p, df2) if np.isfinite(F) else 0.0
            fs.append(f"lag{i}:F={F:7.2f},p={pv:.4f}"
                      + (" *" if i <= L and F > _f_crit(p, df2, 0.05) else ""))
        print(f"{driver} -> {target}: " + "  ".join(fs))

    print("\n--- with errors: one perturb draw (scrambled), F at lags ---")
    sigma_map = dict(S1_PARAMS['age'])
    data_e, t_e, Z_e = generate_submitted(42, True, span=span, orbital=args.orbital)
    res_e = layer2(data_e, mode=args.deconfound, alpha=args.alpha, Z=Z_e)
    flags_e = adf_flags(res_e)
    L_e = aic_lag(res_e, flags_e, args.maxlag)
    print(f"ADF diff flags: {flags_e}   AIC lag L = {L_e}")
    rng = np.random.default_rng(1234)
    for _ in range(3):
        ps = perturb_draw(res_e, t_e, rng, sigma_map)
        for driver, target in pairs:
            Y = _maybe_diff(ps[target], flags_e[target])
            X = _maybe_diff(ps[driver], flags_e[driver])
            n = min(len(Y), len(X))
            stats = _granger_stats(Y[:n], X[:n], args.maxlag)
            fs = []
            for i, (F, p, df2) in enumerate(stats[:args.maxlag], 1):
                pv = _f_pvalue(F, p, df2) if np.isfinite(F) else 0.0
                fs.append(f"{F:6.1f}"
                          + ("*" if i <= L_e and F > _f_crit(p, df2, 0.05) else " "))
            print(f"{driver} -> {target}: " + "  ".join(fs))
        print("---")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true',
                    help='smoke test: 3 realizations x 40 perturbations')
    ap.add_argument('--probe', action='store_true',
                    help='1 clean-world realization with per-pair F/p printout')
    ap.add_argument('--realizations', type=int, default=20)
    ap.add_argument('--perturb', type=int, default=200)
    ap.add_argument('--span', type=float, default=58.0,
                    help='series span end (ka); submitted spec = 58')
    ap.add_argument('--orbital', action='store_true',
                    help='add 40-ka orbital sinusoid to Z (not in submitted spec)')
    ap.add_argument('--alpha', type=float, default=ARC_ALPHA)
    ap.add_argument('--deconfound', choices=['keep', 'altproxy', 'ztrue', 'zraw'],
                    default='keep')
    ap.add_argument('--perdraw', action='store_true',
                    help='run Layer 2 deconfounding inside each perturbation '
                         'draw on the scrambled raw data (paper 3.4 order) '
                         'instead of once on the clean series')
    ap.add_argument('--test', choices=['minp', 'lagonly'], default='minp')
    ap.add_argument('--maxlag', type=int, default=5,
                    help='VAR lag search cap for the AIC selection. Paper '
                         'Section 3.3 says 1-5, but the injected lags '
                         '(0.8/1.5/2.3 ka = 16/30/46 steps) are only visible '
                         'with a cap that covers them; probe with --maxlag 50.')
    ap.add_argument('--errmode', choices=['empirical', 'constant'],
                    default='empirical',
                    help='with-error injection: sample per-point sigma from '
                         'real-record error files, or constant Table S1 value')
    ap.add_argument('--diff', choices=['adf', 'forced'], default='adf',
                    help='differencing policy: ADF unit-root gate (script '
                         'default) or unconditional first-differencing '
                         '(current paper Section 4.1 selected config)')
    ap.add_argument('--params-current', action='store_true',
                    help='override S1_PARAMS with the current (Aug 17) Table S1 '
                         'values: calibrated phi_gen, beta = 0.25 uniform, '
                         'per-record innovation sigma; implies --orbital '
                         '(the current Table S1 note adds the 40-ka sinusoid)')
    ap.add_argument('--seed-base', type=int, default=42)
    args = ap.parse_args()

    if args.params_current:
        # Current supplementary Table S1 (2026-08-17 rewrite): "Generator
        # AR(1) coefficient phi (calibrated, beta = 0.25)", innovation sigma,
        # beta = 0.25 for all records, 40-ka orbital sinusoid in Z.
        S1_PARAMS['phi'] = {'NGRIP': 0.639, 'EDC_dD': 0.852, 'CO2': 0.936,
                            'CH4': 0.910, 'Cariaco': 0.835, 'Cheng': 0.841}
        S1_PARAMS['beta'] = {k: 0.25 for k in VAR_ORDER}
        S1_PARAMS['sigma'] = {'NGRIP': 0.702, 'EDC_dD': 0.299, 'CO2': 0.071,
                              'CH4': 0.111, 'Cariaco': 0.377, 'Cheng': 0.358}
        args.orbital = True

    if args.probe:
        probe(args)
        return

    if args.quick:
        n_real, n_pert = 3, 40
    else:
        n_real, n_pert = args.realizations, args.perturb

    # The empirical per-point distributions are degenerate (constant per
    # record, verified), so the perturbation loop uses the same values.
    sigma_map = dict(S1_PARAMS['age'])
    print(f"Scenario A submitted run: span={args.span} orbital={args.orbital} "
          f"deconfound={args.deconfound} alpha={args.alpha} test={args.test} "
          f"errmode={args.errmode} {n_real} x {n_pert}", flush=True)

    true_pairs = [(a, b) for a, b, _ in TRUE_EDGES]
    all_ctl = [(v1, v2) for v1 in VAR_ORDER for v2 in VAR_ORDER
               if v1 != v2 and (v1, v2) not in true_pairs]

    noerr_hits = {p: 0 for p in true_pairs + all_ctl}
    err_med = {p: [] for p in true_pairs + all_ctl}
    for r in range(n_real):
        noerr_det, err_rob, flags, L, flags_e, L_e, _, _ = run_realization(
            args.seed_base + r, n_pert, sigma_map, float(args.span),
            args.orbital, args)
        for p in noerr_hits:
            noerr_hits[p] += p in noerr_det
        for p, rob in err_rob.items():
            err_med[p].append(rob)
        print(f"  realization {r + 1}/{n_real} done "
              f"(AIC lag {L}, ADF diff {sum(flags.values())}/6)", flush=True)

    rows = []
    for driver, target, lag_ka in TRUE_EDGES:
        sigma_drv = S1_PARAMS['age'][driver]
        rms = float(np.sqrt((sigma_drv ** 2 + S1_PARAMS['age'][target] ** 2) / 2.0))
        noerr = noerr_hits[(driver, target)] / n_real
        err = float(np.median(err_med[(driver, target)]))
        rows.append({
            'causal_edge': f'{driver} -> {target}',
            'true_lag_ka': lag_ka,
            'lag_error_ratio': round(lag_ka / sigma_drv, 1),
            'robustness_no_error': noerr,
            'robustness_with_error': err,
            'passes_95': err >= 0.95,
        })
    ctl_noerr = [noerr_hits[p] / n_real for p in all_ctl]
    ctl_err = [float(np.median(err_med[p])) for p in all_ctl]
    rows.append({
        'causal_edge': '12 negative controls',
        'true_lag_ka': np.nan,
        'lag_error_ratio': np.nan,
        'robustness_no_error': max(ctl_noerr) if ctl_noerr else np.nan,
        'robustness_with_error': max(ctl_err) if ctl_err else np.nan,
        'passes_95': (max(ctl_err) if ctl_err else 0.0) >= 0.95,
    })
    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUTPUT_DIR, 'scenarioA_submitted_table1.csv')
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}\n")
    print(df.to_string(index=False))

    full_rows = []
    for src in VAR_ORDER:
        for dst in VAR_ORDER:
            if src == dst:
                continue
            true = next((t for t in TRUE_EDGES if t[0] == src and t[1] == dst), None)
            full_rows.append({
                'source': src, 'target': dst,
                'prescribed_true_edge': true is not None,
                'lag_ka': true[2] if true else np.nan,
                'robustness_no_error': noerr_hits[(src, dst)] / n_real,
                'robustness_with_error_median': float(np.median(err_med[(src, dst)])),
                'passes_95_with_error': float(np.median(err_med[(src, dst)])) >= 0.95,
            })
    full_csv = os.path.join(OUTPUT_DIR, 'scenarioA_submitted_table1_full.csv')
    pd.DataFrame(full_rows).to_csv(full_csv, index=False)
    print(f"Saved: {full_csv}")


if __name__ == '__main__':
    main()
