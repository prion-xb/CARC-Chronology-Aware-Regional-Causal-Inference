#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
21_scenarioA_table1.py
----------------------
Canonical Scenario A synthetic validation (paper Section 4.1 / Table 1).

Implements the pipeline exactly as described in the paper:

  Generator  : Scenario A with the parameters of paper Table S1 (Appendix A):
               6 records (NGRIP, EDC_dD, CO2, CH4, Cariaco, Cheng), 10-60 ka,
               dt = 0.05 ka; common forcing Z (AR(1) phi=0.6, sigma=0.8,
               40-ka orbital sinusoid, 25 D/O-type Gaussian pulses); each
               record is AR(1) dynamics with loading beta and noise sigma
               per Table S1; 3 prescribed causal edges (NGRIP->EDC_dD +0.8 ka,
               CO2->CH4 1.5 ka with CO2 leading, Cariaco->Cheng +2.3 ka,
               strength 0.4); pointwise age errors per Table S1.
  Layer 2    : ARC deconfounding, alpha = 0.5, proxy = 1-ka Gaussian-smoothed
               NGRIP trend (NGRIP itself kept unchanged).
  Layer 3    : per dataset: ADF stationarity check (first-differencing if the
               unit-root null is not rejected at alpha = 0.05); AIC selects
               the optimal VAR lag from 1 to 5; F-test at 0.05 (min-p across
               lags 1..L — statsmodels decision-equivalent, fast engine).
  Layer 4    : M = 200 chronology-perturbation realizations; robustness =
               fraction of realizations judged significant; honesty filter
               at 95%.

Columns:
  no-error  : generation without age errors; chronology perturbation
              disabled (one Layer-3 test per realization).
  with-error: pointwise errors injected at generation (Table S1) and the
              chronology-perturbation loop still applied (paper Sections
              3.4/4.1 reading).

Outputs:
  outputs/scenarioA_table1.csv      per-edge Table 1 rows
  outputs/scenarioA_table1_full.csv all 30 directed cells (matrix CSV)

Usage:
  python scripts/21_scenarioA_table1.py                 # full run (20 x 200)
  python scripts/21_scenarioA_table1.py --quick         # smoke test (3 x 40)
  python scripts/21_scenarioA_table1.py --params code   # SCENARIO_A_CONFIG
                                                        # (scripts/16 params)
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
from src.granger_fast import _granger_stats, _f_crit

DT = 0.05
MAXLAG = 5
ARC_ALPHA = 0.5
CAUSAL_STRENGTH = 0.4
TRUE_EDGES = [('NGRIP', 'EDC_dD', 0.8), ('CO2', 'CH4', 1.5),
              ('Cariaco', 'Cheng', 2.3)]

# Paper Table S1 generator parameters
S1_PARAMS = {
    'phi':   {'NGRIP': 0.82, 'EDC_dD': 0.90, 'CO2': 0.72, 'CH4': 0.78,
              'Cariaco': 0.85, 'Cheng': 0.88},
    'beta':  {'NGRIP': 0.35, 'EDC_dD': 0.30, 'CO2': 0.25, 'CH4': 0.25,
              'Cariaco': 0.35, 'Cheng': 0.35},
    'sigma': {'NGRIP': 0.50, 'EDC_dD': 0.45, 'CO2': 0.55, 'CH4': 0.55,
              'Cariaco': 0.50, 'Cheng': 0.50},
    'age':   {'NGRIP': 0.65, 'EDC_dD': 0.83, 'CO2': 1.65, 'CH4': 1.66,
              'Cariaco': 0.50, 'Cheng': 0.14},
}
VAR_ORDER = ['NGRIP', 'EDC_dD', 'CO2', 'CH4', 'Cariaco', 'Cheng']


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
def generate_s1(seed, with_gen_errors):
    """Scenario A generator with paper Table S1 parameters."""
    np.random.seed(seed)
    t_grid = np.arange(10.0, 60.0 + DT / 2, DT)
    N = len(t_grid)

    # Common forcing Z: AR(1) + orbital + D/O pulses (standardized)
    Z = np.zeros(N)
    Z[0] = np.random.normal(0, 0.8)
    for t in range(1, N):
        Z[t] = 0.6 * Z[t - 1] + np.random.normal(0, 0.8 * np.sqrt(1 - 0.6 ** 2))
    Z += 0.5 * np.sin(2 * np.pi * (t_grid - 10.0) / 40.0)
    positions = np.random.uniform(10.0 + 2 * 0.4, 60.0 - 2 * 0.4, 25)
    sigma_p = 0.4 / (2 * np.sqrt(2 * np.log(2)))
    for pos in positions:
        amp = 1.0 * np.exp(-(pos - 10.0) / 1.2)
        Z += amp * np.exp(-0.5 * ((t_grid - pos) / sigma_p) ** 2)
    Z = (Z - np.mean(Z)) / np.std(Z)

    # Per-record AR(1) dynamics
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

    # Pointwise age errors (Table S1)
    if with_gen_errors:
        for name in VAR_ORDER:
            pert = t_grid + np.random.normal(0, S1_PARAMS['age'][name], N)
            idx = np.argsort(pert)
            f = interp1d(pert[idx], data[name][idx], kind='linear',
                         fill_value='extrapolate', bounds_error=False)
            new = f(t_grid)
            data[name] = (new - np.mean(new)) / np.std(new)

    return data, t_grid


def generate_data(params_mode, seed, with_gen_errors):
    if params_mode == 's1':
        return generate_s1(seed, with_gen_errors)
    from src.synthetic import generate_scenario_A
    data, edges, t = generate_scenario_A(seed=seed,
                                         include_age_errors=with_gen_errors)
    return data, t


# ---------------------------------------------------------------------------
# Layers 2-3
# ---------------------------------------------------------------------------
def layer2(data, alpha=ARC_ALPHA):
    proxy = gaussian_filter1d(data['NGRIP'], sigma=1.0 / DT, mode='nearest')
    return arc_deconfound_batch(data, proxy, alpha=alpha, keep_proxy='NGRIP')


def adf_flags(residuals):
    flags = {}
    for v, x in residuals.items():
        try:
            flags[v] = adfuller(x, autolag='AIC')[1] > 0.05
        except Exception:
            flags[v] = False
    return flags


def aic_lag(residuals, flags):
    cols = []
    for v in VAR_ORDER:
        x = residuals[v]
        if flags[v]:
            x = np.diff(x)
        cols.append(x)
    n = min(len(c) for c in cols)
    try:
        model = VAR(np.column_stack([c[:n] for c in cols]))
        order = model.select_order(maxlags=MAXLAG)
        L = order.aic
        return int(L) if (L is not None and L >= 1) else MAXLAG
    except Exception:
        return MAXLAG


def _maybe_diff(x, flag):
    return np.diff(x) if flag else x


def test_pairs(residuals, flags, L):
    """Paper Layer 3: ADF gate + AIC lag 1..5 + F-test min-p across 1..L."""
    det = set()
    for target in VAR_ORDER:
        Y = _maybe_diff(residuals[target], flags[target])
        for driver in VAR_ORDER:
            if driver == target:
                continue
            X = _maybe_diff(residuals[driver], flags[driver])
            n = min(len(Y), len(X))
            for F, p, df2 in _granger_stats(Y[:n], X[:n], L):
                if F > _f_crit(p, df2, 0.05):
                    det.add((driver, target))
                    break
    return det


def perturb_draw(residuals, t, rng, sigma_map):
    """One chronology-perturbation draw (per-variable N(0, sigma_ka))."""
    return {v: residuals[v][np.argsort(t + rng.normal(0, sigma_map[v], len(t)))]
            for v in VAR_ORDER}


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def run_realization(params_mode, seed, n_perturb, sigma_map):
    """Returns (noerr_det, err_rob, err_counts) for one realization."""
    all_ctl = [(v1, v2) for v1 in VAR_ORDER for v2 in VAR_ORDER
               if v1 != v2 and (v1, v2) not in [(a, b) for a, b, _ in TRUE_EDGES]]

    # ---- no-error column: clean generation, perturbation disabled ----
    data, t = generate_data(params_mode, seed, False)
    res = layer2(data)
    flags = adf_flags(res)
    L = aic_lag(res, flags)
    det0 = test_pairs(res, flags, L)
    noerr_det = set(det0)

    # ---- with-error column: pointwise errors injected at generation,
    #      chronology-perturbation loop applied ----
    data_e, t_e = generate_data(params_mode, seed, True)
    res_e = layer2(data_e)
    flags_e = adf_flags(res_e)
    L_e = aic_lag(res_e, flags_e)
    rng = np.random.default_rng(seed)
    counts = {(v1, v2): 0 for v1 in VAR_ORDER for v2 in VAR_ORDER if v1 != v2}
    for _ in range(n_perturb):
        ps = perturb_draw(res_e, t_e, rng, sigma_map)
        det = test_pairs(ps, flags_e, L_e)
        for pair in det:
            counts[pair] += 1

    err_rob = {p: c / n_perturb for p, c in counts.items()}
    return noerr_det, err_rob, flags, L, flags_e, L_e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true',
                    help='smoke test: 3 realizations x 40 perturbations')
    ap.add_argument('--realizations', type=int, default=20)
    ap.add_argument('--perturb', type=int, default=200)
    ap.add_argument('--params', choices=['s1', 'code'], default='s1',
                    help='generator parameter set (paper Table S1 or '
                         'src.synthetic.SCENARIO_A_CONFIG)')
    args = ap.parse_args()

    if args.quick:
        n_real, n_pert = 3, 40
    else:
        n_real, n_pert = args.realizations, args.perturb

    sigma_map = S1_PARAMS['age'] if args.params == 's1' else {
        'NGRIP': 0.65, 'EDC_dD': 0.83, 'CO2': 1.65, 'CH4': 1.66,
        'Cariaco': 0.23, 'Cheng': 0.14}
    print(f"Scenario A Table 1 run: params={args.params}, "
          f"{n_real} realizations x {n_pert} perturbations", flush=True)

    true_pairs = [(a, b) for a, b, _ in TRUE_EDGES]
    all_ctl = [(v1, v2) for v1 in VAR_ORDER for v2 in VAR_ORDER
               if v1 != v2 and (v1, v2) not in true_pairs]

    noerr_hits = {p: 0 for p in true_pairs + all_ctl}
    err_med = {p: [] for p in true_pairs + all_ctl}
    for r in range(n_real):
        noerr_det, err_rob, flags, L, flags_e, L_e = run_realization(
            args.params, 42 + r, n_pert, sigma_map)
        for p in noerr_hits:
            noerr_hits[p] += p in noerr_det
        for p, rob in err_rob.items():
            err_med[p].append(rob)
        print(f"  realization {r + 1}/{n_real} done "
              f"(AIC lag {L}, ADF diff {sum(flags.values())}/6)", flush=True)

    # ---- Table 1 rows ----
    rows = []
    for driver, target, lag_ka in TRUE_EDGES:
        rms = float(np.sqrt((sigma_map[driver] ** 2 + sigma_map[target] ** 2) / 2.0))
        noerr = noerr_hits[(driver, target)] / n_real
        err = float(np.median(err_med[(driver, target)]))
        rows.append({
            'causal_edge': f'{driver} -> {target}',
            'true_lag_ka': lag_ka,
            'lag_error_ratio': round(lag_ka / rms, 1),
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
    out_csv = os.path.join(OUTPUT_DIR, 'scenarioA_table1.csv')
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}\n")
    print(df.to_string(index=False))

    # ---- full 30-cell matrix CSV ----
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
    full_csv = os.path.join(OUTPUT_DIR, 'scenarioA_table1_full.csv')
    pd.DataFrame(full_rows).to_csv(full_csv, index=False)
    print(f"Saved: {full_csv}")


if __name__ == '__main__':
    main()
