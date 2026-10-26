#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
22_scenarioA_final.py
---------------------
Canonical Scenario A synthetic validation (paper Section 4.1 / Table 1) —
final, data-grounded parameterization with transparent pipeline selection.

Project principles (2026-08-16):
 1. Normal experimental workflow: parameters are chosen inside the
    experiment, with stated reasons, not anchored to previously reported
    results.
 2. The pipeline is transparent, public, and reproducible: every input,
    seed, intermediate grid, and output CSV is saved.

Design
------
 Generator (benchmark truth):
   - Scenario A structure kept as defined in the paper: 6 records
     (NGRIP, EDC_dD, CO2, CH4, Cariaco, Cheng), 10-60 ka, dt = 0.05 ka;
     common forcing Z = AR(1) (phi=0.6, sigma=0.8) + 40-ka orbital
     sinusoid + 25 D/O-type Gaussian pulses; 3 prescribed edges
     (NGRIP->EDC_dD +0.8 ka, CO2->CH4 1.5 ka, Cariaco->Cheng +2.3 ka,
     strength 0.4).
   - Per-record AR coefficient phi and noise sigma_eps are ESTIMATED from
     the corresponding Tier-1 real record (10-60 ka, resampled to the
     50-yr grid): marginal AR(1) regression. phi is then calibrated by
     bisection so that the GENERATED series' marginal AR(1) coefficient
     matches the fitted value under the scenario common forcing (the
     forcing adds persistence beyond the generator's own phi).
   - Age errors = published 1-sigma age-model uncertainties
     (src.config.TIER1_SIGMA): 0.65/0.83/1.65/1.66/0.23/0.14 ka.
   - Common-forcing loading beta is NOT identifiable from data (the
     forcing is unobserved by premise — Theorems 1-4), so it is a stated
     scenario assumption swept over {0.15, 0.25, 0.35} ka^-1-free units;
     the primary value is 0.25.
 Pipeline (method under test):
   - Axes: ARC alpha in {0, 0.5, 1.0} x differencing policy in
     {none, adf, force} x test in {min-p F, AIC-lag F} x perturbation
     scheme in {gen_only, test_var, test_joint, gen_and_test} x lag
     horizon in {5, 50} (5 = paper value, covers only 0.25 ka; the
     prescribed lags of 0.8-2.3 ka need up to 46 steps, so the horizon
     itself is selected by the objective).
     Both tests use the exact statsmodels-equivalent engine (per-lag QR,
     window T-p at lag p): min-p F scans lags 1..maxlag; AIC-lag F tests
     only the AIC-selected lag with an exact single-lag F (one QR). The
     faster block-Gram variant exists in src.granger_fast but is not used
     here (near-threshold decisions diverge on unit-root draws).
   - Selection objective (stated up front, matching the framework's own
     priority of false-positive control over sensitivity): lexicographic —
     (i) minimize the mean robustness over the 27 directed control cells
     (max as tiebreak), (ii) maximize the mean robustness of the 3 true
     edges. The full 144-config grid is saved to disk.
   - A no-common-forcing variant (use_Z=False) at the selected
     configuration serves as the attribution diagnostic: with Z removed,
     control robustness should collapse, confirming the spurious
     directions are common-forcing-induced.
 Final numbers: 20 realizations x 200 perturbations at the selected
 configuration, for beta in {0.15, 0.25, 0.35} and the noZ diagnostic.

Outputs (OUTPUT_DIR):
  scenarioA_final_params.csv        fitted/calibrated generator parameters
  scenarioA_final_grid.csv          144-config screening grid (4 x 100)
  scenarioA_final_table1.csv        Table 1 rows (selected config)
  scenarioA_final_table1_full.csv   all 30 directed cells
  scenarioA_final_diag_noZ.csv      no-common-forcing diagnostic

Usage:
  python scripts/22_scenarioA_final.py --quick   # smoke test (2 x 40)
  python scripts/22_scenarioA_final.py           # full run
"""

import os, sys, argparse, itertools, warnings
# Single-threaded BLAS inside joblib workers: multithreaded LAPACK per worker
# caused memory blowup and TerminatedWorkerError on this machine (two runs
# both died in the 144-config grid). QR/LAPACK results are bit-identical
# regardless of threading; this only bounds per-worker memory and thrash.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import solve_triangular
from statsmodels.tsa.stattools import adfuller
from joblib import Parallel, delayed

from src.config import OUTPUT_DIR, DATA_DIR, TIER1_SIGMA
from src.arc import arc_deconfound_batch
from src.granger_fast import _f_crit, _lagged_cols, granger_decision_fast

DT = 0.05
MAXLAGS = [5, 50]   # selection-grid axis: lag horizon (5 = paper value,
                    # covers 0.25 ka; 50 covers the longest prescribed lag
                    # of 2.3 ka = 46 steps at dt = 0.05)
VAR_ORDER = ['NGRIP', 'EDC_dD', 'CO2', 'CH4', 'Cariaco', 'Cheng']
TRUE_EDGES = [('NGRIP', 'EDC_dD', 0.8), ('CO2', 'CH4', 1.5),
              ('Cariaco', 'Cheng', 2.3)]
TRUE_PAIRS = [(a, b) for a, b, _ in TRUE_EDGES]
ALL_CTL = [(v1, v2) for v1 in VAR_ORDER for v2 in VAR_ORDER
           if v1 != v2 and (v1, v2) not in TRUE_PAIRS]
BETA_SWEEP = [0.15, 0.25, 0.35]
PRIMARY_BETA = 0.25
CAUSAL_STRENGTH = 0.4

# Published 1-sigma age-model uncertainties (src.config.TIER1_SIGMA)
AGE_ERR = {'NGRIP': TIER1_SIGMA['NGRIP'],
           'EDC_dD': TIER1_SIGMA['EDC_dD'],
           'CO2': TIER1_SIGMA['EDC_CO2'],
           'CH4': TIER1_SIGMA['EDC_CH4'],
           'Cariaco': TIER1_SIGMA['Cariaco'],
           'Cheng': TIER1_SIGMA['Cheng2016']}

REC_FILES = {
    'NGRIP': ('NGRIP_d18O.csv', 'csv'),
    'EDC_dD': ('EDC_dD.csv', 'csv'),
    'CO2': ('EDC_CO2.xlsx', 'xlsx'),
    'CH4': ('EDC_CH4.csv', 'csv'),
    'Cariaco': ('Cariaco_Mo.csv', 'csv'),
    'Cheng': ('Cheng2016_d18O.xlsx', 'xlsx'),
}


# ---------------------------------------------------------------------------
# 1. Parameter estimation from the real Tier-1 records
# ---------------------------------------------------------------------------
def load_record(name):
    fname, typ = REC_FILES[name]
    path = os.path.join(DATA_DIR, fname)
    df = (pd.read_excel(path) if typ == 'xlsx'
          else pd.read_csv(path, header=None, skipinitialspace=True))
    df = df.iloc[:, :2]
    df.columns = ['age', 'value']
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    w = df.dropna()
    w = w[(w.age >= 10) & (w.age <= 60)]
    if len(w) < 30:
        w = df.dropna()
    x = (w.value - w.value.mean()) / w.value.std()
    f = interp1d(w.age.values, x.values, kind='linear',
                 bounds_error=False, fill_value='extrapolate')
    tg = np.arange(max(10.0, w.age.min()), min(60.0, w.age.max()) + DT / 2, DT)
    xr = f(tg)
    return (xr - xr.mean()) / xr.std()


def fit_ar1(x):
    """Marginal AR(1) regression on a standardized record.
    Returns (phi, innovation_std)."""
    phi = float(np.linalg.lstsq(x[:-1, None], x[1:], rcond=None)[0][0])
    resid = x[1:] - phi * x[:-1]
    return phi, float(resid.std())


def _sim_marginal_rho(phi_g, beta, sigma_eps, seed, n=200000):
    """Marginal lag-1 autocorrelation of x_t = phi_g x_{t-1} + beta Z_t + eps_t
    with the scenario Z structure, from a long simulation."""
    np.random.seed(seed)
    # scenario Z on a long grid (orbital + AR(1); pulses negligible here)
    t = np.arange(n) * DT
    Z = np.zeros(n)
    Z[0] = np.random.normal(0, 0.8)
    for i in range(1, n):
        Z[i] = 0.6 * Z[i - 1] + np.random.normal(0, 0.8 * np.sqrt(1 - 0.36))
    Z += 0.5 * np.sin(2 * np.pi * (t - 10.0) / 40.0)
    Z = (Z - Z.mean()) / Z.std()
    x = np.zeros(n)
    x[0] = np.random.normal(0, 1)
    for i in range(1, n):
        x[i] = phi_g * x[i - 1] + beta * Z[i] + np.random.normal(0, sigma_eps)
    x = (x - x.mean()) / x.std()
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def calibrate_phi(target_rho, beta, sigma_eps, seed):
    """Bisection: generator phi such that the generated marginal AR(1)
    equals the record's fitted AR(1) under the scenario forcing."""
    lo, hi = 0.0, min(0.999, target_rho)
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        rho = _sim_marginal_rho(mid, beta, sigma_eps, seed)
        if rho < target_rho:
            lo = mid
        else:
            hi = mid
    phi = 0.5 * (lo + hi)
    return phi, _sim_marginal_rho(phi, beta, sigma_eps, seed)


def build_params():
    """Fit phi/sigma_eps from real records; calibrate phi per beta value.
    Returns {'phi': {beta: {record: phi}}, 'sigma_eps': {...}, 'age': {...}}.
    """
    fitted = {name: fit_ar1(load_record(name)) for name in VAR_ORDER}
    out = {'phi': {}, 'sigma_eps': {n: round(s, 4) for n, (_, s) in fitted.items()},
           'age': AGE_ERR, 'fitted_phi': {n: round(p, 4) for n, (p, _) in fitted.items()}}
    for beta in BETA_SWEEP:
        out['phi'][beta] = {}
        for name in VAR_ORDER:
            target, sig = fitted[name]
            phi_g, rho_achieved = calibrate_phi(target, beta, sig, seed=1234)
            out['phi'][beta][name] = round(phi_g, 4)
            print(f'  beta={beta:.2f} {name:8s}: fitted phi={target:.4f}, '
                  f'generator phi={phi_g:.4f} (marginal rho={rho_achieved:.4f})',
                  flush=True)
    return out


# ---------------------------------------------------------------------------
# 2. Generator (Scenario A structure + fitted parameters)
# ---------------------------------------------------------------------------
def gen_scenario(seed, with_gen_errors, phi_map, beta_map, sig_map,
                 age_map, use_Z=True, causal_strength=None):
    """causal_strength: override of the prescribed-edge strength
    (default None = CAUSAL_STRENGTH). Used by the detectability-phase
    sweep (scripts/26); the canonical run22f path is unchanged."""
    np.random.seed(seed)
    t_grid = np.arange(10.0, 60.0 + DT / 2, DT)
    N = len(t_grid)
    if use_Z:
        Z = np.zeros(N)
        Z[0] = np.random.normal(0, 0.8)
        for t in range(1, N):
            Z[t] = 0.6 * Z[t - 1] + np.random.normal(0, 0.8 * np.sqrt(1 - 0.36))
        Z += 0.5 * np.sin(2 * np.pi * (t_grid - 10.0) / 40.0)
        positions = np.random.uniform(10.0 + 2 * 0.4, 60.0 - 2 * 0.4, 25)
        sigma_p = 0.4 / (2 * np.sqrt(2 * np.log(2)))
        for pos in positions:
            amp = 1.0 * np.exp(-(pos - 10.0) / 1.2)
            Z += amp * np.exp(-0.5 * ((t_grid - pos) / sigma_p) ** 2)
        Z = (Z - Z.mean()) / Z.std()
    else:
        Z = np.zeros(N)
    data = {}
    for name in VAR_ORDER:
        x = np.zeros(N)
        x[0] = np.random.normal(0, 1)
        for t in range(1, N):
            x[t] = (phi_map[name] * x[t - 1] + beta_map[name] * Z[t]
                    + np.random.normal(0, sig_map[name]))
        data[name] = (x - x.mean()) / x.std()
    strength = CAUSAL_STRENGTH if causal_strength is None else causal_strength
    for driver, target, lag_ka in TRUE_EDGES:
        lag_step = int(round(lag_ka / DT))
        causal = np.zeros(N)
        causal[lag_step:] = data[driver][:-lag_step]
        data[target] = data[target] + strength * causal
        data[target] = (data[target] - np.mean(data[target])) / np.std(data[target])
    if with_gen_errors:
        for name in VAR_ORDER:
            pert = t_grid + np.random.normal(0, age_map[name], N)
            idx = np.argsort(pert)
            f = interp1d(pert[idx], data[name][idx], kind='linear',
                         fill_value='extrapolate', bounds_error=False)
            new = f(t_grid)
            data[name] = (new - np.mean(new)) / np.std(new)
    return data, t_grid


# ---------------------------------------------------------------------------
# 3. Pipeline (ported from the paper-described flow; scripts/16 structure)
# ---------------------------------------------------------------------------
def layer2(data, alpha):
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


def aic_lag(series, diff_policy, flags, maxlag):
    """AIC lag selection on the full K-variable VAR system (one shared lag L
    for all pairs). AIC path computed in a single Gram pass instead of
    statsmodels VAR.select_order (which refits K-variable VAR(p) for every
    p = 1..maxlag and is prohibitively slow at maxlag = 50):

        AIC(p) = logdet(Sigma_p) + 2 * K^2 * p / m

    where Sigma_p is the residual covariance of the VAR(p) fit and
    m = T - maxlag is the effective sample held fixed across p (rows
    maxlag..T-1) so the log-likelihoods are comparable; this is the standard
    VAR information criterion (statsmodels uses logdet + 2 k/nobs with
    k = K^2 p)."""
    cols = []
    for v in VAR_ORDER:
        x = series[v]
        if diff_policy == 'force':
            x = np.diff(x)
        elif diff_policy == 'adf' and flags[v]:
            x = np.diff(x)
        cols.append(x)
    n = min(len(c) for c in cols)
    K = len(VAR_ORDER)
    if n <= maxlag + K * maxlag + 2:
        return maxlag
    Y = np.column_stack([c[:n] for c in cols])          # (n, K)
    # lag matrix Z: row t holds [Y_{t-1}, ..., Y_{t-maxlag}] flattened
    Z = np.empty((n - maxlag, K * maxlag))
    for p in range(1, maxlag + 1):
        Z[:, K * (p - 1):K * p] = Y[maxlag - p:n - p]
    G = Z.T @ Z                                          # (K*maxlag)^2
    ZTy = Z.T @ Y[maxlag:]
    Yeff = Y[maxlag:]
    m = len(Yeff)
    try:
        Lch = np.linalg.cholesky(G)
    except np.linalg.LinAlgError:
        Lch = None
    best_L, best_aic = maxlag, np.inf
    for p in range(1, maxlag + 1):
        k = K * p
        if Lch is not None:
            B = solve_triangular(Lch[:k, :k].T,
                                 solve_triangular(Lch[:k, :k], ZTy[:k],
                                                  lower=True))
        else:
            B = np.linalg.lstsq(G[:k, :k], ZTy[:k], rcond=None)[0]
        R = Yeff - Z[:, :k] @ B
        Sigma = (R.T @ R) / m
        sgn, logdet = np.linalg.slogdet(Sigma)
        if sgn <= 0:
            continue
        aic = logdet + 2.0 * (K * K * p) / m
        if aic < best_aic:
            best_aic, best_L = aic, p
    return best_L


def _maybe_diff(x, v, diff_policy, flags):
    if diff_policy == 'force':
        return np.diff(x)
    if diff_policy == 'adf' and flags[v]:
        return np.diff(x)
    return x


def _granger_f_at_lag(Y, X, L):
    """Single-lag F statistic (statsmodels ssr_ftest alignment, one QR).
    Returns (F, df1, df2), or None if the lag is degenerate."""
    T = len(Y)
    n = T - L
    df2 = n - 1 - 2 * L
    if df2 <= 0:
        return None
    design = np.column_stack(
        [np.ones(n), _lagged_cols(Y, L, T), _lagged_cols(X, L, T)])
    Q, _ = np.linalg.qr(design, mode='reduced')
    qy = Q.T @ Y[L:]
    yss = float(Y[L:] @ Y[L:])
    ssr_r = yss - float(qy[:1 + L] @ qy[:1 + L])
    ssr_u = yss - float(qy @ qy)
    if ssr_u <= 0.0:
        F = np.inf if ssr_r > 0.0 else 0.0
    else:
        F = ((ssr_r - ssr_u) / L) / (ssr_u / df2)
    return (F if np.isfinite(F) else np.inf, L, df2)


def test_pair_series(Y, X, diff_policy, flags, test_mode, L, tgt, drv, maxlag):
    Y = _maybe_diff(Y, tgt, diff_policy, flags)
    X = _maybe_diff(X, drv, diff_policy, flags)
    n = min(len(Y), len(X))
    Y, X = Y[:n], X[:n]
    if test_mode == 'minp_f':
        # Exact statsmodels-equivalent engine (per-lag QR, window T-p at
        # lag p). The block-Gram variant (granger_decision_block) is ~50x
        # faster but its near-threshold decisions diverge on unit-root
        # draws (22-38% disagreement measured), so this validation uses
        # the exact engine; the full run parallelizes over realizations
        # to keep the wall time feasible.
        return granger_decision_fast(Y, X, maxlag)
    out = _granger_f_at_lag(Y, X, L)
    return out is not None and out[0] > _f_crit(out[1], out[2], 0.05)


def test_pairs(series, diff_policy, flags, test_mode, L, maxlag):
    det = set()
    for target in VAR_ORDER:
        for driver in VAR_ORDER:
            if driver == target:
                continue
            if test_pair_series(series[target], series[driver], diff_policy,
                                flags, test_mode, L, target, driver, maxlag):
                det.add((driver, target))
    return det


def perturb_draw(series, t, pert, rng, smap):
    n = len(t)
    if pert == 'test_var':
        return {v: series[v][np.argsort(t + rng.normal(0, smap[v], n))]
                for v in VAR_ORDER}
    out = {}
    for target in VAR_ORDER:
        for driver in VAR_ORDER:
            if driver == target:
                continue
            s = float(np.sqrt((smap[driver] ** 2 + smap[target] ** 2) / 2.0))
            idx = np.argsort(t + rng.normal(0, s, n))
            out[(driver, target)] = (series[target][idx], series[driver][idx])
    return out


def realization_block(cfg, seed, phi_map, beta_map, sig_map, age_map,
                      use_Z, n_pert):
    """One realization. Returns (noerr_det set, err_counts dict)."""
    alpha, diff_policy, test_mode, pert, maxlag = cfg
    np.random.seed(seed)
    data, t = gen_scenario(seed, False, phi_map, beta_map, sig_map,
                           age_map, use_Z)
    res = layer2(data, alpha)
    flags = adf_flags(res)
    L = aic_lag(res, diff_policy, flags, maxlag)
    noerr_det = test_pairs(res, diff_policy, flags, test_mode, L, maxlag)

    if pert == 'gen_only':
        np.random.seed(seed)
        data_e, t_e = gen_scenario(seed, True, phi_map, beta_map, sig_map,
                                   age_map, use_Z)
        res_e = layer2(data_e, alpha)
        flags_e = adf_flags(res_e)
        L_e = aic_lag(res_e, diff_policy, flags_e, maxlag)
        return noerr_det, test_pairs(res_e, diff_policy, flags_e, test_mode,
                                     L_e, maxlag)

    res_base, flags_base, L_base = res, flags, L
    if pert == 'gen_and_test':
        np.random.seed(seed)
        data_e, t_e = gen_scenario(seed, True, phi_map, beta_map, sig_map,
                                   age_map, use_Z)
        res_base = layer2(data_e, alpha)
        flags_base = adf_flags(res_base)
        L_base = aic_lag(res_base, diff_policy, flags_base, maxlag)
    rng = np.random.default_rng(seed)
    counts = {(v1, v2): 0 for v1 in VAR_ORDER for v2 in VAR_ORDER if v1 != v2}
    for _ in range(n_pert):
        ps = perturb_draw(res_base, t, pert, rng, age_map)
        if pert == 'test_var':
            det = test_pairs(ps, diff_policy, flags_base, test_mode, L_base,
                             maxlag)
        else:
            det = set()
            for (drv, tgt), (yp, xp) in ps.items():
                if test_pair_series(yp, xp, diff_policy, flags_base,
                                    test_mode, L_base, tgt, drv, maxlag):
                    det.add((drv, tgt))
        for pair in det:
            counts[pair] += 1
    return noerr_det, counts


# ---------------------------------------------------------------------------
# 4. Screening grid (objective: FPR control first, then sensitivity)
# ---------------------------------------------------------------------------
def grid_score(res):
    """Lexicographic selection objective:
    (err_ctl_mean, err_ctl_max, -err_true_mean)."""
    return (res['err_ctl_mean'], res['err_ctl_max'], -res['err_true_mean'])


def run_config(cfg, phi_map, beta_map, sig_map, age_map, use_Z,
               n_real=4, n_pert=100):
    noerr_tr = np.zeros(3)
    noerr_ct = []
    err_tr = np.zeros(3)
    err_ct = []
    for r in range(n_real):
        noerr_det, err_counts = realization_block(
            cfg, 42 + r, phi_map, beta_map, sig_map, age_map, use_Z, n_pert)
        noerr_tr += np.array([1.0 * (e in noerr_det) for e in TRUE_PAIRS])
        noerr_ct.extend(1.0 * (p in noerr_det) for p in ALL_CTL)
        if isinstance(err_counts, dict):
            err_tr += np.array([err_counts[e] for e in TRUE_PAIRS]) / n_pert
            err_ct.extend(err_counts[p] / n_pert for p in ALL_CTL)
        else:
            err_tr += np.array([1.0 * (e in err_counts) for e in TRUE_PAIRS])
            err_ct.extend(1.0 * (p in err_counts) for p in ALL_CTL)
    return {
        'config': ' | '.join(str(c) for c in cfg),
        'arc_alpha': cfg[0], 'diff': cfg[1], 'test': cfg[2], 'pert': cfg[3],
        'maxlag': cfg[4],
        'noerr_true_mean': float(noerr_tr.mean() / n_real),
        'err_true_mean': float(err_tr.mean() / n_real),
        'err_true_edges': [round(float(v / n_real), 3) for v in err_tr],
        'noerr_ctl_mean': float(np.mean(noerr_ct)),
        'noerr_ctl_max': float(np.max(noerr_ct)),
        'err_ctl_mean': float(np.mean(err_ct)),
        'err_ctl_max': float(np.max(err_ct)),
        'err_ctl_pass95': float(np.mean([1.0 * (c >= 0.95) for c in err_ct])),
    }


# ---------------------------------------------------------------------------
# 5. Full run (20 x 200) — per-cell statistics
# ---------------------------------------------------------------------------
def full_run(cfg, phi_map, beta_map, sig_map, age_map, use_Z,
             n_real=20, n_pert=200):
    pairs = TRUE_PAIRS + ALL_CTL
    # Realizations are independent -> parallelized (exact per-lag-QR engine
    # at maxlag=50 is ~0.17 s per pair-test, so 20 x 200 serial would be
    # several hours per block).
    results = Parallel(n_jobs=8, verbose=1)(
        delayed(realization_block)(cfg, 42 + r, phi_map, beta_map, sig_map,
                                   age_map, use_Z, n_pert)
        for r in range(n_real))
    noerr_hits = {p: 0 for p in pairs}
    err_med = {p: [] for p in pairs}
    for noerr_det, err_counts in results:
        for p in pairs:
            noerr_hits[p] += p in noerr_det
            if isinstance(err_counts, dict):
                err_med[p].append(err_counts[p] / n_pert)
            else:
                err_med[p].append(1.0 * (p in err_counts))
    rows = []
    for driver, target, lag_ka in TRUE_EDGES:
        rows.append({
            'causal_edge': f'{driver} -> {target}',
            'true_lag_ka': lag_ka,
            'lag_error_ratio': round(lag_ka / np.sqrt(
                (age_map[driver] ** 2 + age_map[target] ** 2) / 2.0), 1),
            'robustness_no_error': noerr_hits[(driver, target)] / n_real,
            'robustness_with_error': round(
                float(np.median(err_med[(driver, target)])), 4),
            'passes_95': float(np.median(err_med[(driver, target)])) >= 0.95,
        })
    ctl_noerr = [noerr_hits[p] / n_real for p in ALL_CTL]
    ctl_err = [float(np.median(err_med[p])) for p in ALL_CTL]
    rows.append({
        'causal_edge': 'negative controls (27 directed cells)',
        'true_lag_ka': np.nan,
        'lag_error_ratio': np.nan,
        'robustness_no_error': round(max(ctl_noerr), 4),
        'robustness_with_error': round(max(ctl_err), 4),
        'passes_95': bool(np.max(ctl_err) >= 0.95),
    })
    df = pd.DataFrame(rows)
    df['n_ctl_pass95'] = [np.nan] * 3 + [int(np.sum(np.array(ctl_err) >= 0.95))]
    df['n_ctl_total'] = [np.nan] * 3 + [len(ctl_err)]
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
                'robustness_with_error_median': float(
                    np.median(err_med[(src, dst)])),
                'passes_95_with_error': float(
                    np.median(err_med[(src, dst)])) >= 0.95,
            })
    return df, pd.DataFrame(full_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true',
                    help='smoke test: 2 realizations x 40 perturbations')
    ap.add_argument('--skip-calib', action='store_true',
                    help='reuse saved calibration if present')
    args = ap.parse_args()

    if args.quick:
        N_REAL, N_PERT, N_REAL_FULL, N_PERT_FULL = 2, 40, 2, 40
    else:
        N_REAL, N_PERT, N_REAL_FULL, N_PERT_FULL = 4, 100, 20, 200

    print('== 1. Fitting/calibrating generator parameters from real records ==',
          flush=True)
    if args.skip_calib:
        pcsv = os.path.join(OUTPUT_DIR, 'scenarioA_final_params.csv')
        p = pd.read_csv(pcsv).set_index('record')
        params = {
            'phi': {b: {n: float(p.loc[n, f'gen_phi_beta_{b}']) for n in VAR_ORDER}
                    for b in BETA_SWEEP},
            'sigma_eps': {n: float(p.loc[n, 'sigma_eps']) for n in VAR_ORDER},
            'age': AGE_ERR,
            'fitted_phi': {n: float(p.loc[n, 'fitted_phi']) for n in VAR_ORDER},
        }
        print('  (loaded from scenarioA_final_params.csv)')
    else:
        params = build_params()
    pd.DataFrame([
        {'record': n,
         'fitted_phi': params['fitted_phi'][n],
         'sigma_eps': params['sigma_eps'][n],
         'age_err_ka': params['age'][n],
         **{f'gen_phi_beta_{b}': params['phi'][b][n] for b in BETA_SWEEP}}
        for n in VAR_ORDER]).to_csv(
        os.path.join(OUTPUT_DIR, 'scenarioA_final_params.csv'), index=False)

    print('\n== 2. Pipeline selection grid (144 configs) ==', flush=True)
    cfgs = list(itertools.product([0.0, 0.5, 1.0],
                                  ['none', 'adf', 'force'],
                                  ['minp_f', 'aic_f'],
                                  ['gen_only', 'test_var', 'test_joint',
                                   'gen_and_test'],
                                  MAXLAGS))
    prim_phi = params['phi'][PRIMARY_BETA]
    prim_beta = {v: PRIMARY_BETA for v in VAR_ORDER}

    # Chunked execution with per-chunk checkpointing: workers are torn down
    # between chunks (bounds cumulative worker memory — the two earlier full
    # grids died of TerminatedWorkerError after ~56 tasks), and a crash loses
    # at most one chunk. run_config is deterministic per config (fixed seeds),
    # so resuming from the partial file reproduces a single uninterrupted run
    # bit for bit.
    partial_path = os.path.join(OUTPUT_DIR,
                                'scenarioA_final_grid_partial.csv')
    grid_rows = []
    done = set()
    if os.path.exists(partial_path):
        prev = pd.read_csv(partial_path)
        grid_rows = prev.to_dict('records')
        done = set(prev['config'])
        print(f'  (resuming: {len(done)}/144 configs already done)',
              flush=True)
    todo = [c for c in cfgs
            if ' | '.join(str(x) for x in c) not in done]
    CHUNK, NJOBS = 8, 4
    for i0 in range(0, len(todo), CHUNK):
        chunk = todo[i0:i0 + CHUNK]
        first = len(grid_rows) + 1
        last = len(grid_rows) + len(chunk)
        print(f'  dispatching chunk {i0 // CHUNK + 1} '
              f'({first}-{last} of 144) ...', flush=True)
        res_chunk = Parallel(n_jobs=NJOBS, verbose=1)(
            delayed(run_config)(cfg, prim_phi, prim_beta,
                                params['sigma_eps'], params['age'], True,
                                N_REAL, N_PERT)
            for cfg in chunk)
        grid_rows.extend(res_chunk)
        pd.DataFrame(grid_rows).to_csv(partial_path, index=False)
        print(f'  chunk saved ({len(grid_rows)} configs total)', flush=True)
    grid = pd.DataFrame(grid_rows).sort_values(
        by=['err_ctl_mean', 'err_ctl_max', 'err_true_mean'],
        ascending=[True, True, False])
    grid.to_csv(os.path.join(OUTPUT_DIR, 'scenarioA_final_grid.csv'),
                index=False)
    sel = grid.iloc[0]
    sel_cfg = [sel['arc_alpha'], sel['diff'], sel['test'], sel['pert'],
               int(sel['maxlag'])]
    print('\nSelected configuration (objective: min control FPR, then max '
          'true-edge robustness):')
    print(f"  {sel_cfg}  err_true_mean={sel['err_true_mean']:.3f} "
          f"err_ctl_mean={sel['err_ctl_mean']:.3f} "
          f"err_ctl_max={sel['err_ctl_max']:.3f}", flush=True)

    print('\n== 3. Full runs (20 x 200) ==', flush=True)
    t1_rows = []
    for beta in BETA_SWEEP:
        print(f'  beta = {beta:.2f} ...', flush=True)
        df, full = full_run(sel_cfg, params['phi'][beta],
                            {v: beta for v in VAR_ORDER},
                            params['sigma_eps'], params['age'], True,
                            N_REAL_FULL, N_PERT_FULL)
        df.insert(0, 'beta', beta)
        full.insert(0, 'beta', beta)
        t1_rows.append(df)
        full.to_csv(os.path.join(
            OUTPUT_DIR, f'scenarioA_final_table1_full_beta{beta:.2f}.csv'),
            index=False)
    print('  no-common-forcing diagnostic (use_Z=False, beta=0.25) ...',
          flush=True)
    dfz, fullz = full_run(sel_cfg, prim_phi, prim_beta,
                          params['sigma_eps'], params['age'], False,
                          N_REAL_FULL, N_PERT_FULL)
    dfz.insert(0, 'beta', 'noZ')
    fullz.insert(0, 'beta', 'noZ')
    fullz.to_csv(os.path.join(OUTPUT_DIR, 'scenarioA_final_diag_noZ.csv'),
                 index=False)
    t1_rows.append(dfz)

    out = pd.concat(t1_rows, ignore_index=True)
    out.to_csv(os.path.join(OUTPUT_DIR, 'scenarioA_final_table1.csv'),
               index=False)
    print('\n== Table 1 (final) ==')
    print(out.to_string(index=False))
    print(f"\nSelected config: {sel_cfg}")
    print('Saved outputs/scenarioA_final_*.csv', flush=True)


if __name__ == '__main__':
    main()
