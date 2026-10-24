"""
Granger causality with age-perturbation ensemble and robustness scoring.

This module provides the core CARC detection pipeline:
1. VAR lag-order selection
2. Bivariate Granger causality test
3. Age-perturbation robustness
4. Directional detection with honesty filter
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR

from .config import MAX_GRANGER_LAG, ALPHA_LEVEL, STABILITY_THRESHOLD, NOISE_STD

# Module-level RNG: perturbation ensembles are reproducible across runs.
# Callers may override by passing an explicit ``rng``.
_MODULE_RNG = np.random.default_rng(42)


def select_var_lag(df, maxlag=MAX_GRANGER_LAG):
    """
    Select optimal VAR lag order via AIC.

    Returns
    -------
    opt_lag : int or None
    """
    try:
        model = VAR(df)
        lag_order = model.select_order(maxlags=maxlag)
        opt_lag = lag_order.aic
        return opt_lag if (opt_lag is not None and opt_lag >= 1) else None
    except Exception:
        return None


def granger_test_pair(Y, X, maxlag=MAX_GRANGER_LAG, alpha=ALPHA_LEVEL):
    """
    Test whether X Granger-causes Y.

    Parameters
    ----------
    Y, X : np.ndarray (1D)
        Response and predictor series. Must be same length.
    maxlag : int
        Maximum lag order.
    alpha : float
        Significance threshold.

    Returns
    -------
    is_significant : bool
        True if min p-value across lags < alpha.
    """
    if len(Y) != len(X) or len(Y) < maxlag + 3:
        return False

    df = pd.DataFrame({'Y': Y, 'X': X}).dropna()
    if len(df) < maxlag + 3:
        return False

    try:
        res = grangercausalitytests(df, maxlag, verbose=False)
        pvals = [res[lag][0]['ssr_ftest'][1] for lag in range(1, maxlag + 1)]
        return min(pvals) < alpha
    except Exception:
        return False


def granger_min_pvalue(Y, X, maxlag=MAX_GRANGER_LAG):
    """
    Return the minimum Granger p-value across lags.

    Returns
    -------
    p_min : float (1.0 if test fails)
    """
    if len(Y) != len(X) or len(Y) < maxlag + 3:
        return 1.0

    df = pd.DataFrame({'Y': Y, 'X': X}).dropna()
    if len(df) < maxlag + 3:
        return 1.0

    try:
        res = grangercausalitytests(df, maxlag, verbose=False)
        pvals = [res[lag][0]['ssr_ftest'][1] for lag in range(1, maxlag + 1)]
        return min(pvals)
    except Exception:
        return 1.0


def granger_detect(data_dict, maxlag=MAX_GRANGER_LAG, alpha=ALPHA_LEVEL):
    """
    Detect all significant Granger-causal edges in a multivariate dataset.

    Parameters
    ----------
    data_dict : dict of {name: np.ndarray}
        All variables. Each array must be same length.
    maxlag : int
    alpha : float

    Returns
    -------
    edges : list of (driver, target) tuples
    """
    var_names = list(data_dict.keys())
    if len(var_names) < 2:
        return []

    df = pd.DataFrame(data_dict)
    df_diff = df.diff().dropna()
    if len(df_diff) < 10:
        return []

    opt_lag = select_var_lag(df_diff, maxlag)
    if opt_lag is None:
        return []

    detected = []
    for v1 in var_names:
        for v2 in var_names:
            if v1 == v2:
                continue
            try:
                res = grangercausalitytests(df_diff[[v2, v1]], maxlag=opt_lag, verbose=False)
                p_vals = [res[i + 1][0]['ssr_chi2test'][1] for i in range(opt_lag)]
                if min(p_vals) < alpha:
                    detected.append((v1, v2))
            except Exception:
                pass

    return detected


def age_perturb(series, sigma_err, rng=None, dt=1.0):
    """
    Perturb the time axis of a series by N(0, sigma_err), re-sort, and
    interpolate back to the original grid.

    Parameters
    ----------
    series : np.ndarray (1D)
    sigma_err : float
        Age error standard deviation in PHYSICAL ka units (converted to
        grid index units via ``dt``).
    dt : float
        Grid spacing in ka (default 1.0; pass 0.05 for the 10–60 ka grid).
    rng : np.random.Generator or None

    Returns
    -------
    perturbed : np.ndarray (same length as input)
    """
    if rng is None:
        rng = _MODULE_RNG

    n = len(series)
    orig = np.arange(n)
    perturbed = orig + rng.normal(0, sigma_err / dt, n)
    idx = np.argsort(perturbed)
    perturbed_sorted = perturbed[idx]
    series_sorted = series[idx]

    f = interp1d(perturbed_sorted, series_sorted, kind='linear',
                 fill_value='extrapolate', bounds_error=False)
    result = f(orig)
    valid = ~np.isnan(result)
    if valid.sum() < 10:
        return series
    return result


def age_perturb_pair(Y, X, sigma_err, rng=None, dt=1.0):
    """
    Apply age perturbation to two series simultaneously (same perturbed time axis).

    Parameters
    ----------
    sigma_err : float
        Combined 1σ age uncertainty in PHYSICAL ka units. Converted to grid
        index units via ``dt`` (index noise = sigma_err / dt).
    dt : float
        Grid spacing in ka (default 1.0, i.e. series sampled every 1 ka).
        Pass 0.05 for the 10–60 ka high-resolution grid.

    Returns
    -------
    Yp, Xp : np.ndarray (same length, may be slightly shorter if edge NaN removed)
    """
    if rng is None:
        rng = _MODULE_RNG

    n = len(Y)
    orig = np.arange(n)
    perturbed = orig + rng.normal(0, sigma_err / dt, n)
    idx = np.argsort(perturbed)
    perturbed_sorted = perturbed[idx]

    fY = interp1d(perturbed_sorted, Y[idx], kind='linear',
                  fill_value='extrapolate', bounds_error=False)
    fX = interp1d(perturbed_sorted, X[idx], kind='linear',
                  fill_value='extrapolate', bounds_error=False)

    Yp, Xp = fY(orig), fX(orig)
    valid = ~(np.isnan(Yp) | np.isnan(Xp))
    if valid.sum() < 10:
        return Y, X
    return Yp[valid], Xp[valid]


def robustness(Y, X, sigma_err, n_perturb=200, maxlag=MAX_GRANGER_LAG,
               alpha=ALPHA_LEVEL, noise_std=NOISE_STD, rng=None, dt=1.0):
    """
    Compute CARC robustness score: fraction of age-perturbed tests where
    X Granger-causes Y at p < alpha.

    Parameters
    ----------
    Y, X : np.ndarray
        Target and source series.
    sigma_err : float
        Combined age error (ka; see ``age_perturb_pair`` for units).
    dt : float
        Grid spacing in ka (default 1.0).
    n_perturb : int
        Number of perturbation iterations.
    maxlag : int
    alpha : float
    noise_std : float
        Observational noise fraction added per perturbation.
    rng : np.random.Generator or None

    Returns
    -------
    robustness : float (0.0 to 1.0)
    """
    if rng is None:
        rng = _MODULE_RNG

    sig = 0
    for _ in range(n_perturb):
        Yp, Xp = age_perturb_pair(Y, X, sigma_err, rng, dt=dt)
        if noise_std > 0:
            Yp = Yp + rng.normal(0, noise_std * np.std(Yp), size=len(Yp))
            Xp = Xp + rng.normal(0, noise_std * np.std(Xp), size=len(Xp))
        if granger_test_pair(Yp, Xp, maxlag, alpha):
            sig += 1

    return sig / n_perturb


def is_honest(rob_value, threshold=STABILITY_THRESHOLD):
    """Check whether a robustness score passes the honesty filter."""
    return rob_value >= threshold
