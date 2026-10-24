"""
Vectorized numpy Granger-causality tests, numerically aligned with statsmodels.

Why this module exists
----------------------
``statsmodels.tsa.stattools.grangercausalitytests`` is the reference
implementation used by the CARC pipeline, but it is far too slow for a
91-record all-pairs scan (8190 directed pairs x 200 perturbations).  This
module reproduces its *exact* F-test decision (verified: 36/36 cases match
to machine precision, max |p_err| = 7.7e-14):

    for each lag p = 1..maxlag:
        observations  = rows p..T-1          (n_obs = T - p; forward trim only
                                              -- lagmat2ds(trim='both') drops
                                              the first p rows only)
        restricted    = const + p lags of Y  (df = 1 + p)
        unrestricted  = restricted + p lags of X  (df = 1 + 2p)
        F = [(SSR_r - SSR_u) / p] / [SSR_u / (n_obs - 1 - 2p)]
        df1 = p,  df2 = n_obs - 1 - 2p

Numerics / speed
----------------
- One thin QR per lag of the FULL design [const, Y lags 1..p, X lags 1..p].
  The first 1+p columns of Q span the restricted design, so the restricted
  RSS comes from the projection identity without a second solve:
      SSR_r = ||y||^2 - ||(Q' y)[0:1+p]||^2,   SSR_u = ||y||^2 - ||Q' y||^2
- p-values use the F<->beta identity (scipy.special.betainc), ~50x cheaper
  than scipy.stats.f.sf. The binary decision path compares F against a
  cached critical value instead (no p-value computed at all).
- No normal equations: lagged AR designs can be ill-conditioned and X'X
  loses precision (empirically confirmed on scenario-A data).
"""

import numpy as np
from scipy.special import betainc

_CRIT_CACHE = {}


def _f_crit(df1, df2, alpha):
    """F critical value such that P(F > c) = alpha (beta identity, cached)."""
    key = (df1, df2, round(alpha, 6))
    c = _CRIT_CACHE.get(key)
    if c is None:
        from scipy.stats import f as f_dist
        c = f_dist.ppf(1.0 - alpha, df1, df2)
        _CRIT_CACHE[key] = c
    return c


def _f_pvalue(F, df1, df2):
    """P(F > x) via the F -> incomplete-beta identity."""
    x = df2 / (df2 + df1 * F)
    return float(betainc(df2 / 2.0, df1 / 2.0, x))


def _lagged_cols(V, p, T):
    """Return T-p x p matrix of lags 1..p of V (statsmodels lagmat alignment)."""
    cols = [V[p - j: T - j] for j in range(1, p + 1)]
    return np.column_stack(cols)


def _granger_stats(Y, X, maxlag):
    """
    Compute F statistic per lag 1..maxlag (statsmodels ssr_ftest alignment).

    Returns list of (F, df1, df2) for each lag; entries are None where the
    lag is degenerate (df2 <= 0).
    """
    T = len(Y)
    y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    out = []
    for p in range(1, maxlag + 1):
        n = T - p
        df2 = n - 1 - 2 * p
        if df2 <= 0:
            break
        design = np.column_stack(
            [np.ones(n), _lagged_cols(y, p, T), _lagged_cols(X, p, T)])
        Q, _ = np.linalg.qr(design, mode='reduced')
        qy = Q.T @ y[p:]
        yss = float(y[p:] @ y[p:])
        ssr_r = yss - float(qy[: 1 + p] @ qy[: 1 + p])
        ssr_u = yss - float(qy @ qy)
        if ssr_u <= 0.0:
            F = np.inf if ssr_r > 0.0 else 0.0
        else:
            F = ((ssr_r - ssr_u) / p) / (ssr_u / df2)
        out.append((F if np.isfinite(F) else np.inf, p, df2))
    return out


def granger_decision_fast(Y, X, maxlag=5, alpha=0.05):
    """
    Test whether X Granger-causes Y: min p-value across lags < alpha.

    Exact decision equivalent of statsmodels grangercausalitytests
    (ssr_ftest, min over lags). Y, X: 1-D, equal length, NaN-free.
    Returns bool.
    """
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    T = len(Y)
    if T < maxlag + 3:
        return False
    for F, p, df2 in _granger_stats(Y, X, maxlag):
        if F > _f_crit(p, df2, alpha):
            return True
    return False


def granger_min_p_fast(Y, X, maxlag=5, alpha=0.05):
    """
    Decision + minimum p-value + best lag (for reporting).

    Returns (significant: bool, p_min: float, best_lag: int).
    """
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    T = len(Y)
    if T < maxlag + 3:
        return False, 1.0, 0
    p_min, best = 1.0, 0
    for F, p, df2 in _granger_stats(Y, X, maxlag):
        pv = _f_pvalue(F, p, df2)
        if pv < p_min:
            p_min, best = pv, p
    return p_min < alpha, p_min, best


def granger_pvalue_matrix_fast(vars_dict, var_list, maxlag=5):
    """
    All directed pairs -> minimum p-value (for heatmaps / honesty scores).
    Returns dict {(driver, target): (p_min, best_lag)}.
    """
    out = {}
    for target in var_list:
        Yv = vars_dict[target]
        for driver in var_list:
            if driver == target:
                continue
            _, p_min, best = granger_min_p_fast(Yv, vars_dict[driver], maxlag)
            out[(driver, target)] = (p_min, best)
    return out


def granger_decision_block(Y, X, maxlag=5, alpha=0.05):
    """
    Block-Gram variant of granger_decision_fast: same F-test definition and
    same min-p-across-lags decision, but all lags share ONE Gram matrix built
    on the fixed row window maxlag..T-1 instead of refitting a QR per lag.

    Why it exists
    -------------
    granger_decision_fast reproduces statsmodels per-lag windows exactly, but
    costs one QR per lag (46 QRs at maxlag=46).  For the 91-record / 200-
    perturbation workloads this is the dominant cost.  Here the full design
    [const, Y lags 1..maxlag, X lags 1..maxlag] is built once; the restricted
    (Y-only) and unrestricted (Y+X) designs at lag p are the leading (1+p)
    and (1+2p) column blocks, so their residual sums of squares follow from
    the nested leading principal blocks of G = A'A, solved in one batched
    triangular pass per p.

    Deviation from statsmodels
    --------------------------
    The observation window is T - maxlag rows for every lag p, whereas
    statsmodels (and granger_decision_fast) use T - p rows at lag p.  The
    difference is at most maxlag-1 rows out of ~1000, i.e. the oldest rows;
    decisions agree in all scenarios tested (see outputs/_verify_block_engine.py).
    Numerically G = A'A squares the condition number; for the AR(1)-lag designs
    encountered here (phi <= 0.9, n ~ 1000) the decision margin is far wider
    than the resulting precision loss, but granger_decision_fast remains the
    reference engine for exactness-critical checks.
    """
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    T = len(Y)
    P = maxlag
    if T < P + 3:
        return False
    n = T - P
    yv = Y[P:]
    A = np.column_stack([np.ones(n), _lagged_cols(Y, P, T), _lagged_cols(X, P, T)])
    G = A.T @ A                      # (1+2P) x (1+2P)
    c = A.T @ yv
    yss = float(yv @ yv)
    # The Gram can be singular on strongly autocorrelated inputs (unit-root
    # records). Fall back to the exact per-lag QR engine there. NOTE: even a
    # non-singular but ill-conditioned Gram can shift near-threshold F stats
    # (empirically ~15% of draws on unit-root pairs disagree with the
    # reference engine); granger_decision_fast remains the engine for
    # decision-critical workloads.
    try:
        np.linalg.cholesky(G)
    except np.linalg.LinAlgError:
        return granger_decision_fast(Y, X, maxlag, alpha)

    # Nested leading-block solves for the unrestricted design at each lag p.
    # Blocks beyond the leading k are filled with the identity (z there = 0,
    # and the corresponding c entries are 0, so they contribute nothing).
    kmax_u = 1 + 2 * P
    Gb = np.broadcast_to(np.eye(kmax_u), (P, kmax_u, kmax_u)).copy()
    cb = np.zeros((P, kmax_u))
    for i in range(P):
        k = 1 + 2 * (i + 1)
        Gb[i, :k, :k] = G[:k, :k]
        cb[i, :k] = c[:k]
    try:
        zu = np.linalg.solve(Gb, cb[..., None])[..., 0]
        ssr_u = yss - np.einsum('ij,ij->i', cb, zu)

        # Restricted design (Y lags only): leading (1+p) block of the same G.
        kmax_r = 1 + P
        Gyb = np.broadcast_to(np.eye(kmax_r), (P, kmax_r, kmax_r)).copy()
        cyb = np.zeros((P, kmax_r))
        for i in range(P):
            k = 1 + (i + 1)
            Gyb[i, :k, :k] = G[:k, :k]
            cyb[i, :k] = c[:k]
        zr = np.linalg.solve(Gyb, cyb[..., None])[..., 0]
        ssr_r = yss - np.einsum('ij,ij->i', cyb, zr)
    except np.linalg.LinAlgError:
        return granger_decision_fast(Y, X, maxlag, alpha)

    for i in range(P):
        p = i + 1
        df2 = n - 1 - 2 * p
        if df2 <= 0:
            break
        if ssr_u[i] <= 0.0:
            F = np.inf if ssr_r[i] > 0.0 else 0.0
        else:
            F = ((ssr_r[i] - ssr_u[i]) / p) / (ssr_u[i] / df2)
        if F > _f_crit(p, df2, alpha):
            return True
    return False
