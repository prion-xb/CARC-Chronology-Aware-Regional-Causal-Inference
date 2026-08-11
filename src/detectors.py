"""
CausalDetector — Object-oriented causal detection framework.

Provides an abstract base class for causal detection methods and two
concrete implementations:

Class Hierarchy
---------------
CausalDetector (ABC)
├── StandardGrangerDetector  — standard bivariate Granger causality
└── CARCDetector             — CARC: chronology-aware robust causality

CARCDetector extends StandardGrangerDetector by adding ARC (Adaptive
Residual Causality) deconfounding, age-perturbation ensembles, and a
honesty filter that rejects edges failing to reach a stability threshold.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy.interpolate import interp1d
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
from sklearn.linear_model import LinearRegression

from .config import (
    MAX_GRANGER_LAG, ALPHA_LEVEL, STABILITY_THRESHOLD,
    N_PERTURB, ARC_ALPHA, NOISE_STD,
)


# ======================================================================
# Abstract Base Class
# ======================================================================
class CausalDetector(ABC):
    """Abstract base for all causal detection methods.

    Subclasses must implement :meth:`detect`, which takes a dictionary of
    variable-name → 1D-array and returns a list of (driver, target) edges.

    Parameters
    ----------
    name : str
        Human-readable method name.
    params : dict or None
        Method hyper-parameters stored for later inspection.
    """

    def __init__(self, name, params=None):
        self._name = name
        self._params = params if params is not None else {}

    @property
    def name(self):
        """Method name."""
        return self._name

    @property
    def params(self):
        """Method hyper-parameters (read-only copy)."""
        return dict(self._params)

    @abstractmethod
    def detect(self, data):
        """Detect causal edges in `data`.

        Parameters
        ----------
        data : dict of {str: np.ndarray}
            Variable name → 1D time series. All arrays must be same length.

        Returns
        -------
        list of (driver, target) tuples
        """
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self._name}')"


# ======================================================================
# Standard Granger Causality
# ======================================================================
class StandardGrangerDetector(CausalDetector):
    """Bivariate Granger causality with VAR lag-order selection.

    Tests all ordered pairs of variables on differenced data using the
    optimal VAR lag (via AIC), reporting edges where min p < α.

    Parameters
    ----------
    maxlag : int
        Maximum VAR lag order.
    alpha : float
        Significance threshold for the Granger F-test.
    """

    def __init__(self, maxlag=MAX_GRANGER_LAG, alpha=ALPHA_LEVEL):
        super().__init__('Standard Granger', {'maxlag': maxlag, 'alpha': alpha})
        self._maxlag = maxlag
        self._alpha = alpha

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _select_lag(df, maxlag):
        """Select optimal VAR lag via AIC. Returns int or None."""
        try:
            model = VAR(df)
            lag_order = model.select_order(maxlags=maxlag)
            opt = lag_order.aic
            return opt if (opt is not None and opt >= 1) else None
        except Exception:
            return None

    @staticmethod
    def _granger_test_pair(Y, X, maxlag, alpha):
        """Test whether X Granger-causes Y. Returns bool."""
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

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def detect(self, data):
        """Run standard Granger detection on differenced data.

        Parameters
        ----------
        data : dict of {str: np.ndarray}

        Returns
        -------
        list of (driver, target)
        """
        var_names = list(data.keys())
        if len(var_names) < 2:
            return []

        df = pd.DataFrame(data)
        df_diff = df.diff().dropna()
        if len(df_diff) < 10:
            return []

        opt_lag = self._select_lag(df_diff, self._maxlag)
        if opt_lag is None:
            return []

        detected = []
        for v1 in var_names:
            for v2 in var_names:
                if v1 == v2:
                    continue
                try:
                    res = grangercausalitytests(
                        df_diff[[v2, v1]], maxlag=opt_lag, verbose=False
                    )
                    p_vals = [res[i + 1][0]['ssr_chi2test'][1]
                              for i in range(opt_lag)]
                    if min(p_vals) < self._alpha:
                        detected.append((v1, v2))
                except Exception:
                    pass

        return detected


# ======================================================================
# CARC Detector — Chronology-Aware Robustness Causality
# ======================================================================
class CARCDetector(StandardGrangerDetector):
    """Chronology-Aware Robustness Causality (CARC) detector.

    Extends standard Granger causality with three key innovations:

    1. **ARC deconfounding**: Conservatively strips a fraction α of the
       shared common-forcing signal before testing.
    2. **Age-perturbation ensemble**: Repeats Granger tests under N(0, σ)
       perturbations of each record's age model (N times).
    3. **Honesty filter**: Only edges that are significant in ≥ threshold
       fraction of perturbations are accepted.

    Parameters
    ----------
    n_perturb : int
        Number of age-perturbation iterations (default 200).
    stability_threshold : float
        Minimum fraction of perturbations where edge must be significant
        to pass the honesty filter (default 0.95).
    arc_alpha : float
        ARC stripping strength. 0 = no removal, 1 = full removal.
        0.5 = conservative default.
    maxlag : int
        Maximum VAR lag order.
    alpha : float
        Granger significance threshold.
    noise_std : float
        Observational noise fraction added per perturbation.
    """

    def __init__(self,
                 n_perturb=N_PERTURB,
                 stability_threshold=STABILITY_THRESHOLD,
                 arc_alpha=ARC_ALPHA,
                 maxlag=MAX_GRANGER_LAG,
                 alpha=ALPHA_LEVEL,
                 noise_std=NOISE_STD):
        super().__init__(maxlag=maxlag, alpha=alpha)
        self._name = 'CARC'
        self._params.update({
            'n_perturb': n_perturb,
            'stability_threshold': stability_threshold,
            'arc_alpha': arc_alpha,
            'noise_std': noise_std,
        })
        self._n_perturb = n_perturb
        self._stability_threshold = stability_threshold
        self._arc_alpha = arc_alpha
        self._noise_std = noise_std

    # ------------------------------------------------------------------
    # ARC deconfounding
    # ------------------------------------------------------------------
    @staticmethod
    def arc_deconfound(target, proxy, alpha=0.5):
        """Conservatively strip common-forcing signal from `target` using `proxy`.

        Parameters
        ----------
        target : np.ndarray (1D)
        proxy : np.ndarray (1D), same length
        alpha : float
            Stripping strength.

        Returns
        -------
        np.ndarray — target − α·ŷ + α·mean(ŷ)
        """
        if proxy is None or len(target) != len(proxy):
            return target

        X = proxy.reshape(-1, 1)
        X_mat = np.column_stack([np.ones(len(X)), X])
        model = LinearRegression()
        model.fit(X_mat, target)
        y_pred = model.predict(X_mat)
        residual = target - alpha * y_pred + alpha * np.mean(y_pred)
        return residual

    # ------------------------------------------------------------------
    # Age perturbation + robustness
    # ------------------------------------------------------------------
    def _age_perturb_pair(self, Y, X, sigma_err, rng):
        """Apply age perturbation to both series. Returns (Yp, Xp)."""
        n = len(Y)
        orig = np.arange(n)
        pert = orig + rng.normal(0, sigma_err, n)
        idx = np.argsort(pert)

        fY = interp1d(pert[idx], Y[idx], kind='linear',
                      fill_value='extrapolate', bounds_error=False)
        fX = interp1d(pert[idx], X[idx], kind='linear',
                      fill_value='extrapolate', bounds_error=False)

        Yp, Xp = fY(orig), fX(orig)
        valid = ~(np.isnan(Yp) | np.isnan(Xp))
        if valid.sum() < 10:
            return Y, X
        return Yp[valid], Xp[valid]

    def robustness(self, Y, X, sigma_err, rng=None):
        """Compute CARC robustness score for the edge X → Y.

        Fraction of ``n_perturb`` age-perturbed trials where X Granger-causes Y.

        Parameters
        ----------
        Y, X : np.ndarray
            Target and source series.
        sigma_err : float
            Combined age error (ka).
        rng : np.random.Generator or None

        Returns
        -------
        float in [0, 1]
        """
        if rng is None:
            rng = np.random.default_rng()

        sig = 0
        for _ in range(self._n_perturb):
            Yp, Xp = self._age_perturb_pair(Y, X, sigma_err, rng)
            if self._noise_std > 0:
                Yp = Yp + rng.normal(0, self._noise_std * np.std(Yp), size=len(Yp))
                Xp = Xp + rng.normal(0, self._noise_std * np.std(Xp), size=len(Xp))
            if self._granger_test_pair(Yp, Xp, self._maxlag, self._alpha):
                sig += 1

        return sig / self._n_perturb

    def honesty_filter(self, rob_value):
        """Return True if the robustness score passes the honesty threshold."""
        return rob_value >= self._stability_threshold

    # ------------------------------------------------------------------
    # Main detection
    # ------------------------------------------------------------------
    def detect(self, data, forcing_proxy=None, sigma_dict=None, seed=None):
        """Run CARC detection.

        Parameters
        ----------
        data : dict of {str: np.ndarray}
            Variable name → time series.
        forcing_proxy : np.ndarray or None
            Common-forcing proxy (e.g. NGRIP). If provided, ARC is applied first.
        sigma_dict : dict of {str: float} or None
            Age-model uncertainty (ka) per variable.
        seed : int or None
            RNG seed for reproducibility.

        Returns
        -------
        list of (driver, target) tuples
            Edges that pass the honesty filter.
        """
        rng = np.random.default_rng(seed)
        var_names = list(data.keys())

        # 1. Apply ARC deconfounding
        if forcing_proxy is not None:
            working = {}
            for name in var_names:
                working[name] = self.arc_deconfound(
                    data[name], forcing_proxy, self._arc_alpha
                )
        else:
            working = {k: v for k, v in data.items()}

        # 2. Build sigma dict if not provided
        if sigma_dict is None:
            sigma_dict = {v: 1.0 for v in var_names}

        # 3. Pairwise scan with age perturbations
        detected = []
        for i, v1 in enumerate(var_names):
            for v2 in var_names:
                if v1 == v2:
                    continue
                Y = working[v2]
                X = working[v1]
                sigma_combined = max(sigma_dict.get(v1, 1.0),
                                     sigma_dict.get(v2, 1.0))
                rob = self.robustness(Y, X, sigma_combined, rng)
                if self.honesty_filter(rob):
                    detected.append((v1, v2))

        return detected

    # ------------------------------------------------------------------
    # Full pairwise scan with robustness scores
    # ------------------------------------------------------------------
    def pairwise_scan(self, data, forcing_proxy=None, sigma_dict=None, seed=None):
        """Full pairwise scan returning robustness scores for all ordered pairs.

        Returns
        -------
        dict
            {(driver, target): {
                'robustness': float,
                'honest': bool,
                'sigma_combined': float,
            }}
        """
        rng = np.random.default_rng(seed)
        var_names = list(data.keys())

        if forcing_proxy is not None:
            working = {}
            for name in var_names:
                working[name] = self.arc_deconfound(
                    data[name], forcing_proxy, self._arc_alpha
                )
        else:
            working = {k: v for k, v in data.items()}

        if sigma_dict is None:
            sigma_dict = {v: 1.0 for v in var_names}

        results = {}
        for v1 in var_names:
            for v2 in var_names:
                if v1 == v2:
                    continue
                sigma_combined = max(sigma_dict.get(v1, 1.0),
                                     sigma_dict.get(v2, 1.0))
                rob = self.robustness(working[v2], working[v1], sigma_combined, rng)
                results[(v1, v2)] = {
                    'robustness': rob,
                    'honest': self.honesty_filter(rob),
                    'sigma_combined': sigma_combined,
                }
        return results
