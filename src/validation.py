"""
Method evaluation framework: CausalMethodEvaluator and cross-method comparisons.

Provides:
- CausalMethodEvaluator: evaluate TPR/FPR for any causal detection method
- granger_causality(): Standard Granger causality wrapper
- ccm_causality(): Convergent Cross Mapping wrapper
- pcmci_causality(): PCMCI wrapper

Re-exports from figS5_causal_tester_audit.py and paleo_generator.py.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR

# Re-export PaleoDataGenerator
from .paleo_generator import PaleoDataGenerator

# Optional imports
try:
    from skccm import CCM as _CCM
    HAS_CCM = True
except ImportError:
    HAS_CCM = False


def _time_delay_embed(x, E, tau=1):
    """Time-delay embedding for 1D time series → 2D array (n_embed, E)."""
    n = len(x)
    n_embed = n - (E - 1) * tau
    if n_embed <= 0:
        return None
    embedded = np.zeros((n_embed, E))
    for i in range(E):
        embedded[:, i] = x[i * tau : i * tau + n_embed]
    return embedded

try:
    from tigramite import data_processing as _pp
    from tigramite.pcmci import PCMCI as _PCMCI
    try:
        from tigramite.independence_tests.parcorr import ParCorr as _ParCorr
    except ImportError:
        from tigramite.independence_tests import ParCorr as _ParCorr
    HAS_PCMCI = True
except ImportError:
    HAS_PCMCI = False


# ======================================================================
# CausalMethodEvaluator
# ======================================================================
class CausalMethodEvaluator:
    """
    Evaluate a causal detection method on synthetic data with known ground truth.

    Parameters
    ----------
    generator : PaleoDataGenerator
    method : callable
        Function with signature method(data, dt, **kwargs) → list of (src, tgt) edges
    method_name : str
    method_kwargs : dict or None
    """

    def __init__(self, generator, method, method_name, method_kwargs=None):
        self.generator = generator
        self.method = method
        self.method_name = method_name
        self.method_kwargs = method_kwargs if method_kwargs else {}
        self.var_names = generator.var_names
        self.true_edges = generator.causal_edges

        # Build pair lists
        self.all_pairs = []
        for i, v1 in enumerate(self.var_names):
            for v2 in self.var_names[i + 1:]:
                self.all_pairs.append((v1, v2))

        self.true_pairs = set(self.true_edges.keys())
        self.null_pairs = [
            p for p in self.all_pairs
            if p not in self.true_pairs and (p[1], p[0]) not in self.true_pairs
        ]

    def evaluate(self, n_realizations=20, use_age_errors=False):
        """
        Run evaluation over `n_realizations` independent datasets.

        Returns
        -------
        dict with keys: TPR, FPR, TP, FP, N_Null, N_True
        """
        ensemble = self.generator.generate_ensemble(n_realizations)
        tp, fp = 0, 0

        for realization in ensemble:
            data = realization['signals']
            time = realization['time']
            age_errors = realization['age_errors'] if use_age_errors else None

            detected = self.method(
                data, self.generator.dt, time=time, age_errors=age_errors,
                **self.method_kwargs
            )
            for edge in detected:
                if edge in self.true_pairs:
                    tp += 1
                elif (edge[1], edge[0]) not in self.true_pairs:
                    fp += 1

        n_null = len(self.null_pairs) * n_realizations
        n_true = len(self.true_pairs) * n_realizations

        return {
            'TPR': tp / n_true if n_true > 0 else 0.0,
            'FPR': fp / n_null if n_null > 0 else 0.0,
            'TP': tp, 'FP': fp,
            'N_Null': n_null, 'N_True': n_true
        }


# ======================================================================
# Standard method wrappers
# ======================================================================
def granger_causality(data, dt, **kwargs):
    """Standard Granger causality on differenced data with VAR lag selection."""
    df = pd.DataFrame(data)
    non_stat = [col for col in df.columns if adfuller(df[col].dropna())[1] > 0.05]
    df_diff = df.diff().dropna()

    if len(df_diff) < 10:
        return []

    try:
        model = VAR(df_diff)
        lag_order = model.select_order(maxlags=5)
        optimal_lag = lag_order.aic
        if optimal_lag is None:
            return []
    except Exception:
        return []

    detected = []
    var_names = list(data.keys())
    for v1 in var_names:
        for v2 in var_names:
            if v1 == v2:
                continue
            try:
                res = grangercausalitytests(df_diff[[v2, v1]], maxlag=optimal_lag, verbose=False)
                p_vals = [res[i + 1][0]['ssr_chi2test'][1] for i in range(optimal_lag)]
                if min(p_vals) < 0.05:
                    detected.append((v1, v2))
            except Exception:
                pass

    return detected


def ccm_causality(data, dt, **kwargs):
    """CCM (Convergent Cross Mapping) causal detection.

    Uses updated skccm API (scikit-learn-style fit/predict/score).
    """
    if not HAS_CCM:
        return []

    E = kwargs.get('E', 3)
    tau = kwargs.get('tau', 1)
    threshold = kwargs.get('threshold', 0.3)

    detected = []
    var_names = list(data.keys())
    for i, v1 in enumerate(var_names):
        for v2 in var_names[i + 1:]:
            try:
                # Time-delay embedding
                X1_embed = _time_delay_embed(data[v1], E, tau)
                X2_embed = _time_delay_embed(data[v2], E, tau)
                if X1_embed is None or X2_embed is None:
                    continue
                if len(X1_embed) < 20:
                    continue

                # New skccm API: fit → predict → score
                ccm = _CCM(weights='exp', score_metric='corrcoef')
                ccm.fit(X1_embed, X2_embed)
                lib_lengths = np.arange(min(20, len(X1_embed) - 2),
                                       len(X1_embed) - 1,
                                       max(10, (len(X1_embed) - 20) // 8))
                if len(lib_lengths) < 2:
                    lib_lengths = np.array([len(X1_embed) // 3, 2 * len(X1_embed) // 3, len(X1_embed) - 1])
                ccm.predict(X1_embed, X2_embed, lib_lengths)
                s1, s2 = ccm.score()

                # s1: cross-mapping skill from v2 → v1 (using v2's manifold to predict v1)
                # s2: cross-mapping skill from v1 → v2
                rho_1_to_2 = float(s2[-1])  # v1 cross-maps v2 (v1→v2 direction)
                rho_2_to_1 = float(s1[-1])  # v2 cross-maps v1 (v2→v1 direction)

                # If v1 causes v2, then v2 should contain information about v1,
                # so v2→v1 cross-mapping (rho_2_to_1) should be stronger.
                if rho_2_to_1 > rho_1_to_2 and rho_2_to_1 > threshold:
                    detected.append((v1, v2))
                elif rho_1_to_2 > rho_2_to_1 and rho_1_to_2 > threshold:
                    detected.append((v2, v1))
            except Exception:
                pass

    return detected


def pcmci_causality(data, dt, **kwargs):
    """PCMCI causal discovery."""
    if not HAS_PCMCI:
        return []

    var_names = list(data.keys())
    try:
        dataframe = _pp.DataFrame(
            np.column_stack([data[v] for v in var_names]),
            var_names=var_names
        )
        parcorr = _ParCorr(significance='analytic')
        pcmci = _PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)
        results = pcmci.run_pcmci(tau_max=5, pc_alpha=0.05)
        graph = results['graph']

        detected = []
        for i, v1 in enumerate(var_names):
            for j, v2 in enumerate(var_names):
                if i != j and graph[i, j] == '-->':
                    detected.append((v1, v2))
        return detected
    except Exception:
        return []
