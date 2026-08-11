"""
ARC (Adaptive Residual Causality) — conservative common-forcing removal.

The core idea: before testing causality between variables X and Y, we regress
out a fraction α of their shared response to an independent common-forcing
proxy (e.g., NGRIP δ¹⁸O). α = 0.5 is the default "conservative" setting.

Reference: Section 3 of the CARC paper.
"""

import numpy as np
from sklearn.linear_model import LinearRegression


def arc_deconfound(target, proxy, alpha=0.5):
    """
    Conservatively strip the common-forcing signal from `target` using `proxy`.

    Parameters
    ----------
    target : np.ndarray (1D)
        The variable to deconfound.
    proxy : np.ndarray (1D)
        The common-forcing proxy (e.g., NGRIP trend). Must be same length as target.
    alpha : float
        Stripping strength. 0 = no removal, 1 = full linear removal.
        0.5 = default conservative setting.

    Returns
    -------
    residual : np.ndarray (same shape as target)
        target - α·ŷ + α·mean(ŷ), where ŷ is the OLS prediction from proxy.
    """
    if proxy is None or len(target) != len(proxy):
        return target

    X = proxy.reshape(-1, 1)
    X_mat = np.column_stack([np.ones(len(X)), X])

    model = LinearRegression()
    model.fit(X_mat, target)
    y_pred = model.predict(X_mat)

    # Conservative stripping: keep α·mean(y_pred) so the mean is preserved
    residual = target - alpha * y_pred + alpha * np.mean(y_pred)
    return residual


def arc_deconfound_batch(data_dict, proxy, alpha=0.5, keep_proxy=None):
    """
    Apply ARC to all variables in a dict. Optionally skip the proxy variable itself.

    Parameters
    ----------
    data_dict : dict of {name: np.ndarray}
        Variable signals.
    proxy : np.ndarray
        Common-forcing proxy signal.
    alpha : float
        Stripping strength.
    keep_proxy : str or None
        If set, this variable is passed through unchanged (identity for the proxy itself).

    Returns
    -------
    residuals : dict of {name: np.ndarray}
    """
    residuals = {}
    for name, signal in data_dict.items():
        if keep_proxy is not None and name == keep_proxy:
            residuals[name] = signal.copy()
        else:
            res = arc_deconfound(signal, proxy, alpha)
            # Re-standardize
            std = np.std(res)
            if std > 1e-10:
                res = (res - np.mean(res)) / std
            residuals[name] = res
    return residuals


def extract_residuals_adaptive(data_dict, forcing_proxy, alpha=0.5):
    """
    Legacy-compatible wrapper around arc_deconfound_batch.

    Parameters
    ----------
    data_dict : dict of {name: np.ndarray}
    forcing_proxy : np.ndarray (1D)
    alpha : float

    Returns
    -------
    residuals : dict of {name: np.ndarray}
    """
    return arc_deconfound_batch(data_dict, forcing_proxy, alpha)
