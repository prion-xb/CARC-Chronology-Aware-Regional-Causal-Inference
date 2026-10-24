"""
Synthetic data generation for method validation.

Provides:
- generate_scenario_A(): 6-variable event-scale (10–60 ka) synthetic paleoclimate data
- generate_null_data(): Null-causality pairs with controllable common forcing
- PaleoDataGenerator: Class-based generator (imported from paleo_generator.py)
"""

import numpy as np
from scipy.interpolate import interp1d

# Re-export the class-based generator
try:
    from .paleo_generator import PaleoDataGenerator
except ImportError:
    PaleoDataGenerator = None


# ---------------------------------------------------------------------------
# Scenario A: event-scale synthetic data (10–60 ka, DT=0.05 ka)
# ---------------------------------------------------------------------------
SCENARIO_A_CONFIG = {
    't_start': 10.0, 't_end': 60.0, 'dt': 0.05,
    'phi_z': 0.6, 'sigma_z': 0.8,
    'n_pulses': 25, 'pulse_width': 0.4, 'pulse_decay': 1.2, 'pulse_amplitude': 1.0,
    'orbital_amp': 0.5, 'orbital_period': 40.0,
    # Per-record parameters (data-grounded, scripts/22_scenarioA_final.py):
    #   sigma_eps = innovation std of the marginal AR(1) fitted to the
    #     corresponding Tier-1 real record (10-60 ka, 50-yr grid);
    #   phi = CALIBRATED generator AR coefficient whose generated marginal
    #     AR(1) matches the fitted one under the common forcing (bisection,
    #     primary loading beta = 0.25);
    #   beta = common-forcing loading (unidentifiable from data by premise,
    #     swept over {0.15, 0.25, 0.35}; 0.25 = primary value);
    #   age_err = published 1-sigma age-model uncertainty (src.config.TIER1_SIGMA).
    'variables': {
        'NGRIP':   {'phi': 0.6388, 'beta': 0.25, 'sigma_eps': 0.7015, 'age_err': 0.65},
        'EDC_dD':  {'phi': 0.8515, 'beta': 0.25, 'sigma_eps': 0.2988, 'age_err': 0.83},
        'CO2':     {'phi': 0.9355, 'beta': 0.25, 'sigma_eps': 0.0707, 'age_err': 1.65},
        'CH4':     {'phi': 0.9101, 'beta': 0.25, 'sigma_eps': 0.1114, 'age_err': 1.66},
        'Cariaco': {'phi': 0.8345, 'beta': 0.25, 'sigma_eps': 0.3771, 'age_err': 0.23},
        'Cheng':   {'phi': 0.8409, 'beta': 0.25, 'sigma_eps': 0.3575, 'age_err': 0.14}
    },
    'causal_edges': [
        ('NGRIP', 'EDC_dD', 0.8),
        ('CO2', 'CH4', 1.5),
        ('Cariaco', 'Cheng', 2.3)
    ],
    'causal_strength': 0.4
}


def _generate_do_pulses(t_grid, n_pulses, width, decay, amplitude=1.0, seed=None):
    """Generate D/O-type Gaussian pulses with exponential amplitude decay."""
    if seed is not None:
        np.random.seed(seed)
    t_min, t_max = t_grid[0], t_grid[-1]
    positions = np.random.uniform(t_min + 2 * width, t_max - 2 * width, n_pulses)
    pulses = np.zeros_like(t_grid)
    sigma_fwhm = width / (2 * np.sqrt(2 * np.log(2)))
    for pos in positions:
        amp = amplitude * np.exp(-(pos - t_min) / decay)
        pulses += amp * np.exp(-0.5 * ((t_grid - pos) / sigma_fwhm) ** 2)
    return pulses


def generate_scenario_A(seed=None, include_orbital=True, include_pulses=True,
                        include_age_errors=True, return_Z=False):
    """
    Generate synthetic paleoclimate data for Scenario A.

    Parameters
    ----------
    seed : int or None
    include_orbital : bool
    include_pulses : bool
    include_age_errors : bool
    return_Z : bool
        If True, additionally return the common forcing Z (for ARC probes).

    Returns
    -------
    data : dict of {name: np.ndarray}
    causal_edges : list of (driver, target, lag_ka)
    time_grid : np.ndarray
    Z : np.ndarray (only if return_Z=True)
    """
    cfg = SCENARIO_A_CONFIG
    if seed is not None:
        np.random.seed(seed)

    t_grid = np.arange(cfg['t_start'], cfg['t_end'] + cfg['dt'] / 2, cfg['dt'])
    N = len(t_grid)

    # 1. Common forcing Z_t
    Z_ar = np.zeros(N)
    Z_ar[0] = np.random.normal(0, cfg['sigma_z'])
    for t in range(1, N):
        Z_ar[t] = cfg['phi_z'] * Z_ar[t - 1] + \
                  np.random.normal(0, cfg['sigma_z'] * np.sqrt(1 - cfg['phi_z'] ** 2))

    Z = Z_ar.copy()

    if include_orbital:
        orbital = cfg['orbital_amp'] * np.sin(
            2 * np.pi * (t_grid - cfg['t_start']) / cfg['orbital_period'])
        Z += orbital

    if include_pulses:
        # NOTE: no seed passed — pulse positions are drawn from the current
        # RNG stream so the realization is bit-identical to the canonical
        # generator in scripts/22_scenarioA_final.py.
        pulses = _generate_do_pulses(t_grid, cfg['n_pulses'],
                                     width=cfg['pulse_width'],
                                     decay=cfg['pulse_decay'],
                                     amplitude=cfg['pulse_amplitude'])
        Z += pulses

    Z = (Z - np.mean(Z)) / np.std(Z)

    # 2. Generate each variable
    data = {}
    for name, params in cfg['variables'].items():
        phi, beta, sigma_eps = params['phi'], params['beta'], params['sigma_eps']
        x = np.zeros(N)
        x[0] = np.random.normal(0, 1)
        for t in range(1, N):
            x[t] = phi * x[t - 1] + beta * Z[t] + np.random.normal(0, sigma_eps)
        data[name] = (x - np.mean(x)) / np.std(x)

    # 3. Impose causal lags
    for driver, target, lag_ka in cfg['causal_edges']:
        lag_step = int(round(lag_ka / cfg['dt']))
        if 0 < lag_step < N:
            causal_signal = np.zeros(N)
            causal_signal[lag_step:] = data[driver][:-lag_step]
            data[target] = data[target] + cfg['causal_strength'] * causal_signal
            data[target] = (data[target] - np.mean(data[target])) / np.std(data[target])

    # 4. Age uncertainties
    if include_age_errors:
        for name, params in cfg['variables'].items():
            age_err_std = params['age_err']
            if age_err_std <= 0:
                continue
            perturbed = t_grid + np.random.normal(0, age_err_std, size=N)
            idx = np.argsort(perturbed)
            f = interp1d(perturbed[idx], data[name][idx],
                         kind='linear', fill_value='extrapolate', bounds_error=False)
            new_vals = f(t_grid)
            valid = ~np.isnan(new_vals)
            if valid.sum() > N / 2:
                data[name] = new_vals
            data[name] = (data[name] - np.mean(data[name])) / np.std(data[name])

    if return_Z:
        return data, cfg['causal_edges'], t_grid, Z
    return data, cfg['causal_edges'], t_grid


# ---------------------------------------------------------------------------
# Null-causality generator for FPR experiments
# ---------------------------------------------------------------------------
def generate_null_data(n=500, phi=0.85, beta=0.4, sigma_z=1.0, sigma_eps=0.8, seed=None):
    """
    Generate X, Y with no direct causality, linked only through common forcing Z.

    Parameters
    ----------
    n : int — time series length
    phi : float — AR(1) coefficient
    beta : float — common forcing strength
    sigma_z : float — forcing innovation std
    sigma_eps : float — noise std
    seed : int or None

    Returns
    -------
    X, Y, Z : np.ndarray (each length n)
    """
    if seed is not None:
        np.random.seed(seed)

    z = np.zeros(n)
    z[0] = np.random.normal(0, sigma_z)
    for t in range(1, n):
        z[t] = phi * z[t - 1] + np.random.normal(0, sigma_z * np.sqrt(1 - phi ** 2))

    eps_x = np.random.normal(0, sigma_eps, n)
    eps_y = np.random.normal(0, sigma_eps, n)

    X = np.zeros(n)
    Y = np.zeros(n)
    X[0] = np.random.normal(0, 1)
    Y[0] = np.random.normal(0, 1)

    for t in range(1, n):
        X[t] = phi * X[t - 1] + beta * z[t] + eps_x[t]
        Y[t] = phi * Y[t - 1] + beta * z[t] + eps_y[t]

    return X, Y, z
