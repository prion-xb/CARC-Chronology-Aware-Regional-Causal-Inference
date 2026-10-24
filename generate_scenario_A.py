# -*- coding: utf-8 -*-
"""
generate_scenario_A.py
Synthetic data generator for Scenario A (paleoclimate event scale)
Includes: orbital trend, D/O-type pulses, AR(1) common forcing,
          per-variable AR(1) dynamics with loading, causal lags,
          and age uncertainties.
"""

import numpy as np
from scipy.interpolate import interp1d

# ------------------------------------------------------------
# Scenario A parameters (based on Table S4 and main text)
# ------------------------------------------------------------
SCENARIO_A = {
    # Time axis
    't_start': 10.0,       # ka
    't_end': 60.0,         # ka
    'dt': 0.05,            # ka (50-year resolution)

    # Common forcing Z_t: AR(1) + D/O pulses
    'phi_z': 0.6,          # AR(1) coefficient of Z
    'sigma_z': 0.8,        # innovation std of Z
    'n_pulses': 25,        # number of D/O-type pulses
    'pulse_width': 0.4,    # ka (width at half max)
    'pulse_decay': 1.2,    # ka (exponential decay time)
    'pulse_amplitude': 1.0,  # amplitude scaling

    # Orbital trend (sinusoidal, period ~40 ka)
    'orbital_amp': 0.5,
    'orbital_period': 40.0,  # ka

    # Variables: names, AR(1) phi, common forcing loading beta,
    #            noise sigma_eps, age error 1sigma (ka).
    # Data-grounded parameters (scripts/22_scenarioA_final.py): sigma_eps from
    # the marginal AR(1) fitted to each Tier-1 real record (10-60 ka, 50-yr
    # grid); phi is the CALIBRATED generator coefficient whose generated
    # marginal AR(1) matches the fitted one under the common forcing
    # (bisection, primary loading beta = 0.25); age_err = published 1-sigma
    # age-model uncertainties.
    'variables': {
        'NGRIP':      {'phi': 0.6388, 'beta': 0.25, 'sigma_eps': 0.7015, 'age_err': 0.65},
        'EDC_dD':     {'phi': 0.8515, 'beta': 0.25, 'sigma_eps': 0.2988, 'age_err': 0.83},
        'CO2':        {'phi': 0.9355, 'beta': 0.25, 'sigma_eps': 0.0707, 'age_err': 1.65},
        'CH4':        {'phi': 0.9101, 'beta': 0.25, 'sigma_eps': 0.1114, 'age_err': 1.66},
        'Cariaco':    {'phi': 0.8345, 'beta': 0.25, 'sigma_eps': 0.3771, 'age_err': 0.23},
        'Cheng':      {'phi': 0.8409, 'beta': 0.25, 'sigma_eps': 0.3575, 'age_err': 0.14}
    },

    # True causal edges (driver, target, lag in ka)
    'causal_edges': [
        ('NGRIP',   'EDC_dD',   0.8),
        ('CO2',     'CH4',      1.5),
        ('Cariaco', 'Cheng',    2.3)
    ],

    # Causal strength (coefficient multiplying the lagged driver)
    'causal_strength': 0.4
}

# ------------------------------------------------------------
# Helper: generate D/O-type pulses (random positions, Gaussian shape)
# ------------------------------------------------------------
def generate_do_pulses(t_grid, n_pulses, width, decay, amplitude=1.0, seed=None):
    """
    Generate a series of Gaussian pulses with exponential decay in amplitude?
    Actually the paper says "Gaussian pulses (width 0.4 ka, decay 1.2 ka)".
    We interpret as: each pulse is a Gaussian with sigma = width/2.355,
    and the pulse amplitude decays exponentially with time constant decay.
    Or simpler: create a random sequence of pulse positions and add Gaussian shapes.
    """
    if seed is not None:
        np.random.seed(seed)
    t_min, t_max = t_grid[0], t_grid[-1]
    # Random positions (avoid edges)
    positions = np.random.uniform(t_min + 2*width, t_max - 2*width, n_pulses)
    pulses = np.zeros_like(t_grid)
    for pos in positions:
        # Gaussian shape: exp(-0.5 * ((t-pos)/sigma)**2)
        sigma = width / (2 * np.sqrt(2 * np.log(2)))  # FWHM to sigma
        # Amplitude decay? Use decay time as exponential factor:
        # amplitude = amplitude * exp(-(pos - t_min)/decay)
        amp = amplitude * np.exp(-(pos - t_min) / decay)
        pulses += amp * np.exp(-0.5 * ((t_grid - pos) / sigma)**2)
    return pulses

# ------------------------------------------------------------
# Core data generator
# ------------------------------------------------------------
def generate_scenario_A(seed=None, include_orbital=True, include_pulses=True, include_age_errors=True):
    """
    Generate synthetic data for Scenario A.

    Parameters
    ----------
    seed : int, optional
        Random seed.
    include_orbital : bool
        Add orbital trend to common forcing.
    include_pulses : bool
        Add D/O-type pulses to common forcing.
    include_age_errors : bool
        Apply age uncertainties (Gaussian perturbation + linear interpolation).

    Returns
    -------
    data : dict of numpy arrays (each length N_time)
        Keys are variable names.
    causal_edges : list of tuples (driver, target, lag_ka)
    time_grid : 1D array of ages (ka)
    """
    cfg = SCENARIO_A
    if seed is not None:
        np.random.seed(seed)

    # Time grid
    t_grid = np.arange(cfg['t_start'], cfg['t_end'] + cfg['dt']/2, cfg['dt'])
    N = len(t_grid)

    # --------------------------------------------------------
    # 1. Build common forcing Z_t
    # --------------------------------------------------------
    # AR(1) base
    Z_ar = np.zeros(N)
    Z_ar[0] = np.random.normal(0, cfg['sigma_z'])
    for t in range(1, N):
        Z_ar[t] = cfg['phi_z'] * Z_ar[t-1] + np.random.normal(0, cfg['sigma_z'] * np.sqrt(1 - cfg['phi_z']**2))

    Z = Z_ar.copy()

    # Add orbital trend (sinusoidal)
    if include_orbital:
        orbital = cfg['orbital_amp'] * np.sin(2 * np.pi * (t_grid - cfg['t_start']) / cfg['orbital_period'])
        Z += orbital

    # Add D/O-type pulses
    if include_pulses:
        # NOTE: no seed passed — pulse positions are drawn from the current
        # RNG stream so the realization is bit-identical to the canonical
        # generator in scripts/22_scenarioA_final.py.
        pulses = generate_do_pulses(t_grid, cfg['n_pulses'],
                                    width=cfg['pulse_width'],
                                    decay=cfg['pulse_decay'],
                                    amplitude=cfg['pulse_amplitude'])
        Z += pulses

    # Standardize Z to zero mean, unit variance (optional, keep scale consistent)
    Z = (Z - np.mean(Z)) / np.std(Z)

    # --------------------------------------------------------
    # 2. Generate each variable
    # --------------------------------------------------------
    data = {}
    for name, params in cfg['variables'].items():
        phi = params['phi']
        beta = params['beta']
        sigma_eps = params['sigma_eps']

        x = np.zeros(N)
        x[0] = np.random.normal(0, 1)
        for t in range(1, N):
            x[t] = phi * x[t-1] + beta * Z[t] + np.random.normal(0, sigma_eps)

        # Standardize (zero mean, unit variance)
        x = (x - np.mean(x)) / np.std(x)
        data[name] = x

    # --------------------------------------------------------
    # 3. Impose causal lags
    # --------------------------------------------------------
    lag_steps = {}
    for driver, target, lag_ka in cfg['causal_edges']:
        lag_step = int(round(lag_ka / cfg['dt']))
        lag_steps[(driver, target)] = lag_step
        if lag_step > 0 and lag_step < N:
            # Add causal influence: strength * lagged driver
            causal_signal = np.zeros(N)
            causal_signal[lag_step:] = data[driver][:-lag_step]
            data[target] = data[target] + cfg['causal_strength'] * causal_signal
            # Re-standardize
            data[target] = (data[target] - np.mean(data[target])) / np.std(data[target])

    # --------------------------------------------------------
    # 4. Age uncertainties (if requested)
    # --------------------------------------------------------
    if include_age_errors:
        for name, params in cfg['variables'].items():
            age_err_std = params['age_err']  # in ka
            if age_err_std <= 0:
                continue
            # Perturb ages
            orig_time = t_grid
            perturbed = orig_time + np.random.normal(0, age_err_std, size=N)
            # Sort by perturbed time to keep monotonicity
            idx = np.argsort(perturbed)
            perturbed_sorted = perturbed[idx]
            values_sorted = data[name][idx]
            # Interpolate back to original time grid
            f = interp1d(perturbed_sorted, values_sorted,
                         kind='linear', fill_value='extrapolate', bounds_error=False)
            new_vals = f(orig_time)
            # Replace NaNs (if any) with original values
            valid = ~np.isnan(new_vals)
            if valid.sum() > N/2:
                data[name] = new_vals
            # re-standardize after interpolation
            data[name] = (data[name] - np.mean(data[name])) / np.std(data[name])

    return data, cfg['causal_edges'], t_grid


# ------------------------------------------------------------
# Example usage (if run as script)
# ------------------------------------------------------------
if __name__ == '__main__':
    # Generate one realization
    data, edges, t = generate_scenario_A(seed=42, include_orbital=True, include_pulses=True, include_age_errors=True)
    print("Generated variables:", list(data.keys()))
    print("Causal edges:", edges)
    print("Time range: {:.2f} to {:.2f} ka, {} points".format(t[0], t[-1], len(t)))
    for name, series in data.items():
        print(f"{name}: mean={series.mean():.3f}, std={series.std():.3f}")