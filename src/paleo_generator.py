"""
Paleoclimate synthetic data generator.

Generates synthetic datasets with known ground-truth causal structure and
typical paleoclimate statistical properties (high autocorrelation, common
forcing, D/O-like pulses, age-model errors).

Supports three scenarios:
  - Scenario A: event–millennial scale (10–60 ka, 50-yr resolution)
  - Scenario B: orbital scale (0–800 ka, 1-kyr resolution)
  - Scenario C: long-term state transitions (0–2000 ka)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from collections import defaultdict


class PaleoDataGenerator:
    """Generate synthetic paleoclimate datasets with known causal graphs.

    Used for benchmarking causal detection methods (TPR, FPR) under
    controlled conditions with configurable age errors, common forcing,
    and autocorrelation structures.

    Parameters
    ----------
    scenario : str
        'A' = event/millennial scale, 'B' = orbital scale, 'C' = state transitions.
    random_seed : int or None
        Seed for reproducibility.
    """

    def __init__(self, scenario='A', random_seed=None):
        self.scenario = scenario
        self.rng = np.random.default_rng(random_seed)
        self.dt = None
        self.n_points = None
        self.var_names = None

        if scenario == 'A':
            self._init_scenario_A()
        elif scenario == 'B':
            self._init_scenario_B()
        elif scenario == 'C':
            self._init_scenario_C()
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

    # ========================
    # Scenario initialization
    # ========================
    def _init_scenario_A(self):
        """Scenario A: event–millennial scale, high-resolution 10–60 ka records.

        Parameters match :data:`src.synthetic.SCENARIO_A_CONFIG`, which is the
        canonical source used for the paper's synthetic validation (Table S1).
        """
        self.dt = 0.05            # ka → 50-year resolution
        self.total_time = 65.0    # generate 65 ka, trim boundary effects → 50 ka
        self.n_points = int(self.total_time / self.dt)

        self.var_names = ['NGRIP', 'EDC_dD', 'CO2', 'CH4', 'Cariaco', 'Cheng']

        # Autoregressive coefficients (Table S1, φ ∈ [0.72, 0.90])
        self.phi = {
            'NGRIP': 0.82, 'EDC_dD': 0.80, 'CO2': 0.90,
            'CH4': 0.88, 'Cariaco': 0.72, 'Cheng': 0.75
        }

        self.noise_std = {
            'NGRIP': 0.45, 'EDC_dD': 0.55, 'CO2': 0.80,
            'CH4': 0.70, 'Cariaco': 0.30, 'Cheng': 0.25
        }

        # Common forcing loading (Table S1, β)
        self.forcing_weights = {
            'NGRIP': 0.35, 'EDC_dD': 0.35, 'CO2': 0.25,
            'CH4': 0.25, 'Cariaco': 0.40, 'Cheng': 0.40
        }

        # Ground-truth causal graph: (driver, responder) → lag (ka)
        self.causal_edges = {
            ('NGRIP', 'EDC_dD'): 0.8,
            ('CO2', 'CH4'): 1.5,
            ('Cariaco', 'Cheng'): 2.3
        }
        self.causal_strength = {edge: 0.40 for edge in self.causal_edges}

        # Age error std (1σ, ka) per variable (Table S1)
        self.age_error_std = {
            'NGRIP': 0.65, 'EDC_dD': 0.83, 'CO2': 1.65,
            'CH4': 1.66, 'Cariaco': 0.23, 'Cheng': 0.14
        }

    def _init_scenario_B(self):
        """Scenario B: orbital scale, low-resolution 0–800 ka records."""
        self.dt = 1.0    # 1-kyr resolution
        self.total_time = 805.0
        self.n_points = int(self.total_time / self.dt)

        self.var_names = ['d18O', 'CO2', 'CH4', 'Temp']

        # Very strong autocorrelation (ice-sheet inertia)
        self.phi = {'d18O': 0.97, 'CO2': 0.96, 'CH4': 0.95, 'Temp': 0.96}
        self.noise_std = {'d18O': 0.15, 'CO2': 0.12, 'CH4': 0.18, 'Temp': 0.15}

        # Weak common forcing (orbital forcing modeled as direct driver)
        self.forcing_weights = {v: 0.05 for v in self.var_names}

        # Longer lags, higher SNR at orbital scale
        self.causal_edges = {
            ('d18O', 'CO2'): 5.0,
            ('CO2', 'CH4'): 1.5,
            ('d18O', 'Temp'): 3.0
        }
        self.causal_strength = {edge: 0.50 for edge in self.causal_edges}

        self.age_error_std = {'d18O': 2.0, 'CO2': 4.0, 'CH4': 4.0, 'Temp': 2.0}

    def _init_scenario_C(self):
        """Scenario C: long-term climate state transitions, 0–2000 ka."""
        self.dt = 2.0
        self.total_time = 2010.0
        self.n_points = int(self.total_time / self.dt)

        self.var_names = ['d18O', 'd13C', 'SST']
        self.phi = {v: 0.92 for v in self.var_names}
        self.noise_std = {v: 0.18 for v in self.var_names}
        self.forcing_weights = {v: 0.10 for v in self.var_names}

        # Scenario C uses state-dependent correlations, not fixed lags
        self.causal_edges = {}
        self.causal_strength = {}

        self.age_error_std = {'d18O': 10.0, 'd13C': 10.0, 'SST': 10.0}

    # ========================
    # Core generation
    # ========================
    def _generate_ar1(self, n, phi, noise_std):
        """Generate AR(1) process: x_t = phi * x_{t-1} + ε_t."""
        x = np.zeros(n)
        noise = self.rng.normal(0, noise_std, n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + noise[t]
        return x

    def _generate_common_forcing(self, t):
        """Generate common external forcing with orbital trends and D/O-like pulses."""
        forcing = np.zeros_like(t)

        # 1. Low-frequency orbital trend (100-kyr cycle)
        forcing += 0.5 * np.sin(2 * np.pi * t / 100.0)

        # 2. D/O-type pulse events (scenario-dependent)
        if self.scenario == 'A':
            n_do_events = 25
            do_start_times = self.rng.uniform(5, 55, n_do_events)
            for do_time in do_start_times:
                # Asymmetric Gaussian: rapid warming, slow cooling
                pulse = np.exp(-0.5 * ((t - do_time) / 0.4) ** 2)
                pulse[t < do_time] = 0
                decay = np.exp(-(t - do_time) / 1.2)
                decay[t < do_time] = 0
                forcing += 1.5 * pulse + 0.6 * decay
        elif self.scenario == 'B':
            # Realistic Milankovitch period combination
            forcing += 0.8 * np.sin(2 * np.pi * t / 41.0)   # obliquity
            forcing += 0.6 * np.sin(2 * np.pi * t / 23.0)   # precession
            forcing += 0.4 * np.sin(2 * np.pi * t / 100.0)  # eccentricity

        return forcing

    # ========================
    # Main data generation
    # ========================
    def generate_dataset(self):
        """Generate a complete synthetic dataset.

        Returns
        -------
        t_valid : np.ndarray
            Valid time grid (boundary effects trimmed).
        signals_valid : dict of {name: np.ndarray}
            Standardized variable signals.
        forcing_valid : np.ndarray
            Common forcing signal on the valid time grid.
        """
        t = np.arange(0, self.total_time, self.dt)
        n = len(t)
        common_forcing = self._generate_common_forcing(t)

        # 1. Generate AR(1) baseline + noise for each variable
        baselines = {}
        for var in self.var_names:
            baselines[var] = self._generate_ar1(n, self.phi[var], self.noise_std[var])

        # 2. Build signals (before causal lags)
        signals = {}
        for var in self.var_names:
            signal = baselines[var].copy()
            signal += self.forcing_weights[var] * common_forcing
            signals[var] = signal

        # 3. Impose causal lags following topological order
        processed = set()
        drivers = set(edge[0] for edge in self.causal_edges)
        responders = set(edge[1] for edge in self.causal_edges)
        start_nodes = drivers - responders

        queue = list(start_nodes) + [
            v for v in self.var_names
            if v not in start_nodes and v not in responders
        ]

        while queue:
            var = queue.pop(0)
            if var in processed:
                continue

            dependencies = [d for d, r in self.causal_edges if r == var]
            if not all(d in processed for d in dependencies):
                queue.append(var)
                continue

            for driver in dependencies:
                lag_ka = self.causal_edges[(driver, var)]
                lag_steps = int(lag_ka / self.dt)
                strength = self.causal_strength[(driver, var)]

                if lag_steps > 0 and lag_steps < n:
                    rolled = np.zeros_like(signals[var])
                    rolled[lag_steps:] = signals[driver][:-lag_steps]
                    signals[var] += strength * rolled

            processed.add(var)

        # Handle pure responders not reached in the queue
        for var in self.var_names:
            if var not in processed:
                dependencies = [d for d, r in self.causal_edges if r == var]
                for driver in dependencies:
                    if driver in processed:
                        lag_ka = self.causal_edges[(driver, var)]
                        lag_steps = int(lag_ka / self.dt)
                        strength = self.causal_strength[(driver, var)]
                        if 0 < lag_steps < n:
                            rolled = np.zeros_like(signals[var])
                            rolled[lag_steps:] = signals[driver][:-lag_steps]
                            signals[var] += strength * rolled
                processed.add(var)

        # Standardize
        for var in self.var_names:
            mean = np.mean(signals[var])
            std = np.std(signals[var])
            if std > 0:
                signals[var] = (signals[var] - mean) / std

        # Trim boundary effects
        if self.scenario == 'A':
            mask = (t >= 10.0) & (t <= 60.0)
        elif self.scenario == 'B':
            mask = (t >= 5.0) & (t <= 800.0)
        else:
            mask = (t >= 10.0) & (t <= 2000.0)

        t_valid = t[mask]
        signals_valid = {var: sig[mask] for var, sig in signals.items()}
        forcing_valid = common_forcing[mask]

        return t_valid, signals_valid, forcing_valid

    def inject_age_errors(self, t, signals):
        """Perturb each variable's time axis by its age-model uncertainty.

        Returns perturbed age labels per variable for downstream Monte Carlo.
        """
        perturbed = {}
        for var in self.var_names:
            sigma = self.age_error_std[var]
            perturbed[var] = t + self.rng.normal(0, sigma, len(t))
        return perturbed

    def generate_ensemble(self, n_realizations=100):
        """Generate multiple independent realizations for statistical evaluation.

        Returns
        -------
        ensemble : list of dict
            Each dict contains: time, signals, common_forcing, age_errors,
            causal_edges, causal_strength.
        """
        ensemble = []
        for i in range(n_realizations):
            t, signals, forcing = self.generate_dataset()
            age_errors = self.inject_age_errors(t, signals)

            ensemble.append({
                'time': t,
                'signals': signals,
                'common_forcing': forcing,
                'age_errors': age_errors,
                'causal_edges': self.causal_edges,
                'causal_strength': self.causal_strength
            })
        return ensemble
