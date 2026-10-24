#!/usr/bin/env python3
"""
08_regional_phase_lag.py
------------------------
Phase-lag analysis via cross-correlation of regional stacks across
climate windows, plus sliding-window network coupling analysis.

Requires regional stacking data.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

from src.config import (DATA_DIR, OUTPUT_DIR, AGE_GRID_10_60, DT_HIGH_RES,
                         CLIMATE_WINDOWS, N_PERTURB, ARC_ALPHA, STABILITY_THRESHOLD,
                         MIN_POINTS_IN_EVENT)
from src.io import load_all_records, standardize
from src.stacking import build_region_dict, stack_region
from src.arc import arc_deconfound

AGE_GRID = AGE_GRID_10_60
DT = DT_HIGH_RES
MIN_LAG_KA = 0.2
MAX_LAG_KA = 2.0
WINDOW_WIDTH = 2.0
WINDOW_STEP = 0.1

# Canonical region order — matches 09_regional_robustness.py.
PAPER_REGIONS = ['North_Atlantic', 'Antarctic', 'Tropical',
                 'East_Asian_Monsoon', 'South_America', 'Other_Lakes']

# ---------------------------------------------------------------------------
# Phase lag via cross-correlation
# ---------------------------------------------------------------------------
def compute_phase_lag(x, y, dt, max_lag_ka=MAX_LAG_KA):
    """Estimate the phase lag between x and y via cross-correlation.

    Sign convention (matches synthetic.py Scenario A, where an edge
    driver -> target with lag tau is implemented as
    target(t) += c * driver(t - tau)):

        lag > 0  =>  x leads y (y's response is delayed by |lag| ka behind x)
        lag < 0  =>  y leads x

    Returns
    -------
    lag : float (ka)
    """
    lags_ka = np.arange(-max_lag_ka, max_lag_ka + dt, dt)
    shifts = (lags_ka / dt).astype(int)
    cors = np.full_like(lags_ka, np.nan, dtype=float)
    for i, shift in enumerate(shifts):
        if shift < 0:
            x_s, y_s = x[:shift], y[-shift:]
        elif shift > 0:
            x_s, y_s = x[shift:], y[:-shift]
        else:
            x_s, y_s = x, y
        min_len = min(len(x_s), len(y_s))
        if min_len < 10: continue
        cors[i] = np.corrcoef(x_s[:min_len], y_s[:min_len])[0, 1]
    best = np.nanargmax(cors)
    return -lags_ka[best]


def phase_analysis_window(arc_stacks, region_pairs, window_start, window_end,
                          n_pert, dt=DT, lowpass_sigma=None):
    idx_start = np.searchsorted(AGE_GRID, window_start)
    idx_end = np.searchsorted(AGE_GRID, window_end)
    if idx_end - idx_start < 10:
        return None

    results = {}
    for r1, r2 in region_pairs:
        lags = np.zeros(n_pert)
        for pert in range(n_pert):
            x = arc_stacks[r1][pert, idx_start:idx_end].copy()
            y = arc_stacks[r2][pert, idx_start:idx_end].copy()
            if lowpass_sigma is not None:
                sigma_pts = lowpass_sigma / dt
                x = gaussian_filter1d(x, sigma=sigma_pts, mode='nearest')
                y = gaussian_filter1d(y, sigma=sigma_pts, mode='nearest')
            lags[pert] = compute_phase_lag(x, y, dt)
        results[(r1, r2)] = lags
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading records...")
    records, _ = load_all_records(DATA_DIR, age_min=10.0, age_max=60.0)

    # Merge Southern Ocean records into the Antarctic composite,
    # matching the paper's six composites ("Antarctic/Southern Ocean").
    records = [(i, s, 'Antarctic' if r == 'Southern_Ocean' else r, n)
               for i, s, r, n in records]

    region_dict = build_region_dict(records)
    print(f"Regions: {list(region_dict.keys())}")

    # Stack
    rng = np.random.default_rng(42)
    region_stacks = {}
    for region, recs in region_dict.items():
        region_stacks[region] = stack_region(recs, AGE_GRID, N_PERTURB, rng, smooth_sigma=0.5)

    # ARC
    proxy_region = 'North_Atlantic' if 'North_Atlantic' in region_stacks else list(region_stacks.keys())[0]
    print(f"ARC proxy: {proxy_region}")
    proxy_series = region_stacks[proxy_region]
    arc_stacks = {}
    for region, stack in region_stacks.items():
        if region == proxy_region:
            arc_stacks[region] = stack.copy()
        else:
            residual = np.zeros_like(stack)
            for pert in range(N_PERTURB):
                y, X = stack[pert], proxy_series[pert]
                residual[pert] = arc_deconfound(y, X, ARC_ALPHA)
            arc_stacks[region] = residual
        for i in range(arc_stacks[region].shape[0]):
            arc_stacks[region][i] = standardize(arc_stacks[region][i])

    regions = [r for r in PAPER_REGIONS if r in arc_stacks]
    # All C(6,2) = 15 unordered region pairs, in the canonical region order —
    # same enumeration as 09_regional_robustness.py.
    focus_pairs = [(a, b) for i, a in enumerate(regions)
                   for b in regions[i + 1:]]
    print(f"Region pairs analyzed: {len(focus_pairs)} "
          f"(= C({len(regions)}, 2) of {PAPER_REGIONS})")

    # Phase analysis per climate window
    print("\nPhase-lag analysis (lowpass σ=0.5 ka):")
    print("Sign convention: lag > 0 => r1 leads r2; lag < 0 => r2 leads r1")
    lowpass_sigma = 0.5
    for label, (start, end) in CLIMATE_WINDOWS.items():
        print(f"\n{label} ({start}–{end} ka):")
        phase_results = phase_analysis_window(arc_stacks, focus_pairs, start, end,
                                              N_PERTURB, DT, lowpass_sigma=lowpass_sigma)
        if phase_results is None:
            print("  Window too short, skipped")
            continue
        for r1, r2 in focus_pairs:
            if (r1, r2) not in phase_results:
                continue
            lags = phase_results[(r1, r2)]
            med_lag = np.median(lags)
            lo, hi = np.percentile(lags, [16, 84])
            sign_pos = np.sum(lags > MIN_LAG_KA) / N_PERTURB
            sign_neg = np.sum(lags < -MIN_LAG_KA) / N_PERTURB
            if sign_pos >= STABILITY_THRESHOLD:
                direction = f"{r1} → {r2}"
            elif sign_neg >= STABILITY_THRESHOLD:
                direction = f"{r2} → {r1}"
            else:
                direction = "unstable"
            print(f"  {r1} ↔ {r2}: median lag={med_lag:+.3f} ka "
                  f"(16–84%: {lo:+.3f}…{hi:+.3f}), "
                  f"+={sign_pos:.1%}, -={sign_neg:.1%} → {direction}")

    print("\nDone.")


if __name__ == "__main__":
    main()
