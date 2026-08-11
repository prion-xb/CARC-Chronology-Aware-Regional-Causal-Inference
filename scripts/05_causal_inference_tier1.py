#!/usr/bin/env python3
"""
05_causal_inference_tier1.py
----------------------------
Causal inference on Tier 1 paleoclimate records (NGRIP, EDC δD, EDC CO₂,
EDC CH₄, Cariaco, Cheng2016) with ARC preprocessing and age-perturbation
ensemble.

Compares: with ARC vs without ARC (baseline).

Output: outputs/pathC_discovered_edges.csv
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from src.config import (DATA_DIR, OUTPUT_DIR, TIER1_FILES, TIER1_SIGMA,
                         AGE_GRID_10_60, DT_HIGH_RES,
                         N_PERTURB, STABILITY_THRESHOLD, MAX_GRANGER_LAG, ARC_ALPHA)
from src.io import load_record, interpolate_to_grid, standardize
from src.arc import arc_deconfound_batch
from src.granger import granger_detect

# ---------------------------------------------------------------------------
# Load and preprocess Tier 1 data
# ---------------------------------------------------------------------------
print("Loading Tier 1 records...")
raw = {}
for name, fname in TIER1_FILES.items():
    age, val = load_record(fname, DATA_DIR)
    if age is not None:
        raw[name] = (age, val)
        print(f"  {name}: {len(age)} pts, {age.min():.1f}–{age.max():.1f} ka")

AGE_GRID = AGE_GRID_10_60
DT = DT_HIGH_RES
var_names = [v for v in ['NGRIP', 'EDC_dD', 'EDC_CO2', 'EDC_CH4', 'Cariaco', 'Cheng2016'] if v in raw]

# Interpolate to common grid
interp_data = {}
for var in var_names:
    age, val = raw[var]
    interp = interpolate_to_grid(age, val, AGE_GRID)
    mask = np.isfinite(interp)
    if np.sum(mask) < 20:
        print(f"  ⚠ {var}: too few valid points, skipping")
        continue
    # Fill NaNs
    valid = interp[mask]
    nan_idx = np.where(~mask)[0]
    for i in nan_idx:
        if i > 0:
            interp[i] = interp[i - 1]
    interp_data[var] = standardize(interp)
    print(f"  {var}: interpolated to {len(AGE_GRID)} pts")

# ARC preprocessing: strip NGRIP common forcing
print("\nARC preprocessing (α=0.5, proxy=NGRIP trend)...")
ngrip_trend = gaussian_filter1d(interp_data['NGRIP'], sigma=1.0 / DT, mode='nearest')
residual_data = arc_deconfound_batch(interp_data, ngrip_trend, alpha=ARC_ALPHA, keep_proxy='NGRIP')

# ---------------------------------------------------------------------------
# Run age-perturbation Granger
# ---------------------------------------------------------------------------
def run_perturbation_granger(data_dict, label):
    """Run the CARC perturbation loop and report detected edges."""
    var_names_list = list(data_dict.keys())
    pairs = [(v1, v2) for i, v1 in enumerate(var_names_list)
             for v2 in var_names_list[i + 1:]]
    edge_counts = {pair: {'v1→v2': 0, 'v2→v1': 0} for pair in pairs}

    print(f"  Running {N_PERTURB} perturbations ({label})...")
    for k in range(N_PERTURB):
        if (k + 1) % 50 == 0:
            print(f"    {label}: {k + 1}/{N_PERTURB}")

        perturbed = {}
        for var in var_names_list:
            sigma = TIER1_SIGMA.get(var, 1.0)
            age_pert = AGE_GRID + np.random.normal(0, sigma, len(AGE_GRID))
            idx = np.argsort(age_pert)
            perturbed[var] = data_dict[var][idx]

        edges = granger_detect(perturbed, maxlag=MAX_GRANGER_LAG)
        for v1, v2 in edges:
            if (v1, v2) in pairs:
                edge_counts[(v1, v2)]['v1→v2'] += 1
            elif (v2, v1) in pairs:
                edge_counts[(v2, v1)]['v2→v1'] += 1

    detected = []
    for (v1, v2), counts in edge_counts.items():
        rate_12 = counts['v1→v2'] / N_PERTURB
        rate_21 = counts['v2→v1'] / N_PERTURB
        if rate_12 >= STABILITY_THRESHOLD:
            detected.append((v1, v2, rate_12))
        if rate_21 >= STABILITY_THRESHOLD:
            detected.append((v2, v1, rate_21))
    return detected


# ---------------------------------------------------------------------------
# Run with ARC and without ARC
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("With ARC preprocessing")
print("=" * 60)
detected_arc = run_perturbation_granger(residual_data, "ARC")

print("\n" + "=" * 60)
print("Without ARC (baseline)")
print("=" * 60)
detected_no_arc = run_perturbation_granger(interp_data, "No-ARC")

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Results Summary")
print("=" * 60)
print(f"With ARC:    {len(detected_arc)} edges detected")
for src, dst, rate in detected_arc:
    print(f"  {src} → {dst}: {rate:.1%}")
print(f"Without ARC: {len(detected_no_arc)} edges detected")
for src, dst, rate in detected_no_arc:
    print(f"  {src} → {dst}: {rate:.1%}")

if detected_arc:
    pd.DataFrame(detected_arc, columns=['Source', 'Target', 'Stability']).to_csv(
        os.path.join(OUTPUT_DIR, 'pathC_discovered_edges.csv'), index=False)
    print(f"\nResults saved to {os.path.join(OUTPUT_DIR, 'pathC_discovered_edges.csv')}")

print("\nDone.")
