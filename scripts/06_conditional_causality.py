#!/usr/bin/env python3
"""
06_conditional_causality.py
---------------------------
Conditional causality analysis: test causal direction within distinct
climate states (H1, H4, LGM, Bølling-Allerød, GI-12, GS-12).

Output: figures/conditional_causality.png
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from src.config import (DATA_DIR, FIGURE_DIR, TIER1_FILES, TIER1_SIGMA,
                         CLIMATE_WINDOWS, N_PERTURB, ARC_ALPHA, MAX_GRANGER_LAG)
from src.io import load_record
from src.arc import arc_deconfound
from src.granger import granger_detect
from src.visualization import (apply_style, CARC_COLORS, CARC_COLORS_LIGHT,
                                add_panel_label, add_minor_ticks, save_figure)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading Tier 1 records...")
raw = {}
for name, fname in TIER1_FILES.items():
    age, val = load_record(fname, DATA_DIR)
    if age is not None:
        raw[name] = (age, val)
        print(f"  {name}: {len(age)} pts")

var_names = [v for v in ['NGRIP', 'EDC_dD', 'EDC_CO2', 'EDC_CH4', 'Cariaco', 'Cheng2016'] if v in raw]
PAIRS_OF_INTEREST = [('NGRIP', 'EDC_dD'), ('EDC_CO2', 'EDC_dD'), ('Cariaco', 'Cheng2016')]

# ---------------------------------------------------------------------------
# Window-by-window analysis
# ---------------------------------------------------------------------------
def analyze_window(window_start, window_end, label, dt=0.05):
    grid = np.arange(window_start, window_end + dt, dt)
    if len(grid) < 10:
        return None

    # Interpolate all variables to window grid
    interp = {}
    for v in var_names:
        age, val = raw[v]
        f = interp1d(age, val, kind='linear', bounds_error=False, fill_value='extrapolate')
        sig = f(grid)
        mask = np.isfinite(sig)
        if np.sum(mask) < 10:
            return None
        valid = sig[mask]
        nan_idx = np.where(~mask)[0]
        for i in nan_idx:
            j = max(0, i - 1)
            sig[i] = sig[j] if j < len(sig) else valid[-1]
        interp[v] = (sig - np.mean(sig)) / np.std(sig)

    all_pairs = [(v1, v2) for i, v1 in enumerate(var_names) for v2 in var_names[i + 1:]]
    edge_counts = {p: [0, 0] for p in all_pairs}

    for pert in range(N_PERTURB):
        # Age perturbation
        perturbed = {}
        for v in var_names:
            sigma = TIER1_SIGMA.get(v, 1.0)
            age_pert = grid + np.random.normal(0, sigma, len(grid))
            idx = np.argsort(age_pert)
            perturbed[v] = interp[v][idx]

        # ARC: strip NGRIP using library function
        ngrip_trend = gaussian_filter1d(perturbed['NGRIP'], sigma=1.0 / dt, mode='nearest')
        residual = {}
        for v in var_names:
            if v == 'NGRIP':
                residual[v] = perturbed[v]
            else:
                res = arc_deconfound(perturbed[v], ngrip_trend, ARC_ALPHA)
                residual[v] = (res - np.mean(res)) / np.std(res)

        # Granger
        edges = granger_detect(residual, maxlag=MAX_GRANGER_LAG)
        for v1, v2 in edges:
            if (v1, v2) in all_pairs:
                edge_counts[(v1, v2)][0] += 1
            elif (v2, v1) in all_pairs:
                edge_counts[(v2, v1)][1] += 1

    result = {}
    for (v1, v2), (c12, c21) in edge_counts.items():
        result[(v1, v2)] = {'v1→v2': c12 / N_PERTURB, 'v2→v1': c21 / N_PERTURB}
    return result


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print("\nRunning conditional causality analysis...")
all_window_results = {}
for label, (start, end) in CLIMATE_WINDOWS.items():
    print(f"  {label} ({start}–{end} ka)...")
    res = analyze_window(start, end, label)
    if res:
        all_window_results[label] = res

# Report
print("\n" + "=" * 70)
print("Conditional Causality Summary")
print("=" * 70)
for v1, v2 in PAIRS_OF_INTEREST:
    print(f"\nPair: {v1} ↔ {v2}")
    print(f"{'Climate State':<25} {'v1→v2':>8} {'v2→v1':>8} {'Dominant':>12}")
    print("-" * 60)
    for label in CLIMATE_WINDOWS:
        if label in all_window_results and (v1, v2) in all_window_results[label]:
            rates = all_window_results[label][(v1, v2)]
            r12, r21 = rates['v1→v2'], rates['v2→v1']
            dom = f"{v1}→{v2}" if r12 > r21 else f"{v2}→{v1}" if r21 > r12 else "none"
            print(f"{label:<25} {r12:>8.1%} {r21:>8.1%} {dom:>12}")

# Plot
apply_style("paper", font_scale=1.05)

fig, axes = plt.subplots(len(PAIRS_OF_INTEREST), 1, figsize=(14, 4.5 * len(PAIRS_OF_INTEREST)))
if len(PAIRS_OF_INTEREST) == 1:
    axes = [axes]

labels = list(CLIMATE_WINDOWS.keys())
width = 0.32

for idx, (v1, v2) in enumerate(PAIRS_OF_INTEREST):
    ax = axes[idx]
    rates12, rates21, valid_labels = [], [], []
    for label in labels:
        if label in all_window_results and (v1, v2) in all_window_results[label]:
            rates12.append(all_window_results[label][(v1, v2)]['v1→v2'])
            rates21.append(all_window_results[label][(v1, v2)]['v2→v1'])
            valid_labels.append(label)
    x_local = np.arange(len(valid_labels))
    ax.bar(x_local - width / 2, rates12, width, label=f'{v1}→{v2}',
           color=CARC_COLORS["red"], alpha=0.82, edgecolor="#333333", linewidth=0.5)
    ax.bar(x_local + width / 2, rates21, width, label=f'{v2}→{v1}',
           color=CARC_COLORS["blue"], alpha=0.82, edgecolor="#333333", linewidth=0.5)
    ax.set_xticks(x_local)
    ax.set_xticklabels(valid_labels, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('Directional Stability', fontsize=11)
    ax.set_title(f'Conditional Causality: {v1} ↔ {v2}', fontweight='bold', fontsize=12)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.axhline(0.95, color='grey', linestyle='--', alpha=0.6, linewidth=1.2)
    ax.text(len(valid_labels) - 0.5, 0.955, '95% stability', fontsize=7.5,
            color='grey', ha='right', va='bottom')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.2)
    add_minor_ticks(ax)
    add_panel_label(ax, f"({'a' if idx == 0 else 'b' if idx == 1 else 'c'})")

fig.suptitle("CARC Conditional Causality — Directional Stability by Climate State",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
save_figure(fig, "FigS_Conditional_Causality", fmts=("pdf", "png"))
plt.close()
print("Done.")
