#!/usr/bin/env python3
"""
09_regional_robustness.py
-------------------------
Directional robustness analysis for regional pairs (0-800 ka).
Produces grouped bar charts and asymmetry scatter plots.

Output: figures/Figure_S7_enhanced.png
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from src.config import (DATA_DIR, OUTPUT_DIR, FIGURE_DIR, N_PERTURB,
                         ARC_ALPHA, MAX_GRANGER_LAG, NOISE_STD)
from src.arc import arc_deconfound
from src.granger import robustness, granger_test_pair
from src.visualization import (apply_style, CARC_COLORS, CARC_COLORS_LIGHT,
                                add_panel_label, add_minor_ticks, save_figure)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
STACKED_DIR = DATA_DIR  # override if regional stack files are elsewhere
REGION_PAIRS = [
    ('North_Atlantic', 'Antarctic'),
    ('North_Atlantic', 'Tropical_Marine'),
    ('Antarctic', 'South_American_Monsoon'),
    ('North_Atlantic', 'South_American_Monsoon')
]
WINDOWS = {'0-800 ka': (0, 800)}


# ---------------------------------------------------------------------------
# Data loading (from stacked CSV files or proxy records)
# ---------------------------------------------------------------------------
def load_stacked(filepath):
    """Load a stacked regional CSV file."""
    df = pd.read_csv(filepath, header=None)
    if df.shape[1] >= 3:
        age = df.iloc[:, 0].astype(float).values
        val = df.iloc[:, 1].astype(float).values
        err = df.iloc[:, 2].astype(float).values
        valid = ~(np.isnan(age) | np.isnan(val))
        return age[valid], val[valid], err[valid]
    return np.array([]), np.array([]), np.array([])


def main():
    # Try loading stacked data; fall back to computing on-the-fly
    region_data = {}
    all_regions = set()
    for a, b in REGION_PAIRS:
        all_regions.add(a); all_regions.add(b)

    for region in all_regions:
        f = os.path.join(STACKED_DIR, f"{region}_stacked.csv")
        if os.path.exists(f):
            age, val, err = load_stacked(f)
            if len(age) > 10:
                region_data[region] = {'age': age, 'value': val, '1sigma': err}
                print(f"Loaded {region}: {len(age)} pts")

    if len(region_data) < 2:
        print("Not enough stacked data. Run regional stacking first.")
        print(f"Searched in: {STACKED_DIR}")
        return

    # Compute robustness for each pair
    results = {}
    for win_name, (tmin, tmax) in WINDOWS.items():
        print(f"\nWindow: {win_name}")
        for regA, regB in REGION_PAIRS:
            if regA not in region_data or regB not in region_data:
                continue
            dA = region_data[regA]; dB = region_data[regB]
            dfA = pd.DataFrame({'age': dA['age'], 'valueA': dA['value'], 'sigmaA': dA['1sigma']})
            dfB = pd.DataFrame({'age': dB['age'], 'valueB': dB['value'], 'sigmaB': dB['1sigma']})
            merged = pd.merge(dfA, dfB, on='age', how='inner')
            merged = merged[(merged['age'] >= tmin) & (merged['age'] <= tmax)]
            if len(merged) < 10:
                print(f"  Skip {regA} vs {regB}: too few points")
                continue

            A = merged['valueA'].values; B = merged['valueB'].values
            sigma_err = np.sqrt(merged['sigmaA'].median() ** 2 + merged['sigmaB'].median() ** 2) / 2

            rob_AB = robustness(B, A, sigma_err, n_perturb=N_PERTURB)
            rob_BA = robustness(A, B, sigma_err, n_perturb=N_PERTURB)
            results[(f"{regA}→{regB}", win_name)] = rob_AB
            results[(f"{regB}→{regA}", win_name)] = rob_BA
            print(f"  {regA}→{regB}: {rob_AB:.3f}, {regB}→{regA}: {rob_BA:.3f}")

    # Plot
    win_name = list(WINDOWS.keys())[0]
    pair_names, rob_ab, rob_ba = [], [], []
    for regA, regB in REGION_PAIRS:
        r1 = results.get((f"{regA}→{regB}", win_name))
        r2 = results.get((f"{regB}→{regA}", win_name))
        if r1 is None or r2 is None: continue
        pair_names.append(f"{regA.replace('_',' ')} vs {regB.replace('_',' ')}")
        rob_ab.append(r1); rob_ba.append(r2)

    if not pair_names:
        print("No valid data to plot.")
        return

    apply_style("paper", font_scale=1.08)

    fig = plt.figure(figsize=(14, 11))
    # Bar chart
    ax1 = fig.add_subplot(2, 1, 1)
    x = np.arange(len(pair_names)); width = 0.32
    ax1.bar(x - width / 2, rob_ab, width, label='A→B',
            color=CARC_COLORS["red"], alpha=0.8, edgecolor="#333333", linewidth=0.5)
    ax1.bar(x + width / 2, rob_ba, width, label='B→A',
            color=CARC_COLORS["blue"], alpha=0.8, edgecolor="#333333", linewidth=0.5)
    ax1.set_ylabel('Robustness Score', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(pair_names, rotation=35, ha='right', fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=10, framealpha=0.8)
    ax1.grid(axis='y', linestyle='dotted', alpha=0.25)
    ax1.axhline(0.95, color='grey', linestyle='--', alpha=0.5, linewidth=1.2)
    add_minor_ticks(ax1)
    ax1.set_title('Directional Robustness by Regional Pair', fontweight='bold', fontsize=13)
    add_panel_label(ax1, '(a)')

    # Scatter
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.scatter(rob_ab, rob_ba, c=CARC_COLORS["purple"], s=100, alpha=0.8,
               zorder=5, edgecolors="white", linewidth=1)
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1.2, alpha=0.7, label='y=x (symmetry)')
    ax2.fill_between([0, 1], [0, 1], [1, 1], color=CARC_COLORS_LIGHT["red"], alpha=0.06)
    ax2.fill_between([0, 1], [0, 0], [0, 1], color=CARC_COLORS_LIGHT["blue"], alpha=0.06)
    for i, name in enumerate(pair_names):
        ax2.annotate(name, (rob_ab[i], rob_ba[i]), xytext=(12, 10),
                     textcoords='offset points', fontsize=8, ha='left',
                     arrowprops=dict(arrowstyle='->', lw=0.5, color='gray'),
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                               edgecolor='#cccccc', alpha=0.9))
    ax2.set_xlabel('Robustness (A→B)', fontsize=12)
    ax2.set_ylabel('Robustness (B→A)', fontsize=12)
    ax2.set_xlim(0, 1.05); ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle='dotted', alpha=0.25)
    ax2.set_aspect('equal')
    ax2.legend(fontsize=10, framealpha=0.8)
    add_minor_ticks(ax2)
    ax2.set_title('Directional Asymmetry Analysis', fontweight='bold', fontsize=13)
    add_panel_label(ax2, '(b)')

    fig.suptitle("CARC Directional Robustness — Regional Causal Asymmetry",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_figure(fig, "FigS_Directional_Robustness", fmts=("pdf", "png"))
    plt.close()
    print(f"Figure saved.")


if __name__ == "__main__":
    main()
