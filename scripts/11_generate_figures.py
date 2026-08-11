#!/usr/bin/env python3
"""
11_generate_figures.py
----------------------
Quick-look demo validation figure using synthetic/illustrative data.

This script produces a multi-panel validation summary for rapid visual
inspection. All data are synthetic approximations — NOT computed from the
actual CARC pipeline.

For the publication-grade version (with real data loading, result caching,
and fallback handling), use ``13_publication_figures.py`` instead.

Output: figures/CARC_Validation_MultiPanel.pdf (DEMO)
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from src.config import FIGURE_DIR
from src.visualization import (apply_style, CARC_COLORS, CARC_COLORS_LIGHT,
                                add_panel_label, add_minor_ticks, save_figure)

os.makedirs(FIGURE_DIR, exist_ok=True)


def main():
    apply_style("paper", font_scale=1.1)

    fig = plt.figure(figsize=(19, 13))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 1.2], hspace=0.4, wspace=0.25)

    # ---- (a) SNR vs FPR ----
    ax_a = fig.add_subplot(gs[0, 0])
    lambda_eff = np.linspace(0.1, 5, 100)
    fpr_theory = 1 - stats.norm.cdf(1.645 - np.sqrt(lambda_eff / 2))
    fpr_sim = fpr_theory + np.random.normal(0, 0.01, len(lambda_eff))
    ax_a.scatter(lambda_eff, fpr_sim, color=CARC_COLORS["blue"], s=18, edgecolor='white', alpha=0.6, label='Simulated FPR')
    ax_a.plot(lambda_eff, fpr_theory, color=CARC_COLORS["red"], linewidth=2.2, label='Theoretical Lower Bound')
    ax_a.fill_between(lambda_eff, fpr_theory - 0.02, fpr_theory + 0.02, color=CARC_COLORS["red"], alpha=0.1)
    ax_a.set_xlabel(r'Effective SNR ($\lambda_{eff}$)', fontsize=11)
    ax_a.set_ylabel('False Positive Rate', fontsize=11)
    ax_a.legend(frameon=True, fontsize=9); ax_a.grid(True, linestyle='dotted', alpha=0.25)
    add_minor_ticks(ax_a)
    ax_a.set_title('Theoretical FPR Bound', fontweight='bold')
    add_panel_label(ax_a, '(a)')

    # ---- (b) α scan ----
    ax_b = fig.add_subplot(gs[0, 1])
    alpha_scan = np.linspace(0, 1.0, 50)
    fpr_scan = 0.21 - 0.15 * alpha_scan + 0.2 * alpha_scan ** 2
    ar_scan = 0.82 - 0.75 * alpha_scan
    ax_b.plot(alpha_scan, fpr_scan, color=CARC_COLORS["blue"], linewidth=2.2)
    ax_b.set_xlabel(r'ARC stripping strength $\alpha$', fontsize=11)
    ax_b.set_ylabel('False Positive Rate', color=CARC_COLORS["blue"], fontsize=11)
    ax_b.tick_params(axis='y', labelcolor=CARC_COLORS["blue"])
    ax_b2 = ax_b.twinx()
    ax_b2.plot(alpha_scan, ar_scan, color=CARC_COLORS["red"], linestyle='--', linewidth=2.2)
    ax_b2.set_ylabel(r'Residual AR coefficient $\phi$', color=CARC_COLORS["red"], fontsize=11)
    ax_b2.tick_params(axis='y', labelcolor=CARC_COLORS["red"])
    ax_b.axvline(0.5, color=CARC_COLORS["green"], linestyle=':', linewidth=2)
    ax_b.annotate(r'Optimal $\alpha=0.5$', xy=(0.5, 0.14), xytext=(0.25, 0.20),
                  arrowprops=dict(facecolor='black', arrowstyle='->'),
                  fontsize=9, fontweight='bold',
                  bbox=dict(boxstyle='round', facecolor='white', edgecolor=CARC_COLORS["green"], alpha=0.9))
    ax_b.grid(True, linestyle='dotted', alpha=0.25)
    add_minor_ticks(ax_b)
    ax_b.set_title(r'ARC Hyperparameter Sensitivity', fontweight='bold')
    add_panel_label(ax_b, '(b)')

    # ---- (c) Robustness surface ----
    ax_c = fig.add_subplot(gs[1, 0])
    sig_x, sig_y = np.meshgrid(np.linspace(0, 2.0, 30), np.linspace(0, 2.0, 30))
    rob_surf = 0.95 * np.exp(-0.5 * (sig_x ** 2 + sig_y ** 2) / 0.3) + 0.05
    contour = ax_c.contourf(sig_x, sig_y, rob_surf, levels=10, cmap='RdBu_r')
    ax_c.set_xlabel('Age Error Proxy A (ka)', fontsize=11)
    ax_c.set_ylabel('Age Error Proxy B (ka)', fontsize=11)
    cbar = plt.colorbar(contour, ax=ax_c, pad=0.02)
    cbar.set_label('Robustness Score', fontsize=11)
    ax_c.set_title('Robustness vs Age Errors', fontweight='bold')
    add_panel_label(ax_c, '(c)')

    # ---- (d) Climate window robustness ----
    ax_d = fig.add_subplot(gs[1, 1])
    windows = ['H1', 'LGM', 'H4', 'GI12', 'GS12']
    ranges = [(14, 18), (19, 26), (36, 40), (44, 47), (47, 49)]
    data_ab, data_ba = [], []
    rng = np.random.default_rng(42)
    for start, end in ranges:
        n = int((end - start) * 10)
        data_ab.append(0.12 + rng.normal(0, 0.03, max(n, 5)))
        data_ba.append(0.08 + rng.normal(0, 0.02, max(n, 5)))
    bp1 = ax_d.boxplot(data_ab, positions=np.arange(len(windows)) - 0.15, widths=0.3, patch_artist=True)
    bp2 = ax_d.boxplot(data_ba, positions=np.arange(len(windows)) + 0.15, widths=0.3, patch_artist=True)
    for element in ['boxes', 'whiskers', 'fliers', 'caps', 'medians']:
        plt.setp(bp1[element], color=CARC_COLORS["red"]); plt.setp(bp2[element], color=CARC_COLORS["blue"])
    for patch in bp1['boxes']: patch.set_facecolor(CARC_COLORS_LIGHT["red"])
    for patch in bp2['boxes']: patch.set_facecolor(CARC_COLORS_LIGHT["blue"])
    ax_d.set_xticks(range(len(windows))); ax_d.set_xticklabels(windows)
    ax_d.set_xlabel('Climate Windows', fontsize=11)
    ax_d.set_ylabel('Robustness Score', fontsize=11)
    ax_d.grid(True, linestyle='dotted', alpha=0.25)
    ax_d.legend([bp1['boxes'][0], bp2['boxes'][0]], ['A→B', 'B→A'], fontsize=9)
    add_minor_ticks(ax_d)
    ax_d.set_title('Window-Specific Robustness', fontweight='bold')
    add_panel_label(ax_d, '(d)')

    # ---- (e) Method comparison ----
    ax_e = fig.add_subplot(gs[2, 0])
    snr_range = np.linspace(0.1, 1.0, 30)
    ax_e.plot(snr_range, 1 / (1 + np.exp(-5 * (snr_range - 0.4))), color=CARC_COLORS["red"], linewidth=2.2, marker='o', markersize=4, markerfacecolor='white', markeredgewidth=1.2, label='Granger')
    ax_e.plot(snr_range, 1 / (1 + np.exp(-3 * (snr_range - 0.5))), color=CARC_COLORS["blue"], linewidth=2.2, marker='s', markersize=4, markerfacecolor='white', markeredgewidth=1.2, label='CCM')
    ax_e.plot(snr_range, 1 / (1 + np.exp(-4 * (snr_range - 0.45))), color=CARC_COLORS["green"], linewidth=2.2, marker='^', markersize=4, markerfacecolor='white', markeredgewidth=1.2, label='PCMCI')
    ax_e.set_xlabel('Signal-to-Noise Ratio (SNR)', fontsize=11)
    ax_e.set_ylabel('Robustness Score', fontsize=11)
    ax_e.legend(frameon=True, fontsize=9); ax_e.grid(True, linestyle='dotted', alpha=0.25)
    add_minor_ticks(ax_e)
    ax_e.set_title('Method Comparison', fontweight='bold')
    add_panel_label(ax_e, '(e)')

    # ---- (f) Window size sensitivity ----
    ax_f = fig.add_subplot(gs[2, 1])
    window_sizes = [2, 4, 6, 8, 10, 12]
    sens_data = [rng.normal(0.12, 0.04, 50) for _ in window_sizes]
    bp3 = ax_f.boxplot(sens_data, positions=window_sizes, widths=0.8, patch_artist=True)
    plt.setp(bp3['boxes'], facecolor=CARC_COLORS_LIGHT["grey"], color=CARC_COLORS["blue"])
    ax_f.set_xlabel('Sliding Window Size (ka)', fontsize=11)
    ax_f.set_ylabel('Robustness Score', fontsize=11)
    ax_f.grid(True, linestyle='dotted', alpha=0.25)
    add_minor_ticks(ax_f)
    ax_f.set_title('Window Size Sensitivity', fontweight='bold')
    add_panel_label(ax_f, '(f)')

    fig.suptitle("CARC Method Validation and Sensitivity Analysis",
                 fontsize=14, fontweight="bold", y=1.0)
    plt.subplots_adjust(hspace=0.4, wspace=0.25)
    save_figure(fig, "CARC_Validation_MultiPanel", fmts=("pdf", "png"))
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
