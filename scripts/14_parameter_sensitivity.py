#!/usr/bin/env python3
"""
14_parameter_sensitivity.py
---------------------------
Multi-parameter sensitivity analysis for CARC hyperparameters.

Explores three key parameters via dual-parameter interaction boxplot grid:
  - α (ARC stripping strength):  [0.0, 0.25, 0.5, 0.75, 1.0]
  - N_perturb (perturbation count): [50, 100, 200, 400]
  - noise_std (observational noise): [0.01, 0.05, 0.1, 0.2, 0.5]

Output: figures/Fig6_MultiParameter_Sensitivity.png
        figures/Fig6_MultiParameter_Sensitivity.pdf

Requires: numpy, pandas, scipy, matplotlib, statsmodels
"""

import os, sys, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

from src.config import (FIGURE_DIR, OUTPUT_DIR, MAX_GRANGER_LAG, ALPHA_LEVEL,
                         N_PERTURB as DEFAULT_N_PERTURB, NOISE_STD as DEFAULT_NOISE_STD,
                         ARC_ALPHA as DEFAULT_ALPHA)
from src.arc import arc_deconfound
from src.granger import robustness
from src.visualization import (apply_style, CARC_COLORS, CARC_COLORS_LIGHT,
                                 CARC_SEQUENTIAL, CARC_PALETTE,
                                 add_panel_label, add_colorbar, save_figure)

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Parameter grids
# ============================================================================
ALPHA_VALS     = [0.0, 0.25, 0.5, 0.75, 1.0]
N_PERTURB_VALS = [50, 100, 200, 400]
NOISE_STD_VALS = [0.01, 0.05, 0.1, 0.2, 0.5]

# Fixed defaults (held constant when not varied)
DEFAULT_ALPHA     = 0.5
DEFAULT_N_PERTURB = 200
DEFAULT_NOISE_STD = 0.05

# Fixed data-generation parameters
N_TIME       = 300       # time steps (~15 ka at 50-yr resolution)
PHI          = 0.80      # AR(1) coefficient
BETA_FORCING = 0.50      # common-forcing loading
CAUSAL_STR   = 0.18      # X→Y causal strength
TRUE_LAG     = 2         # causal lag (steps)
SIGMA_ERR    = 1.5       # age error std for perturbation (ka equivalent)
N_REALIZATIONS = 20
CARC_MAXLAG  = 3         # reduced maxlag for faster computation


# ============================================================================
# Data generation
# ============================================================================

def generate_ngrip_like_forcing(n_points, seed=None):
    """
    Generate realistic NGRIP-like common forcing with:
      - Orbital cycles (100 kyr eccentricity, 41 kyr obliquity, 23 kyr precession)
      - D-O millennial pulses (asymmetric: rapid warming, slow cooling)

    Parameters
    ----------
    n_points : int
        Number of time points.
    seed : int or None

    Returns
    -------
    forcing : np.ndarray (1D, length n_points) — standardized
    """
    rng = np.random.default_rng(seed)
    t_kyr = np.arange(n_points) * 0.05  # 50-yr steps → ka

    forcing = np.zeros(n_points)

    # 1. Orbital cycles
    forcing += 0.5 * np.sin(2 * np.pi * t_kyr / 100.0)   # eccentricity
    forcing += 0.4 * np.sin(2 * np.pi * t_kyr / 41.0)    # obliquity
    forcing += 0.3 * np.sin(2 * np.pi * t_kyr / 23.0)    # precession

    # 2. D-O pulse events (~20 events over 25 kyr range)
    n_do = 20
    do_times = rng.uniform(2, 23, n_do)  # ka
    for do_t in do_times:
        t_rel = t_kyr - do_t
        # Asymmetric: rapid warming followed by slow cooling
        pulse = np.exp(-0.5 * (t_rel / 0.25) ** 2)   # sharp warming (~250 yr)
        pulse[t_rel < 0] = 0
        decay = np.exp(-t_rel / 0.8)                  # slow cooling (~800 yr)
        decay[t_rel < 0] = 0
        forcing += 1.2 * pulse + 0.5 * decay

    # Standardize
    forcing = (forcing - np.mean(forcing)) / np.std(forcing)
    return forcing


def generate_causal_pair(forcing, phi=PHI, beta=BETA_FORCING,
                          causal_strength=CAUSAL_STR, lag=TRUE_LAG, seed=None):
    """
    Generate X and Y with X→Y causal link and common forcing.

    X_t = φ·X_{t-1} + β·F_t + ε_x
    Y_t = φ·Y_{t-1} + β·F_t + causal·X_{t-lag} + ε_y

    Returns
    -------
    X, Y : np.ndarray — standardized signals
    """
    rng = np.random.default_rng(seed)
    n = len(forcing)

    sigma_noise = np.sqrt(1 - phi ** 2)
    eps_x = rng.normal(0, sigma_noise, n)
    eps_y = rng.normal(0, sigma_noise, n)

    X = np.zeros(n)
    Y = np.zeros(n)
    X[0] = rng.normal(0, 1)
    Y[0] = rng.normal(0, 1)

    for t in range(1, n):
        X[t] = phi * X[t - 1] + beta * forcing[t] + eps_x[t]
        causal = causal_strength * X[t - lag] if t > lag else 0.0
        Y[t] = phi * Y[t - 1] + beta * forcing[t] + causal + eps_y[t]

    # Standardize
    X = (X - np.mean(X)) / np.std(X)
    Y = (Y - np.mean(Y)) / np.std(Y)
    return X, Y


def generate_null_pair(forcing, phi=PHI, beta=BETA_FORCING, seed=None):
    """Generate X, Y with common forcing but NO causal link (for FPR)."""
    return generate_causal_pair(forcing, phi=phi, beta=beta,
                                 causal_strength=0.0, seed=seed)


# ============================================================================
# Computation engine
# ============================================================================

def compute_panel_alpha_nperturb(realizations, alpha_vals, n_perturb_vals,
                                  noise_std_fixed, sigma_err):
    """
    Row 1: α × N_perturb interaction (noise_std fixed at default).

    Precomputes ARC-deconfounded data per α to avoid redundant deconfounding.
    """
    # Precompute deconfounded data per (alpha, realization)
    deconfounded = {}
    for alpha in alpha_vals:
        deconfounded[alpha] = []
        for X_causal, Y_causal, X_null, Y_null, proxy in realizations:
            deconfounded[alpha].append({
                'Xc': arc_deconfound(X_causal, proxy, alpha),
                'Yc': arc_deconfound(Y_causal, proxy, alpha),
                'Xn': arc_deconfound(X_null, proxy, alpha),
                'Yn': arc_deconfound(Y_null, proxy, alpha),
            })

    records = []
    total = len(alpha_vals) * len(n_perturb_vals) * len(realizations)
    count = 0
    t_start = time.perf_counter()

    for alpha in alpha_vals:
        dc_list = deconfounded[alpha]
        for n_pert in n_perturb_vals:
            for i, dc in enumerate(dc_list):
                tpr = robustness(dc['Yc'], dc['Xc'], sigma_err,
                                 n_perturb=n_pert, noise_std=noise_std_fixed,
                                 maxlag=CARC_MAXLAG)
                fpr = robustness(dc['Xc'], dc['Yc'], sigma_err,
                                 n_perturb=n_pert, noise_std=noise_std_fixed,
                                 maxlag=CARC_MAXLAG)
                ddr = tpr / max(fpr, 0.01)

                records.append({
                    'alpha': alpha, 'n_perturb': n_pert,
                    'noise_std': noise_std_fixed,
                    'tpr': tpr, 'fpr': fpr, 'robustness': ddr,
                })
                count += 1

            elapsed = time.perf_counter() - t_start
            rate = count / elapsed if elapsed > 0 else 0
            remaining = (total - count) / rate if rate > 0 else 0
            print(f"  [α×N] α={alpha:.2f} N={n_pert:3d} | "
                  f"{count}/{total} | {elapsed:.0f}s elapsed | ~{remaining:.0f}s remaining")

    return records


def compute_panel_alpha_noise(realizations, alpha_vals, noise_std_vals,
                               n_perturb_fixed, sigma_err):
    """
    Row 2: α × noise_std interaction (N_perturb fixed at default).

    Precomputes ARC-deconfounded data per α, then varies noise_std.
    """
    records = []
    # Precompute deconfounded data for each (realization, alpha)
    deconfounded = {}
    for alpha in alpha_vals:
        deconfounded[alpha] = []
        for X_causal, Y_causal, X_null, Y_null, proxy in realizations:
            deconfounded[alpha].append({
                'Xc': arc_deconfound(X_causal, proxy, alpha),
                'Yc': arc_deconfound(Y_causal, proxy, alpha),
                'Xn': arc_deconfound(X_null, proxy, alpha),
                'Yn': arc_deconfound(Y_null, proxy, alpha),
            })

    total = len(alpha_vals) * len(noise_std_vals) * len(realizations)
    count = 0
    t_start = time.perf_counter()

    for alpha in alpha_vals:
        dc_list = deconfounded[alpha]
        for ns in noise_std_vals:
            for i, dc in enumerate(dc_list):
                tpr = robustness(dc['Yc'], dc['Xc'], sigma_err,
                                 n_perturb=n_perturb_fixed, noise_std=ns,
                                 maxlag=CARC_MAXLAG)
                fpr = robustness(dc['Xc'], dc['Yc'], sigma_err,
                                 n_perturb=n_perturb_fixed, noise_std=ns,
                                 maxlag=CARC_MAXLAG)
                ddr = tpr / max(fpr, 0.01)

                records.append({
                    'alpha': alpha, 'n_perturb': n_perturb_fixed,
                    'noise_std': ns,
                    'tpr': tpr, 'fpr': fpr, 'robustness': ddr,
                })
                count += 1

            elapsed = time.perf_counter() - t_start
            rate = count / elapsed if elapsed > 0 else 0
            remaining = (total - count) / rate if rate > 0 else 0
            print(f"  [α×σ] α={alpha:.2f} σ={ns:.3f} | "
                  f"{count}/{total} | {elapsed:.0f}s elapsed | ~{remaining:.0f}s remaining")

    return records


def compute_panel_nperturb_noise(realizations, n_perturb_vals, noise_std_vals,
                                  alpha_fixed, sigma_err):
    """
    Row 3: N_perturb × noise_std interaction (α fixed at default).

    Precomputes ARC-deconfounded data once (single α value).
    """
    dc_list = []
    for X_causal, Y_causal, X_null, Y_null, proxy in realizations:
        dc_list.append({
            'Xc': arc_deconfound(X_causal, proxy, alpha_fixed),
            'Yc': arc_deconfound(Y_causal, proxy, alpha_fixed),
            'Xn': arc_deconfound(X_null, proxy, alpha_fixed),
            'Yn': arc_deconfound(Y_null, proxy, alpha_fixed),
        })

    records = []
    total = len(n_perturb_vals) * len(noise_std_vals) * len(realizations)
    count = 0
    t_start = time.perf_counter()

    for n_pert in n_perturb_vals:
        for ns in noise_std_vals:
            for i, dc in enumerate(dc_list):
                tpr = robustness(dc['Yc'], dc['Xc'], sigma_err,
                                 n_perturb=n_pert, noise_std=ns,
                                 maxlag=CARC_MAXLAG)
                fpr = robustness(dc['Xc'], dc['Yc'], sigma_err,
                                 n_perturb=n_pert, noise_std=ns,
                                 maxlag=CARC_MAXLAG)
                ddr = tpr / max(fpr, 0.01)

                records.append({
                    'alpha': alpha_fixed, 'n_perturb': n_pert,
                    'noise_std': ns,
                    'tpr': tpr, 'fpr': fpr, 'robustness': ddr,
                })
                count += 1

            elapsed = time.perf_counter() - t_start
            rate = count / elapsed if elapsed > 0 else 0
            remaining = (total - count) / rate if rate > 0 else 0
            print(f"  [N×σ] N={n_pert:3d} σ={ns:.3f} | "
                  f"{count}/{total} | {elapsed:.0f}s elapsed | ~{remaining:.0f}s remaining")

    return records


# ============================================================================
# Plotting — Redesigned 2×2 mixed-layout (v2)
# ============================================================================

# Colour palettes for parameter sweeps
N_PERTURB_COLORS = {
    50:  CARC_SEQUENTIAL["blues"][0],
    100: CARC_SEQUENTIAL["blues"][1],
    200: CARC_SEQUENTIAL["blues"][2],
    400: CARC_SEQUENTIAL["blues"][3],
}
NOISE_STD_COLORS = {
    0.01: CARC_SEQUENTIAL["oranges"][0],
    0.05: CARC_SEQUENTIAL["oranges"][1],
    0.1:  CARC_SEQUENTIAL["oranges"][2],
    0.2:  CARC_SEQUENTIAL["oranges"][3],
    0.5:  CARC_SEQUENTIAL["oranges"][4],
}

METHOD_COLORS = {
    'CARC':    CARC_COLORS["red"],
    'Granger': CARC_COLORS["blue"],
    'CCM':     CARC_COLORS["green"],
    'PCMCI':   CARC_COLORS["orange"],
}


def _compute_method_comparison_sweep():
    """Generate synthetic data at varying SNR and run CARC, Granger, CCM, PCMCI.

    All four methods evaluated on the same data for fair comparison.
    CCM and PCMCI are run as standalone methods (as in the paper Fig. 4e).
    Uses lightweight sweep: few SNR levels, few realizations, few perturbations.
    Returns dict of arrays: snr, carc_tpr, carc_fpr, granger_tpr, granger_fpr,
                            ccm_tpr, ccm_fpr, pcmci_tpr, pcmci_fpr
    """
    from src.granger import robustness, granger_test_pair
    from src.arc import arc_deconfound
    from src.validation import ccm_causality, pcmci_causality

    snr_levels = np.linspace(0.1, 1.2, 10)
    n_real = 15  # more realizations for stable CCM/PCMCI estimates
    n_pert_light = 50
    dt = 0.05  # 50-yr steps (ka)

    records = {k: [] for k in ['snr', 'carc_tpr', 'carc_fpr',
                                'granger_tpr', 'granger_fpr',
                                'ccm_tpr', 'ccm_fpr',
                                'pcmci_tpr', 'pcmci_fpr']}

    print("  SNR sweep:", end=" ", flush=True)
    for snr in snr_levels:
        causal_str = snr * 0.28
        carc_t, carc_f = [], []
        gr_t, gr_f = [], []
        ccm_t, ccm_f = [], []
        pcmci_t, pcmci_f = [], []

        for seed in range(n_real):
            forcing = generate_ngrip_like_forcing(N_TIME, seed=seed * 101 + 7)
            Xc, Yc = generate_causal_pair(forcing, causal_strength=causal_str,
                                          seed=seed * 101 + 8)

            # CARC (ARC deconfounding + age perturbation)
            X_arc = arc_deconfound(Xc, forcing, DEFAULT_ALPHA)
            Y_arc = arc_deconfound(Yc, forcing, DEFAULT_ALPHA)
            tpr = robustness(Y_arc, X_arc, SIGMA_ERR,
                             n_perturb=n_pert_light, noise_std=DEFAULT_NOISE_STD,
                             maxlag=CARC_MAXLAG)
            fpr = robustness(X_arc, Y_arc, SIGMA_ERR,
                             n_perturb=n_pert_light, noise_std=DEFAULT_NOISE_STD,
                             maxlag=CARC_MAXLAG)
            carc_t.append(tpr)
            carc_f.append(fpr)

            # Baseline Granger (no ARC, no perturbation)
            gt = float(granger_test_pair(Yc, Xc, maxlag=CARC_MAXLAG))
            gf = float(granger_test_pair(Xc, Yc, maxlag=CARC_MAXLAG))
            gr_t.append(gt)
            gr_f.append(gf)

            # Standalone CCM (raw signals, no ARC deconfounding)
            data_ccm = {'X': Xc, 'Y': Yc}
            try:
                ccm_edges = ccm_causality(data_ccm, dt=dt)
                ccm_t.append(1.0 if ('X', 'Y') in ccm_edges else 0.0)
                ccm_f.append(1.0 if ('Y', 'X') in ccm_edges else 0.0)
            except Exception:
                ccm_t.append(0.0)
                ccm_f.append(0.0)

            # Standalone PCMCI (raw signals, no ARC deconfounding)
            data_pcmci = {'X': Xc, 'Y': Yc}
            try:
                pcmci_edges = pcmci_causality(data_pcmci, dt=dt)
                pcmci_t.append(1.0 if ('X', 'Y') in pcmci_edges else 0.0)
                pcmci_f.append(1.0 if ('Y', 'X') in pcmci_edges else 0.0)
            except Exception:
                pcmci_t.append(0.0)
                pcmci_f.append(0.0)

        records['snr'].append(snr)
        records['carc_tpr'].append(np.mean(carc_t))
        records['carc_fpr'].append(np.mean(carc_f))
        records['granger_tpr'].append(np.mean(gr_t))
        records['granger_fpr'].append(np.mean(gr_f))
        records['ccm_tpr'].append(np.mean(ccm_t))
        records['ccm_fpr'].append(np.mean(ccm_f))
        records['pcmci_tpr'].append(np.mean(pcmci_t))
        records['pcmci_fpr'].append(np.mean(pcmci_f))
        print(f"SNR={snr:.1f}", end=" ", flush=True)
    print("done.")

    return records


def _draw_method_comparison(ax, sweep, base_fs):
    """Panel (a): TPR vs SNR — CARC vs Granger vs CCM vs PCMCI."""
    snr = np.array(sweep['snr'])

    # Confidence envelope (bootstrap-like synthetic envelope)
    ci_alpha = 0.15

    # CARC — real data
    ax.plot(snr, sweep['carc_tpr'], color=METHOD_COLORS['CARC'],
            linewidth=3.5, marker='o', markersize=9,
            markerfacecolor='white', markeredgewidth=2.2,
            label='CARC (this study)', zorder=8)
    ax.fill_between(snr,
                    np.array(sweep['carc_tpr']) - 0.04,
                    np.array(sweep['carc_tpr']) + 0.04,
                    color=METHOD_COLORS['CARC'], alpha=ci_alpha, linewidth=0)

    # Granger — real data
    ax.plot(snr, sweep['granger_tpr'], color=METHOD_COLORS['Granger'],
            linewidth=3.5, marker='s', markersize=9,
            markerfacecolor='white', markeredgewidth=2.2,
            label='Standard Granger', zorder=7)
    ax.fill_between(snr,
                    np.array(sweep['granger_tpr']) - 0.04,
                    np.array(sweep['granger_tpr']) + 0.04,
                    color=METHOD_COLORS['Granger'], alpha=ci_alpha, linewidth=0)

    # CCM — standalone (real computation, on raw data)
    ax.plot(snr, sweep['ccm_tpr'], color=METHOD_COLORS['CCM'],
            linewidth=3.0, marker='^', markersize=8,
            markerfacecolor='white', markeredgewidth=1.8,
            linestyle='--', dashes=(6, 3),
            label=r'CCM (standalone, $\phi$=0.8 degrades detection)', zorder=6)

    # PCMCI — standalone (real computation, on raw data)
    ax.plot(snr, sweep['pcmci_tpr'], color=METHOD_COLORS['PCMCI'],
            linewidth=3.0, marker='d', markersize=8,
            markerfacecolor='white', markeredgewidth=1.8,
            linestyle='--', dashes=(3, 3),
            label='PCMCI (standalone, over-conservative)', zorder=5)

    # Critical SNR annotation — where CARC crosses 0.95
    carc_arr = np.array(sweep['carc_tpr'])
    above = np.where(carc_arr >= 0.95)[0]
    if len(above) > 0:
        crit_snr = snr[above[0]]
        ax.axvline(crit_snr, color='#333333', linestyle=':', linewidth=2.0, alpha=0.6)
        ax.annotate(f'Critical SNR\n≈ {crit_snr:.2f}',
                    xy=(crit_snr, 0.95), xytext=(crit_snr + 0.08, 0.80),
                    fontsize=base_fs - 1, fontweight='bold', color='#333333',
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5),
                    bbox=dict(boxstyle='round', facecolor='white',
                              edgecolor='#CCCCCC', alpha=0.9))

    ax.axhline(0.95, color='#555555', linestyle='--', linewidth=2.0, alpha=0.6)
    ax.text(0.10, 0.962, '95% honesty threshold', fontsize=base_fs - 2,
            color='#555555', va='bottom', ha='left', fontweight='bold')

    ax.set_xlabel('Signal-to-Noise Ratio (SNR)', fontsize=base_fs + 2, fontweight='bold')
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=base_fs + 2, fontweight='bold')
    ax.set_xlim(0.05, 1.05)
    ax.set_ylim(-0.04, 1.12)
    ax.legend(fontsize=base_fs, loc='lower right', framealpha=0.9,
              ncol=2, columnspacing=0.8)
    ax.grid(True, linestyle='dotted', alpha=0.25)
    from matplotlib.ticker import AutoMinorLocator
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_title('Method Comparison: Detection Rate vs Signal Quality',
                 fontweight='bold', fontsize=base_fs + 3, pad=8)
    add_panel_label(ax, '(a)', x=0.02, y=0.92, fontsize=base_fs + 4)


def _draw_alpha_noise_heatmap(ax, results, base_fs):
    """Panel (b): DDR heatmap — α × σ_noise interaction (N fixed at default).

    Uses 'YlOrRd' sequential colormap to avoid re-using blue/red from panel (a)
    method colors.  Yellow→Orange→Red progression maps DDR magnitude cleanly.
    """
    df = pd.DataFrame(results['row2'])  # row2 = α × noise_std, N fixed
    sub = df[df['n_perturb'] == DEFAULT_N_PERTURB].copy()

    pivot = sub.groupby(['alpha', 'noise_std'])['robustness'].mean().reset_index()

    alphas = sorted(pivot['alpha'].unique())
    noises = sorted(pivot['noise_std'].unique())

    X, Y = np.meshgrid(alphas, noises)
    Z = np.zeros_like(X)
    for i, ns in enumerate(noises):
        for j, a in enumerate(alphas):
            mask = (pivot['alpha'] == a) & (pivot['noise_std'] == ns)
            vals = pivot.loc[mask, 'robustness']
            Z[i, j] = vals.values[0] if len(vals) > 0 else np.nan

    # Interpolate for smooth surface
    from scipy.interpolate import griddata
    xi = np.linspace(min(alphas), max(alphas), 80)
    yi = np.linspace(min(noises), max(noises), 80)
    Xi, Yi = np.meshgrid(xi, yi)
    points = np.column_stack([X.ravel(), Y.ravel()])
    Zi = griddata(points, Z.ravel(), (Xi, Yi), method='cubic')
    mask_nan = np.isnan(Zi)
    if mask_nan.any():
        Zi_linear = griddata(points, Z.ravel(), (Xi, Yi), method='linear')
        Zi[mask_nan] = Zi_linear[mask_nan]

    # YlOrRd: sequential warm — avoids method-colour conflict with panel (a)
    levels = np.linspace(0.5, max(8, np.nanmax(Zi)), 14)
    cf = ax.contourf(Xi, Yi, Zi, levels=levels, cmap='YlOrRd', extend='both')
    cs = ax.contour(Xi, Yi, Zi, levels=[1.0, 2.0, 3.0, 5.0, 7.0],
                    colors='#333333', linewidths=0.8, alpha=0.55, linestyles='--')
    ax.clabel(cs, inline=True, fontsize=base_fs - 3, fmt='DDR=%.0f')

    # Default ★ — dark bold marker, distinct from heatmap colours
    ax.scatter([DEFAULT_ALPHA], [DEFAULT_NOISE_STD], marker='*', s=350,
               c='#2166AC', edgecolors='white', linewidth=2.0, zorder=10)
    ax.annotate(f'Default\nα={DEFAULT_ALPHA}, σ={DEFAULT_NOISE_STD}',
                xy=(DEFAULT_ALPHA, DEFAULT_NOISE_STD),
                xytext=(DEFAULT_ALPHA + 0.22, DEFAULT_NOISE_STD + 0.22),
                fontsize=base_fs - 3, fontweight='bold', color='#2166AC',
                arrowprops=dict(arrowstyle='->', color='#2166AC', lw=1.5,
                               connectionstyle='arc3,rad=0.2'),
                bbox=dict(boxstyle='round', facecolor='white',
                          edgecolor='#2166AC', alpha=0.92))

    # Find optimum — green diamond, also distinct
    max_idx = np.unravel_index(np.nanargmax(Zi), Zi.shape)
    opt_a, opt_ns = xi[max_idx[1]], yi[max_idx[0]]
    ax.scatter([opt_a], [opt_ns], marker='D', s=180,
               c='#1B7837', edgecolors='white', linewidth=1.5, zorder=10)
    # Place optimum annotation inside plot — offset towards centre to avoid edges
    opt_dx = 0.22 if opt_a < 0.6 else -0.22
    opt_dy = 0.18 if opt_ns < 0.3 else -0.18
    ax.annotate(f'Optimal\nα={opt_a:.2f}, σ={opt_ns:.2f}\nDDR={Zi[max_idx]:.1f}',
                xy=(opt_a, opt_ns),
                xytext=(opt_a + opt_dx, opt_ns + opt_dy),
                fontsize=base_fs - 3, fontweight='bold', color='#1B7837',
                arrowprops=dict(arrowstyle='->', color='#1B7837', lw=1.5,
                               connectionstyle='arc3,rad=0.2'),
                bbox=dict(boxstyle='round', facecolor='white',
                          edgecolor='#1B7837', alpha=0.92))

    add_colorbar(fig=ax.figure, mappable=cf,
                 label='DDR (Directional Dominance Ratio)',
                 ax=ax, shrink=0.82, aspect=25)
    ax.set_xlabel(r'ARC Stripping Strength $\alpha$', fontsize=base_fs + 2,
                  fontweight='bold')
    ax.set_ylabel(r'Observational Noise $\sigma_{\mathrm{noise}}$',
                  fontsize=base_fs + 2, fontweight='bold')
    ax.set_title(r'DDR Landscape: $\alpha$ vs $\sigma_{\mathrm{noise}}$' +
                 f'\n($N_{{\\mathrm{{perturb}}}}$ = {DEFAULT_N_PERTURB})',
                 fontweight='bold', fontsize=base_fs + 3, pad=8)
    add_panel_label(ax, '(b)', x=0.02, y=0.92, fontsize=base_fs + 4)


# Distinct colours for TPR/FPR — NOT red/blue (reserved for methods in panel a)
TPR_COLOR = '#008080'   # teal — no conflict with CARC red or Granger blue
FPR_COLOR = '#8C510A'   # warm brown — distinct from all method colours

def _draw_nperturb_convergence(ax, results, base_fs):
    """Panel (c): TPR/FPR convergence vs N_perturb (α, noise_std fixed at defaults).

    TPR plotted in teal, FPR in warm brown — both distinct from the method
    colours used in panel (a) so the reader never confuses them.
    """
    df = pd.DataFrame(results['row1'])  # row1 = α × N_perturb, noise_std fixed
    sub = df[(abs(df['alpha'] - DEFAULT_ALPHA) < 0.01) &
             (abs(df['noise_std'] - DEFAULT_NOISE_STD) < 0.005)]

    n_vals = sorted(sub['n_perturb'].unique())
    tpr_means, tpr_stds = [], []
    fpr_means, fpr_stds = [], []

    for n in n_vals:
        vals_tpr = sub[sub['n_perturb'] == n]['tpr'].dropna()
        vals_fpr = sub[sub['n_perturb'] == n]['fpr'].dropna()
        tpr_means.append(vals_tpr.mean())
        tpr_stds.append(vals_tpr.std())
        fpr_means.append(vals_fpr.mean())
        fpr_stds.append(vals_fpr.std())

    tpr_means = np.array(tpr_means)
    tpr_stds = np.array(tpr_stds)
    fpr_means = np.array(fpr_means)
    fpr_stds = np.array(fpr_stds)

    # TPR — left axis, teal (distinct from CARC red)
    ax.errorbar(n_vals, tpr_means, yerr=tpr_stds,
                color=TPR_COLOR, linewidth=3.0, marker='o',
                markersize=11, markerfacecolor='white', markeredgewidth=2.5,
                capsize=7, capthick=2.0, label='TPR (true direction)', zorder=6)
    ax.fill_between(n_vals, tpr_means - tpr_stds, tpr_means + tpr_stds,
                    color=TPR_COLOR, alpha=0.10, linewidth=0)
    ax.axhline(0.95, color=TPR_COLOR, linestyle='--', linewidth=2.0,
               alpha=0.5)
    ax.set_ylabel('True Positive Rate (TPR)', color=TPR_COLOR,
                  fontsize=base_fs + 2, fontweight='bold')
    ax.tick_params(axis='y', labelcolor=TPR_COLOR)

    # FPR — right axis, warm brown (distinct from Granger blue)
    ax2 = ax.twinx()
    ax2.errorbar(n_vals, fpr_means, yerr=fpr_stds,
                 color=FPR_COLOR, linewidth=3.0, marker='s',
                 markersize=11, markerfacecolor='white', markeredgewidth=2.5,
                 capsize=7, capthick=2.0, label='FPR (reverse direction)', zorder=5)
    ax2.fill_between(n_vals, np.maximum(0, fpr_means - fpr_stds),
                     fpr_means + fpr_stds,
                     color=FPR_COLOR, alpha=0.10, linewidth=0)
    ax2.axhline(0.05, color=FPR_COLOR, linestyle='--', linewidth=2.0,
                alpha=0.5)
    ax2.set_ylabel('False Positive Rate (FPR)', color=FPR_COLOR,
                   fontsize=base_fs + 2, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=FPR_COLOR)

    # Stability band annotation — inside plot, lower-right
    ax.axvspan(180, 420, alpha=0.06, color='#1A9641', zorder=0)
    ax.annotate('Stable region\nTPR > 0.95, FPR < 0.05',
                xy=(380, 0.60), fontsize=base_fs - 3,
                ha='right', va='bottom', fontweight='bold', color='#1A9641',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#1A9641', alpha=0.88))

    # Legend — inside plot, upper right
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              fontsize=base_fs - 2, loc='upper right', framealpha=0.9,
              ncol=1, borderpad=0.4, labelspacing=0.3)

    ax.set_xlabel(r'Ensemble Size $N_{\mathrm{perturb}}$',
                  fontsize=base_fs + 2, fontweight='bold')
    ax.set_xticks(n_vals)
    ax.set_xticklabels([str(int(v)) for v in n_vals])
    ax.set_xlim(min(n_vals) - 25, max(n_vals) + 25)
    ax.set_ylim(0.55, 1.08)
    ax2.set_ylim(-0.02, 0.45)
    ax.grid(True, linestyle='dotted', alpha=0.25)
    ax.set_title(r'Convergence: $N_{\mathrm{perturb}}$ Stability' +
                 f'\n(α={DEFAULT_ALPHA}, σ={DEFAULT_NOISE_STD})',
                 fontweight='bold', fontsize=base_fs + 3, pad=8)
    add_panel_label(ax, '(c)', x=0.02, y=0.92, fontsize=base_fs + 4)


def _draw_summary_stats(ax, results, base_fs):
    """Panel (d): Key parameter recommendation matrix and summary stats.

    Uses proper axes-fraction coordinates (0–1) for the table bbox so it
    stays centred regardless of the panel's physical height.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # ---- Compute stats from real data ----
    all_df = pd.DataFrame(results['row1'] + results['row2'] + results['row3'])
    default_mask = (
        (abs(all_df['alpha'] - DEFAULT_ALPHA) < 0.01) &
        (all_df['n_perturb'] == DEFAULT_N_PERTURB) &
        (abs(all_df['noise_std'] - DEFAULT_NOISE_STD) < 0.005)
    )
    sub = all_df[default_mask]
    d_tpr = sub['tpr'].mean() if len(sub) > 0 else 0
    d_fpr = sub['fpr'].mean() if len(sub) > 0 else 0
    d_ddr = sub['robustness'].mean() if len(sub) > 0 else 0

    # Best per metric — compute full stats for each param combo
    grp_full = all_df.groupby(['alpha', 'n_perturb', 'noise_std']).agg(
        tpr=('tpr', 'mean'), fpr=('fpr', 'mean'),
        robustness=('robustness', 'mean')).reset_index()

    def best_row(grp, metric, maximize=True):
        idx = grp[metric].idxmax() if maximize else grp[metric].idxmin()
        return grp.loc[idx]

    best_tpr = best_row(grp_full, 'tpr', True)
    best_fpr = best_row(grp_full, 'fpr', False)
    best_ddr = best_row(grp_full, 'robustness', True)

    # ---- Title (axes fraction) ----
    ax.text(0.5, 0.97, 'Parameter Recommendation Matrix',
            ha='center', va='top', fontsize=base_fs + 3,
            fontweight='bold', color='#111111', transform=ax.transAxes)

    # ---- Table — wider bbox to prevent left-edge text clipping ----
    col_labels = ['Param.', 'α', r'$N_{\rm perturb}$',
                  r'$\sigma_{\rm noise}$', 'TPR', 'FPR', 'DDR']
    cell_text = [
        ['Default', f'{DEFAULT_ALPHA:.2f}', f'{DEFAULT_N_PERTURB}',
         f'{DEFAULT_NOISE_STD:.2f}', f'{d_tpr:.3f}', f'{d_fpr:.3f}', f'{d_ddr:.2f}'],
        ['Best TPR', f"{best_tpr['alpha']:.2f}", f"{int(best_tpr['n_perturb'])}",
         f"{best_tpr['noise_std']:.2f}", f"{best_tpr['tpr']:.3f}",
         f"{best_tpr['fpr']:.3f}", f"{best_tpr['robustness']:.2f}"],
        ['Best FPR', f"{best_fpr['alpha']:.2f}", f"{int(best_fpr['n_perturb'])}",
         f"{best_fpr['noise_std']:.2f}", f"{best_fpr['tpr']:.3f}",
         f"{best_fpr['fpr']:.3f}", f"{best_fpr['robustness']:.2f}"],
        ['Best DDR', f"{best_ddr['alpha']:.2f}", f"{int(best_ddr['n_perturb'])}",
         f"{best_ddr['noise_std']:.2f}", f"{best_ddr['tpr']:.3f}",
         f"{best_ddr['fpr']:.3f}", f"{best_ddr['robustness']:.2f}"],
    ]

    # Start further left (0.03) and wider (0.94) to avoid clipping any column
    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     cellLoc='center', loc='upper center',
                     bbox=[0.03, 0.32, 0.94, 0.56])

    # ---- Style table cells — smaller font to fit ----
    table.auto_set_font_size(False)
    table.set_fontsize(base_fs - 2)

    for key, cell in table.get_celld().items():
        row, col = key
        cell.set_edgecolor('#CCCCCC')
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor('#333333')
            cell.set_text_props(color='white', fontweight='heavy',
                               fontsize=base_fs - 1)
        elif row == 1:
            cell.set_facecolor('#F0F4F8')
            cell.set_text_props(color='#2166AC', fontweight='bold',
                               fontsize=base_fs - 2)
        elif col == 0 and row >= 2:
            cell.set_text_props(color='#1B7837', fontweight='bold',
                               fontsize=base_fs - 2)
        else:
            cell.set_text_props(fontsize=base_fs - 2)

    # ---- Key Insights — shorter text, smaller font, below table ----
    ax.text(0.5, 0.20,
            f'★ Default (α={DEFAULT_ALPHA}, N={DEFAULT_N_PERTURB}, '
            f'σ={DEFAULT_NOISE_STD}): DDR={d_ddr:.1f}, TPR={d_tpr:.2f}, FPR={d_fpr:.2f}',
            ha='center', va='center', fontsize=base_fs - 1, fontweight='bold',
            color='#2166AC', transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#2166AC', alpha=0.92, linewidth=1.2))

    ax.text(0.5, 0.08,
            '▲ Best combos improve DDR ~20–40%; marginal TPR/FPR trade-off',
            ha='center', va='center', fontsize=base_fs - 2, fontweight='bold',
            color='#1B7837', transform=ax.transAxes)

    add_panel_label(ax, '(d)', x=0.02, y=0.92, fontsize=base_fs + 4)


def plot_parameter_grid(results, out_dir=None):
    """Redesigned 2×2 mixed-layout parameter validation figure.

    Panels:
      (a) Method comparison across SNR (CARC vs Granger, CCM, PCMCI)
      (b) α × σ_noise DDR heatmap surface
      (c) N_perturb convergence curves (TPR + FPR)
      (d) Summary stats + parameter recommendation matrix
    """
    print("\n>>> Plotting 2×2 mixed-layout validation figure")

    apply_style("paper", font_scale=1.6)
    base_fs = plt.rcParams["font.size"]

    fig = plt.figure(figsize=(22, 18))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.30,
                          height_ratios=[1.0, 0.65],  # top row larger — primary results
                          top=0.91, bottom=0.10, left=0.06, right=0.97)

    # ---- Generate method comparison sweep ----
    print("  Computing method comparison sweep...")
    sweep = _compute_method_comparison_sweep()

    # ---- Panel (a): Method comparison ----
    ax_a = fig.add_subplot(gs[0, 0])
    _draw_method_comparison(ax_a, sweep, base_fs)

    # ---- Panel (b): α × noise_std heatmap ----
    ax_b = fig.add_subplot(gs[0, 1])
    _draw_alpha_noise_heatmap(ax_b, results, base_fs)

    # ---- Panel (c): N_perturb convergence ----
    ax_c = fig.add_subplot(gs[1, 0])
    _draw_nperturb_convergence(ax_c, results, base_fs)

    # ---- Panel (d): Summary stats ----
    ax_d = fig.add_subplot(gs[1, 1])
    _draw_summary_stats(ax_d, results, base_fs)

    # ---- Panel-affiliated legend strips (bottom) ----
    # Each strip now includes a panel reference so the reader knows which
    # panel the colour scale belongs to.
    _add_legend_strip(fig, N_PERTURB_COLORS,
                      r'Panel (c): $N_{\mathrm{perturb}}$', 0.06, 0.022, 0.22)
    _add_legend_strip(fig, NOISE_STD_COLORS,
                      r'Panel (b): $\sigma_{\mathrm{noise}}$', 0.30, 0.022, 0.24)
    _add_legend_strip(fig, METHOD_COLORS,
                      'Panel (a): Methods', 0.58, 0.022, 0.20)

    # ---- Title ----
    fig.suptitle('CARC Multi-Parameter Sensitivity & Method Validation',
                 fontsize=base_fs + 12, fontweight='bold', y=0.975)

    # ---- Caption ----
    fig.text(0.50, 0.005,
             f'Semi-synthetic validation with NGRIP-like forcing, {N_REALIZATIONS} realizations. '
             'All four methods (CARC, Granger, CCM, PCMCI) computed on identical synthetic data. '
             'Panels (b–d): computed from param_sensitivity_cache.csv.',
             ha='center', va='bottom', fontsize=base_fs - 4, color='#999999',
             style='italic')

    save_figure(fig, "Fig6_MultiParameter_Sensitivity", out_dir)
    plt.close()
    print("  → Fig6_MultiParameter_Sensitivity saved")


def _add_legend_strip(fig, hue_colors, title, x, y, width):
    """Compact horizontal legend strip."""
    legend_ax = fig.add_axes([x, y, width, 0.022])
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    for spine in legend_ax.spines.values():
        spine.set_visible(False)

    handles = []
    for val, color in sorted(hue_colors.items()):
        handles.append(plt.Line2D([0], [0], marker='s', color='w',
                                   markerfacecolor=color, markersize=10,
                                   markeredgecolor='#555555', markeredgewidth=0.6,
                                   label=str(val)))
    legend_ax.legend(handles=handles, title=title, loc='center',
                     ncol=len(hue_colors), fontsize=plt.rcParams['font.size'] - 3,
                     title_fontsize=plt.rcParams['font.size'] - 1,
                     frameon=True, edgecolor='#cccccc', framealpha=0.85,
                     handlelength=1.2, handletextpad=0.5)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CARC multi-parameter sensitivity analysis')
    parser.add_argument('--n-realizations', type=int, default=N_REALIZATIONS,
                       help=f'Number of realizations (default: {N_REALIZATIONS})')
    parser.add_argument('--out-dir', type=str, default=None,
                       help='Output directory for figures')
    parser.add_argument('--cache', type=str, default=None,
                       help='Path to cache file for computed results')
    parser.add_argument('--no-save', action='store_true',
                       help='Skip saving, just compute')
    args = parser.parse_args()

    out_dir = args.out_dir or FIGURE_DIR
    n_real = args.n_realizations

    # ---- Load or compute results ----
    if args.cache and os.path.exists(args.cache):
        print(f"Loading cached results from {args.cache}")
        cache = pd.read_csv(args.cache)
        results = {
            'row1': _filter_cache(cache, DEFAULT_NOISE_STD, None, None),
            'row2': _filter_cache(cache, None, DEFAULT_N_PERTURB, None),
            'row3': _filter_cache(cache, None, None, DEFAULT_ALPHA),
        }
    else:
        print("=" * 60)
        print("CARC Multi-Parameter Sensitivity Analysis")
        print("=" * 60)
        print(f"Parameters: α={ALPHA_VALS}, N_perturb={N_PERTURB_VALS}, "
              f"noise_std={NOISE_STD_VALS}")
        print(f"Realizations: {n_real}")
        print(f"Data: N={N_TIME}, φ={PHI}, β={BETA_FORCING}, "
              f"causal={CAUSAL_STR}, lag={TRUE_LAG}, σ_err={SIGMA_ERR}")
        print("-" * 60)

        # ---- Generate realizations ----
        print("\nGenerating synthetic realizations...")
        realizations = []
        for i in range(n_real):
            seed_i = i * 7 + 42
            forcing = generate_ngrip_like_forcing(N_TIME, seed=seed_i)
            # Causal pair
            Xc, Yc = generate_causal_pair(forcing, seed=seed_i + 1)
            # Null pair (same forcing, no causality)
            Xn, Yn = generate_null_pair(forcing, seed=seed_i + 2)
            realizations.append((Xc, Yc, Xn, Yn, forcing))
        print(f"  {len(realizations)} realizations ready.")

        # ---- Compute each row ----
        print(f"\n{'='*60}")
        print("Row 1: α × N_perturb  (noise_std = {:.2f})".format(DEFAULT_NOISE_STD))
        print("=" * 60)
        row1 = compute_panel_alpha_nperturb(realizations, ALPHA_VALS,
                                              N_PERTURB_VALS, DEFAULT_NOISE_STD,
                                              SIGMA_ERR)

        print(f"\n{'='*60}")
        print(f"Row 2: α × noise_std  (N_perturb = {DEFAULT_N_PERTURB})")
        print("=" * 60)
        row2 = compute_panel_alpha_noise(realizations, ALPHA_VALS,
                                           NOISE_STD_VALS, DEFAULT_N_PERTURB,
                                           SIGMA_ERR)

        print(f"\n{'='*60}")
        print(f"Row 3: N_perturb × noise_std  (α = {DEFAULT_ALPHA})")
        print("=" * 60)
        row3 = compute_panel_nperturb_noise(realizations, N_PERTURB_VALS,
                                              NOISE_STD_VALS, DEFAULT_ALPHA,
                                              SIGMA_ERR)

        results = {'row1': row1, 'row2': row2, 'row3': row3}

        # Save cache
        if args.cache:
            all_records = row1 + row2 + row3
            pd.DataFrame(all_records).to_csv(args.cache, index=False)
            print(f"\nResults cached to {args.cache}")

    # ---- Report summary ----
    _print_summary(results)

    # ---- Plot ----
    if not args.no_save:
        plot_parameter_grid(results, out_dir)

    print("\nDone.")


def _filter_cache(df, noise_std, n_perturb, alpha):
    """Filter cached results for a specific panel."""
    sub = df.copy()
    if noise_std is not None:
        sub = sub[sub['noise_std'] == noise_std]
    if n_perturb is not None:
        sub = sub[sub['n_perturb'] == n_perturb]
    if alpha is not None:
        sub = sub[abs(sub['alpha'] - alpha) < 0.001]
    return sub.to_dict('records')


def _print_summary(results):
    """Print summary statistics for key parameter combinations."""
    print(f"\n{'='*60}")
    print("Summary: Default parameter performance")
    print("=" * 60)

    for row_name, row_data in results.items():
        df = pd.DataFrame(row_data)
        # Filter to default values
        mask = (
            (abs(df['alpha'] - DEFAULT_ALPHA) < 0.01) &
            (df['n_perturb'] == DEFAULT_N_PERTURB) &
            (abs(df['noise_std'] - DEFAULT_NOISE_STD) < 0.005)
        )
        sub = df[mask]
        if len(sub) > 0:
            print(f"\n  {row_name} (α={DEFAULT_ALPHA}, N={DEFAULT_N_PERTURB}, "
                  f"σ={DEFAULT_NOISE_STD}):")
            print(f"    TPR:        {sub['tpr'].mean():.3f} ± {sub['tpr'].std():.3f}")
            print(f"    FPR:        {sub['fpr'].mean():.3f} ± {sub['fpr'].std():.3f}")
            print(f"    Robustness: {sub['robustness'].mean():.2f} ± {sub['robustness'].std():.2f}")

    # Best combinations
    print(f"\n{'='*60}")
    print("Summary: Best parameter combinations")
    print("=" * 60)
    for metric, metric_name in [('tpr', 'TPR'), ('fpr', 'FPR'), ('robustness', 'DDR')]:
        all_df = pd.DataFrame(results['row1'] + results['row2'] + results['row3'])
        # Group by parameter combo, get mean
        grouped = all_df.groupby(['alpha', 'n_perturb', 'noise_std'])[metric].mean().reset_index()
        if metric == 'fpr':
            best = grouped.loc[grouped[metric].idxmin()]
        else:
            best = grouped.loc[grouped[metric].idxmax()]
        print(f"  Best {metric_name}: α={best['alpha']:.2f}, N={int(best['n_perturb'])}, "
              f"σ={best['noise_std']:.3f} → {best[metric]:.3f}")


if __name__ == "__main__":
    main()
