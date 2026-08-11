#!/usr/bin/env python3
"""
15_sliding_window_carc.py
-------------------------
Sliding-window CARC causality analysis across the 10–60 ka interval.

For each window position t and width W, computes:
  1. ARC deconfounding (α=0.5) using NGRIP as common-forcing proxy
  2. CARC robustness for X→Y (TPR) and Y→X (FPR)
  3. Directional Dominance Ratio (DDR)

Like Fig.4 in correlation-fluctuation literature: different window widths
(W=2,4,8 ka) shown as different coloured time series, revealing the
trade-off between temporal resolution and detection stability.

Output: figures/Fig7_SlidingWindow_CARC.png
"""

import os, sys, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

from src.config import (DATA_DIR, OUTPUT_DIR, FIGURE_DIR, TIER1_FILES, TIER1_SIGMA,
                         AGE_GRID_10_60, DT_HIGH_RES,
                         N_PERTURB as DEFAULT_N_PERTURB,
                         ARC_ALPHA, MAX_GRANGER_LAG, NOISE_STD,
                         CLIMATE_WINDOWS, STABILITY_THRESHOLD)
from src.io import load_record, interpolate_to_grid, standardize
from src.arc import arc_deconfound
from src.granger import robustness
from src.visualization import (apply_style, CARC_COLORS, CARC_COLORS_LIGHT,
                                 CARC_SEQUENTIAL, save_figure)

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Configuration
# ============================================================================
WINDOW_WIDTHS = [2, 4, 8]          # ka — like the paper's l=10,40,400
WINDOW_STEP   = 0.2                # ka
N_PERTURB     = 100                # reduced for speed; still statistically valid
CARC_MAXLAG   = 3                  # reduced maxlag
CARC_ALPHA    = 0.5                # ARC stripping strength
CARC_NOISE    = 0.05               # observational noise fraction
AGE_START     = 10.0               # ka
AGE_END       = 60.0               # ka
DT            = DT_HIGH_RES        # 0.05 ka = 50 yr

# Colour scheme for window widths (matches the paper's grey/blue/red)
WINDOW_COLORS = {
    2: '#333333',  # dark charcoal (was #555555 — stronger for readability)
    4: '#0D4A8A',  # dark blue
    8: '#B31B1B',  # dark red
}
WINDOW_COLORS_LIGHT = {
    2: '#E0E0E0',
    4: '#D1E5F0',
    8: '#FDDBC7',
}

# ============================================================================
# Data loading (reuse pattern from 05_causal_inference_tier1.py)
# ============================================================================

def load_tier1_data():
    """Load Tier 1 records, interpolate to uniform grid, standardize."""
    print("Loading Tier 1 records...")
    raw = {}
    for name, fname in TIER1_FILES.items():
        age, val = load_record(fname, DATA_DIR)
        if age is not None:
            raw[name] = (age, val)
            print(f"  {name}: {len(age)} pts, {age.min():.1f}–{age.max():.1f} ka")

    age_grid = AGE_GRID_10_60  # 10–60 ka at 50-yr steps
    var_names = [v for v in ['NGRIP', 'EDC_dD', 'EDC_CO2', 'EDC_CH4',
                              'Cariaco', 'Cheng2016'] if v in raw]

    interp_data = {}
    for var in var_names:
        age, val = raw[var]
        interp = interpolate_to_grid(age, val, age_grid)
        mask = np.isfinite(interp)
        if np.sum(mask) < 20:
            print(f"  SKIP {var}: too few valid points")
            continue
        # Forward-fill NaN edges
        valid_vals = interp[mask]
        nan_idx = np.where(~mask)[0]
        for i in nan_idx:
            interp[i] = interp[i - 1] if i > 0 else valid_vals[0]
        interp_data[var] = standardize(interp)
        print(f"  {var}: {len(age_grid)} pts on grid")

    return age_grid, interp_data


# ============================================================================
# Sliding-window CARC engine
# ============================================================================

def sliding_window_carc(age_grid, signals, pair, proxy_var,
                         window_widths, window_step,
                         n_perturb=N_PERTURB, alpha=CARC_ALPHA,
                         noise_std=CARC_NOISE, maxlag=CARC_MAXLAG):
    """
    Slide a window across the time axis and compute CARC robustness at each step.

    Parameters
    ----------
    age_grid : np.ndarray — uniform age grid (ka)
    signals : dict — {var_name: np.ndarray} standardized signals
    pair : tuple — (X_name, Y_name) where X→Y is the hypothesized causal direction
    proxy_var : str — variable to use as common-forcing proxy
    window_widths : list of float — window widths in ka
    window_step : float — step size in ka
    n_perturb, alpha, noise_std, maxlag : method parameters

    Returns
    -------
    results : dict of {width: DataFrame} with columns:
        time, tpr, fpr, ddr
    """
    X_name, Y_name = pair
    X_full = signals[X_name]
    Y_full = signals[Y_name]
    proxy_full = signals[proxy_var]

    dt = age_grid[1] - age_grid[0]  # time step
    t_min, t_max = age_grid[0], age_grid[-1]
    sigma_err = TIER1_SIGMA.get(X_name, 1.5)  # combined age error estimate

    results = {}

    for W in window_widths:
        print(f"\n  Window W={W} ka:")
        W_steps = int(W / dt)  # window size in index steps
        step_steps = int(window_step / dt)
        t_starts = np.arange(t_min, t_max - W + window_step, window_step)

        records = []
        n_total = len(t_starts)
        t_start_wall = time.perf_counter()

        for i, t_start in enumerate(t_starts):
            # Find indices in the full grid
            idx_start = np.searchsorted(age_grid, t_start)
            idx_end = idx_start + W_steps
            if idx_end > len(age_grid):
                break

            # Extract windowed data
            X_win = X_full[idx_start:idx_end].copy()
            Y_win = Y_full[idx_start:idx_end].copy()
            P_win = proxy_full[idx_start:idx_end].copy()

            if len(X_win) < 20:
                records.append({'time': t_start + W/2, 'tpr': np.nan,
                               'fpr': np.nan, 'ddr': np.nan})
                continue

            # ARC deconfound within this window
            X_arc = arc_deconfound(X_win, P_win, alpha)
            Y_arc = arc_deconfound(Y_win, P_win, alpha)

            # Robustness for X→Y (TPR direction)
            tpr = robustness(Y_arc, X_arc, sigma_err,
                            n_perturb=n_perturb, noise_std=noise_std,
                            maxlag=maxlag)
            # Robustness for Y→X (FPR direction)
            fpr = robustness(X_arc, Y_arc, sigma_err,
                            n_perturb=n_perturb, noise_std=noise_std,
                            maxlag=maxlag)
            ddr = tpr / max(fpr, 0.01)

            records.append({'time': t_start + W/2, 'tpr': tpr,
                           'fpr': fpr, 'ddr': ddr})

            # Progress
            if (i + 1) % 25 == 0 or i == n_total - 1:
                elapsed = time.perf_counter() - t_start_wall
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (n_total - i - 1) / rate if rate > 0 else 0
                print(f"    {i+1}/{n_total} | {elapsed:.0f}s elapsed | "
                      f"~{remaining:.0f}s remaining | "
                      f"t={t_start:.1f} ka, TPR={tpr:.2f}, FPR={fpr:.2f}")

        results[W] = pd.DataFrame(records)
        print(f"  W={W} ka done in {time.perf_counter()-t_start_wall:.0f}s")

    return results


# ============================================================================
# Plotting — Redesigned v4: time-series-integrated layout
# ============================================================================

SMOOTH_SIGMA_TS = 3    # proxy time series smoothing
SMOOTH_SIGMA_CS = 4    # causality metric smoothing (reduced from 6: less blur)

# Sub-stage category colours — warm (interstadial), hot (Heinrich), cool (stadial)
# Deliberately distinct from MIS band colours below.
SUBSTAGE_TYPE_COLORS = {
    'interstadial': '#D73027',   # warm red
    'heinrich':     '#E66101',   # vivid orange
    'stadial':      '#4393C3',   # cool medium-blue
}
SUBSTAGE_BG_ALPHA = 0.16   # slightly more visible

SUBSTAGES = [
    (12.7, 14.7, 'B-A',        'interstadial'),
    (14.0, 18.0, 'H1',         'heinrich'),
    (19.0, 26.0, 'LGM',        'stadial'),
    (36.0, 40.0, 'H4',         'heinrich'),
    (44.0, 47.0, 'GI-12',      'interstadial'),
    (47.0, 49.0, 'GS-12',      'stadial'),
]

# MIS colours — distinct from sub-stage category colours
# Warm interglacials in magenta-pink/green; cold glacials in blue/brown
MIS_STAGES = [
    (10.0, 14.0,  'MIS 1', '#C51B7D',  'Term. I'),              # magenta-pink (interglacial)
    (14.0, 29.0,  'MIS 2', '#2166AC',  'LGM'),                  # deep blue (glacial)
    (29.0, 57.0,  'MIS 3', '#4DAF4A',  'Interstadial Complex'), # green
    (57.0, 60.0,  'MIS 4', '#8C510A',  'Early Glacial'),        # brown (glacial)
]

MIS_GLACIAL_COLORS = {
    'interglacial': '#D1E5F0',  # odd MIS = warm
    'glacial':      '#FDDBC7',  # even MIS = cold
}

PHASES = [
    (15.0, 22.0, 'Deglacial\nPhase I'),
    (37.0, 52.0, 'Interstadial\nPhase II'),
]



def _load_proxy_for_plotting(pair, age_grid):
    """Load and standardize the proxy time series for the chosen pair."""
    from src.io import load_record, interpolate_to_grid, standardize

    X_name, Y_name = pair
    # Map pair names to config keys
    name_to_file = {
        'NGRIP': ('NGRIP', 'Erhardt_NGRIP_aerosol_d18O_step1.csv'),
        'EDC_dD': ('EDC_dD', '32_Jouzel-2007-dD.csv'),
        'EDC_CO2': ('EDC_CO2', '32_Bereiter et al.-2015-GRL-revised CO2 .xlsx'),
        'EDC_CH4': ('EDC_CH4', '32_CH4 over past 800kyr.csv'),
        'Cariaco': ('Cariaco', '16_MD2622ODP1002-Mo.csv'),
        'Cheng2016': ('Cheng2016', '14_cheng2016composite-noaa.xlsx'),
    }

    def _load_one(name):
        if name in name_to_file:
            cfg_name, fname = name_to_file[name]
        else:
            return None
        age, val = load_record(fname, DATA_DIR)
        if age is None:
            return None
        interp = interpolate_to_grid(age, val, age_grid)
        # Fill NaN edges
        mask = np.isfinite(interp)
        if mask.sum() < 20:
            return None
        valid = interp[mask]
        for i in np.where(~mask)[0]:
            interp[i] = interp[i - 1] if i > 0 else valid[0]
        return standardize(interp)

    return {X_name: _load_one(X_name), Y_name: _load_one(Y_name)}


def _load_orbital_10_60():
    """Load orbital forcing for the 10–60 ka window."""
    import os as _os
    orb_path = _os.path.join(_os.path.dirname(__file__), '..',
                             'data', 'orbital', 'La04_800ka.csv')
    if not _os.path.exists(orb_path):
        return None
    orb = pd.read_csv(orb_path)
    age = orb['age_ka'].values
    mask = (age >= AGE_START - 1) & (age <= AGE_END + 1)
    return {
        'age': age[mask],
        'ecc': orb['eccentricity'].values[mask],
        'obl': orb['obliquity_rad'].values[mask] * 180 / np.pi,
        'pre': np.sin(orb['pibar_rad'].values[mask]),
    }


# ============================================================================
# Main plot function
# ============================================================================

def plot_sliding_window(results, pair, out_dir=None):
    """Fig.7 v5 — Per-W columns: TPR/FPR/DDR stacked vertically within each W.

    Layout:
      (a) Proxy time series (full width)
      (b) Orbital forcing (compact)
      (c-e) W=2,4,8 ka columns — each with TPR/FPR/DDR sub-panels
      (f) MIS/substage ruler (compact strip)
    """
    print("\n>>> Plotting per-W column sliding-window figure")
    X_name, Y_name = pair

    apply_style("paper", font_scale=1.6)
    base_fs = plt.rcParams["font.size"]

    WW = sorted(WINDOW_WIDTHS)

    # ---- Load proxy data ----
    age_grid_full = np.arange(AGE_START, AGE_END + DT, DT)
    proxy_signals = _load_proxy_for_plotting(pair, age_grid_full)
    orbital = _load_orbital_10_60()

    # ---- Figure grid ----
    fig = plt.figure(figsize=(27, 20))

    gs = fig.add_gridspec(4, 1,
        height_ratios=[1.0, 0.32, 2.8, 0.70],  # taller ruler for clear MIS/event separation
        hspace=0.18,
        top=0.958, bottom=0.050, left=0.055, right=0.985)

    # Panels
    ax_ts  = fig.add_subplot(gs[0])   # (a) proxy time series
    ax_orb = fig.add_subplot(gs[1])   # (b) orbital forcing
    ax_mis = fig.add_subplot(gs[3])   # (f) MIS/substage ruler

    # ---- Row 2: 3-column W panels, each column has 3 sub-rows ----
    gs_w = gs[2].subgridspec(3, 3, hspace=0.09, wspace=0.11)

    # ---- (a) Proxy Time Series ----
    _draw_proxy_timeseries(ax_ts, age_grid_full, proxy_signals, X_name, Y_name, base_fs)

    # ---- (b) Orbital Forcing ----
    _draw_orbital_compact(ax_orb, orbital, base_fs)

    # ---- (c–e) Per-W causality columns ----
    w_labels = ['(c) W = 2 ka', '(d) W = 4 ka', '(e) W = 8 ka']
    metric_configs = [
        ('tpr', 'TPR', (-0.05, 1.12), 0.95, '95% threshold'),
        ('fpr', 'FPR', (-0.05, 1.12), 0.05, '5% significance'),
        ('ddr', 'DDR', (-1, 11),     3.0,  'DDR = 3'),
    ]

    for col_idx, (W, wlabel) in enumerate(zip(WW, w_labels)):
        df_w = results[W]
        color = WINDOW_COLORS[W]

        for row_idx, (metric, ylabel, (ylo, yhi), ref_val, ref_label) in enumerate(
            metric_configs):
            ax = fig.add_subplot(gs_w[row_idx, col_idx])
            is_top = (row_idx == 0)
            is_bot = (row_idx == 2)

            # Sub-stage background
            for start, end, _short, category in SUBSTAGES:
                ax.axvspan(start, end, 0, 1,
                           facecolor=SUBSTAGE_TYPE_COLORS[category],
                           alpha=SUBSTAGE_BG_ALPHA, zorder=0, edgecolor='none')

            # Draw data (single W — thick smooth, thin raw)
            _draw_single_window_metric(ax, df_w, metric, color)

            # Axes
            ax.set_ylabel(ylabel, fontsize=base_fs + 1, fontweight='bold',
                          color='#222222', labelpad=6)
            ax.set_ylim(ylo, yhi)
            ax.set_xlim(AGE_END, AGE_START)

            # Reference line
            if metric == 'ddr':
                ax.axhline(ref_val, color='#333333', linestyle='--',
                          linewidth=1.6, alpha=0.55)
                ax.axhline(1.0, color='#999999', linestyle=':',
                          linewidth=1.0, alpha=0.35)
            else:
                ax.axhline(ref_val, color='#333333', linestyle='--',
                          linewidth=1.6, alpha=0.55)

            # Grid
            ax.grid(True, which='major', linestyle='solid', alpha=0.22,
                    linewidth=0.6, color='#999999')
            ax.grid(True, which='minor', linestyle='dotted', alpha=0.10,
                    linewidth=0.4, color='#BBBBBB')
            from matplotlib.ticker import AutoMinorLocator
            ax.yaxis.set_minor_locator(AutoMinorLocator(4))
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))

            # Spines
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['left'].set_linewidth(1.4)
            ax.spines['left'].set_color('#444444')
            ax.tick_params(axis='y', which='major', direction='out',
                           length=4, width=1.0, colors='#333333')

            # X-axis ticks (bottom row only)
            if is_bot:
                ax.spines['bottom'].set_visible(True)
                ax.spines['bottom'].set_linewidth(1.6)
                ax.spines['bottom'].set_color('#444444')
                ax.set_xticks(np.arange(AGE_START, AGE_END + 0.01, 10))
                ax.tick_params(axis='x', which='major', direction='out',
                               length=6, width=1.3, colors='#333333',
                               labelsize=base_fs)
                ax.set_xticks(np.arange(AGE_START, AGE_END + 0.01, 5), minor=True)
                ax.tick_params(axis='x', which='minor', direction='out',
                               length=3, width=0.6, colors='#888888')
                ax.set_xlabel('Age (ka BP)', fontsize=base_fs + 1,
                              fontweight='bold', color='#111111', labelpad=8)
            else:
                ax.tick_params(labelbottom=False)

            # Column title on top sub-panel
            if is_top:
                ax.set_title(wlabel, fontweight='heavy', fontsize=base_fs + 3,
                             color=color, pad=8)

            # Phase blocks on TPR panel
            if metric == 'tpr':
                for x0, x1, label in PHASES:
                    _add_phase_block(ax, x0, x1,
                                     y_lo=0.78, y_hi=0.99,
                                     label=label, n_fade=12)

            # ---- Stats summary (DDR panel, bottom of each column) ----
            if is_bot:
                tpr_v = df_w['tpr'].dropna()
                fpr_v = df_w['fpr'].dropna()
                ddr_v = df_w['ddr'].dropna()
                ddr_v = ddr_v[ddr_v < 20]
                honest = (tpr_v >= 0.95).mean()

                # Compact one-line stats — avoids the dense multi-line box
                stat_str = (f'μ TPR={tpr_v.mean():.2f}  '
                           f'FPR={fpr_v.mean():.2f}  '
                           f'DDR={ddr_v.mean():.1f}  '
                           f'honest={honest:.0%}')
                ax.text(0.98, 0.96, stat_str, transform=ax.transAxes,
                        fontsize=base_fs - 3, family='monospace',
                        va='top', ha='right', color=color,
                        bbox=dict(boxstyle='round,pad=0.35',
                                  facecolor='white',
                                  edgecolor=color,
                                  alpha=0.90, linewidth=1.2),
                        zorder=30)

                # Data-driven annotations
                _add_column_annotations(ax, df_w, color, base_fs)

    # ---- (f) MIS/Substage compact ruler ----
    _draw_mis_substage_ruler(ax_mis, base_fs)

    # ---- Window-width comparison legend ----
    legend_ax = fig.add_axes([0.18, 0.032, 0.64, 0.020])
    legend_ax.set_xticks([]); legend_ax.set_yticks([])
    for spine in legend_ax.spines.values():
        spine.set_visible(False)
    legend_elements = []
    for W in WW:
        legend_elements.append(
            plt.Line2D([0], [0], color=WINDOW_COLORS[W], linewidth=4.0,
                       solid_capstyle='round',
                       label=f'W = {W} ka  (honest = {(results[W]["tpr"].dropna() >= 0.95).mean():.0%})')
        )
    legend_ax.legend(handles=legend_elements, loc='center', ncol=3,
                     fontsize=base_fs - 2, frameon=True,
                     edgecolor='#CCCCCC', framealpha=0.9)

    # ---- Title ----
    fig.suptitle(f'Sliding-Window CARC Causality:  {X_name} → {Y_name}  '
                 f'(10–60 ka BP, ARC α={CARC_ALPHA})',
                 fontsize=base_fs + 7, fontweight='bold', y=0.992, color='#111111')

    save_figure(fig, "Fig7_SlidingWindow_CARC", out_dir)
    plt.close()
    print("  → Fig7_SlidingWindow_CARC saved")


# ============================================================================
# Single-window metric drawing (thick smooth, thin raw)
# ============================================================================

def _draw_single_window_metric(ax, df, col, color):
    """Draw ONE window width on ONE axis.

    Key visual rule: smoothed line is HEAVIER (thick, opaque) than raw line
    (thin, semi-transparent). The smoothed trend is what the reader focuses on.
    """
    time = df['time'].values
    raw = df[col].values.copy()

    # Fill NaN gaps for smoothing
    nan_mask = np.isnan(raw)
    valid = ~nan_mask
    if valid.sum() > 10:
        xx = np.arange(len(raw))
        filled = np.interp(xx, xx[valid], raw[valid])
        smooth = gaussian_filter1d(filled, sigma=SMOOTH_SIGMA_CS)
        smooth[~valid] = np.nan
    else:
        smooth = raw.copy()

    if col == 'ddr':
        raw = np.clip(raw, 0, 10)
        smooth = np.clip(smooth, 0, 10)

    # 1. Raw detail: THIN, SEMI-TRANSPARENT, behind
    ax.plot(time, raw, color=color, linewidth=1.0, alpha=0.38,
            zorder=2, solid_capstyle='round')

    # 2. Smoothed trend: THICK, OPAQUE, on top — visually dominant
    ax.plot(time, smooth, color=color, linewidth=5.5, alpha=0.92,
            zorder=4, solid_capstyle='round')


# ============================================================================
# Column-level annotations
# ============================================================================

def _add_column_annotations(ax_ddr, df, color, base_fs):
    """Data-driven leader-line annotations for one W column's DDR panel.

    Peak label is placed to avoid the stats line at the top-right corner.
    """
    time = df['time'].values
    ddr_f = np.nan_to_num(df['ddr'].values, nan=0.0)
    ddr_c = np.clip(ddr_f, 0, 10)

    # DDR peak — only annotate if clearly above noise
    peak = ddr_c > 5.0
    if peak.sum() > 3:
        pt, pv = time[peak], ddr_c[peak]
        idx = np.nanargmax(pv)
        tp, vp = pt[idx], pv[idx]
        if 12 < tp < 56:
            # Place annotation to the LEFT by default (avoids stats at top-right)
            ann_x = tp - 6
            if ann_x < AGE_START + 3:
                ann_x = tp + 6
            ann_y_peak = min(vp, 8.5)
            # Drop text well below peak to avoid crowding the top
            ann_y_text = max(ann_y_peak - 3.5, 1.0)
            ax_ddr.annotate(f'DDR = {vp:.1f}',
                            xy=(tp, ann_y_peak),
                            xytext=(ann_x, ann_y_text),
                            fontsize=base_fs - 5, fontweight='heavy',
                            color=color, ha='center', va='center',
                            arrowprops=dict(arrowstyle='-', color=color, lw=1.5),
                            zorder=30,
                            bbox=dict(boxstyle='round,pad=0.3',
                                      facecolor='white',
                                      edgecolor=color, alpha=0.88,
                                      linewidth=1.2))

def _draw_proxy_timeseries(ax, age_grid, signals, X_name, Y_name, base_fs):
    """Draw NGRIP and target proxy time series with MIS shading."""
    # MIS glacial/interglacial shading
    mis_windows_10_60 = [
        ("1", 10.0, 14.0), ("2", 14.0, 29.0),
        ("3", 29.0, 57.0), ("4", 57.0, 60.0),
    ]
    for name, s, e in mis_windows_10_60:
        mis_num = int(name)
        bg = MIS_GLACIAL_COLORS['interglacial'] if mis_num % 2 == 1 else MIS_GLACIAL_COLORS['glacial']
        ax.axvspan(s, e, facecolor=bg, alpha=0.35, zorder=0, edgecolor='none')

    ax.axhline(0, color='#999999', linewidth=0.8, linestyle=':', alpha=0.5, zorder=1)

    # Plot each proxy
    signal_colors = {
        'NGRIP': CARC_COLORS['red'],
        'EDC_dD': CARC_COLORS['blue'],
        'EDC_CO2': CARC_COLORS['green'],
        'EDC_CH4': CARC_COLORS['orange'],
        'Cariaco': CARC_COLORS['teal'],
        'Cheng2016': CARC_COLORS['purple'],
    }

    for name, color in signal_colors.items():
        if name in signals and signals[name] is not None:
            sig = signals[name]
            smooth = gaussian_filter1d(sig, sigma=SMOOTH_SIGMA_TS)
            ax.plot(age_grid, sig, color=color, linewidth=0.6, alpha=0.35, zorder=2)
            ax.plot(age_grid, smooth, color=color, linewidth=2.2, alpha=0.92,
                    label=name, zorder=3)

    ax.set_ylabel('Standardized\nProxy Value', fontsize=base_fs + 1,
                  fontweight='bold', color='#222222', labelpad=6)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlim(AGE_END, AGE_START)

    # Legend
    ax.legend(fontsize=base_fs, loc='upper left', framealpha=0.85, ncol=2)

    # MIS labels on top
    for name, s, e in mis_windows_10_60:
        mid = (s + e) / 2
        ax.text(mid, 3.25, f'MIS {name}', fontsize=base_fs - 3,
                color='#666666', ha='center', va='bottom',
                fontstyle='italic', zorder=5)

    # Styling
    ax.grid(True, alpha=0.12, linestyle='dotted')
    ax.tick_params(labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['left'].set_color('#444444')
    ax.tick_params(axis='y', which='major', direction='out',
                   length=5, width=1.2, colors='#333333')

    title_str = f'(a) Proxy Records: {X_name} & {Y_name}  (10–60 ka BP)'
    ax.set_title(title_str, fontweight='bold', fontsize=base_fs + 2,
                 loc='left', pad=6)


# ============================================================================
# Panel (b): Orbital Forcing (compact)
# ============================================================================

def _draw_orbital_compact(ax, orbital, base_fs):
    """Compact orbital forcing panel — eccentricity + obliquity."""
    if orbital is None:
        ax.text(0.5, 0.5, 'Orbital data not available', ha='center', va='center',
                fontsize=base_fs, color='#999999', transform=ax.transAxes)
        ax.set_ylabel('Orbital\nForcing', fontsize=base_fs + 1, fontweight='bold')
        return

    age = orbital['age']
    ecc = orbital['ecc']
    obl = orbital['obl']
    obl_smooth = gaussian_filter1d(obl, sigma=3)

    # Eccentricity (fill + line)
    ax.fill_between(age, 0, ecc, color=CARC_COLORS['blue'], alpha=0.22, linewidth=0)
    ax.plot(age, ecc, color=CARC_COLORS['blue'], linewidth=1.8, alpha=0.85,
            label='Eccentricity')
    ecc_smooth = gaussian_filter1d(ecc, sigma=5)
    ax.plot(age, ecc_smooth, color=CARC_COLORS['navy'], linewidth=2.2, alpha=0.85)

    ax.set_ylabel('Eccentricity', color=CARC_COLORS['blue'], fontsize=base_fs - 1,
                  fontweight='bold')
    ax.tick_params(axis='y', labelcolor=CARC_COLORS['blue'], labelsize=base_fs - 3)

    # Obliquity on twin axis
    ax2 = ax.twinx()
    ax2.plot(age, obl_smooth, color=CARC_COLORS['red'], linewidth=2.0, alpha=0.80,
             label='Obliquity')
    ax2.set_ylabel('Obliquity (°)', color=CARC_COLORS['red'], fontsize=base_fs - 1,
                   fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=CARC_COLORS['red'], labelsize=base_fs - 3)

    ax.set_xlim(AGE_END, AGE_START)
    ax.tick_params(labelbottom=False)

    # Styling
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['left'].set_color(CARC_COLORS['blue'])
    ax.tick_params(axis='y', which='major', direction='out', length=4)

    ax.grid(True, alpha=0.10, linestyle='dotted')

    # Legend
    from matplotlib.lines import Line2D
    orb_lines = [
        Line2D([0], [0], color=CARC_COLORS['blue'], linewidth=1.8, label='Eccentricity'),
        Line2D([0], [0], color=CARC_COLORS['red'], linewidth=2.0, label='Obliquity (sm.)'),
    ]
    ax.legend(handles=orb_lines, loc='upper right', fontsize=base_fs - 4,
             framealpha=0.7, ncol=2)

    ax.set_title('(b) Orbital Forcing  (Laskar et al. 2004)',
                 fontweight='bold', fontsize=base_fs, loc='left', pad=4)


# ============================================================================
# Panel (f): MIS/Substage Combined Ruler
# ============================================================================

def _draw_mis_substage_ruler(ax, base_fs):
    """Combined MIS + substage ruler — MIS on top, climate events below.

    Layout (top to bottom):
      - MIS stage bands (4→3→2→1, oldest→youngest left→right)
      - Age tick marks with "Age (ka BP)" label
      - Climate event bands (GS-12, GI-12, H4, LGM, H1, B-A)
      - Three category strips (Interstadial / Heinrich / Stadial) as legend
    """
    ax.set_xlim(AGE_END, AGE_START)
    ax.set_ylim(-0.55, 1.05)

    N_GRAD = 8  # number of gradient strips per edge

    # ====================================================================
    # Helper: draw a band with gradient-faded left & right edges
    # ====================================================================
    def _gradient_band(x0, x1, y_lo, y_hi, color, fade_frac=0.08, alpha_main=0.85):
        """Draw a horizontal band whose left & right edges fade to transparent."""
        w = x1 - x0
        fade_w = w * fade_frac
        # Main solid body
        ax.axvspan(x0 + fade_w, x1 - fade_w, y_lo, y_hi,
                   facecolor=color, alpha=alpha_main, zorder=2, edgecolor='none')
        # Left fade
        for i in range(N_GRAD):
            frac = (i + 0.5) / N_GRAD
            a = alpha_main * (np.sin(frac * np.pi / 2) ** 2)
            a0 = x0 + (i / N_GRAD) * fade_w
            a1 = x0 + ((i + 1) / N_GRAD) * fade_w
            if a1 > a0:
                ax.axvspan(a0, a1, y_lo, y_hi,
                           facecolor=color, alpha=a, zorder=2, edgecolor='none')
        # Right fade
        for i in range(N_GRAD):
            frac = (i + 0.5) / N_GRAD
            a = alpha_main * (np.cos(frac * np.pi / 2) ** 2)
            a0 = x1 - fade_w + (i / N_GRAD) * fade_w
            a1 = x1 - fade_w + ((i + 1) / N_GRAD) * fade_w
            if a1 > a0:
                ax.axvspan(a0, a1, y_lo, y_hi,
                           facecolor=color, alpha=a, zorder=2, edgecolor='none')

    # ====================================================================
    # 1. MIS bands (top: y=0.55–0.94) — MIS 4→3→2→1 left→right
    # ====================================================================
    for x0, x1, label, color, _desc in MIS_STAGES:
        _gradient_band(x0, x1, 0.55, 0.94, color, alpha_main=0.88)
        mid = (x0 + x1) / 2
        w = x1 - x0
        fs_label = base_fs if w > 8 else base_fs - 2
        ax.text(mid, 0.745, label, ha='center', va='center',
                fontsize=fs_label, fontweight='heavy', color='white', zorder=4,
                path_effects=[pe.withStroke(linewidth=1.2, foreground='black')])
    # "MIS" label — right edge, inside MIS band
    ax.text(AGE_END + 2.5, 0.745, 'MIS', ha='right', va='center',
            fontsize=base_fs, fontweight='heavy', color='white', zorder=5,
            clip_on=False,
            path_effects=[pe.withStroke(linewidth=1.5, foreground='#333333')])

    # ====================================================================
    # 2. Age tick marks (y=0.49–0.54) — thin strip, tight between MIS and events
    # ====================================================================
    # Dark gray horizontal axis line at base of age ruler
    ax.axhline(0.49, color='#555555', linewidth=1.8, alpha=0.85,
               zorder=5, clip_on=False)
    for ta in np.arange(AGE_START, AGE_END + 0.01, 5):
        ax.plot([ta, ta], [0.49, 0.54], color='#222222', linewidth=1.8,
                solid_capstyle='butt', zorder=5, clip_on=False)
        ax.text(ta, 0.545, f'{ta:.0f}', ha='center', va='bottom',
                fontsize=base_fs - 1, fontweight='bold', color='#111111',
                zorder=5, clip_on=False)
    for ta in np.arange(AGE_START, AGE_END + 0.01, 1):
        if ta % 5 != 0:
            ax.plot([ta, ta], [0.49, 0.52], color='#888888', linewidth=0.5,
                    solid_capstyle='butt', zorder=4, clip_on=False)
    # "Age (ka BP)" label — right edge
    ax.text(AGE_END + 2.5, 0.515, 'Age\n(ka BP)', ha='right', va='center',
            fontsize=base_fs - 1, fontweight='bold', color='#111111',
            zorder=5, clip_on=False)

    # ====================================================================
    # 3. Climate Event bands (y=0.14–0.48) — tight against age ruler above
    # ====================================================================
    for start, end, short, category in SUBSTAGES:
        color = SUBSTAGE_TYPE_COLORS[category]
        # Use reduced fade for narrow bands so text stays inside solid colour
        w = end - start
        fade_frac = 0.04 if w < 3.0 else 0.08
        _gradient_band(start, end, 0.14, 0.48, color, fade_frac, alpha_main=0.85)
        mid = (start + end) / 2
        if w >= 2.5:
            ax.text(mid, 0.31, short, ha='center', va='center',
                    fontsize=base_fs - 1, fontweight='heavy', color='white', zorder=4,
                    path_effects=[pe.withStroke(linewidth=1.0, foreground='#333333')])
        elif w >= 1.0:
            ax.text(mid, 0.31, short, ha='center', va='center',
                    fontsize=base_fs - 2, fontweight='heavy', color='white', zorder=4,
                    path_effects=[pe.withStroke(linewidth=1.0, foreground='#333333')])
    # "Climate Events" label — placed in blank gap at right edge (x=11, below MIS 1)
    ax.text(11.5, 0.31, 'Climate\nEvents', ha='center', va='center',
            fontsize=base_fs - 1, fontweight='bold', fontstyle='italic',
            color='#333333', zorder=5, clip_on=False)

    # ====================================================================
    # 4. Category strips (bottom: y=-0.35 to 0.00) — staircase label layout
    # ====================================================================
    cat_strips = [
        ('Interstadial', SUBSTAGE_TYPE_COLORS['interstadial'], -0.00, -0.12,  15),
        ('Heinrich',     SUBSTAGE_TYPE_COLORS['heinrich'],     -0.12, -0.24,  35),
        ('Stadial',      SUBSTAGE_TYPE_COLORS['stadial'],      -0.24, -0.35,  52),
    ]
    cat_y_lookup = {
        'interstadial': (-0.00 + -0.12) / 2,
        'heinrich':     (-0.12 + -0.24) / 2,
        'stadial':      (-0.24 + -0.35) / 2,
    }
    for label, color, y_lo, y_hi, x_label in cat_strips:
        ax.axhspan(y_lo, y_hi, 0.0, 1.0,
                   facecolor=color, alpha=0.78, zorder=2, edgecolor='white',
                   linewidth=1.0, clip_on=False)
        ax.text(x_label, (y_lo + y_hi) / 2, label,
                ha='center', va='center', fontsize=base_fs - 1,
                fontweight='heavy', color='white', zorder=5, clip_on=False,
                path_effects=[pe.withStroke(linewidth=1.2, foreground='#333333')])

    # ====================================================================
    # 5. Vertical connectors: each event → its category strip
    #    From event band bottom (0.14) down to category strip
    # ====================================================================
    for start, end, short, category in SUBSTAGES:
        color = SUBSTAGE_TYPE_COLORS[category]
        cat_y_mid = cat_y_lookup.get(category)
        if cat_y_mid is None:
            continue
        mid = (start + end) / 2
        ax.plot([mid, mid], [0.14, cat_y_mid], color=color,
                linewidth=1.8, alpha=0.50, solid_capstyle='round',
                zorder=3, clip_on=False)
        ax.scatter(mid, 0.14, s=20, c=color, edgecolors='white',
                   linewidth=0.8, zorder=6, clip_on=False)

    # ====================================================================
    # 6. Separator lines between zones
    # ====================================================================
    ax.axhline(0.55, color='#CCCCCC', linewidth=0.8, alpha=0.4,
               zorder=3, clip_on=False)  # MIS bottom / age top
    ax.axhline(0.48, color='#CCCCCC', linewidth=0.8, alpha=0.4,
               zorder=3, clip_on=False)  # age bottom / events top
    ax.axhline(0.12, color='#CCCCCC', linewidth=0.8, alpha=0.4,
               zorder=3, clip_on=False)  # events bottom / categories

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


# ============================================================================
# Phase block annotation
# ============================================================================

def _add_phase_block(ax, x_start, x_end, y_lo=0, y_hi=1, label='', n_fade=20):
    """Gradient-faded annotation block.

    y_lo / y_hi are in *axes fraction* (same as axvspan ymin/ymax).
    Text is placed using a blended transform so its y is also in axes fraction,
    guaranteeing it stays centred inside the shaded block.
    """
    from matplotlib.transforms import blended_transform_factory
    x_span = x_end - x_start
    fw = x_span * 0.12
    for i in range(n_fade):
        frac = (i + 0.5) / n_fade
        a = np.sin(frac * np.pi / 2) ** 1.5
        x0 = x_start + (i / n_fade) * fw
        x1 = x_start + ((i + 1) / n_fade) * fw
        if x1 > x0:
            ax.axvspan(x0, x1, y_lo, y_hi,
                       facecolor='#1A1A1A', alpha=a * 0.50, zorder=20, edgecolor='none')
    fl, fr = x_start + fw, x_end - fw
    if fl < fr:
        ax.axvspan(fl, fr, y_lo, y_hi,
                   facecolor='#1A1A1A', alpha=0.50, zorder=20, edgecolor='none')
    for i in range(n_fade):
        frac = (i + 0.5) / n_fade
        a = np.cos(frac * np.pi / 2) ** 1.5
        x0 = fr + (i / n_fade) * fw
        x1 = fr + ((i + 1) / n_fade) * fw
        if x1 > x0:
            ax.axvspan(x0, x1, y_lo, y_hi,
                       facecolor='#1A1A1A', alpha=a * 0.50, zorder=20, edgecolor='none')
    mid = (x_start + x_end) / 2
    ym = (y_lo + y_hi) / 2
    # Blended transform: x in data, y in axes fraction — matches axvspan
    trans_blend = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(mid, ym, label, ha='center', va='center',
            fontsize=plt.rcParams['font.size'] + 1,
            fontweight='heavy', color='white', zorder=23,
            transform=trans_blend)


# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Sliding-window CARC causality analysis')
    parser.add_argument('--pair', type=str, default='NGRIP,EDC_dD',
                       help='Variable pair to analyze (default: NGRIP,EDC_dD)')
    parser.add_argument('--out-dir', type=str, default=None,
                       help='Output directory for figures')
    parser.add_argument('--cache', type=str, default='outputs/sliding_window_cache.csv',
                       help='Cache file path')
    parser.add_argument('--no-save', action='store_true',
                       help='Skip saving')
    args = parser.parse_args()

    out_dir = args.out_dir or FIGURE_DIR
    pair = tuple(args.pair.split(','))

    # ---- Load or compute ----
    if args.cache and os.path.exists(args.cache):
        print(f"Loading cached results from {args.cache}")
        cache_df = pd.read_csv(args.cache)
        results = {}
        for W in WINDOW_WIDTHS:
            results[W] = cache_df[cache_df['window_width'] == W].copy()
        print(f"  Loaded {len(cache_df)} records across {len(results)} window widths")
    else:
        print("=" * 60)
        print("Sliding-Window CARC Causality Analysis")
        print("=" * 60)
        print(f"Pair: {pair[0]} ↔ {pair[1]}")
        print(f"Window widths: {WINDOW_WIDTHS} ka, step: {WINDOW_STEP} ka")
        print(f"Parameters: α={CARC_ALPHA}, N_perturb={N_PERTURB}, "
              f"maxlag={CARC_MAXLAG}, noise_std={CARC_NOISE}")
        print("-" * 60)

        # Load data
        age_grid, signals = load_tier1_data()

        if pair[0] not in signals or pair[1] not in signals:
            print(f"ERROR: Pair {pair} not both in available data: {list(signals.keys())}")
            sys.exit(1)

        proxy_var = 'NGRIP'
        if proxy_var not in signals:
            proxy_var = list(signals.keys())[0]
        print(f"Proxy for ARC: {proxy_var}")

        # Run sliding window
        print(f"\nRunning sliding-window CARC ({len(WINDOW_WIDTHS)} widths × "
              f"~{int((AGE_END-AGE_START)/WINDOW_STEP)} positions)...")
        t0 = time.perf_counter()
        results = sliding_window_carc(age_grid, signals, pair, proxy_var,
                                       WINDOW_WIDTHS, WINDOW_STEP,
                                       n_perturb=N_PERTURB, alpha=CARC_ALPHA,
                                       noise_std=CARC_NOISE, maxlag=CARC_MAXLAG)
        elapsed = time.perf_counter() - t0
        print(f"\nTotal computation: {elapsed:.0f}s ({elapsed/60:.1f} min)")

        # Save cache
        if args.cache:
            all_dfs = []
            for W, df in results.items():
                df = df.copy()
                df['window_width'] = W
                all_dfs.append(df)
            cache_df = pd.concat(all_dfs, ignore_index=True)
            cache_df.to_csv(args.cache, index=False)
            print(f"Results cached to {args.cache}")

    # ---- Report ----
    _print_summary(results, pair)

    # ---- Plot ----
    if not args.no_save:
        plot_sliding_window(results, pair, out_dir)

    print("\nDone.")


def _print_summary(results, pair):
    """Print summary statistics."""
    print(f"\n{'='*60}")
    print(f"Summary: {pair[0]} ↔ {pair[1]}")
    print(f"{'='*60}")
    for W in sorted(results.keys()):
        df = results[W]
        tpr = df['tpr'].dropna()
        fpr = df['fpr'].dropna()
        ddr = df['ddr'].dropna()
        ddr = ddr[ddr < 20]
        frac = (tpr >= 0.95).mean()
        print(f"  W={W} ka: TPR={tpr.mean():.3f}±{tpr.std():.3f}, "
              f"FPR={fpr.mean():.3f}±{fpr.std():.3f}, "
              f"DDR={ddr.mean():.1f}±{ddr.std():.1f}, "
              f"honest={frac:.1%} ({len(tpr)} windows)")


if __name__ == "__main__":
    main()
