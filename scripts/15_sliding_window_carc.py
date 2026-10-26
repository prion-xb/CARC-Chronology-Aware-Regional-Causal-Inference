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
from matplotlib.patches import Rectangle
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
# Axis / grid treatment — matches the reference figure FigS_Stacking_0-800:
# dark charcoal axis family (#2a2927 frame / #282626 ticks) plus a visible
# (~60% opacity) thin horizontal grid. The previous #444444 1.4-pt frame
# washed out at print size; this palette survives 300-dpi downsampling.
# ============================================================================
AX_COL       = '#2a2927'   # spines / frame
AX_TICK_COL  = '#282626'   # tick marks
AX_LABEL_COL = '#111111'   # axis titles / tick labels
GRID_COL     = '#3b3e3e'
GRID_ALPHA   = 0.35   # v11: 0.30→0.35 — hairline grid must stay clearly visible (user could not see the v10 thinning)
GRID_LW      = 0.3    # v10: hairline grid (requested: keep gridlines thin) — 0.6pt read thick
SPINE_LW     = 2.0

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
# Plotting — v7 restack: CENOGRID (Westerhold 2020 Fig.1) composition
# ============================================================================

SMOOTH_SIGMA_TS = 1    # proxy time series smoothing (Westerhold-level sharpness:
                       # sigma 1 only clips 50-yr noise spikes, no airbrush)
SMOOTH_SIGMA_CS = 1    # causality metric smoothing (single thin line, v8)

# Sub-stage category colours — Westerhold-2020-level muted family (v8,
# user feedback 2026-09-26: the saturated red/orange/blue read as cheap).
# Low chroma, unified mid lightness = the "premium" look of CENOGRID.
SUBSTAGE_TYPE_COLORS = {
    'interstadial': '#BD5954',   # muted soft red (extracted from Westerhold)
    'heinrich':     '#C0925E',   # muted warm ochre
    'stadial':      '#7A8FB8',   # muted steel blue (extracted from Westerhold)
}
SUBSTAGE_BG_ALPHA = 0.12   # light: full-width panels keep the shading recessive

SUBSTAGES = [
    (12.7, 14.7, 'B-A',        'interstadial'),
    (14.0, 18.0, 'H1',         'heinrich'),
    (19.0, 26.0, 'LGM',        'stadial'),
    (36.0, 40.0, 'H4',         'heinrich'),
    (44.0, 47.0, 'GI-12',      'interstadial'),
    (47.0, 49.0, 'GS-12',      'stadial'),
]

# MIS stages — v10: mirror the Westerhold state-bar's colour PROPORTIONS
# (user feedback 2026-09-26: the source Epoch bar reads brown-first; its
# deep coolhouse blue is only a small sliver at the oldest/left edge).
# Warm stages = warmhouse tans (brown family dominates, as in the source),
# cold stages = pale cool blues, deep blue reserved for the narrow
# left-edge MIS 4 — exactly where the source puts its deep blue.
MIS_STAGES = [
    (10.0, 14.0,  'MIS 1', '#CCA480',  'Term. I'),              # warmhouse tan (interglacial)
    (14.0, 29.0,  'MIS 2', '#8CC6E8',  'LGM'),                  # coolhouse2 pale blue (glacial)
    (29.0, 57.0,  'MIS 3', '#DBAB70',  'Interstadial Complex'), # warmhouse2 ochre (mild)
    (57.0, 60.0,  'MIS 4', '#3F74A7',  'Early Glacial'),        # coolhouse deep blue (oldest edge)
]

MIS_GLACIAL_COLORS = {
    'interglacial': '#D1E5F0',  # odd MIS = warm
    'glacial':      '#FDDBC7',  # even MIS = cold
}

# v13: phase bands deleted entirely — the event bar above carries the
# event alignment at the same x's, so the grey bands duplicated it (the
# last redundancy left over from v12); the phase vocabulary belongs in
# the caption/text only.



def _load_proxy_for_plotting(pair, age_grid):
    """Load and standardize the proxy time series for the chosen pair."""
    from src.io import load_record, interpolate_to_grid, standardize

    X_name, Y_name = pair
    # Map pair names to config keys
    name_to_file = {
        'NGRIP': ('NGRIP', 'NGRIP_d18O.csv'),
        'EDC_dD': ('EDC_dD', 'EDC_dD.csv'),
        'EDC_CO2': ('EDC_CO2', 'EDC_CO2.xlsx'),
        'EDC_CH4': ('EDC_CH4', 'EDC_CH4.csv'),
        'Cariaco': ('Cariaco', 'Cariaco_Mo.csv'),
        'Cheng2016': ('Cheng2016', 'Cheng2016_d18O.xlsx'),
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


# ============================================================================
# Main plot function
# ============================================================================

def plot_sliding_window(results, pair, out_dir=None):
    """Fig.7 v7 — Westerhold-2020-style restack (user feedback 2026-09-22).

    Layout — single column of full-width panels, ONE shared time axis:
      TOP: a thin UNTITLED stratigraphy band (user feedback 2026-09-22:
      the geological axis must neither be its own panel nor be folded
      into panel (a)) — two thin Westerhold GTS colour bars (MIS stages
      + climate sub-stages, labels inside), the shared ink age axis and
      the category key row. No panel letter, no title, no frame: it
      reads as the figure's axis header, not as a panel.
      (a) Proxy time series — MIS shading ties to the MIS bar above
      (b) TPR / (c) FPR / (d) DDR — three W series per panel

    Vertically compact (user feedback 2026-09-22): panels kept short,
    inter-panel gaps tightened (hspace 0.26) and titles pulled in
    (pad 7).

    v6's 3×3 per-W columns were too crowded at 178 mm print width (user
    feedback); the W series colours already encode window width, so the
    columns were redundant. Composition follows Westerhold et al. 2020
    Fig.1: full-width stacked panels, height proportional to information
    density, no suptitle — panel letters/titles + caption carry context.
    """
    print("\n>>> Plotting restacked sliding-window figure")
    X_name, Y_name = pair

    apply_style("paper", font_scale=1.0)  # print size: base ~8 pt
    base_fs = plt.rcParams["font.size"]

    WW = sorted(WINDOW_WIDTHS)

    # ---- Load proxy data ----
    age_grid_full = np.arange(AGE_START, AGE_END + DT, DT)
    proxy_signals = _load_proxy_for_plotting(pair, age_grid_full)

    # ---- Figure grid: GTS band on TOP, panels, age axis at the BOTTOM ----
    # v15: the shared age axis moves to the figure bottom (user 2026-09-27:
    # "axis belongs at the bottom" — the age numbers must sit under the whole figure).
    # Row 0 = stratigraphy band (pure bar, NO axis of its own), rows 1-4 =
    # panels (a)-(d), row 5 = the single age-axis row (black line +
    # downward ticks + numbers + unit, Westerhold p2 geomag-bar style).
    # Black lines are separators: the ink line under the band divides the
    # GTS bar zone from the curve zone, the bottom axis line closes the
    # column — Westerhold p4 layout.
    fig = plt.figure(figsize=(6.93, 6.10))  # 176 mm wide (Science column)

    gs = fig.add_gridspec(6, 1,
        height_ratios=[0.85, 1.40, 0.95, 0.95, 0.95, 0.42],
        hspace=0.26,
        top=0.965, bottom=0.05, left=0.085, right=0.985)

    ax_strip = fig.add_subplot(gs[0])                        # untitled GTS band
    ax_ts    = fig.add_subplot(gs[1])                        # (a) proxy
    axs      = [fig.add_subplot(gs[2 + i]) for i in range(3)]  # (b)(c)(d)
    ax_age   = fig.add_subplot(gs[5])                        # shared age axis

    # ---- Top: thin untitled stratigraphy band (no panel letter/title) ----
    _draw_stratigraphy_band(ax_strip, base_fs)

    # ---- Bottom: the figure's ONE shared age axis (numbers live here) ----
    _draw_age_axis(ax_age, base_fs)

    # ---- (a) Proxy Time Series (v13: W legend moved to (b)) ----
    _draw_proxy_timeseries(ax_ts, age_grid_full, proxy_signals,
                           X_name, Y_name, base_fs, results)

    # ---- (b–d) TPR / FPR / DDR — three W series per panel ----
    metric_configs = [
        ('tpr', 'TPR', (-0.05, 1.12), 0.95, '95% threshold'),
        ('fpr', 'FPR', (-0.05, 1.12), 0.05, '5% significance'),
        ('ddr', 'DDR', (-1, 11),     3.0,  'DDR = 3'),
    ]

    for idx, (metric, ylabel, (ylo, yhi), ref_val, ref_label) in enumerate(
        metric_configs):
        ax = axs[idx]

        # Sub-stage background (paper dictionary colours, light).
        # The colour blocks are named by the sub-stage bar in the
        # stratigraphy band above (same x alignment), so no in-panel
        # labels are needed — in-panel labels sat on the data curves
        # (occlusion, user feedback).
        for n_ss, (start, end, _short, category) in enumerate(SUBSTAGES):
            span = ax.axvspan(start, end, 0, 1,
                              facecolor=SUBSTAGE_TYPE_COLORS[category],
                              alpha=SUBSTAGE_BG_ALPHA, zorder=0,
                              edgecolor='none')
            span.set_gid(f'layer-shading-sub-{n_ss:02d}')

        # Data — one thick-smooth / thin-raw pair per window width
        for W in WW:
            _draw_single_window_metric(ax, results[W], metric,
                                       WINDOW_COLORS[W])

        # Axes — v12: no rotated ylabel; the title line '({letter}) TPR'
        # already names the metric (the old ylabel repeated it verbatim)
        ax.set_ylim(ylo, yhi)
        ax.set_xlim(AGE_END, AGE_START)
        # v11: every-5-ka x ticks anchor the VERTICAL grid lines but stay
        # invisible themselves — the band above still carries the shared
        # age axis (grid only draws where ticks exist, so set_xticks([])
        # had silently killed the vertical grid).
        ax.set_xticks(np.arange(AGE_START, AGE_END + 0.01, 5.0))
        ax.tick_params(axis='x', which='major', labelbottom=False, length=0)

        # Reference line(s) — left end, white halo label
        if metric == 'ddr':
            ref1 = ax.axhline(ref_val, color='#333333', linestyle='--',
                              linewidth=1.0, alpha=0.55)
            ref1.set_gid('layer-ref-main')
            ref2 = ax.axhline(1.0, color='#999999', linestyle=':',
                              linewidth=1.0, alpha=0.35)
            ref2.set_gid('layer-ref-unit')
            lbl_y, lbl_va = ref_val + 0.25, 'bottom'
            # unit-skill line (DDR = 1) gets a muted companion label
            unit_lbl = ax.text(AGE_END - 1.2, 1.0 - 0.15, 'DDR = 1',
                               ha='left', va='top',
                               fontsize=base_fs - 1, color='#777777',
                               bbox=dict(boxstyle='round,pad=0.22',
                                         facecolor='white', edgecolor='none',
                                         alpha=0.75), zorder=25)
            unit_lbl.set_gid('layer-ref-label')
        else:
            ref = ax.axhline(ref_val, color='#333333', linestyle='--',
                             linewidth=1.0, alpha=0.55)
            ref.set_gid('layer-ref-main')
            # v13: TPR label hugs its dashed line (the phase bands are
            # gone — nothing occupies the strip under the 95% line, so
            # the old 0.14-below placement stranded the label far from
            # the line it names)
            if metric == 'tpr':
                lbl_y, lbl_va = ref_val - 0.02, 'top'
            else:
                lbl_y, lbl_va = ref_val + 0.06, 'bottom'
        ref_lbl = ax.text(AGE_END - 1.2, lbl_y, ref_label,
                          ha='left', va=lbl_va, fontsize=base_fs - 1,
                          color='#333333',
                          bbox=dict(boxstyle='round,pad=0.28',
                                    facecolor='white', edgecolor='none',
                                    alpha=0.85), zorder=25)
        ref_lbl.set_gid('layer-ref-label')

        # Grid — v11: full hairline grid, both directions (user: the
        # vertical lines were missing, only horizontal ones drawn)
        ax.grid(True, which='major', axis='both', linestyle='solid',
                alpha=GRID_ALPHA, linewidth=GRID_LW, color=GRID_COL)
        from matplotlib.ticker import AutoMinorLocator
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))

        # Spines — dark charcoal, print-visible thickness
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_linewidth(SPINE_LW)
        ax.spines['left'].set_color(AX_COL)
        ax.tick_params(axis='y', which='major', direction='out',
                       length=4, width=1.5, colors=AX_TICK_COL)

        # Panel letter + metric title (no suptitle — CENOGRID keeps panels bare)
        ax.set_title(f'({chr(98 + idx)}) {ylabel}', loc='left',
                     fontweight='bold', fontsize=base_fs + 1,
                     color='#111111', pad=7)

        # v13: W legend lives HERE (panel b, top-right) — the window
        # colours only appear in the metric panels, so the old placement
        # in (a) sat in a panel where nothing was window-coloured
        # (misleading). The v12 grey phase bands are gone too: the event
        # bar directly above, at the same x's, already carries the event
        # alignment, and the bands were the figure's last unexplained
        # boxes — with them away the top of (b) is free for this legend.
        if metric == 'tpr':
            handles = [plt.Line2D([], [], color=WINDOW_COLORS[W],
                                  linewidth=1.5, solid_capstyle='round',
                                  label=f'W = {W} ka')
                       for W in WW]
            ax.legend(handles=handles, loc='upper right',
                      fontsize=base_fs - 1, frameon=True,
                      framealpha=0.85, borderaxespad=0.35,
                      handlelength=1.5)

        # v12: no per-panel 'mean (W=2/4/8)' stats strings — three ~30-char
        # haloed lines floating over the upper-right curve area; the curve
        # levels already show the means graphically and the numbers belong
        # in the text/table (easily restored if reviewers want them).

        # DDR peak annotation — the single most prominent peak
        if metric == 'ddr':
            _add_ddr_annotation(ax, results, base_fs)

    # semantic SVG layers for the Adobe Illustrator workflow — see AI_EDITING.md
    fig._svg_layer_order = ('frame', 'background', 'shading', 'ref',
                            'data', 'phase', 'markers', 'text',
                            'stats', 'annotations', 'legend')

    # SVG with editable text (svg.fonttype "none") for the Inkscape/AI workflow
    plt.rcParams["svg.fonttype"] = "none"
    save_figure(fig, "Fig7_SlidingWindow_CARC", out_dir,
                fmts=("pdf", "png", "tif", "svg"))
    plt.close()
    print("  → Fig7_SlidingWindow_CARC saved")


# ============================================================================
# Single-window metric drawing (thick smooth, thin raw)
# ============================================================================

def _draw_single_window_metric(ax, df, col, color):
    """Draw ONE window width on ONE axis.

    v8: ONE thin full-opacity line (σ=1) — the thick-smooth + thin-raw
    double stack blurred the panel; a single hairline reads sharper
    (Westerhold-style), and the three W series still separate by colour.
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
        smooth = np.clip(smooth, 0, 10)

    ax.plot(time, smooth, color=color, linewidth=0.8, alpha=1.0,
            zorder=4, solid_capstyle='round')


# ============================================================================
# Column-level annotations
# ============================================================================

def _add_ddr_annotation(ax_ddr, results, base_fs):
    """Single leader-line annotation: the most prominent DDR peak overall.

    v6 annotated one peak per W column; with all three W series in one
    panel only the global maximum is labelled, so the panel stays clean.
    """
    best = None  # (time, value, W)
    for W in sorted(results):
        ddr_c = np.clip(np.nan_to_num(results[W]['ddr'].values, nan=0.0),
                        0, 10)
        peak = ddr_c > 5.0
        if peak.sum() < 3:
            continue
        i = int(np.argmax(ddr_c * peak))
        if best is None or ddr_c[i] > best[1]:
            best = (results[W]['time'].values[i], ddr_c[i], W)
    if best is None or not (12 < best[0] < 56):
        return
    tp, vp, W = best
    color = WINDOW_COLORS[W]
    # Label to the LEFT by default (clear of the stats line at top-right)
    ann_x = tp - 7
    if ann_x < AGE_START + 3:
        ann_x = tp + 7
    ann_y_peak = min(vp, 8.5)
    ann_y_text = max(ann_y_peak - 3.5, 1.0)
    ann = ax_ddr.annotate(f'DDR = {vp:.1f} (W = {W} ka)',
                          xy=(tp, ann_y_peak),
                          xytext=(ann_x, ann_y_text),
                          fontsize=base_fs - 1, fontweight='heavy',
                          color=color, ha='center', va='center',
                          arrowprops=dict(arrowstyle='-', color=color,
                                          lw=0.8),
                          zorder=30,
                          bbox=dict(boxstyle='round,pad=0.3',
                                    facecolor='white',
                                    edgecolor=color, alpha=0.88,
                                    linewidth=0.8))
    ann.set_gid('layer-stats-ann')

def _draw_proxy_timeseries(ax, age_grid, signals, X_name, Y_name, base_fs,
                           results=None):
    """Panel (a): proxy time series with MIS shading (compact, no header).

    The stratigraphy band above this panel carries the shared age axis,
    so this panel has no time axis of its own. `results` (optional) adds
    the window-width legend (top-right).
    """
    # MIS shading — v9: α0.10 tints of the SAME Westerhold stage colours
    # used by the MIS bar above (restores the bar↔panel colour tie that
    # the v8 monochrome strip broke; the tints stay subtle so the curves
    # keep the visual foreground).
    for x0, x1, _label, color, _desc in MIS_STAGES:
        span = ax.axvspan(x0, x1, facecolor=color, alpha=0.10,
                          zorder=0, edgecolor='none')
        span.set_gid(f'layer-shading-mis-{x0:.0f}-{x1:.0f}')

    # No separate zero line — the horizontal grid line at y = 0 (major
    # tick) already marks the zero reference; the old dotted axhline sat
    # underneath it and only produced a faint ripple on the grid line.

    # Plot each proxy
    signal_colors = {
        'NGRIP': CARC_COLORS['red'],
        'EDC_dD': CARC_COLORS['blue'],
        'EDC_CO2': CARC_COLORS['green'],
        'EDC_CH4': CARC_COLORS['orange'],
        'Cariaco': CARC_COLORS['teal'],
        'Cheng2016': CARC_COLORS['purple'],
    }
    # v9: journal notation via Unicode (δ, superscripts) — NOT mathtext:
    # mathtext turns glyphs into vector paths and the exported SVG text
    # stops being editable in Illustrator (breaks the AI-editing
    # workflow). Unicode glyphs are real font text, Arial ships them all.
    SIGNAL_DISPLAY = {
        'NGRIP': 'NGRIP δ¹⁸O',
        'EDC_dD': 'EDC δD',
        'EDC_CO2': 'EDC CO₂',
        'EDC_CH4': 'EDC CH₄',
    }

    # v8: ONE thin full-opacity line per proxy (σ=1 kills only the 50-yr
    # noise spikes). The old 2.2pt-smooth + 0.6pt-raw stack read as an
    # airbrushed double edge — the Westerhold figure's sharpness comes
    # from single hairlines at full data frequency.
    for name, color in signal_colors.items():
        if name in signals and signals[name] is not None:
            sig = signals[name]
            smooth = gaussian_filter1d(sig, sigma=SMOOTH_SIGMA_TS)
            ax.plot(age_grid, smooth, color=color, linewidth=0.8, alpha=1.0,
                    label=SIGNAL_DISPLAY.get(name, name), zorder=3)

    ax.set_ylabel('Standardized\nProxy Value', fontsize=base_fs + 1,
                  fontweight='bold', color=AX_LABEL_COL, labelpad=6)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlim(AGE_END, AGE_START)

    # Legend
    leg_proxy = ax.legend(fontsize=base_fs, loc='upper left',
                          framealpha=0.85, ncol=2)

    # No in-panel MIS labels — the band's MIS bar above names the stages
    # (the old labels collided with both legends and sat on the curves).

    # v13: no W legend in (a) — its curves are proxy-coloured and never
    # use WINDOW_COLORS, so the legend sat in the one panel where nothing
    # was window-coloured (misleading). It moved to (b), top-right, where
    # the window-coloured curves actually live.

    # Styling — same dark-charcoal axis + visible grid treatment as (b–d)
    # v11: full hairline grid, both directions (vertical lines restored)
    ax.grid(True, which='major', axis='both', linestyle='solid',
            alpha=GRID_ALPHA, linewidth=GRID_LW, color=GRID_COL)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_linewidth(SPINE_LW)
    ax.spines['left'].set_color(AX_COL)
    ax.tick_params(axis='y', which='major', direction='out',
                   length=5, width=1.5, colors=AX_TICK_COL)

    # No time axis of its own — the stratigraphy band above carries the
    # figure's single shared age axis; the panel keeps a bare top frame
    # line and its title sits directly below it
    ax.spines['top'].set_visible(True)
    ax.spines['top'].set_linewidth(SPINE_LW)
    ax.spines['top'].set_color(AX_COL)
    # v11: every-5-ka x ticks anchor the vertical grid (labels stay hidden)
    ax.set_xticks(np.arange(AGE_START, AGE_END + 0.01, 5.0))
    ax.tick_params(axis='x', which='major', labelbottom=False, length=0)

    # v12: short title — the legend right below already names the two
    # proxies and the band above carries the shared 10–60 ka age axis;
    # the old ~50-char title repeated both verbatim.
    ax.set_title('(a) Proxy Records', fontweight='bold', fontsize=base_fs + 1,
                 loc='left', pad=7)


# ============================================================================
# Stratigraphy band — thin UNTITLED row at the figure top, not a panel
# ============================================================================

def _draw_stratigraphy_band(ax, base_fs):
    """Thin untitled stratigraphy band at the top of the figure —
    Westerhold 2020 Fig. 1 GTS-bar style (user feedback 2026-09-22:
    the geological axis must neither be its own panel nor be folded
    into panel (a); it reads as the figure's axis header).

    Drawn in absolute data-y in its own short axes (v15: ylim -2.0..4.8,
    x inverted 60→10), no panel letter, no title, no frame, NO axis:
      - MIS stage bar — v10: per-stage fills in the Westerhold Epoch-bar
        state palette with the source's PROPORTIONS (brown family
        dominant; deep blue only as the narrow oldest-edge sliver),
        α=1 solid, ink labels inside
      - climate sub-stages bar (muted Westerhold-level category colours,
        6.5 pt ink labels inside; 6.5 pt keeps the narrow GI-12/GS-12
        pair from touching)
      - category key (solid colour patch + ink name per category)
      - v15: bottom-edge ink separator line — black lines are SEPARATORS
        ("black lines are separators"); the age axis itself (line + ticks +
        numbers + 'Age (ka BP)') moved to the figure BOTTOM, see
        _draw_age_axis
    The CENOGRID geological bar is ONE thin strip of solid colour with
    labels inside — not a tall stack of shaded rows, leaders and
    floating labels.
    """
    ink = '#24201F'   # Westerhold ink (not pure black)

    y_lo, y_hi = -2.0, 4.8   # v15: key row + separator strip only — the
                             # axis/ticks/labels moved to the figure bottom
    y_span = y_hi - y_lo
    ax.set_xlim(AGE_END, AGE_START)
    ax.set_ylim(y_lo, y_hi)

    # 1. MIS bar — v10: one solid fill per stage mirroring the source
    #    Epoch bar's proportions (browns first, deep blue only at the
    #    oldest edge), ink stage labels inside.
    #    axvspan's ymin/ymax are *axes fractions* (0–1), so convert.
    frac_mis = ((3.0 - y_lo) / y_span, (4.6 - y_lo) / y_span)
    for x0, x1, label, color, _desc in MIS_STAGES:
        ax.axvspan(x0, x1, *frac_mis, facecolor=color, alpha=1.0,
                   zorder=2, edgecolor='none')
        ax.text((x0 + x1) / 2, 3.8, label, ha='center', va='center',
                fontsize=base_fs, fontweight='bold', color=ink, zorder=4)

    # 2. Climate sub-stages bar — one thin row, every segment labelled
    #    inside (6.5 pt keeps the narrow GI-12/GS-12 pair from touching).
    #    H1 is drawn after B-A and covers their 14–14.7 ka overlap, as in
    #    the previous version (near-solid α keeps the overlap blend
    #    visible — data honesty). No path effects: matplotlib's SVG
    #    backend drops texts that carry them (they stop being editable).
    frac_ev = ((1.2 - y_lo) / y_span, (2.8 - y_lo) / y_span)
    for start, end, short, category in SUBSTAGES:
        color = SUBSTAGE_TYPE_COLORS[category]
        ax.axvspan(start, end, *frac_ev, facecolor=color, alpha=0.85,
                   zorder=2, edgecolor='none')
        ax.text((start + end) / 2, 2.0, short, ha='center', va='center',
                fontsize=base_fs - 1.5, fontweight='bold', color=ink,
                zorder=4)

    # v15: no frame, no axis here — the band is a bare GTS bar. Black
    # lines are SEPARATORS ("black lines are separators"): the one ink line at
    # the band's bottom edge (step 3) divides the bar zone from the
    # curve zone; the bottom age-axis line closes the column below
    # (Westerhold p4: state bar floats over the panel, the panel's own
    # ink axis at the bottom carries the numbers).

    # v13→v15: vertical gridlines threaded through the band (every 5 ka —
    # the SAME x's as the panel grids below — drawn UNDER the bars).
    # They surface in the band's white margins (the 2.8–3.0 slit and
    # the strip between key and event bar), tying the band into the
    # column-wide grid rising from the bottom age axis.
    for gx in np.arange(AGE_START + 5, AGE_END, 5):
        ax.plot([gx, gx], [y_lo, 4.6], color=GRID_COL, lw=GRID_LW,
                alpha=GRID_ALPHA, zorder=1)

    # 3. Separator (v15): ONE black line at the band's bottom edge — the
    #    ink separator dividing the GTS bar zone from the curve zone
    #    below ("black lines are separators"). The age axis itself (line + ticks
    #    + numbers + unit) MOVED to the figure bottom (_draw_age_axis);
    #    this band carries no axis of its own.
    ax.plot([AGE_END, AGE_START], [y_lo, y_lo], color=ink, lw=1.4,
            zorder=4, clip_on=False)

    # 4. Category key — solid colour patch + ink name per category, so
    #    every colour block in the bars has an explicit pointer (the
    #    research annotation for the band). Text extends toward
    #    decreasing age (rightward), so the key is laid out from its
    #    left edge with x_cur decreasing — that centres it. (v15: sits
    #    between the bars and the separator; 'Age (ka BP)' moved down to
    #    the bottom axis row.)
    key_items = [('Interstadial', SUBSTAGE_TYPE_COLORS['interstadial']),
                 ('Heinrich', SUBSTAGE_TYPE_COLORS['heinrich']),
                 ('Stadial', SUBSTAGE_TYPE_COLORS['stadial'])]
    key_w = sum(0.55 + 0.3 + len(l) * 0.5 for l, _c in key_items) \
        + 1.0 * (len(key_items) - 1)
    x_cur = 35.0 + key_w / 2
    for label, color in key_items:
        ax.add_patch(Rectangle((x_cur - 0.55, -1.15 - 0.28), 0.55, 0.56,
                               facecolor=color, edgecolor='none',
                               zorder=4))
        ax.text(x_cur - 0.85, -1.15, label, ha='left', va='center',
                fontsize=base_fs - 1, fontweight='bold', color=ink,
                zorder=4)
        x_cur -= 0.55 + 0.3 + len(label) * 0.5 + 1.0

    # No frame, no ticks, no labels — this row is an axis header, not a
    # panel; only the ink separator line above is visible.
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_age_axis(ax, base_fs):
    """v15: the figure's ONE shared age axis, in its own slim row at the
    very bottom (user 2026-09-27: "axis belongs at the bottom" — the age numbers must
    sit under the whole figure). Westerhold p2 geomag-bar style: black
    line, major/minor ticks pointing DOWN from it, numbers under the
    ticks, unit centred below. The GTS band on top is now a pure bar and
    carries no axis; this row is the only place age numbers appear.

    ylim (0, 1): line at 0.8, major ticks 0.8→0.70 (every 10 ka), minor
    0.8→0.75 (every 5 ka), labels 0.60 va='top', unit 0.12 va='center'.
    """
    ink = '#24201F'   # Westerhold ink (not pure black)

    ax.set_xlim(AGE_END, AGE_START)
    ax.set_ylim(0, 1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.plot([AGE_END, AGE_START], [0.8, 0.8], color=ink, lw=1.4, zorder=4)
    for ta in np.arange(AGE_START, AGE_END + 0.01, 10):
        ax.plot([ta, ta], [0.8, 0.70], color=ink, lw=1.0, zorder=4)
        ax.text(ta, 0.60, f'{ta:.0f}', ha='center', va='top',
                fontsize=base_fs - 1, color=ink, zorder=4)
    for ta in np.arange(AGE_START, AGE_END + 0.01, 5):
        if ta % 10:
            ax.plot([ta, ta], [0.8, 0.75], color=ink, lw=0.5, zorder=4)
    ax.text(35.0, 0.12, 'Age (ka BP)', ha='center', va='center',
            fontsize=base_fs - 1, fontweight='bold', color=ink, zorder=4)


# Main
# ============================================================================

def main():
    # Fixed seed: makes the perturbation ensembles exactly reproducible
    np.random.seed(42)

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
