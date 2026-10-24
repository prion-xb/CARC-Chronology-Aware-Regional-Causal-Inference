#!/usr/bin/env python3
"""
16_scenarioA_matrix.py
----------------------
Full 6x6 directed robustness matrix for Scenario A: all C(6,2)=15 pairs in
both directions (30 directed cells), under two conditions:
  (a) ideal — no chronological error (generation and perturbations error-free)
  (b) realistic — published pointwise age errors injected at generation time
      and propagated through the CARC perturbation ensemble

Layer-3 test is UNIFIED across all cells (the paper-described style):
ADF stationarity check with conditional first-differencing, then a Granger
F-test across lags 1..MAX_GRANGER_LAG (min p < 0.05), computed with the
fast numpy engine in src/granger_fast.py (statsmodels-equivalent decisions).

Each off-diagonal cell (row, col) shows the robustness of row -> col:
  - 20 dots: robustness per realization (20 realizations x 200 perturbations)
  - thick short line: median across realizations
  - dashed hairline: 95% honesty threshold
  - yellow box: prescribed causal edge (3 true edges)
  - dashed gray box: reverse (spurious) direction of a prescribed edge

Output:
  figures/scenarioA_pairwise_matrix.png
  outputs/scenarioA_pairwise_matrix.csv
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from src.config import FIGURE_DIR, OUTPUT_DIR, N_PERTURB, ARC_ALPHA, MAX_GRANGER_LAG
from src.synthetic import generate_scenario_A
from src.arc import arc_deconfound_batch
from src.granger_fast import granger_decision_fast
from statsmodels.tsa.stattools import adfuller
from scipy.ndimage import gaussian_filter1d
from joblib import Parallel, delayed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
np.random.seed(42)

VAR_ORDER = ['NGRIP', 'EDC_dD', 'CO2', 'CH4', 'Cariaco', 'Cheng']
SIGMA_LABEL = {'NGRIP': 0.65, 'EDC_dD': 0.83, 'CO2': 1.65, 'CH4': 1.66,
               'Cariaco': 0.23, 'Cheng': 0.14}
TRUE_EDGES = [('NGRIP', 'EDC_dD', 0.8), ('CO2', 'CH4', 1.5), ('Cariaco', 'Cheng', 2.3)]
N_REALIZATIONS = 20
THRESHOLD = 0.95

# Palette (validated light-mode reference)
INK, INK_SEC, INK_MUTED = '#0b0b0b', '#52514e', '#898781'
SURFACE, GRID = '#fcfcfb', '#c3c2b7'
TRUE_EDGE = '#eda100'          # yellow slot: prescribed-edge emphasis
BLUE_LO, BLUE_HI = '#86b6ef', '#104281'   # sequential blue ramp endpoints


def _hex_lerp(c1, c2, t):
    """Linear interpolation between two hex colors."""
    a = np.array([int(c1[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    b = np.array([int(c2[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    v = a + (b - a) * np.clip(t, 0.0, 1.0)
    return '#' + ''.join(f'{int(round(x)):02x}' for x in v)


def lag_error_ratio(lag, s1, s2):
    """|tau| / RMS(sigma1, sigma2), as defined in the paper (Table 1 note)."""
    return lag / np.sqrt((s1 ** 2 + s2 ** 2) / 2.0)


# ---------------------------------------------------------------------------
# Unified Layer-3 test (paper-described): ADF stationarity check with
# first-differencing, then Granger F-test across lags 1..MAX_GRANGER_LAG.
# The F-test uses the fast numpy engine (src/granger_fast.py), verified to
# reproduce statsmodels grangercausalitytests decisions to machine precision.
# ---------------------------------------------------------------------------
def _adf_diff_flags(residuals):
    """Per-series differencing decisions (ADF), computed once per realization.

    Layer 3 differences any series whose ADF test does not reject a unit
    root. The decision is computed on the unperturbed residuals: with
    n = 1000, age perturbations only reorder values and do not flip the
    ADF outcome, so caching it per realization reproduces the per-call
    check without 200x redundant ADF regressions.
    """
    flags = {}
    for var, vals in residuals.items():
        try:
            flags[var] = adfuller(vals, autolag='AIC')[1] > 0.05
        except Exception:
            flags[var] = False
    return flags


def _test_pairs(residuals, diff_flags, maxlag):
    """Unified Layer-3 test on one (perturbed) dataset.

    Returns the set of directed pairs with min-p < 0.05 across lags.
    """
    det = set()
    for target in VAR_ORDER:
        Y = np.diff(residuals[target]) if diff_flags[target] else residuals[target]
        for driver in VAR_ORDER:
            if driver == target:
                continue
            X = np.diff(residuals[driver]) if diff_flags[driver] else residuals[driver]
            n = min(len(Y), len(X))
            if granger_decision_fast(Y[:n], X[:n], maxlag):
                det.add((driver, target))
    return det


def _perturb_and_test(t_grid, residuals, noise_k, diff_flags, maxlag):
    """One perturbation draw -> set of detected directed pairs (joblib worker)."""
    pert = {var: residuals[var][np.argsort(t_grid + noise_k[var])]
            for var in residuals}
    return _test_pairs(pert, diff_flags, maxlag)


# ---------------------------------------------------------------------------
# Experiment: per-realization robustness for all 30 directed pairs
# ---------------------------------------------------------------------------
def run_scenario(use_age_errors):
    """Return dict {(src, dst): np.ndarray of per-realization robustness}."""
    per_pair = {(v1, v2): np.zeros(N_REALIZATIONS)
                for v1 in VAR_ORDER for v2 in VAR_ORDER if v1 != v2}

    for r in range(N_REALIZATIONS):
        data, _, t_grid = generate_scenario_A(seed=42 + r,
                                              include_age_errors=use_age_errors)

        # ARC preprocessing (Layer 2): strip half the NGRIP-trend component
        ngrip_trend = gaussian_filter1d(data['NGRIP'], sigma=1.0 / 0.05, mode='nearest')
        residuals = arc_deconfound_batch(data, ngrip_trend, alpha=ARC_ALPHA,
                                         keep_proxy='NGRIP')
        diff_flags = _adf_diff_flags(residuals)

        n_pts = len(t_grid)
        counts = {pair: 0 for pair in per_pair}
        if not use_age_errors:
            # Perturbation with sigma=0 is the identity: test once per realization.
            for src, dst in _test_pairs(residuals, diff_flags, MAX_GRANGER_LAG):
                counts[(src, dst)] += N_PERTURB
        else:
            # Pre-draw the perturbation stream in the main process
            # (deterministic under np.random.seed(42)), then fan out.
            noise = {var: np.random.normal(0, SIGMA_LABEL[var], (N_PERTURB, n_pts))
                     for var in VAR_ORDER}
            results = Parallel(n_jobs=-1)(
                delayed(_perturb_and_test)(t_grid, residuals,
                                           {var: noise[var][k] for var in VAR_ORDER},
                                           diff_flags, MAX_GRANGER_LAG)
                for k in range(N_PERTURB))
            for det in results:
                for pair in det:
                    counts[pair] += 1

        for pair, cnt in counts.items():
            per_pair[pair][r] = cnt / N_PERTURB

        print(f"  realization {r + 1}/{N_REALIZATIONS} done "
              f"(errors={'on' if use_age_errors else 'off'})")

    return per_pair


def _save_results_npz(path, results_noerr, results_err):
    """Cache per-realization robustness arrays for --plot-only reuse."""
    payload = {}
    for cond, res in (('noerr', results_noerr), ('err', results_err)):
        for (src, dst), arr in res.items():
            payload[f'{cond}__{src}__{dst}'] = arr
    np.savez(path, **payload)


def _load_results_npz(path):
    z = np.load(path)
    out_noerr, out_err = {}, {}
    for k in z.files:
        cond, src, dst = k.split('__')
        (out_noerr if cond == 'noerr' else out_err)[(src, dst)] = z[k]
    return out_noerr, out_err


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def draw_matrix_panel(fig, gs, results, show_labels):
    """Draw one 6x6 directed robustness matrix into a GridSpec."""
    axes = {}
    for i, src in enumerate(VAR_ORDER):
        for j, dst in enumerate(VAR_ORDER):
            ax = fig.add_subplot(gs[i, j])
            axes[(i, j)] = ax
            ax.set_xlim(0, N_REALIZATIONS)
            ax.set_ylim(-0.06, 1.06)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if i == j:  # diagonal: name + sigma
                ax.set_facecolor('none')
                # 0.62/0.32: at the printed cell height the name and sigma
                # boxes (9 pt + 7 pt) need a few px of clear air between
                # them — 0.58/0.36 left the two boxes touching (layout QA)
                ax.text(0.5, 0.62, src, ha='center', va='center',
                        fontsize=9, fontweight='bold', color=INK,
                        transform=ax.transAxes)
                ax.text(0.5, 0.32, f'σ = {SIGMA_LABEL[src]:.2f} ka', ha='center',
                        va='center', fontsize=7, color=INK_MUTED,
                        transform=ax.transAxes)
                continue

            values = results[(src, dst)]
            is_true = any(src == t[0] and dst == t[1] for t in TRUE_EDGES)
            is_reverse = any(src == t[1] and dst == t[0] for t in TRUE_EDGES)

            # cell background / border
            if is_true:
                ax.set_facecolor(TRUE_EDGE)
                ax.patch.set_alpha(0.08)
                ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                       fill=False, edgecolor=TRUE_EDGE,
                                       linewidth=2.2, clip_on=False))
            elif is_reverse:
                ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                       fill=False, edgecolor=GRID,
                                       linewidth=1.0, linestyle=(0, (4, 3)),
                                       clip_on=False))

            # 95% threshold hairline
            ax.axhline(THRESHOLD, color=GRID, linewidth=0.7, linestyle=(0, (5, 3)),
                       zorder=1)

            # per-realization dots
            xs = np.arange(1, N_REALIZATIONS + 1)
            colors = [_hex_lerp(BLUE_LO, BLUE_HI, v) if is_true else INK_MUTED
                      for v in values]
            ax.scatter(xs, values, s=12, c=colors, alpha=0.75,
                       edgecolors='none', zorder=3, linewidths=0)

            # median marker
            med = np.median(values)
            ax.plot([0, N_REALIZATIONS], [med, med],
                    color=BLUE_HI if is_true else INK_SEC,
                    linewidth=2.0, zorder=4, solid_capstyle='round')

            # true-edge annotation
            if is_true:
                lag = next(t[2] for t in TRUE_EDGES if t[0] == src and t[1] == dst)
                ratio = lag_error_ratio(lag, SIGMA_LABEL[src], SIGMA_LABEL[dst])
                ax.text(0.04, 0.88, f'lag {lag} ka\nlag/σ ≈ {ratio:.1f}',
                        ha='left', va='top', fontsize=7, color=INK_SEC,
                        transform=ax.transAxes, zorder=5)

            # row & column labels
            if show_labels:
                if j == 0:
                    ax.text(-0.42, 0.5, src, ha='right', va='center',
                            fontsize=8.5, color=INK, rotation=0,
                            transform=ax.transAxes)
                if i == 0:
                    ax.text(0.5, 1.32, dst, ha='center', va='bottom',
                            fontsize=8.5, color=INK, transform=ax.transAxes)
    return axes


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Scenario A full 6x6 directed robustness matrix')
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip the simulations; load per-realization '
                             'results from outputs/scenarioA_pairwise_results.npz '
                             'and redraw the figure')
    args = parser.parse_args()

    print("Scenario A: full 15-pair robustness matrix")
    print("=" * 60)

    npz_path = os.path.join(OUTPUT_DIR, 'scenarioA_pairwise_results.npz')
    if args.plot_only:
        results_noerr, results_err = _load_results_npz(npz_path)
        print(f"Loaded cached per-realization results from {npz_path}")
    else:
        print("\n[1/2] Ideal case (no chronological error)")
        results_noerr = run_scenario(use_age_errors=False)

        print("\n[2/2] Realistic case (published age errors)")
        results_err = run_scenario(use_age_errors=True)

        _save_results_npz(npz_path, results_noerr, results_err)
        print(f"Saved per-realization results to {npz_path}")

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------
    rows = []
    for src in VAR_ORDER:
        for dst in VAR_ORDER:
            if src == dst:
                continue
            true = next((t for t in TRUE_EDGES if t[0] == src and t[1] == dst), None)
            rows.append({
                'source': src, 'target': dst,
                'prescribed_true_edge': true is not None,
                'lag_ka': true[2] if true else np.nan,
                'lag_error_ratio': (lag_error_ratio(true[2], SIGMA_LABEL[src],
                                                    SIGMA_LABEL[dst])
                                    if true else np.nan),
                'robustness_no_error_median': np.median(results_noerr[(src, dst)]),
                'robustness_with_error_median': np.median(results_err[(src, dst)]),
                'passes_95_with_error': np.median(results_err[(src, dst)]) >= THRESHOLD,
            })
    df = pd.DataFrame(rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_DIR, 'scenarioA_pairwise_matrix.csv'), index=False)
    print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'scenarioA_pairwise_matrix.csv')}")

    print("\nSummary (median robustness, with errors):")
    for _, r in df.iterrows():
        flag = '  <-- PASS' if r['passes_95_with_error'] else ''
        mark = 'T' if r['prescribed_true_edge'] else ' '
        print(f"  [{mark}] {r['source']:>8} -> {r['target']:<8} "
              f"{r['robustness_with_error_median']:.2f}{flag}")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    plt.rcParams.update({'font.size': 9, 'font.family': 'Arial',
                         'axes.edgecolor': GRID, 'text.color': INK})
    fig = plt.figure(figsize=(7.0, 3.837), facecolor=SURFACE)  # 178 mm wide, same aspect

    # left=0.10: the row labels sit 0.42 cell-widths to the left of the
    # leftmost cells, so the left margin must absorb that offset (layout QA:
    # EDC_dD / Cariaco labels used to fall off the canvas edge).
    # top=0.84: the strip above the grids must hold the suptitle, both panel
    # titles and the column headers without any of the three touching
    # (layout QA: '(a)' used to sit on the EDC_dD/CO2/CH4/Cariaco headers)
    gs_outer = fig.add_gridspec(1, 2, wspace=0.10, left=0.10, right=0.965,
                                bottom=0.115, top=0.84)
    gs_a = gs_outer[0].subgridspec(6, 6, wspace=0.07, hspace=0.07)
    gs_b = gs_outer[1].subgridspec(6, 6, wspace=0.07, hspace=0.07)

    draw_matrix_panel(fig, gs_a, results_noerr, True)
    draw_matrix_panel(fig, gs_b, results_err, False)

    # baselines 0.933 / 0.971: three stacked text rows (suptitle 10 pt,
    # panel titles 10 pt, headers 8.5 pt) must clear each other inside the
    # 0.84..1.0 strip; the long parenthetical used to push the suptitle
    # past the right canvas edge (the dot meanings repeat in the legend)
    fig.text(0.10, 0.933, '(a) Ideal — no chronological error',
             fontsize=10, fontweight='bold', color=INK)
    fig.text(0.515, 0.933, '(b) Realistic — published age errors',
             fontsize=10, fontweight='bold', color=INK)
    fig.text(0.10, 0.971, 'Scenario A: robustness of every directed pair',
             fontsize=10, fontweight='bold', color=INK)

    # legend
    handles = [
        Line2D([], [], marker='o', linestyle='none', ms=5, mfc=INK_MUTED,
               mec='none', alpha=0.75, label='per-realization robustness\n(control pair)'),
        Line2D([], [], marker='o', linestyle='none', ms=5, mfc=BLUE_HI,
               mec='none', alpha=0.75, label='per-realization robustness\n(prescribed edge)'),
        Line2D([0], [0], color=BLUE_HI, linewidth=2, label='median (20 realizations)'),
        Line2D([0], [0], color=GRID, linewidth=1, linestyle=(0, (5, 3)),
               label='95% honesty threshold'),
        Rectangle((0, 0), 1, 1, fill=False, edgecolor=TRUE_EDGE, linewidth=2,
                  label='prescribed causal edge'),
        Rectangle((0, 0), 1, 1, fill=False, edgecolor=GRID, linewidth=1,
                  linestyle=(0, (4, 3)), label='reverse direction of a true edge'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, 0.012), handlelength=1.4,
               columnspacing=1.2)

    # route through save_figure: flattens the RGBA canvas to RGB PNG and
    # writes the LZW-compressed TIFF journals require
    from src.visualization import save_figure
    save_figure(fig, 'scenarioA_pairwise_matrix', fmts=('png', 'tif'), dpi=300)
    plt.close(fig)
    print(f"\nSaved: scenarioA_pairwise_matrix.png/.tif")


if __name__ == "__main__":
    main()
