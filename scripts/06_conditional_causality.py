#!/usr/bin/env python3
"""
06_conditional_causality.py
---------------------------
Conditional causality analysis: test causal direction within distinct
climate states (H1, H4, LGM, Bølling-Allerød, GI-12, GS-12).

Outputs:
  outputs/conditional_causality_allpairs.csv   (15 pairs x 6 windows)
  figures/FigS_Conditional_Causality.pdf/.png  (bar panels per pair of interest)
  figures/FigS_SixRecord_Pairwise_Matrix.pdf/.png (6x6 directed robustness matrix)

Usage:
  python scripts/06_conditional_causality.py            # full re-run (heavy)
  python scripts/06_conditional_causality.py --plot-only  # rebuild figures from CSV
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Fixed seed: makes the perturbation ensemble exactly reproducible
np.random.seed(42)

from src.config import (DATA_DIR, OUTPUT_DIR, FIGURE_DIR, TIER1_FILES, TIER1_SIGMA,
                        CLIMATE_WINDOWS, N_PERTURB, ARC_ALPHA, MAX_GRANGER_LAG)
from src.io import load_record
from src.arc import arc_deconfound
from src.granger import granger_detect
from src.visualization import (apply_style, CARC_COLORS, CARC_COLORS_LIGHT,
                               add_panel_label, add_minor_ticks, save_figure)

PAIRS_OF_INTEREST = [('NGRIP', 'EDC_dD'), ('EDC_dD', 'EDC_CO2'), ('Cariaco', 'Cheng2016')]


# ---------------------------------------------------------------------------
# Analysis (heavy: N_PERTURB x 15 pairs x 6 windows)
# ---------------------------------------------------------------------------
def run_analysis():
    print("Loading Tier 1 records...")
    raw = {}
    for name, fname in TIER1_FILES.items():
        age, val = load_record(fname, DATA_DIR)
        if age is not None:
            raw[name] = (age, val)
            print(f"  {name}: {len(age)} pts")

    var_names = [v for v in ['NGRIP', 'EDC_dD', 'EDC_CO2', 'EDC_CH4', 'Cariaco', 'Cheng2016'] if v in raw]
    # All C(6,2) = 15 undirected pairs (order: var_names[i] -> var_names[i+1:])
    ALL_PAIRS = [(v1, v2) for i, v1 in enumerate(var_names) for v2 in var_names[i + 1:]]

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

        edge_counts = {p: [0, 0] for p in ALL_PAIRS}

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
                if (v1, v2) in ALL_PAIRS:
                    edge_counts[(v1, v2)][0] += 1
                elif (v2, v1) in ALL_PAIRS:
                    edge_counts[(v2, v1)][1] += 1

        result = {}
        for (v1, v2), (c12, c21) in edge_counts.items():
            result[(v1, v2)] = {'v1→v2': c12 / N_PERTURB, 'v2→v1': c21 / N_PERTURB}
        return result

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

    # --- Full 15-pair x 6-window table (all C(6,2) edges) ---
    csv_rows = []
    for label in CLIMATE_WINDOWS:
        if label not in all_window_results:
            continue
        for (v1, v2), rates in sorted(all_window_results[label].items()):
            csv_rows.append({'window': label, 'record_A': v1, 'record_B': v2,
                             'A_to_B': rates['v1→v2'], 'B_to_A': rates['v2→v1']})
    csv_df = pd.DataFrame(csv_rows)
    out_csv = os.path.join(OUTPUT_DIR, 'conditional_causality_allpairs.csv')
    csv_df.to_csv(out_csv, index=False)
    print(f"\nSaved full {len(csv_rows)}-row table (15 pairs x {len(all_window_results)} "
          f"windows): {out_csv}")
    return all_window_results, var_names


def load_plot_data():
    """Rebuild the analysis dict from the saved CSV (fast, no compute)."""
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'conditional_causality_allpairs.csv'))
    var_names = list(dict.fromkeys(df['record_A']))
    results = defaultdict(dict)
    for row in df.itertuples(index=False):
        results[row.window][(row.record_A, row.record_B)] = {
            'v1→v2': row.A_to_B, 'v2→v1': row.B_to_A}
    return dict(results), var_names


# ---------------------------------------------------------------------------
# Figure 1: paired bars per pair of interest
# ---------------------------------------------------------------------------
def plot_bars(all_window_results, var_names):
    apply_style("paper", font_scale=1.0)  # print size: base ~8 pt

    fig, axes = plt.subplots(len(PAIRS_OF_INTEREST), 1, figsize=(7.0, 0.5 * 4.5 * len(PAIRS_OF_INTEREST)))  # 178 mm wide, same aspect
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
        ax.set_title(f'Conditional Causality: {v1} ↔ {v2}', fontweight='bold', fontsize=10)
        ax.legend(fontsize=10, framealpha=0.8)
        ax.axhline(0.95, color='grey', linestyle='--', alpha=0.6, linewidth=1.2)
        ax.text(len(valid_labels) - 0.5, 0.955, '95% stability', fontsize=7.5,
                color='grey', ha='right', va='bottom')
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.2)
        add_minor_ticks(ax)
        add_panel_label(ax, f"({'a' if idx == 0 else 'b' if idx == 1 else 'c'})")

    # suptitle inside the reserved top strip: tight_layout rect leaves room,
    # so the title can never collide with the top panel title
    fig.suptitle("CARC Conditional Causality — Directional Stability by Climate State",
                 fontsize=10, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "FigS_Conditional_Causality", fmts=("pdf", "png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: record-level 6x6 pairwise robustness matrix (all C(6,2) = 15 edges)
# ---------------------------------------------------------------------------
def plot_matrix(all_window_results, var_names):
    WINDOW_ORDER = ['Bølling-Allerød', 'H1 event', 'Last Glacial Maximum',
                    'H4 event', 'GI-12 (interstadial)', 'GS-12 (stadial)']
    INK, INK_SEC, GRID_C = '#0b0b0b', '#52514e', '#c3c2b7'
    BLUE, BLUE_HI = '#2a78d6', '#104281'
    ACCENT = '#eda100'
    ASYM_MIN = 0.30

    rates_rm = {}
    for label, res in all_window_results.items():
        for (v1, v2), r in res.items():
            rates_rm[(v1, v2, label)] = r['v1→v2']
            rates_rm[(v2, v1, label)] = r['v2→v1']

    plt.rcParams.update({'font.size': 9, 'font.family': 'Arial',
                         'axes.edgecolor': GRID_C, 'text.color': INK})
    fig2 = plt.figure(figsize=(7.0, 6.337), facecolor='#fcfcfb')  # 178 mm wide, same aspect
    n = len(var_names)
    # left margin holds the row labels; top margin holds title + subtitle +
    # the top row's column labels
    gs2 = fig2.add_gridspec(n, n, wspace=0.10, hspace=0.12,
                            left=0.21, right=0.94, bottom=0.075, top=0.86)

    for i, src in enumerate(var_names):
        for j, dst in enumerate(var_names):
            ax = fig2.add_subplot(gs2[i, j])
            ax.set_xlim(-0.5, len(WINDOW_ORDER) - 0.5)
            ax.set_ylim(-0.06, 1.06)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if i == j:  # diagonal: record name
                ax.set_facecolor('none')
                ax.text(0.5, 0.5, src, ha='center', va='center',
                        fontsize=9, fontweight='bold', color=INK,
                        transform=ax.transAxes)
                continue

            vals = np.array([rates_rm.get((src, dst, w), np.nan)
                             for w in WINDOW_ORDER])
            med = np.nanmedian(vals)

            # asymmetry flag (vs the mirror cell)
            rev = np.array([rates_rm.get((dst, src, w), np.nan)
                            for w in WINDOW_ORDER])
            if abs(np.nanmedian(vals) - np.nanmedian(rev)) >= ASYM_MIN:
                ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                       fill=False, edgecolor=ACCENT,
                                       linewidth=1.8, clip_on=False))

            ax.axhline(0.95, color=GRID_C, linewidth=0.7,
                       linestyle=(0, (5, 3)), zorder=1)
            ax.axhline(0.80, color=GRID_C, linewidth=0.5,
                       linestyle=(0, (1, 4)), zorder=1)

            xs = np.arange(len(WINDOW_ORDER))
            ok = np.isfinite(vals)
            ax.plot(xs[ok], vals[ok], color=BLUE, linewidth=1.2,
                    zorder=2, alpha=0.85)
            ax.scatter(xs[ok], vals[ok], s=16, c=BLUE, alpha=0.85,
                       edgecolors='none', zorder=3, linewidths=0)

            # classical pairs of interest (both directions)
            if (src, dst) in PAIRS_OF_INTEREST or (dst, src) in PAIRS_OF_INTEREST:
                for k, w in enumerate(WINDOW_ORDER):
                    if np.isfinite(vals[k]):
                        ax.scatter([k], [vals[k]], marker='*', s=60, c='white',
                                   edgecolors=INK, linewidths=0.8, zorder=4)

            ax.plot([-0.35, len(WINDOW_ORDER) - 0.65], [med, med],
                    color=BLUE_HI, linewidth=2.2, zorder=5,
                    solid_capstyle='round')

            if j == 0:
                ax.text(-0.60, 0.5, src, ha='right', va='center',
                        fontsize=9, color=INK, transform=ax.transAxes)
            if i == 0:
                ax.text(0.5, 1.14, dst, ha='center', va='bottom',
                        fontsize=9, color=INK, transform=ax.transAxes)

    fig2.text(0.5, 0.965,
              'Directional robustness of the six Tier-1 records '
              '(cell = row → col)',
              fontsize=10, fontweight='bold', color=INK, ha='center')
    fig2.text(0.5, 0.928,
              'each cell: robustness in the six climate windows '
              '(BA, H1, LGM, H4, GI-12, GS-12) · bar = window median',
              fontsize=9, color=INK_SEC, ha='center')

    handles2 = [
        Line2D([], [], marker='o', linestyle='-', ms=4.5, mfc=BLUE, mec='none',
               color=BLUE, linewidth=1.2,
               label='per-window robustness\n(chronological order)'),
        Line2D([0], [0], color=BLUE_HI, linewidth=2.2,
               label='median across windows'),
        Line2D([0], [0], color=GRID_C, linewidth=1, linestyle=(0, (5, 3)),
               label='95% honesty threshold'),
        Line2D([0], [0], color=GRID_C, linewidth=1, linestyle=(0, (1, 4)),
               label='80% threshold (relaxed)'),
        Line2D([], [], marker='*', linestyle='none', ms=9, mfc='white',
               mec=INK, mew=0.8, label='classical pair\nof interest'),
        Rectangle((0, 0), 1, 1, fill=False, edgecolor=ACCENT, linewidth=1.8,
                  label='asymmetric pair\n(Δmedian ≥ 0.30)'),
    ]
    fig2.legend(handles=handles2, loc='lower center', ncol=6, frameon=False,
                fontsize=7.5, bbox_to_anchor=(0.5, 0.012), handlelength=1.4,
                columnspacing=1.2)

    save_figure(fig2, "FigS_SixRecord_Pairwise_Matrix", fmts=("pdf", "png"))
    plt.close(fig2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plot-only", action="store_true",
                    help="rebuild figures from outputs/conditional_causality_allpairs.csv")
    args = ap.parse_args()

    if args.plot_only:
        print("Plot-only mode: loading saved table...")
        all_window_results, var_names = load_plot_data()
    else:
        all_window_results, var_names = run_analysis()

    plot_bars(all_window_results, var_names)
    plot_matrix(all_window_results, var_names)
    print("Done.")
