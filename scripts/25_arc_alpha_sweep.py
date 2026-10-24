#!/usr/bin/env python3
"""
25_arc_alpha_sweep.py
---------------------
De-confounding effectiveness scan (Layer 2, ARC): FPR and residual AR(1)
coefficient as functions of the ARC strength alpha in {0, 0.25, 0.5, 0.75,
1.0}, on the canonical Scenario A pipeline (run22f config: force / aic_f /
test_joint / maxlag=50, beta=0.25, with age errors, 20 x 200).

This restores the trade-off panel of the manuscript's Section 3.3 argument.
Observed result on the canonical pipeline (alpha=0.5 adopted): with the
*observable* millennial-resolution NGRIP record as the deconfounding
proxy, both the control FPR (5.9-8.0%) and the residual AR(1) coefficient
(0.89-0.91) are nearly flat in alpha -- the sharp FPR/AR trade-off of the
Section 3.3 preliminary scan (AR dropping toward ~0.1 at full
deconfounding) appears only when the proxy is the true common forcing,
which is unobservable in practice. The adopted alpha=0.5 therefore sits at
a robust midpoint of a weak trade-off rather than a steep optimum. The
alpha=0.5 row must reproduce the run22f Table 1 beta=0.25 column (true
edges 6.0 / 6.5 / 4.5%, control max 7.25%, 0/27 controls pass 95%).

Usage:
  python scripts/25_arc_alpha_sweep.py --quick      # smoke (2 x 40)
  python scripts/25_arc_alpha_sweep.py              # full (20 x 200)
  python scripts/25_arc_alpha_sweep.py --plot-only  # redraw from CSV
Output: outputs/arc_alpha_sweep.csv, FigS9_ARC_Alpha_Tradeoff.
"""

import os
import sys
import argparse
import importlib.util

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from joblib import Parallel, delayed

from src.config import OUTPUT_DIR
from src.visualization import apply_style, save_figure, CARC_COLORS

SCRIPTS = os.path.dirname(os.path.abspath(__file__))


def load_s22():
    spec = importlib.util.spec_from_file_location(
        's22', os.path.join(SCRIPTS, '22_scenarioA_final.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S22 = load_s22()

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
CFG_TAIL = ('force', 'aic_f', 'test_joint', 50)   # alpha filled per sweep
BETA = 0.25


def load_params():
    p = pd.read_csv(os.path.join(OUTPUT_DIR, 'scenarioA_final_params.csv')) \
        .set_index('record')
    return {
        'phi': {b: {n: float(p.loc[n, f'gen_phi_beta_{b}'])
                    for n in S22.VAR_ORDER}
                for b in S22.BETA_SWEEP},
        'sigma_eps': {n: float(p.loc[n, 'sigma_eps']) for n in S22.VAR_ORDER},
        'age': S22.AGE_ERR,
    }


def realization_block(alpha, seed, params, n_pert):
    """One realization: returns per-pair perturbation counts + the mean
    AR(1) coefficient of the deconfounded series."""
    cfg = (alpha,) + CFG_TAIL
    phi_map = params['phi'][BETA]
    beta_map = {v: BETA for v in S22.VAR_ORDER}
    sig_map, age_map = params['sigma_eps'], params['age']
    np.random.seed(seed)
    data, t = S22.gen_scenario(seed, False, phi_map, beta_map, sig_map,
                               age_map, True)
    res = S22.layer2(data, alpha)
    ars = [S22.fit_ar1(res[v])[0] for v in S22.VAR_ORDER]
    flags = S22.adf_flags(res)
    L = S22.aic_lag(res, 'force', flags, 50)
    rng = np.random.default_rng(seed)
    counts = {(v1, v2): 0 for v1 in S22.VAR_ORDER
              for v2 in S22.VAR_ORDER if v1 != v2}
    for _ in range(n_pert):
        ps = S22.perturb_draw(res, t, 'test_joint', rng, age_map)
        det = set()
        for (drv, tgt), (yp, xp) in ps.items():
            if S22.test_pair_series(yp, xp, 'force', flags, 'aic_f', L,
                                    tgt, drv, 50):
                det.add((drv, tgt))
        for pair in det:
            counts[pair] += 1
    return counts, float(np.mean(ars))


def run_alpha(alpha, params, n_real, n_pert):
    results = Parallel(n_jobs=8, verbose=0)(
        delayed(realization_block)(alpha, 42 + r, params, n_pert)
        for r in range(n_real))
    err_med = {p: [] for p in S22.TRUE_PAIRS + S22.ALL_CTL}
    ars = []
    for counts, ar in results:
        ars.append(ar)
        for p in err_med:
            err_med[p].append(counts[p] / n_pert)
    ctl = [float(np.median(err_med[p])) for p in S22.ALL_CTL]
    tru = [float(np.median(err_med[p])) for p in S22.TRUE_PAIRS]
    return {
        'alpha': alpha,
        'ctl_mean': float(np.mean(ctl)),
        'ctl_max': float(np.max(ctl)),
        'ctl_n_pass95': int(np.sum(np.array(ctl) >= 0.95)),
        'true_mean': float(np.mean(tru)),
        **{f'true_{a}_{b}': round(float(np.median(
            err_med[(a, b)])), 4) for a, b in S22.TRUE_PAIRS},
        'ar_coeff_mean': float(np.mean(ars)),
        'ar_coeff_std': float(np.std(ars)),
    }


def plot_from_csv(df):
    apply_style('paper')
    fig, ax1 = plt.subplots(figsize=(7.2, 4.6))

    ax1.plot(df['alpha'], df['ctl_max'] * 100, 'o-',
             color=CARC_COLORS['red'], lw=2.2, ms=6,
             label='FPR: control robustness, max (left)')
    ax1.plot(df['alpha'], df['ctl_mean'] * 100, 's--',
             color=CARC_COLORS['red'], lw=1.6, ms=5, alpha=0.65,
             label='FPR: control robustness, mean (left)')
    ax1.axhline(5.0, color=CARC_COLORS['grey'], lw=1.0, ls='-.')
    ax1.text(0.03, 5.9, 'nominal 5%', fontsize=8,
             color=CARC_COLORS['grey'])
    ax1.set_xlabel(r'ARC de-confounding strength $\alpha$')
    ax1.set_ylabel('False-positive rate (%)', color=CARC_COLORS['red'])
    ax1.tick_params(axis='y', labelcolor=CARC_COLORS['red'])
    ax1.set_ylim(0, 15)

    ax2 = ax1.twinx()
    ax2.errorbar(df['alpha'], df['ar_coeff_mean'],
                 yerr=df['ar_coeff_std'], fmt='d-',
                 color=CARC_COLORS['blue'], lw=2.2, ms=6,
                 capsize=4, label='residual AR(1) coefficient (right)')
    ax2.set_ylabel('Residual AR(1) coefficient', color=CARC_COLORS['blue'])
    ax2.tick_params(axis='y', labelcolor=CARC_COLORS['blue'])
    ax2.set_ylim(0.85, 0.95)

    ax1.axvline(0.5, color=CARC_COLORS['navy'], lw=1.0, ls='--', alpha=0.7)
    ax1.annotate('adopted $\\alpha$ = 0.5',
                 xy=(0.5, 7.5), xytext=(0.55, 12.5),
                 fontsize=8, color=CARC_COLORS['navy'],
                 arrowprops=dict(arrowstyle='-', color=CARC_COLORS['navy'],
                                 lw=0.8))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
               fontsize=8, frameon=True)
    ax1.grid(True, alpha=0.25, lw=0.6)
    ax1.set_xticks(df['alpha'].values)

    fig.tight_layout()
    save_figure(fig, 'FigS9_ARC_Alpha_Tradeoff')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true',
                    help='smoke test: 2 realizations x 40 perturbations')
    ap.add_argument('--plot-only', action='store_true',
                    help='redraw the figure from outputs/arc_alpha_sweep.csv')
    args = ap.parse_args()

    csv = os.path.join(OUTPUT_DIR, 'arc_alpha_sweep.csv')
    if args.plot_only:
        plot_from_csv(pd.read_csv(csv))
        return

    params = load_params()
    if args.quick:
        alphas, n_real, n_pert = [0.0, 0.5, 1.0], 2, 40
    else:
        alphas, n_real, n_pert = ALPHAS, 20, 200

    rows = []
    for a in alphas:
        print(f'  alpha = {a} ...', flush=True)
        rows.append(run_alpha(a, params, n_real, n_pert))
    df = pd.DataFrame(rows)
    df.to_csv(csv, index=False)
    print(f'Saved {csv}', flush=True)

    # Anchor check against run22f Table 1 beta=0.25 column
    t1 = pd.read_csv(os.path.join(OUTPUT_DIR, 'scenarioA_final_table1.csv'))
    ref = t1[(t1['beta'] == '0.25') & t1['causal_edge'].str.startswith(
        'negative controls')].iloc[0]
    got = df[df['alpha'] == 0.5].iloc[0]
    print(f"\nAnchor check (alpha=0.5): ctl_max here={got['ctl_max']:.4f} "
          f"vs run22f={ref['robustness_with_error']:.4f}; "
          f"n_pass95 here={got['ctl_n_pass95']} vs run22f="
          f"{int(ref['n_ctl_pass95'])}", flush=True)
    print('\n== arc_alpha_sweep ==')
    print(df.to_string(index=False))

    plot_from_csv(df)


if __name__ == '__main__':
    main()
