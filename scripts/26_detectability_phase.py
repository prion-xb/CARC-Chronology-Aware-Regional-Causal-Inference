#!/usr/bin/env python3
"""
26_detectability_phase.py
-------------------------
Detectability phase diagram (paper Discussion / escape-path analysis,
Section 5.3): robustness of the prescribed NGRIP -> EDC_dD edge on a grid
of {causal strength} x {age-error scale}, on the canonical Scenario A
pipeline (run22f config: alpha=0.5 / force / aic_f / test_joint /
maxlag=50, beta=0.25, with age errors).

The 95% robustness contour of this grid is the empirical "escape
boundary": how strong a causal signal must be, given the chronological
precision, before CARC's honesty filter would pass it. The canonical
position (strength 0.4, scale 1.0, lag/error ~1.1) sits deep in the
sub-threshold regime; the cell (0.4, 1.0) must reproduce the run22f
Table 1 beta=0.25 value for this edge (6.0%).

The grid feeds Fig 5b (contour map) and the real-data projection of
Fig 5c, assembled by scripts/31_figure5_escape.py.

Usage:
  python scripts/26_detectability_phase.py --quick   # smoke (2 x 40)
  python scripts/26_detectability_phase.py           # full (10 x 200)
Output: outputs/detectability_phase.csv
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
import warnings
warnings.filterwarnings('ignore')

from joblib import Parallel, delayed

from src.config import OUTPUT_DIR

SCRIPTS = os.path.dirname(os.path.abspath(__file__))


def load_s22():
    spec = importlib.util.spec_from_file_location(
        's22', os.path.join(SCRIPTS, '22_scenarioA_final.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S22 = load_s22()

BETA = 0.25
ALPHA = 0.5
STRENGTHS = [0.1, 0.2, 0.4, 0.8, 1.6]
SCALES = [0.2, 0.5, 1.0, 2.0, 5.0]
FOCUS = ('NGRIP', 'EDC_dD')
FOCUS_LAG_KA = 0.8
# RMS 1-sigma error of the focus pair at scale=1 (published errors)
FOCUS_RMS_ERR = float(np.sqrt((S22.AGE_ERR['NGRIP'] ** 2
                               + S22.AGE_ERR['EDC_dD'] ** 2) / 2.0))


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


def realization_block(strength, scale, seed, params, n_pert):
    phi_map = params['phi'][BETA]
    beta_map = {v: BETA for v in S22.VAR_ORDER}
    sig_map = params['sigma_eps']
    age_map = {v: scale * params['age'][v] for v in S22.VAR_ORDER}
    np.random.seed(seed)
    data, t = S22.gen_scenario(seed, False, phi_map, beta_map, sig_map,
                               age_map, True, causal_strength=strength)
    res = S22.layer2(data, ALPHA)
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
    return counts


def run_cell(strength, scale, params, n_real, n_pert):
    results = Parallel(n_jobs=8, verbose=0)(
        delayed(realization_block)(strength, scale, 42 + r, params, n_pert)
        for r in range(n_real))
    rob = {}
    for p in S22.TRUE_PAIRS + S22.ALL_CTL:
        rob[p] = float(np.median([c[p] for c in results]) / n_pert)
    ctl = [rob[p] for p in S22.ALL_CTL]
    ratio = FOCUS_LAG_KA / (FOCUS_RMS_ERR * scale)
    print(f'  strength={strength:.1f} scale={scale:.1f} '
          f'ratio={ratio:.2f} focus={rob[FOCUS]:.3f} '
          f'ctl_max={max(ctl):.3f}', flush=True)
    return {
        'causal_strength': strength,
        'error_scale': scale,
        'lag_error_ratio': round(ratio, 2),
        'robustness_focus': rob[FOCUS],
        'control_max': float(max(ctl)),
        'control_mean': float(np.mean(ctl)),
        'control_n_pass95': int(np.sum(np.array(ctl) >= 0.95)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true',
                    help='smoke test: 2 realizations x 40 perturbations')
    args = ap.parse_args()

    params = load_params()
    if args.quick:
        strengths, scales = [0.2, 0.4], [0.5, 1.0]
        n_real, n_pert = 2, 40
    else:
        strengths, scales = STRENGTHS, SCALES
        n_real, n_pert = 10, 200

    rows = []
    for s in strengths:
        for sc in scales:
            rows.append(run_cell(s, sc, params, n_real, n_pert))
    df = pd.DataFrame(rows)
    csv = os.path.join(OUTPUT_DIR, 'detectability_phase.csv')
    df.to_csv(csv, index=False)
    print(f'Saved {csv}', flush=True)

    # Anchor check: canonical cell (0.4, 1.0) vs run22f Table 1 beta=0.25
    t1 = pd.read_csv(os.path.join(OUTPUT_DIR, 'scenarioA_final_table1.csv'))
    ref = t1[(t1['beta'] == '0.25') & t1['causal_edge'].str.startswith(
        'NGRIP')].iloc[0]['robustness_with_error']
    got = df[(df['causal_strength'] == 0.4)
             & (df['error_scale'] == 1.0)].iloc[0]['robustness_focus']
    print(f'\nAnchor check (strength=0.4, scale=1.0): focus={got:.4f} '
          f'vs run22f NGRIP->EDC_dD={ref:.4f}', flush=True)


if __name__ == '__main__':
    main()
