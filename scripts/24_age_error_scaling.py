#!/usr/bin/env python3
"""
24_age_error_scaling.py
-----------------------
Age-error scaling sweep on the canonical Scenario A pipeline (run22f):
robustness of the three prescribed true edges (and the 27 negative
controls) as the published 1-sigma age-model errors are scaled by
{0.1, 0.2, 0.5, 1.0, 2.0, 5.0}.

Reuses the canonical generator and test engine of scripts/22_scenarioA_final.py
unchanged (config alpha=0.5 / force / aic_f / test_joint / maxlag=50,
primary beta=0.25, 20 realizations x 200 perturbations, seeds 42+r) --
the scale=1.0 row must reproduce the run22f Table 1 beta=0.25 column
(6.0 / 6.5 / 4.5%, control max 7.25%, 0/27 controls pass).

This quantifies the escape path "improve chronological precision"
(Section 5.3): how far the 1-sigma errors must be compressed before the
robustness scores reach the 95% honesty filter.

Usage:
  python scripts/24_age_error_scaling.py --quick      # smoke (2 x 40)
  python scripts/24_age_error_scaling.py              # full (20 x 200)
Output: outputs/age_error_scaling.csv (3 edges + control summaries)
        outputs/age_error_scaling_controls.csv (27 per-control curves, for
        the Figure 7 escape-path control band)
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

from src.config import OUTPUT_DIR

SCRIPTS = os.path.dirname(os.path.abspath(__file__))


def load_s22():
    """Import scripts/22_scenarioA_final.py as a module (canonical engine)."""
    spec = importlib.util.spec_from_file_location(
        's22', os.path.join(SCRIPTS, '22_scenarioA_final.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S22 = load_s22()

CFG = (0.5, 'force', 'aic_f', 'test_joint', 50)   # canonical run22f config
BETA = 0.25                                       # primary beta
SCALES = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]


def load_params():
    p = pd.read_csv(os.path.join(OUTPUT_DIR, 'scenarioA_final_params.csv')) \
        .set_index('record')
    params = {
        'phi': {b: {n: float(p.loc[n, f'gen_phi_beta_{b}'])
                    for n in S22.VAR_ORDER}
                for b in S22.BETA_SWEEP},
        'sigma_eps': {n: float(p.loc[n, 'sigma_eps']) for n in S22.VAR_ORDER},
        'age': S22.AGE_ERR,
    }
    return params


def run_scale(params, scale, n_real, n_pert):
    age_map = {v: scale * params['age'][v] for v in S22.VAR_ORDER}
    df, full = S22.full_run(CFG, params['phi'][BETA],
                            {v: BETA for v in S22.VAR_ORDER},
                            params['sigma_eps'], age_map, True,
                            n_real, n_pert)
    ctl = full[~full['prescribed_true_edge']]
    assert len(ctl) == 27, f'expected 27 controls, got {len(ctl)}'
    rows = []
    for _, r in df.iterrows():
        if r['causal_edge'].startswith('negative controls'):
            continue
        rows.append({'error_scale': scale, 'pair': r['causal_edge'],
                     'robustness': float(r['robustness_with_error'])})
    rows.append({'error_scale': scale, 'pair': 'control_max',
                 'robustness': float(ctl['robustness_with_error_median'].max())})
    rows.append({'error_scale': scale, 'pair': 'control_mean',
                 'robustness': float(ctl['robustness_with_error_median'].mean())})
    rows.append({'error_scale': scale, 'pair': 'control_n_pass95',
                 'robustness': int((ctl['robustness_with_error_median'] >= 0.95)
                                   .sum())})
    ctl_rows = [
        {'error_scale': scale,
         'control': f"{r['source']} -> {r['target']}",
         'robustness': float(r['robustness_with_error_median'])}
        for _, r in ctl.iterrows()]
    return pd.DataFrame(rows), pd.DataFrame(ctl_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true',
                    help='smoke test: 2 realizations x 40 perturbations')
    args = ap.parse_args()

    params = load_params()
    print('Canonical generator params loaded from scenarioA_final_params.csv',
          flush=True)

    if args.quick:
        scales, n_real, n_pert = [0.5, 1.0], 2, 40
    else:
        scales, n_real, n_pert = SCALES, 20, 200

    out, ctl_out = [], []
    for s in scales:
        print(f'  error scale = {s} ...', flush=True)
        d, c = run_scale(params, s, n_real, n_pert)
        out.append(d)
        ctl_out.append(c)
    df = pd.concat(out, ignore_index=True)
    cdf = pd.concat(ctl_out, ignore_index=True)
    csv = os.path.join(OUTPUT_DIR, 'age_error_scaling.csv')
    cctl_csv = os.path.join(OUTPUT_DIR, 'age_error_scaling_controls.csv')

    if args.quick:
        # smoke only: NEVER touch the canonical CSVs (the 2026-09-13 lesson -
        # a quick run once overwrote age_error_scaling.csv with its 2-scale
        # subset; the bit-match check below caught it, but only because the
        # full file still existed to compare against)
        print('quick smoke done (no files written)', flush=True)
        print(df.to_string(index=False))
        print(f'controls: {len(cdf)} rows (not saved)', flush=True)
        return

    # The summary rows are deterministic (seeds 42+r): before overwriting,
    # check them against the existing CSV so a drift fails loudly instead
    # of silently changing the canonical summary.
    if os.path.exists(csv):
        old = pd.read_csv(csv)
        chk = old.merge(df, on=['error_scale', 'pair'],
                        suffixes=('_old', '_new'))
        assert len(chk) == len(df) == len(old), \
            'summary rows drifted vs existing CSV'
        assert np.allclose(chk['robustness_old'], chk['robustness_new'],
                           atol=1e-12, rtol=0), \
            'summary rows drifted vs existing CSV'
        print('Summary rows bit-match the existing age_error_scaling.csv',
              flush=True)
    df.to_csv(csv, index=False)
    cdf.to_csv(cctl_csv, index=False)
    print(f'Saved {csv} (+{len(cdf)} per-control rows -> {cctl_csv})',
          flush=True)

    # Cross-check against the run22f Table 1 beta=0.25 column
    t1 = pd.read_csv(os.path.join(OUTPUT_DIR, 'scenarioA_final_table1.csv'))
    ref = t1[(t1['beta'] == '0.25') & t1['causal_edge'].str.startswith(
        'negative controls')].iloc[0]
    got = df[(df['error_scale'] == 1.0) & (df['pair'] == 'control_max')].iloc[0]
    got_pass = int(df[(df['error_scale'] == 1.0)
                      & (df['pair'] == 'control_n_pass95')].iloc[0]
                   ['robustness'])
    print(f"\nAnchor check (scale=1.0): control_max here={got['robustness']:.4f}"
          f" vs run22f={ref['robustness_with_error']:.4f} "
          f"n_pass95 here={got_pass} "
          f"vs run22f={int(ref['n_ctl_pass95'])}", flush=True)
    print('\n== age_error_scaling ==')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
