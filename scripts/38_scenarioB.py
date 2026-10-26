#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
38_scenarioB.py
-------------------
Scenario B rebuild (supplementary Section S6, Tables S5-S6): robustness of
the sub-threshold finding to parameter variation under "more favourable
statistical conditions".

Generator overrides vs Scenario A (scripts/34; Table S5):
  * phi     0.50 (all records)       [A: 0.72-0.90]
  * beta    0.15 (all records)       [A: 0.25-0.35]
  * rho_F   0.50                     [A: 0.60]
  * lags    0.3 ka (all 3 edges)     [A: 0.8 / 1.5 / 2.3 ka]
  * sigma_e 0.40 (all records)       [A: 0.45-0.55; used as innovation std,
                                      consistent with the Scenario A rebuild]
  * age errors: same empirical distribution as Scenario A
    (0.65 / 0.83 / 1.65 / 1.66 / 0.23 / 0.14 ka)

Edge sign convention: Table S6 lists NGRIP->EDC +0.3, CO2->CH4 -0.3,
Cariaco->Cheng +0.3. Granger detection is sign-symmetric, so the rebuild
uses |lag| = 0.3 ka for all three (the sign already flips between the
main-text Table 1 and Table S6 conventions in Scenario A; documented).

Lag visibility note: 0.3 ka = 6 steps at dt = 0.05, i.e. just outside the
AIC VAR lag cap 1-5. Layer 3 here uses the per-pair AIC-selected lag L as
the test cap (min-p across lags 1..L) -- the same semantics that the P3
probe proved for the regional Section 4.3 numbers. With the cap at 5 the
injected lag is outside the test window, so all robustness scores (true
edges included) sit at the min-p false-positive floor -- the ~18% the
paper reports; Cariaco->Cheng bleeds most into the visible lags, matching
its slightly higher published score (20.9). --maxlags 50 (the canonical
Scenario A config) brings the lag fully into the window for contrast.

Table S6 targets (200 perturbations, ARC alpha=0.5):
  NGRIP -> EDC      18.5
  CO2 -> CH4        17.0
  Cariaco -> Cheng  20.9
  12 negative controls 15-17   (Theorem 3 FPR bound for Scenario B: ~0.18)

Outputs:
  outputs/scenarioB_tableS6.csv       per-edge Table S6 rows x maxlag
  outputs/scenarioB_tableS6_full.csv all 30 directed cells x maxlag

Usage:
  python scripts/38_scenarioB.py --quick   # smoke test (3 x 40)
  python scripts/38_scenarioB.py           # full run (20 x 200)
"""

import os, sys, argparse, warnings, importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from src.config import OUTPUT_DIR

# reuse the Scenario A generator + pipeline machinery (scripts/34);
# Scenario B overrides the module-level parameter tables below.
_spec = importlib.util.spec_from_file_location(
    'reg34', os.path.join(os.path.dirname(__file__), '34_scenarioA_submitted.py'))
reg34 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reg34)

DT = reg34.DT
TRUE_EDGES_B = [('NGRIP', 'EDC_dD', 0.3), ('CO2', 'CH4', 0.3),
                ('Cariaco', 'Cheng', 0.3)]


def scenarioB_params(noise_mode='std'):
    """Table S5 Scenario B parameter tables in the scripts/34 format.

    noise_mode 'std'     : sigma_e 0.40 used directly as the innovation std
                           (consistent with the Scenario A rebuild, where the
                           Table S1 values 0.45-0.55 are used as std);
    noise_mode 'variance': sigma_e = sqrt(0.40) taking the paper's column
                           header "sigma^2_eps" literally (probe variant).
    """
    sigma = 0.40 if noise_mode == 'std' else float(np.sqrt(0.40))
    return {
        'phi':   {v: 0.50 for v in reg34.VAR_ORDER},
        'beta':  {v: 0.15 for v in reg34.VAR_ORDER},
        'sigma': {v: sigma for v in reg34.VAR_ORDER},
        'age':   dict(reg34.S1_PARAMS['age']),     # same empirical errors
    }


def apply_overrides(params):
    reg34.S1_PARAMS = params
    reg34.TRUE_EDGES = list(TRUE_EDGES_B)
    reg34.RHO_F = 0.50


def aic_lag_pair(y, x, maxlag=5):
    """AIC VAR order on one pair (scripts/36 semantics)."""
    n = min(len(y), len(x))
    try:
        model = VAR(np.column_stack([y[:n], x[:n]]))
        order = model.select_order(maxlags=maxlag)
        L = order.aic
        return int(L) if (L is not None and L >= 1) else maxlag
    except Exception:
        return maxlag


def test_pairs_pair_aic(residuals, flags, maxlag, test='minp'):
    """Layer 3 for Scenario B: per-pair AIC lag L caps the min-p window
    (lags 1..L). Replaces reg34.test_pairs, whose fixed test window is the
    Scenario A config; here the window must track the AIC cap so that the
    F-test floor matches the paper's Table S6 FPR numbers (~18%)."""
    det = set()
    for target in reg34.VAR_ORDER:
        Y = reg34._maybe_diff(residuals[target], flags[target])
        for driver in reg34.VAR_ORDER:
            if driver == target:
                continue
            X = reg34._maybe_diff(residuals[driver], flags[driver])
            L = aic_lag_pair(Y, X, maxlag)
            if reg34._pair_significant(Y, X, L, test):
                det.add((driver, target))
    return det


def run_config(n_real, n_pert, span, maxlag, test, noise_mode,
               perdraw, seed_base, quick_label=''):
    params = scenarioB_params(noise_mode)
    apply_overrides(params)

    sigma_map = dict(params['age'])
    true_pairs = [(a, b) for a, b, _ in TRUE_EDGES_B]
    all_ctl = [(v1, v2) for v1 in reg34.VAR_ORDER for v2 in reg34.VAR_ORDER
               if v1 != v2 and (v1, v2) not in true_pairs]

    err_med = {p: [] for p in true_pairs + all_ctl}
    for r in range(n_real):
        # per realization: Layer 2 once on the clean series; the ADF and
        # per-pair AIC gates are fixed for the 200 draws (scripts/35
        # structure: only the perturbation varies across draws).
        data, t, Z = reg34.generate_submitted(seed_base + r, False, span=span)
        res = reg34.layer2(data, mode='keep', alpha=reg34.ARC_ALPHA, Z=Z)
        flags = reg34.adf_flags(res)
        Lmap = {}
        for drv in reg34.VAR_ORDER:
            for tgt in reg34.VAR_ORDER:
                if drv == tgt:
                    continue
                Y = reg34._maybe_diff(res[tgt], flags[tgt])
                X = reg34._maybe_diff(res[drv], flags[drv])
                Lmap[(drv, tgt)] = aic_lag_pair(Y, X, maxlag)
        rng = np.random.default_rng(seed_base * 1000 + 7 + r)
        counts = {p: 0 for p in err_med}
        for _ in range(n_pert):
            if perdraw:
                raw = {v: data[v][np.argsort(
                    t + rng.normal(0, sigma_map[v], len(t)))]
                    for v in reg34.VAR_ORDER}
                ps = reg34.layer2(raw, mode='keep',
                                  alpha=reg34.ARC_ALPHA, Z=Z)
            else:
                ps = reg34.perturb_draw(res, t, rng, sigma_map)
            for drv, tgt in counts:
                Y = reg34._maybe_diff(ps[tgt], flags[tgt])
                X = reg34._maybe_diff(ps[drv], flags[drv])
                if reg34._pair_significant(Y, X, Lmap[(drv, tgt)], test):
                    counts[(drv, tgt)] += 1
        for p, c in counts.items():
            err_med[p].append(c / n_pert)
        if not quick_label:
            print(f"  realization {r + 1}/{n_real} done "
                  f"(ADF diff {sum(flags.values())}/6)", flush=True)

    rows = []
    for driver, target, lag_ka in TRUE_EDGES_B:
        sigma_drv = params['age'][driver]
        rob = float(np.median(err_med[(driver, target)]))
        rows.append({
            'maxlag': maxlag, 'noise_mode': noise_mode,
            'causal_edge': f'{driver} -> {target}',
            'true_lag_ka': lag_ka,
            'lag_error_ratio': round(lag_ka / sigma_drv, 2),
            'robustness': rob,
            'passes_95': rob >= 0.95,
        })
    ctl = [float(np.median(err_med[p])) for p in all_ctl]
    rows.append({
        'maxlag': maxlag, 'noise_mode': noise_mode,
        'causal_edge': '12 negative controls',
        'true_lag_ka': np.nan,
        'lag_error_ratio': np.nan,
        'robustness': max(ctl),
        'robustness_min': min(ctl),
        'robustness_median': float(np.median(ctl)),
        'passes_95': max(ctl) >= 0.95,
    })
    full = []
    for src in reg34.VAR_ORDER:
        for dst in reg34.VAR_ORDER:
            if src == dst:
                continue
            true = next((t for t in TRUE_EDGES_B
                         if t[0] == src and t[1] == dst), None)
            full.append({
                'maxlag': maxlag, 'noise_mode': noise_mode,
                'source': src, 'target': dst,
                'prescribed_true_edge': true is not None,
                'lag_ka': true[2] if true else np.nan,
                'robustness_with_error_median':
                    float(np.median(err_med[(src, dst)])),
            })
    return rows, full


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--quick', action='store_true',
                    help='smoke test: 3 realizations x 40 perturbations')
    ap.add_argument('--realizations', type=int, default=20)
    ap.add_argument('--perturb', type=int, default=200)
    ap.add_argument('--span', type=float, default=58.0)
    ap.add_argument('--maxlags', nargs='+', type=int, default=[5, 50],
                    help='VAR lag search caps to sweep (paper cap 1-5 vs '
                         'the canonical Scenario A config 50)')
    ap.add_argument('--tests', nargs='+', choices=['minp', 'lagonly'],
                    default=['lagonly'],
                    help='Layer 3 F-test semantics to sweep; canonical '
                         'Scenario B = lagonly (paper Section 3.3)')
    ap.add_argument('--noise-modes', nargs='+', choices=['std', 'variance'],
                    default=['std'],
                    help='sigma_e reading of Table S5 "sigma^2_eps": std '
                         '(consistent with the Scenario A rebuild) and/or '
                         'variance')
    ap.add_argument('--no-perdraw', action='store_true',
                    help='run Layer 2 once on the clean series instead of '
                         'inside each perturbation draw (paper 3.4 order)')
    ap.add_argument('--seed-base', type=int, default=42)
    args = ap.parse_args()

    if args.quick:
        n_real, n_pert = 3, 40
        label = 'QUICK'
    else:
        n_real, n_pert = args.realizations, args.perturb
        label = ''

    print(f"Scenario B run: span={args.span} tests={args.tests} "
          f"noise_modes={args.noise_modes} perdraw={not args.no_perdraw} "
          f"maxlags={args.maxlags} {n_real} x {n_pert}", flush=True)

    all_rows, all_full = [], []
    for maxlag in args.maxlags:
        for noise_mode in args.noise_modes:
            for test in args.tests:
                print(f"\n=== maxlag={maxlag} noise_mode={noise_mode} "
                      f"test={test} ===", flush=True)
                rows, full = run_config(
                    n_real, n_pert, args.span, maxlag, test, noise_mode,
                    not args.no_perdraw, args.seed_base, label)
                all_rows.extend(rows)
                all_full.extend(full)

    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(OUTPUT_DIR, 'scenarioB_tableS6.csv')
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}\n")
    print(df.to_string(index=False))
    print("\nPaper Table S6 targets: NGRIP->EDC 18.5, CO2->CH4 17.0, "
          "Cariaco->Cheng 20.9, controls 15-17 "
          "(Theorem 3 FPR bound for Scenario B: ~0.18)")

    full_csv = os.path.join(OUTPUT_DIR, 'scenarioB_tableS6_full.csv')
    pd.DataFrame(all_full).to_csv(full_csv, index=False)
    print(f"Saved: {full_csv}")


if __name__ == '__main__':
    main()
