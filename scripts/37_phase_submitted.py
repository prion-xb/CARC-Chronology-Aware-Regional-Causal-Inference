#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
37_phase_submitted.py
---------------
submitted-pipeline phase-difference analysis on the millennial-trend band
(paper Section 4.4, Table 5) plus the sliding-window network-coupling
peak analysis (2.0 ka wide, 0.1 ka step).

Method (paper 4.4):
  * per draw m (M = 200, same seed/stream as scripts/36): Layer 1 median
    stack per region with Gaussian low-pass 0.5 ka;
  * within each climate window, cross-correlation lag between the two
    low-passed regional series is searched over +-2.0 ka;
    lag > 0 means region A leads region B;
  * sign consistency = fraction of the 200 draws in which A leads B;
    a window-pair combination is a "temporal tendency" when consistency
    >= 95%; the median lag over draws is reported;
  * sliding-window network coupling: the same lag search in 2.0-ka-wide
    windows stepped by 0.1 ka over 10-60 ka; the paper reports that no
    direction attains 95% sign consistency.

Paper Table 5 targets (current manuscript, 21 rows; region names mapped to
the 6-region assignment CSV):
  H1:  EAM->NA 100.0 +2.000 | NA->HLL 100.0 +1.950 | ASO->SAM 95.5 +1.275 |
       ASO->HLL 98.0 +2.000 | Trop->EAM 96.5 +2.000 | SAM->Trop 98.0 +2.000 |
       EAM->SAM 99.5 +1.950 | SAM->HLL 100.0 +1.350 | EAM->ASO 97.5 +1.325
  H4:  EAM->HLL 95.5 +2.000
  B-A: NA->EAM 100.0 +1.350 | ASO->EAM 100.0 +0.950 | EAM->Trop 100.0 +1.200 |
       EAM->SAM 100.0 +1.050 | EAM->HLL 100.0 +1.050
  LGM: NA->EAM 98.5 +2.000 | Trop->EAM 95.0 +2.000 | EAM->ASO 100.0 +2.000 |
       EAM->SAM 100.0 +2.000 | EAM->HLL 100.0 +2.000
  GI-12: SAM->EAM 96.5 +1.300
  (positive lag = A leads B; +2.000 sits exactly on the search boundary)
The current-manuscript note: low-pass sigma = 0.5 ka, deconfounding against
the North Atlantic composite -> single 0.5-ka pass (no second smoothing in
the phase loop) and the --na-raw variant of scripts/36.

Outputs:
  outputs/phase_submitted_table5.csv      per pair-window consistency + median lag
  outputs/phase_submitted_table5_sliding.csv     sliding-window max consistency per pair
"""

import os, sys, re, argparse, warnings, importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from src.config import (DATA_DIR, OUTPUT_DIR, AGE_GRID_10_60, CLIMATE_WINDOWS)

DT = float(AGE_GRID_10_60[1] - AGE_GRID_10_60[0])
MAX_LAG_KA = 2.0
SLIDE_WIDTH = 2.0
SLIDE_STEP = 0.1
M_DRAWS = 200
SEED = 42
SMOOTH_EVENT = 0.5 / DT

# Paper Table 5 targets (current manuscript): (pair, window) ->
# (consistency %, median lag ka)
PAPER_T5 = {
    ('East_Asian_Monsoon', 'North_Atlantic', 'H1 event'): (100.0, 2.000),
    ('North_Atlantic', 'HighLat_Lakes', 'H1 event'): (100.0, 1.950),
    ('Antarctic_SouthernOcean', 'South_America', 'H1 event'): (95.5, 1.275),
    ('Antarctic_SouthernOcean', 'HighLat_Lakes', 'H1 event'): (98.0, 2.000),
    ('Tropical', 'East_Asian_Monsoon', 'H1 event'): (96.5, 2.000),
    ('South_America', 'Tropical', 'H1 event'): (98.0, 2.000),
    ('East_Asian_Monsoon', 'South_America', 'H1 event'): (99.5, 1.950),
    ('South_America', 'HighLat_Lakes', 'H1 event'): (100.0, 1.350),
    ('East_Asian_Monsoon', 'Antarctic_SouthernOcean',
     'H1 event'): (97.5, 1.325),
    ('East_Asian_Monsoon', 'HighLat_Lakes', 'H4 event'): (95.5, 2.000),
    ('North_Atlantic', 'East_Asian_Monsoon', 'Bølling-Allerød'): (100.0, 1.350),
    ('Antarctic_SouthernOcean', 'East_Asian_Monsoon',
     'Bølling-Allerød'): (100.0, 0.950),
    ('East_Asian_Monsoon', 'Tropical', 'Bølling-Allerød'): (100.0, 1.200),
    ('East_Asian_Monsoon', 'South_America',
     'Bølling-Allerød'): (100.0, 1.050),
    ('East_Asian_Monsoon', 'HighLat_Lakes',
     'Bølling-Allerød'): (100.0, 1.050),
    ('North_Atlantic', 'East_Asian_Monsoon',
     'Last Glacial Maximum'): (98.5, 2.000),
    ('Tropical', 'East_Asian_Monsoon',
     'Last Glacial Maximum'): (95.0, 2.000),
    ('East_Asian_Monsoon', 'Antarctic_SouthernOcean',
     'Last Glacial Maximum'): (100.0, 2.000),
    ('East_Asian_Monsoon', 'South_America',
     'Last Glacial Maximum'): (100.0, 2.000),
    ('East_Asian_Monsoon', 'HighLat_Lakes',
     'Last Glacial Maximum'): (100.0, 2.000),
    ('South_America', 'East_Asian_Monsoon', 'GI-12 (interstadial)'): (96.5, 1.300),
}

# reuse scripts/36 loading + per-draw stack machinery (same seed -> same draws)
_spec = importlib.util.spec_from_file_location(
    'reg36', os.path.join(os.path.dirname(__file__), '36_regional_submitted.py'))
reg36 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reg36)


def compute_phase_lag(x, y, dt, max_lag_ka=MAX_LAG_KA):
    """Cross-correlation lag; > 0 means x leads y (same convention as
    scripts/08 and Scenario A: target(t) += c * driver(t - tau))."""
    lags_ka = np.arange(-max_lag_ka, max_lag_ka + dt / 2, dt)
    shifts = np.round(lags_ka / dt).astype(int)
    cors = np.full_like(lags_ka, np.nan)
    for i, s in enumerate(shifts):
        if s < 0:
            xs, ys = x[:s], y[-s:]
        elif s > 0:
            xs, ys = x[s:], y[:-s]
        else:
            xs, ys = x, y
        n = min(len(xs), len(ys))
        if n < 10:
            continue
        cors[i] = np.corrcoef(xs[:n], ys[:n])[0, 1]
    return -lags_ka[np.nanargmax(cors)]


def build_draw_stacks(n_draws, rng, na_proxy_mode='ngrip1k',
                      assignment='region_assignment_91_proposed.csv',
                      na_raw=False, obs_noise=False):
    rec2file = reg36._build_record_file_map()
    assign = pd.read_csv(os.path.join('data', assignment))
    regions = {}
    for _, row in assign.iterrows():
        files = rec2file.get(row.Record)
        if not files:
            continue
        series = reg36.load_grid_series(files[0], AGE_GRID_10_60)
        if series[0] is None:
            continue
        regions.setdefault(row.Region, []).append(
            (series[0], series[1], row.Record))
    regions = {r: v for r, v in regions.items() if len(v) >= 2}
    ngrip = reg36.load_grid_series(
        'Erhardt_NGRIP_aerosol_d18O_step1.csv', AGE_GRID_10_60)
    if ngrip[0] is None:
        raise RuntimeError('NGRIP not loadable')
    ngrip_trend = gaussian_filter1d(ngrip[0],
                                    sigma=reg36.TREND_1K, mode='nearest')
    stacks = [reg36.draw_deconfound(regions, ngrip_trend, rng, na_proxy_mode,
                                    na_raw=na_raw, obs_noise=obs_noise)
              for _ in range(n_draws)]
    return regions, stacks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--quick', action='store_true',
                    help='30 draws, first 3 windows')
    ap.add_argument('--draws', type=int, default=M_DRAWS)
    ap.add_argument('--no-sliding', action='store_true',
                    help='skip the sliding-window analysis')
    ap.add_argument('--assignment', default='region_assignment_91_proposed.csv',
                    help='region assignment CSV under data/ '
                         '(current manuscript: region_assignment_80_paper.csv)')
    ap.add_argument('--na-raw', action='store_true',
                    help='NA composite undeconfounded, others deconfounded '
                         'against the per-draw NA stack (current Sec 5.4)')
    ap.add_argument('--obs-noise', action='store_true',
                    help='5% observational noise per perturbation member')
    ap.add_argument('--out', default='phase_submitted_table5',
                    help='output basename under outputs/')
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    regions, stacks = build_draw_stacks(args.draws, rng,
                                        assignment=args.assignment,
                                        na_raw=args.na_raw,
                                        obs_noise=args.obs_noise)
    rnames = sorted(regions)
    pairs = [(a, b) for i, a in enumerate(rnames) for b in rnames[i + 1:]]

    windows = CLIMATE_WINDOWS
    if args.quick:
        windows = dict(list(windows.items())[:3])

    # --- Table 5: windowed phase-difference analysis ---
    rows = []
    print("Phase-difference analysis (A->B sign consistency, median lag):")
    for wlabel, (w0, w1) in windows.items():
        i0 = int(np.searchsorted(AGE_GRID_10_60, w0))
        i1 = int(np.searchsorted(AGE_GRID_10_60, w1))
        for a, b in pairs:
            lags = np.zeros(args.draws)
            for m, st in enumerate(stacks):
                # single low-pass 0.5 ka: Layer 1 already smoothed the stacks
                x = st[a][i0:i1]
                y = st[b][i0:i1]
                lags[m] = compute_phase_lag(x, y, DT)
            cons = np.mean(lags > 0)
            med = float(np.median(lags))
            target = PAPER_T5.get((a, b, wlabel))
            mark = '  <-- >=95%' if cons >= 0.95 else ''
            if target:
                mark += f'  [paper {target[0]:.1f}% / +{target[1]:.3f} ka]'
            if cons >= 0.95 or target:
                print(f"  {wlabel:<24} {a:<24} -> {b:<24} "
                      f"{cons:6.1%}  {med:+.3f} ka{mark}")
            rows.append({'window': wlabel, 'region_A': a, 'region_B': b,
                         'consistency_A_leads': cons, 'median_lag_ka': med,
                         'paper_consistency': target[0] if target else np.nan,
                         'paper_lag': target[1] if target else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, args.out + '.csv'), index=False)
    print(f"\nSaved: outputs/{args.out}.csv")

    # --- sliding-window network coupling (2.0 ka, 0.1 ka step) ---
    if args.no_sliding:
        return
    print("\nSliding-window sign consistency (2.0 ka wide, 0.1 ka step):")
    starts = np.arange(AGE_GRID_10_60[0],
                       AGE_GRID_10_60[-1] - SLIDE_WIDTH + 1e-9, SLIDE_STEP)
    n_slide = int(np.round(SLIDE_WIDTH / DT))
    slide_rows, worst = [], []
    for a, b in pairs:
        pair_max = 0.0
        pair_max_info = None
        for s0 in starts:
            i0 = int(np.searchsorted(AGE_GRID_10_60, s0))
            i1 = i0 + n_slide
            if i1 > len(AGE_GRID_10_60):
                continue
            signs = np.zeros(args.draws)
            for m, st in enumerate(stacks):
                # single low-pass 0.5 ka (Layer 1 already smoothed)
                x = st[a][i0:i1]
                y = st[b][i0:i1]
                signs[m] = 1.0 if compute_phase_lag(x, y, DT) > 0 else 0.0
            cons = max(np.mean(signs), 1 - np.mean(signs))
            if cons > pair_max:
                pair_max = cons
                pair_max_info = (round(s0, 1), round(s0 + SLIDE_WIDTH, 1))
            slide_rows.append({'region_A': a, 'region_B': b,
                               'window_start': round(s0, 2),
                               'max_directional_consistency': cons})
        print(f"  {a:<24} <-> {b:<24} peak {pair_max:6.1%}"
              f"  at {pair_max_info}")
        worst.append({'region_A': a, 'region_B': b,
                      'peak_consistency': pair_max,
                      'peak_window_ka': pair_max_info})
    sdf = pd.DataFrame(slide_rows)
    sdf.to_csv(os.path.join(OUTPUT_DIR, args.out + '_sliding.csv'),
               index=False)
    wdf = pd.DataFrame(worst)
    print(f"\nPeak directional consistency across all pair-windows: "
          f"{wdf.peak_consistency.max():.1%} "
          f"(paper: no direction attains 95%)")
    print(f"Saved: outputs/{args.out}_sliding.csv")


if __name__ == '__main__':
    main()
