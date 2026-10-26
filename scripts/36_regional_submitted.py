#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
36_regional_submitted.py
------------------
submitted-pipeline regional stacking + robustness (paper Section 4.3), plus the
record-level Lapa -> ODP983 pairing and the P4 error-scaling analysis
(paper Section 5.3: compress the 1sigma errors to ~35% -> 95% robustness).

Record set: default data/region_assignment_91_proposed.csv (91 records /
7 regions per the old supplementary S2/S4). The current manuscript
(Aug-2026 edited version) describes 80 records / 6 regions (Table S2/S4):
pass ``--assignment data/region_assignment_80_paper.csv`` for the rebuilt
6-region set (counts 13/18/34/6/3/6, tiers 7/48/25, reconstructed from the
Table S4 key-site list, judgment calls documented in the CSV Basis column).
Records whose data files are absent from proxy_records/ are listed and
excluded.

Pipeline (paper Section 3, per-draw order):
  * grid 10-60 ka at 0.05 ka; each record interpolated and Z-standardized;
  * each of M = 200 draws:
      Layer 1: per-record Gaussian age perturbation N(0, sigma_record)
               (argsort permutation), median stack per region,
               Gaussian low-pass 0.5 ka;
      Layer 2: ARC deconfounding alpha = 0.5 against the 1-ka Gaussian-
               smoothed NGRIP trend (fixed, unperturbed). For pairs
               involving the North Atlantic stack (which contains NGRIP),
               a second variant uses the NA stack's own >5 ka orbital
               component as the proxy (paper Section 5.4).
               Current-manuscript variant (--na-raw): the NA composite is
               left undeconfounded and every OTHER composite is deconfounded
               against the per-draw NA stack itself (1-ka smoothed); with
               --obs-noise each draw adds 5% observational noise (5% of the
               deconfounded stack SD) per perturbation member, per the
               current Section 3.4;
      Layer 3: ADF unit-root gate (0.05, differencing), AIC VAR lag
               (cap 5), F-test at 0.05. Default ``test=minp``: any lag
               1..L significant. The Tier-1 six-record analysis (Table 4)
               is the single F at the AIC lag only (lagonly, scripts/35,
               P2 finding); the regional numbers of Section 4.3 (95-100%
               bidirectional data-rich windows, Lapa->ODP983 89.5%) are
               only reachable with min-p across lags (probe:
               outputs/_probe36.py), so the lost implementation evidently
               used different F-test semantics in the two sections.
  * robustness = fraction of the 200 draws with a significant edge.

Record-level pairing Lapa -> ODP983 (paper 4.3/5.3): sigma 0.39 ka (Lapa)
and 0.50 ka (ODP983-IRD), pair RMS 0.45 ka = mean(0.39, 0.50). The paper
target: H1 event robustness 89.5% (Lapa -> ODP983). Error-scaling sweeps
scales {1.0..0.2} x sigma and reports the first scale whose robustness
reaches 95% (paper: about 0.35).

Outputs:
  outputs/regional_submitted_windows.csv       region pairs x windows robustness
  outputs/lapa_odp983.csv            record-level pairing, 6 windows
  outputs/lapa_odp983_scaling.csv        P4 error-scaling sweep (H1)

Usage:
  python scripts/36_regional_submitted.py --quick       # 3 windows x 30 draws
  python scripts/36_regional_submitted.py               # full run (min-p, Sec 4.3)
  python scripts/36_regional_submitted.py --test lagonly  # Table 4 semantics
"""

import os, sys, re, argparse, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR

from src.config import (DATA_DIR, OUTPUT_DIR, AGE_GRID_10_60, CLIMATE_WINDOWS,
                        get_sigma)
from src.io import safe_load_csv, identify_columns, read_record_df
from src.stacking import classify_record
from src.arc import arc_deconfound
from src.granger_fast import _granger_stats, _f_crit, granger_decision_block

DT = float(AGE_GRID_10_60[1] - AGE_GRID_10_60[0])   # 0.05 ka
ARC_ALPHA = 0.5
M_DRAWS = 200
MAXLAG = 5
SEED = 42
SMOOTH_EVENT = 0.5 / DT        # Gaussian low-pass 0.5 ka (Layer 1)
TREND_1K = 1.0 / DT            # 1-ka NGRIP millennial trend (Layer 2)
ORBITAL_5K = 5.0 / DT          # >5 ka orbital component (Layer 2, NA pairs)

# Paper targets (Section 4.3 / 5.3)
PAPER = {
    'regional_data_rich': '95-100% bidirectional in H1/H4/LGM',
    'lapa_odp983_H1': 89.5,             # Lapa -> ODP983, H1 event
    'lapa_odp983_rms': 0.45,            # = mean(0.39, 0.50)
    'ngrip_edc_rms': 0.74,              # = sqrt(mean(0.65^2, 0.83^2))
    'scale_crossing': 0.35,             # error compression -> 95%
}
SIGMA_LAPA, SIGMA_ODP983 = 0.39, 0.50   # paper line 161 scalars


# ---------------------------------------------------------------------------
# Record -> file resolution (inverted RECORD_SITE map from scripts/20 +
# explicit additions)
# ---------------------------------------------------------------------------
def _build_record_file_map():
    src = open(os.path.join(os.path.dirname(__file__),
                            '20_map_atlas.py'), encoding='utf-8').read()
    block = re.search(r'RECORD_SITE = \{(.*?)\n\}', src, re.S).group(1)

    class _C(dict):
        def __missing__(self, k):
            return k
    ns = {}
    exec('RECORD_SITE = {' + block + '}', {'RECORD_COORDS': _C()}, ns)
    rec_site = ns['RECORD_SITE']

    files = [f for f in os.listdir(DATA_DIR)
             if f.endswith('.csv') or f.endswith('.xlsx')]
    rec2file = {}
    for f in files:
        stem = os.path.splitext(f)[0].strip()
        r = rec_site.get(stem)
        if r:
            rec2file.setdefault(r, []).append(f)
    # explicit additions not covered by RECORD_SITE
    rec2file.setdefault('Lapa_d18O', []).append('Lapa_d18O.csv')
    return rec2file


# ---------------------------------------------------------------------------
# Loading and Layer 1/2 helpers
# ---------------------------------------------------------------------------
def load_grid_series(fname, grid):
    """Read one record file, interpolate onto grid, ffill/bfill, standardize.

    Returns (series, sigma) or (None, None).
    """
    path = os.path.join(DATA_DIR, fname)
    try:
        if fname.endswith('.xlsx'):
            df = read_record_df(path)   # handles Pangaea-style preambles
            ac, vc = identify_columns(df)
            if ac is None or vc is None:
                return None, None
            age = df[ac].astype(float).values
            val = df[vc].astype(float).values
        else:
            df = safe_load_csv(path)      # headerless two-column, no row loss
            if len(df) < 10:
                return None, None
            age, val = df['age'].values, df['value'].values
    except Exception:
        return None, None

    m = np.isfinite(age) & np.isfinite(val)
    age, val = age[m], val[m]
    if len(age) < 10 or not (age.min() <= grid[-1] and age.max() >= grid[0]):
        return None, None
    order = np.argsort(age)
    age, val = age[order], val[order]
    f = interp1d(age, val, kind='linear', bounds_error=False,
                 fill_value=np.nan)
    sig = f(grid)
    valid = np.where(np.isfinite(sig))[0]
    if valid.size == 0:
        return None, None
    if valid.size < len(sig):
        sig[:valid[0]] = sig[valid[0]]
        sig[valid[-1] + 1:] = sig[valid[-1]]
    sig = np.nan_to_num(sig, nan=np.nanmean(sig))
    if np.std(sig) < 1e-6:
        return None, None

    typ, _ = classify_record(fname)
    sigma = get_sigma(os.path.splitext(fname)[0], archive_type=typ)
    return (sig - sig.mean()) / sig.std(), sigma


def adf_flag(x):
    try:
        return adfuller(x, autolag='AIC')[1] > 0.05
    except Exception:
        return False


def aic_lag(res1, res2, maxlag=MAXLAG):
    n = min(len(res1), len(res2))
    try:
        model = VAR(np.column_stack([res1[:n], res2[:n]]))
        order = model.select_order(maxlags=maxlag)
        L = order.aic
        return int(L) if (L is not None and L >= 1) else maxlag
    except Exception:
        return maxlag


def lagonly_significant(Y, X, L, alpha=0.05):
    n = min(len(Y), len(X))
    if n < L + 3:
        return False
    stats = _granger_stats(Y[:n], X[:n], L)
    if not stats:
        return False
    F, p, df2 = stats[L - 1]
    return F > _f_crit(p, df2, alpha)


def pair_significant(Y, X, L, test='minp', alpha=0.05):
    """Layer 3 decision for one direction.

    test='minp'   : significant if ANY lag 1..L rejects at alpha
                    (Section 4.3 semantics; probe outputs/_probe36.py)
    test='lagonly': single F at the AIC-selected lag only
                    (Table 4 Tier-1 semantics; P2 finding)
    """
    if test == 'lagonly':
        return lagonly_significant(Y, X, L, alpha)
    return bool(granger_decision_block(Y, X, L, alpha))


# ---------------------------------------------------------------------------
# Per-draw Layer 1 + Layer 2 over the full 10-60 ka grid
# ---------------------------------------------------------------------------
def draw_deconfound(regions, ngrip_trend, rng, na_proxy_mode,
                    na_raw=False, obs_noise=False, na_proxy_smooth='1k'):
    """One draw: Layer 1 stack + Layer 2 deconfound for every region.

    regions : {name: [(grid_series, sigma), ...]}
    na_raw  : current-manuscript variant -- leave the North_Atlantic stack
              undeconfounded and deconfound every other region against the
              per-draw NA stack itself (1-ka smoothed).
    obs_noise: add 5% observational noise per perturbation member (5% of the
              deconfounded stack SD), per current Section 3.4.
    Returns {name: deconfounded stack (1001,)}.
    """
    out = {}
    for rname, recs in regions.items():
        members = []
        for sig, sigma, _rec in recs:
            pert = AGE_GRID_10_60 + rng.normal(0, sigma, len(AGE_GRID_10_60))
            members.append(sig[np.argsort(pert)])
        stack = np.median(np.vstack(members), axis=0)
        stack = gaussian_filter1d(stack, sigma=SMOOTH_EVENT, mode='nearest')
        out[rname] = stack
    # Layer 2
    if na_raw:
        if na_proxy_smooth == '1k':
            na_proxy = gaussian_filter1d(out['North_Atlantic'], sigma=TREND_1K,
                                         mode='nearest')
        else:                                 # literal Sec 5.4 reading:
            na_proxy = out['North_Atlantic']  # raw 0.5-ka NA composite
        for rname, stack in out.items():
            if rname == 'North_Atlantic':
                continue                      # NA left undeconfounded (Sec 5.4)
            out[rname] = arc_deconfound(stack, na_proxy, alpha=ARC_ALPHA)
    else:
        for rname, stack in out.items():
            if na_proxy_mode == 'ngrip5k' and rname == 'North_Atlantic':
                proxy = gaussian_filter1d(stack, sigma=ORBITAL_5K,
                                          mode='nearest')
            else:
                proxy = ngrip_trend
            out[rname] = arc_deconfound(stack, proxy, alpha=ARC_ALPHA)
    if obs_noise:
        for rname, stack in out.items():
            out[rname] = stack + rng.normal(0, 0.05 * np.std(stack),
                                            len(stack))
    return out


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run_analysis(n_draws, windows, rng, na_proxy_mode='ngrip1k',
                 do_scaling=False, test='minp',
                 assignment='region_assignment_91_proposed.csv',
                 na_raw=False, obs_noise=False, out_name='regional_submitted_windows',
                 do_lapa=True, na_proxy_smooth='1k'):
    rec2file = _build_record_file_map()
    assign = pd.read_csv(os.path.join('data', assignment))

    # --- resolve records to files ---
    missing = []
    regions = {}
    for _, row in assign.iterrows():
        rname, region = row.Record, row.Region
        files = rec2file.get(rname)
        if not files:
            missing.append((rname, region))
            continue
        # first file wins if several share a semantic name
        series = load_grid_series(files[0], AGE_GRID_10_60)
        if series[0] is None:
            missing.append((rname, region))
            continue
        regions.setdefault(region, []).append((series[0], series[1], rname))

    regions = {r: v for r, v in regions.items() if len(v) >= 2}
    print("Regions (records with files; >=2 required):")
    for r, v in regions.items():
        print(f"  {r}: {len(v)} records")
    print(f"\nExcluded (missing file or unloadable): {len(missing)}")
    for r, g in missing:
        print(f"  {r}  ({g})")

    # --- NGRIP trend proxy (fixed across draws) ---
    ngrip_series = load_grid_series('Erhardt_NGRIP_aerosol_d18O_step1.csv',
                                    AGE_GRID_10_60)
    if ngrip_series[0] is None:
        raise RuntimeError('NGRIP record not loadable')
    ngrip_trend = gaussian_filter1d(ngrip_series[0], sigma=TREND_1K,
                                    mode='nearest')

    # --- per-draw stacks + deconfounding ---
    draw_stacks = [draw_deconfound(regions, ngrip_trend, rng, na_proxy_mode,
                                   na_raw=na_raw, obs_noise=obs_noise,
                                   na_proxy_smooth=na_proxy_smooth)
                   for _ in range(n_draws)]

    # --- Layer 3+4 on region pairs ---
    rnames = sorted(regions)
    pairs = [(a, b) for i, a in enumerate(rnames) for b in rnames[i + 1:]]
    rows = []
    for wlabel, (w0, w1) in windows.items():
        idx0 = int(np.searchsorted(AGE_GRID_10_60, w0))
        idx1 = int(np.searchsorted(AGE_GRID_10_60, w1))
        for a, b in pairs:
            c_ab = c_ba = 0
            for stacks in draw_stacks:
                ya = stacks[a][idx0:idx1]
                yb = stacks[b][idx0:idx1]
                fa, fb = adf_flag(ya), adf_flag(yb)
                Ya = np.diff(ya) if fa else ya
                Xa = np.diff(yb) if fb else yb
                Yb = np.diff(yb) if fb else yb
                Xb = np.diff(ya) if fa else ya
                c_ab += pair_significant(Ya, Xa, aic_lag(Ya, Xa), test)
                c_ba += pair_significant(Yb, Xb, aic_lag(Yb, Xb), test)
            rows.append({'window': wlabel, 'region_A': a, 'region_B': b,
                         'A_to_B': c_ab / n_draws, 'B_to_A': c_ba / n_draws})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUTPUT_DIR, out_name + '.csv'), index=False)
    print("\nRegional pair robustness (A->B / B->A):")
    for wlabel, (w0, w1) in windows.items():
        sub = res[res.window == wlabel]
        print(f"\n{wlabel} ({w0}-{w1} ka):")
        for _, r in sub.iterrows():
            flag = "  <-- bidirectional >=95%" if (
                r.A_to_B >= 0.95 and r.B_to_A >= 0.95) else ""
            print(f"  {r.region_A:<24} <-> {r.region_B:<24} "
                  f"{r.A_to_B:6.1%} / {r.B_to_A:6.1%}{flag}")

    # --- record-level Lapa -> ODP983 (legacy paper section; opt-in) ---
    lapa = load_grid_series('Lapa_d18O.csv', AGE_GRID_10_60)
    odp = load_grid_series('2_ODP983-IRD.csv', AGE_GRID_10_60)
    pair_rows, scaling_rows = [], []
    if do_lapa and lapa[0] is not None and odp[0] is not None:
        pair_rows = record_pair_windows(lapa[0], odp[0], ngrip_trend,
                                        windows, n_draws, rng, test=test)
        pdf = pd.DataFrame(pair_rows)
        pdf.to_csv(os.path.join(OUTPUT_DIR, 'lapa_odp983.csv'),
                   index=False)
        print("\nRecord-level Lapa -> ODP983 (sigma 0.39/0.50 ka):")
        for _, r in pdf.iterrows():
            print(f"  {r.window:<24} Lapa->ODP983 {r.Lapa_to_ODP983:6.1%}"
                  f"   ODP983->Lapa {r.ODP983_to_Lapa:6.1%}"
                  f"   (paper H1: {PAPER['lapa_odp983_H1']}%)")
        if do_scaling:
            scaling_rows = error_scaling(lapa[0], odp[0], ngrip_trend,
                                         windows['H1 event'], n_draws, rng,
                                         test=test)
            sdf = pd.DataFrame(scaling_rows)
            sdf.to_csv(os.path.join(OUTPUT_DIR, 'lapa_odp983_scaling.csv'),
                       index=False)

    print("\nPaper arithmetic checks:")
    print(f"  Lapa-ODP983 pair RMS = mean(0.39, 0.50) = "
          f"{(SIGMA_LAPA + SIGMA_ODP983) / 2:.2f} ka "
          f"(paper about {PAPER['lapa_odp983_rms']})")
    print(f"  NGRIP-EDC pair RMS = sqrt(mean(0.65^2, 0.83^2)) = "
          f"{np.sqrt((0.65 ** 2 + 0.83 ** 2) / 2):.2f} ka "
          f"(paper about {PAPER['ngrip_edc_rms']})")
    return res, pair_rows, scaling_rows


def record_pair_windows(lapa, odp, ngrip_trend, windows, n_draws, rng,
                        test='minp'):
    rows = []
    for wlabel, (w0, w1) in windows.items():
        idx0 = int(np.searchsorted(AGE_GRID_10_60, w0))
        idx1 = int(np.searchsorted(AGE_GRID_10_60, w1))
        c_lo = c_ol = 0
        for _ in range(n_draws):
            pl = AGE_GRID_10_60 + rng.normal(0, SIGMA_LAPA,
                                             len(AGE_GRID_10_60))
            po = AGE_GRID_10_60 + rng.normal(0, SIGMA_ODP983,
                                             len(AGE_GRID_10_60))
            yl = lapa[np.argsort(pl)]
            yo = odp[np.argsort(po)]
            yl = arc_deconfound(yl, ngrip_trend, alpha=ARC_ALPHA)
            yo = arc_deconfound(yo, ngrip_trend, alpha=ARC_ALPHA)
            yl, yo = yl[idx0:idx1], yo[idx0:idx1]
            fl, fo = adf_flag(yl), adf_flag(yo)
            Yl = np.diff(yl) if fl else yl
            Xl = np.diff(yo) if fo else yo
            Yo = np.diff(yo) if fo else yo
            Xo = np.diff(yl) if fl else yl
            c_lo += pair_significant(Yl, Xl, aic_lag(Yl, Xl), test)
            c_ol += pair_significant(Yo, Xo, aic_lag(Yo, Xo), test)
        rows.append({'window': wlabel, 'Lapa_to_ODP983': c_lo / n_draws,
                     'ODP983_to_Lapa': c_ol / n_draws})
    return rows


def error_scaling(lapa, odp, ngrip_trend, window, n_draws, rng,
                  test='minp'):
    """P4: sweep sigma scale for the H1 Lapa -> ODP983 pairing."""
    w0, w1 = window
    idx0 = int(np.searchsorted(AGE_GRID_10_60, w0))
    idx1 = int(np.searchsorted(AGE_GRID_10_60, w1))
    print("\nError-scaling (fraction of present 1sigma errors -> robustness):")
    rows, crossing = [], None
    for scale in [1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.2]:
        c = 0
        for _ in range(n_draws):
            pl = AGE_GRID_10_60 + rng.normal(0, SIGMA_LAPA * scale,
                                             len(AGE_GRID_10_60))
            po = AGE_GRID_10_60 + rng.normal(0, SIGMA_ODP983 * scale,
                                             len(AGE_GRID_10_60))
            yl = lapa[np.argsort(pl)]
            yo = odp[np.argsort(po)]
            yl = arc_deconfound(yl, ngrip_trend, alpha=ARC_ALPHA)
            yo = arc_deconfound(yo, ngrip_trend, alpha=ARC_ALPHA)
            yl, yo = yl[idx0:idx1], yo[idx0:idx1]
            fl, fo = adf_flag(yl), adf_flag(yo)
            Yl = np.diff(yl) if fl else yl
            Xl = np.diff(yo) if fo else yo
            c += pair_significant(Yl, Xl, aic_lag(Yl, Xl), test)
        r = c / n_draws
        rows.append({'scale': scale, 'robustness': r,
                     'sigma_lapa': SIGMA_LAPA * scale,
                     'sigma_odp983': SIGMA_ODP983 * scale})
        print(f"  {scale:.2f} -> {r:.1%}")
        if crossing is None and r >= 0.95:
            crossing = scale
    if crossing is not None:
        print(f"  95% threshold crossed at ~{crossing:.0%} of present errors "
              f"(paper: about {PAPER['scale_crossing']:.0%})")
    else:
        print("  95% threshold not reached at any tested scale")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--quick', action='store_true',
                    help='smoke test: 3 windows x 30 draws')
    ap.add_argument('--draws', type=int, default=M_DRAWS)
    ap.add_argument('--windows', nargs='*', default=None,
                    help='subset of climate windows')
    ap.add_argument('--na-proxy', choices=['ngrip1k', 'ngrip5k'],
                    default='ngrip1k',
                    help='Layer 2 proxy for the North Atlantic stack '
                         '(contains NGRIP): 1-ka NGRIP trend (paper 3.3 '
                         'default) or its own >5 ka orbital component '
                         '(paper 5.4)')
    ap.add_argument('--test', choices=['lagonly', 'minp'], default='minp',
                    help='Layer 3 F-test semantics: min-p across lags 1..AIC '
                         '(Section 4.3, default) or single F at the AIC lag '
                         '(Table 4 Tier-1, scripts/35)')
    ap.add_argument('--scale', action='store_true',
                    help='run the P4 error-scaling sweep')
    ap.add_argument('--assignment', default='region_assignment_91_proposed.csv',
                    help='region assignment CSV under data/ '
                         '(default: 91/7 proposal; current manuscript: '
                         'region_assignment_80_paper.csv)')
    ap.add_argument('--na-raw', action='store_true',
                    help='current-manuscript variant: NA composite left '
                         'undeconfounded, all others deconfounded against '
                         'the per-draw NA stack (Sec 5.4)')
    ap.add_argument('--obs-noise', action='store_true',
                    help='add 5% observational noise per perturbation '
                         'member (current Sec 3.4)')
    ap.add_argument('--na-proxy-smooth', choices=['1k', 'raw'], default='1k',
                    help='with --na-raw: smooth the NA composite proxy with '
                         '1-ka Gaussian (analogy to the six-record Layer 2) '
                         'or use the raw 0.5-ka NA stack (literal Sec 5.4)')
    ap.add_argument('--out', default='regional_submitted_windows',
                    help='output basename under outputs/')
    ap.add_argument('--no-lapa', action='store_true',
                    help='skip the legacy record-level Lapa section')
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    windows = CLIMATE_WINDOWS
    if args.windows:
        windows = {k: v for k, v in CLIMATE_WINDOWS.items()
                   if k in args.windows}
    if args.quick:
        windows = dict(list(windows.items())[:3])
        args.draws = 30

    run_analysis(args.draws, windows, rng, na_proxy_mode=args.na_proxy,
                 do_scaling=args.scale, test=args.test,
                 assignment=args.assignment, na_raw=args.na_raw,
                 obs_noise=args.obs_noise, out_name=args.out,
                 do_lapa=not args.no_lapa,
                 na_proxy_smooth=args.na_proxy_smooth)


if __name__ == '__main__':
    main()
