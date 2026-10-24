#!/usr/bin/env python3
"""
19_full_pairwise_scan.py
------------------------
Record-level all-pairs CARC scan for all N records in the 10-60 ka window
(N*(N-1) directed pairs), with layered ablations, dual honesty thresholds,
and three methods (Granger / PCMCI / CCM).

Pipeline per pair (src -> dst), mirroring scripts/09 at record level:

  Layer 1  age perturbation: one shared time-axis draw N(0, sig_err),
           sig_err = sqrt((sigma_src^2 + sigma_dst^2) / 2) in PHYSICAL ka
           units (grid spacing 0.05 ka, so index noise = sig_err / dt).
           This matches src.granger.age_perturb_pair, which now takes the
           same physical-ka convention (sigma / dt internally) and an
           explicit ``dt`` argument, as does generate_scenario_A.
  Layer 2  ARC deconfounding (alpha = ARC_ALPHA) of every record against a
           1-ka Gaussian-smoothed NGRIP record (the NGRIP record itself is
           passed through unchanged).
  Layer 3  ADF stationarity check (per record, per variant, computed once)
           with conditional first-differencing, then Granger F-test across
           lags 1..MAX_GRANGER_LAG using the fast numpy engine
           (src/granger_fast.py, statsmodels-equivalent decisions).

Ablations (same perturbation draws, so differences are attributable):
  full      canonical pipeline above
  nonoise   same without observational noise injection
  noage     no age perturbation (noise injection only)
  noarc     raw records (no ARC deconfounding)
  alldiff   both series force-differenced (ADF bypassed). Edges that pass
            full but fail alldiff are carried by shared low-frequency
            structure (trend leakage), not by short-lag lead-lag.

Robustness rates are reported for each variant, with pass/fail flags at the
95% (STABILITY_THRESHOLD) and relaxed 80% honesty thresholds.

Phases:
  0  metadata (sigma / region / resolution / SNR) + ARC variants + ADF flags
  1  main scan -- sharded CSV under outputs/pairwise_scan/ (resumable)
  2  per-record aggregation -> outputs/pairwise_scan/record_profile.csv
  3  PCMCI (tigramite ParCorr) on the ARC'd dataset -> pcmci_pairs.csv
  4  CCM (pyEDM) on the top Granger-surviving pairs -> ccm_pairs.csv

Usage:
  python scripts/19_full_pairwise_scan.py            # full run
  python scripts/19_full_pairwise_scan.py --quick    # smoke test (40 pert)
  python scripts/19_full_pairwise_scan.py --only 2   # skip to phase 2
"""

import os, sys, glob, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from src.config import (DATA_DIR, OUTPUT_DIR, N_PERTURB, ARC_ALPHA,
                        MAX_GRANGER_LAG, NOISE_STD, STABILITY_THRESHOLD)
from src.io import load_all_records, load_record
from src.arc import arc_deconfound_batch
from src.granger_fast import granger_decision_fast
from statsmodels.tsa.stattools import adfuller
from scipy.ndimage import gaussian_filter1d
from joblib import Parallel, delayed

SCAN_DIR = os.path.join(OUTPUT_DIR, 'pairwise_scan')
THRESH_80 = 0.80
RNG_BASE = 20260815          # deterministic per-pair perturbation streams


# ---------------------------------------------------------------------------
# Phase 0: data preparation
# ---------------------------------------------------------------------------
def prepare_data():
    """Load records, ARC variants, ADF flags, per-record metadata."""
    records, grid = load_all_records(DATA_DIR, age_min=10.0, age_max=60.0)
    names = [n for _, _, _, n in records]
    print(f"Loaded {len(records)} records, grid {grid[0]:.1f}-{grid[-1]:.1f} ka "
          f"(dt={grid[1] - grid[0]:.3f})")

    raw = {n: np.asarray(v, float) for v, _, _, n in records}
    sigmas = {n: s for _, s, _, n in records}
    regions = {n: r for _, _, r, n in records}

    # ARC proxy: 1-ka Gaussian-smoothed NGRIP record
    ngrip = [n for n in names if 'GRIP' in n.upper()]
    if not ngrip:
        raise SystemExit("No NGRIP record found")
    ngrip = ngrip[0]
    proxy = gaussian_filter1d(raw[ngrip], sigma=1.0 / (grid[1] - grid[0]),
                              mode='nearest')
    arc_dict = arc_deconfound_batch(raw, proxy, alpha=ARC_ALPHA, keep_proxy=ngrip)
    print(f"ARC proxy: {ngrip} (alpha={ARC_ALPHA})")

    def diff_flags(d):
        out = {}
        for n, v in d.items():
            try:
                out[n] = adfuller(v, autolag='AIC')[1] > 0.05
            except Exception:
                out[n] = False
        return out

    flags_full = diff_flags(arc_dict)
    flags_raw = diff_flags(raw)
    print(f"ADF: {sum(flags_full.values())}/{len(flags_full)} records differenced "
          f"(ARC); {sum(flags_raw.values())}/{len(flags_raw)} (raw)")

    # per-record metadata: resolution (median raw spacing) and SNR proxy
    meta_rows = []
    for n in names:
        age, val = load_record(n, DATA_DIR)
        if age is not None and len(age) > 1:
            inr = (age >= 10.0) & (age <= 60.0)
            age = age[inr]
        res_ka = float(np.median(np.diff(age))) if (age is not None and len(age) > 1) else np.nan
        v = raw[n]
        dv = np.diff(v)
        snr = float(np.std(v) / np.std(dv)) if np.std(dv) > 1e-10 else np.nan
        meta_rows.append({'record': n, 'region': regions[n], 'sigma_ka': sigmas[n],
                          'resolution_ka': res_ka, 'snr_proxy': snr,
                          'diff_arc': bool(flags_full[n]),
                          'diff_raw': bool(flags_raw[n])})
    meta = pd.DataFrame(meta_rows)
    os.makedirs(SCAN_DIR, exist_ok=True)
    meta.to_csv(os.path.join(SCAN_DIR, 'record_meta.csv'), index=False)

    arrays = {'raw': raw, 'arc': arc_dict}
    flags = {'raw': flags_raw, 'arc': flags_full}
    return arrays, flags, meta, grid, ngrip


def reload_arrays():
    """Rebuild the record arrays (used by --only phases)."""
    records, grid = load_all_records(DATA_DIR, age_min=10.0, age_max=60.0)
    names = [n for _, _, _, n in records]
    raw = {n: np.asarray(v, float) for v, _, _, n in records}
    ngrip = [n for n in names if 'GRIP' in n.upper()][0]
    proxy = gaussian_filter1d(raw[ngrip], sigma=1.0 / (grid[1] - grid[0]),
                              mode='nearest')
    arrays = {'arc': arc_deconfound_batch(raw, proxy, alpha=ARC_ALPHA,
                                          keep_proxy=ngrip), 'raw': raw}
    meta = pd.read_csv(os.path.join(SCAN_DIR, 'record_meta.csv'))
    return arrays, meta, grid


# ---------------------------------------------------------------------------
# Phase 1: main scan (worker)
# ---------------------------------------------------------------------------
def _run_draw(aY, aX, bY, bX, idx, ny, nx, dy_full, dx_full, dy_raw, dx_raw,
              maxlag):
    """Run the five variant tests for one shared perturbation draw."""
    def test(Y, X, dy, dx):
        if dy:
            Y = np.diff(Y)
        if dx:
            X = np.diff(X)
        m = min(len(Y), len(X))
        return granger_decision_fast(Y[:m], X[:m], maxlag)

    Yp, Xp = aY[idx], aX[idx]
    return (test(Yp + ny, Xp + nx, dy_full, dx_full),   # full
            test(Yp, Xp, dy_full, dx_full),             # nonoise
            test(aY + ny, aX + nx, dy_full, dx_full),   # noage
            test(bY[idx] + ny, bX[idx] + nx, dy_raw, dx_raw),  # noarc
            test(np.diff(Yp + ny), np.diff(Xp + nx), False, False))  # alldiff


def scan_pair(pair_idx, aY, aX, bY, bX, sig_ka, dt,
              dy_full, dx_full, dy_raw, dx_raw,
              n_perturb, n_abl, maxlag, noise_std):
    """Full CARC scan for one directed pair (src -> dst)."""
    rng = np.random.default_rng(RNG_BASE + pair_idx)
    n = len(aY)
    sig_idx = sig_ka / dt          # physical ka -> index units
    rates = np.zeros(5)            # full, nonoise, noage, noarc, alldiff
    n_full = n_perturb
    for k in range(n_perturb):
        idx = np.argsort(np.arange(n) + rng.normal(0, sig_idx, n))
        Yp, Xp = aY[idx], aX[idx]
        ny = rng.normal(0, noise_std * np.std(Yp), n)
        nx = rng.normal(0, noise_std * np.std(Xp), n)
        if k < n_abl:
            r = _run_draw(aY, aX, bY, bX, idx, ny, nx,
                          dy_full, dx_full, dy_raw, dx_raw, maxlag)
            rates += [int(v) for v in r]
        else:
            r = _run_draw(aY, aX, bY, bX, idx, ny, nx,
                          dy_full, dx_full, dy_raw, dx_raw, maxlag)[0]
            rates[0] += int(r)
    return {'rate_full': rates[0] / n_perturb,
            'rate_nonoise': rates[1] / n_abl,
            'rate_noage': rates[2] / n_abl,
            'rate_noarc': rates[3] / n_abl,
            'rate_alldiff': rates[4] / n_abl,
            'pass95_full': rates[0] / n_perturb >= STABILITY_THRESHOLD,
            'pass80_full': rates[0] / n_perturb >= THRESH_80,
            'pass95_nonoise': rates[1] / n_abl >= STABILITY_THRESHOLD,
            'pass80_nonoise': rates[1] / n_abl >= THRESH_80,
            'pass95_noage': rates[2] / n_abl >= STABILITY_THRESHOLD,
            'pass80_noage': rates[2] / n_abl >= THRESH_80,
            'pass95_noarc': rates[3] / n_abl >= STABILITY_THRESHOLD,
            'pass80_noarc': rates[3] / n_abl >= THRESH_80,
            'pass95_alldiff': rates[4] / n_abl >= STABILITY_THRESHOLD,
            'pass80_alldiff': rates[4] / n_abl >= THRESH_80}


def scan_batch(batch, arrays, flags, sig_lookup, region_lookup, dt,
               n_perturb, n_abl, maxlag, noise_std):
    """Scan a batch of (src, dst, pair_idx) rows; return list of row dicts."""
    raw, arc = arrays['raw'], arrays['arc']
    f_raw, f_arc = flags['raw'], flags['arc']
    rows = []
    for src, dst, pair_idx in batch:
        aY, aX = arc[src], arc[dst]
        bY, bX = raw[src], raw[dst]
        sig_ka = float(np.sqrt((sig_lookup[src] ** 2 + sig_lookup[dst] ** 2) / 2))
        r = scan_pair(pair_idx, aY, aX, bY, bX, sig_ka, dt,
                      f_arc[src], f_arc[dst], f_raw[src], f_raw[dst],
                      n_perturb, n_abl, maxlag, noise_std)
        r.update({'source': src, 'target': dst, 'pair_idx': pair_idx,
                  'region_src': region_lookup[src], 'region_dst': region_lookup[dst],
                  'sigma_src': sig_lookup[src], 'sigma_dst': sig_lookup[dst],
                  'sig_err_ka': sig_ka})
        rows.append(r)
    return rows


def run_scan(arrays, flags, meta, grid, n_perturb, n_abl, n_jobs=-1,
             max_batches=None):
    names = meta['record'].tolist()
    pairs = [(s, d, i * (len(names) - 1) + j)
             for i, s in enumerate(names)
             for j, d in enumerate(names) if s != d]
    print(f"Main scan: {len(pairs)} directed pairs x {n_perturb} perturbations "
          f"(ablation variants on first {n_abl} draws)")

    sig_lookup = dict(zip(meta['record'], meta['sigma_ka']))
    region_lookup = dict(zip(meta['record'], meta['region']))
    dt = float(grid[1] - grid[0])

    batch_size = 20
    batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
    existing = {os.path.basename(p) for p in glob.glob(os.path.join(SCAN_DIR, 'shard_*.csv'))}
    todo = [(bi, b) for bi, b in enumerate(batches)
            if f'shard_{bi:04d}.csv' not in existing]
    if max_batches is not None:
        todo = todo[:max_batches]
    print(f"Resume: {len(batches) - len(todo)}/{len(batches)} shards already done; "
          f"processing {len(todo)}")

    results = Parallel(n_jobs=n_jobs)(
        delayed(scan_batch)(b, arrays, flags, sig_lookup, region_lookup, dt,
                            n_perturb, n_abl, MAX_GRANGER_LAG, NOISE_STD)
        for _, b in todo)
    for (bi, _), out in zip(todo, results):
        pd.DataFrame(out).to_csv(os.path.join(SCAN_DIR, f'shard_{bi:04d}.csv'),
                                 index=False)
        print(f"  shard {bi:04d} done ({len(out)} pairs)", flush=True)
    print("Scan complete.")


# ---------------------------------------------------------------------------
# Phase 2: per-record aggregation
# ---------------------------------------------------------------------------
def aggregate(meta):
    shards = sorted(glob.glob(os.path.join(SCAN_DIR, 'shard_*.csv')))
    if not shards:
        print("No shards found -- run phase 1 first")
        return
    df = pd.concat([pd.read_csv(p) for p in shards], ignore_index=True)
    print(f"Aggregating {len(df)} pair rows from {len(shards)} shards")

    names = meta['record'].tolist()
    rows = []
    for rec in names:
        out = df[df['source'] == rec]
        inc = df[df['target'] == rec]
        r = {'record': rec,
             'n_out': len(out), 'n_in': len(inc),
             'mean_rate_out': out['rate_full'].mean(),
             'mean_rate_in': inc['rate_full'].mean(),
             'max_rate_out': out['rate_full'].max(),
             'max_rate_in': inc['rate_full'].max(),
             'n_out_pass95': int(out['pass95_full'].sum()),
             'n_in_pass95': int(inc['pass95_full'].sum()),
             'n_out_pass80': int(out['pass80_full'].sum()),
             'n_in_pass80': int(inc['pass80_full'].sum()),
             'n_out_pass95_nonoise': int(out['pass95_nonoise'].sum()),
             'n_in_pass95_nonoise': int(inc['pass95_nonoise'].sum()),
             'n_out_pass95_noage': int(out['pass95_noage'].sum()),
             'n_in_pass95_noage': int(inc['pass95_noage'].sum()),
             'n_out_pass95_noarc': int(out['pass95_noarc'].sum()),
             'n_in_pass95_noarc': int(inc['pass95_noarc'].sum()),
             'n_out_pass95_alldiff': int(out['pass95_alldiff'].sum()),
             'n_in_pass95_alldiff': int(inc['pass95_alldiff'].sum()),
             # edges that FAIL full but recover under each ablation
             'recov_nonoise': int(((~out['pass95_full']) & out['pass95_nonoise']).sum()
                                  + ((~inc['pass95_full']) & inc['pass95_nonoise']).sum()),
             'recov_noage': int(((~out['pass95_full']) & out['pass95_noage']).sum()
                                + ((~inc['pass95_full']) & inc['pass95_noage']).sum()),
             'recov_noarc': int(((~out['pass95_full']) & out['pass95_noarc']).sum()
                                + ((~inc['pass95_full']) & inc['pass95_noarc']).sum()),
             # edges that PASS full but collapse under forced differencing
             # (i.e. carried by shared low-frequency structure = trend leakage)
             'n_trend_leak95': int((out['pass95_full'] & ~out['pass95_alldiff']).sum()
                                   + (inc['pass95_full'] & ~inc['pass95_alldiff']).sum()),
             }
        r.update({c: meta.loc[meta['record'] == rec, c].iloc[0]
                  for c in ['region', 'sigma_ka', 'resolution_ka', 'snr_proxy']})
        rows.append(r)
    prof = pd.DataFrame(rows)
    prof.to_csv(os.path.join(SCAN_DIR, 'record_profile.csv'), index=False)
    print("Saved record profile:")
    print(prof[['record', 'region', 'n_out_pass80', 'n_in_pass80',
                'n_out_pass95', 'n_in_pass95']].to_string(index=False))


# ---------------------------------------------------------------------------
# Phase 3: PCMCI on the ARC'd dataset
# ---------------------------------------------------------------------------
def run_pcmci(arrays, meta, grid):
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr

    names = meta['record'].tolist()
    arc = arrays['arc']
    data = np.column_stack([arc[n] for n in names])
    dataframe = pp.DataFrame(data, datatime=grid, var_names=names)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr())
    print(f"PCMCI ParCorr on {len(names)} records x {len(grid)} points "
          f"(tau_max={MAX_GRANGER_LAG}, alpha=0.05)...")
    results = pcmci.run_pcmci(tau_min=0, tau_max=MAX_GRANGER_LAG, pc_alpha=0.05)

    rows = []
    for i, src in enumerate(names):
        for j, dst in enumerate(names):
            if src == dst:
                continue
            p_tau = results['p_matrix'][i, j, :]
            lag = int(np.argmin(p_tau))
            rows.append({'source': src, 'target': dst,
                         'pcmci_min_p': float(np.min(p_tau)),
                         'pcmci_lag_tau': lag,
                         'pcmci_significant': bool(np.min(p_tau) < 0.05),
                         'pcmci_lagged_tau_ge1': bool((p_tau[1:] < 0.05).any())})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(SCAN_DIR, 'pcmci_pairs.csv'), index=False)
    print(f"PCMCI: {int(out['pcmci_significant'].sum())} significant pairs "
          f"(tau>=0); {int(out['pcmci_lagged_tau_ge1'].sum())} with lag tau>=1")


# ---------------------------------------------------------------------------
# Phase 4: CCM (pyEDM) on Granger-surviving pairs
# ---------------------------------------------------------------------------
def _simplex_rho(vals, E):
    """Leave-one-out nearest-neighbor forecast skill at embedding E.

    pyEDM 2.5's EmbedDimension/Embed hang on Windows (parallel machinery),
    so E selection uses this direct KDTree simplex projection: embed the
    series with lag-1 coordinates, forecast one step ahead from the nearest
    embedded neighbor (excluding self), return the correlation between
    forecast and observed. Same quantity EmbedDimension maximizes.
    """
    v = np.asarray(vals, float)
    n = len(v)
    if n < E + 20:
        return np.nan
    emb = np.column_stack([v[i: n - E + i] for i in range(E)])
    tgt = v[E:]
    from scipy.spatial import cKDTree
    _, idx = cKDTree(emb).query(emb, k=2)   # k=1 is self
    return float(np.corrcoef(tgt[idx[:, 1]], tgt)[0, 1])


def run_ccm(arrays, meta, grid, n_pairs=400):
    # pyEDM 2.5: legacy=True (sequential CCM_24; the new parallel CCM hangs
    # on Windows), and record names must be sanitized to 'SRC'/'DST' because
    # pyEDM splits column lists on whitespace.
    from pyEDM import CCM as pyCCM

    shards = sorted(glob.glob(os.path.join(SCAN_DIR, 'shard_*.csv')))
    if not shards:
        print("No shards found -- run phase 1 first")
        return
    df = pd.concat([pd.read_csv(p) for p in shards], ignore_index=True)
    surv = df[df['pass80_full']].sort_values('rate_full', ascending=False)
    surv = surv.head(n_pairs)
    print(f"CCM on {len(surv)} pairs surviving 80% full Granger")

    names = meta['record'].tolist()
    arc = arrays['arc']
    df_all = pd.DataFrame({'Time': grid})
    for n in names:
        df_all[n] = arc[n]
    E_best = {}
    for n in names:
        try:
            E_best[n] = max(int(np.nanargmax(
                [_simplex_rho(arc[n], E) for E in range(1, 7)])) + 1, 2)
        except Exception:
            E_best[n] = 3
    print(f"E selected: {dict(sorted(E_best.items(), key=lambda kv: -kv[1])[:6])}... "
          f"(mean {np.mean(list(E_best.values())):.1f})")

    rows = []
    for _, r in surv.iterrows():
        s, d = r['source'], r['target']
        sub = pd.DataFrame({'Time': grid, 'SRC': arc[s], 'DST': arc[d]})
        try:
            ccm = pyCCM(dataFrame=sub, E=E_best[s], Tp=0, columns='SRC',
                        target='DST',
                        libSizes="50 100 200 300 400 500 600 700",
                        sample=25, showPlot=False, legacy=True)
            fwd = float(ccm.iloc[-1]['SRC:DST'])
            rev = float(ccm.iloc[-1]['DST:SRC'])
        except Exception:
            fwd = rev = np.nan
        rows.append({'source': s, 'target': d,
                     'granger_rate': r['rate_full'],
                     'ccm_fwd': fwd, 'ccm_rev': rev,
                     'E_src': E_best[s], 'E_dst': E_best[d]})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(SCAN_DIR, 'ccm_pairs.csv'), index=False)
    ccm_ok = out.dropna(subset=['ccm_fwd', 'ccm_rev'])
    agree = ((ccm_ok['ccm_fwd'] > ccm_ok['ccm_rev'])).mean() if len(ccm_ok) else np.nan
    print(f"CCM: {len(ccm_ok)} pairs computed; fwd>rev agreement {agree:.0%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true',
                    help='smoke test: 40 perturbations, 20 ablation draws')
    ap.add_argument('--only', type=int, default=0,
                    help='skip to phase N (2=aggregate, 3=pcmci, 4=ccm)')
    args = ap.parse_args()

    n_perturb = 40 if args.quick else N_PERTURB
    n_abl = 20 if args.quick else 100
    max_batches = 2 if args.quick else None

    arrays = flags = meta = grid = None
    if args.only < 2:
        arrays, flags, meta, grid, ngrip = prepare_data()
        run_scan(arrays, flags, meta, grid, n_perturb, n_abl,
                 max_batches=max_batches)
    else:
        meta = pd.read_csv(os.path.join(SCAN_DIR, 'record_meta.csv'))

    if args.only <= 2:
        aggregate(meta)

    if args.only <= 3:
        if arrays is None:
            arrays, meta, grid = reload_arrays()
        run_pcmci(arrays, meta, grid)

    if args.only <= 4:
        if arrays is None:
            arrays, meta, grid = reload_arrays()
        run_ccm(arrays, meta, grid)

    print("\nAll phases done.")


if __name__ == "__main__":
    main()
