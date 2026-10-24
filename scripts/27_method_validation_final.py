#!/usr/bin/env python3
"""
27_method_validation_final.py
-----------------------------
Final CCM / PCMCI re-run for paper Table 2 on the canonical Scenario A
generator (scripts/22, run22f parameters), replacing the provisional
"earlier generator parameter set" columns.

Design (documented for the Table 2 footnote):
  - Generator: canonical Scenario A, beta = 0.25, per-record fitted
    phi/sigma_eps from scenarioA_final_params.csv, published age errors.
  - Preprocessing: ARC deconfounding (alpha = 0.5, NGRIP millennial proxy),
    no differencing (CCM/PCMCI condition on lags internally).
  - Age-error robustness: CCM uses the test_joint perturbation scheme of
    the Granger column (each pair re-ordered by its joint RMS sigma);
    PCMCI runs on the six records jointly, each re-ordered by its own
    published sigma (test_var analog). 8 realizations x 80 perturbations
    per method.
  - Grid: dt = 0.1 ka decimation of the 10-60 ka grid (500 points), so
    the longest prescribed lag (2.3 ka) is covered by the lag horizons
    below within feasible runtime; the Granger column uses the native
    dt = 0.05 grid (Table 1).
  - CCM: pyEDM 2.5 legacy=True (the new parallel CCM hangs on Windows),
    sanitized SRC/DST columns, E per record via leave-one-out simplex
    skill (_simplex_rho, scripts/19), libSizes "50 100 200 300 400",
    sample=15. Edge SRC->DST detected if cross-map skill SRC:DST
    (predict SRC from DST) exceeds DST:SRC and the 0.3 skill threshold.
  - PCMCI: tigramite ParCorr (analytic significance), 6 variables,
    tau_max = 24, pc_alpha = 0.05, max_conds_dim = 4; edge i->j detected
    if significant at any lag tau in 1..24 (contemporaneous links ignored).

Usage:
  python scripts/27_method_validation_final.py --quick      # smoke
  python scripts/27_method_validation_final.py              # full (8 x 80)
  python scripts/27_method_validation_final.py --plot-only  # redraw S13
Output: outputs/method_validation_final.csv, FigS13_Method_Comparison.
"""

import os
import sys
import json
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

BETA = 0.25
ALPHA = 0.5
DT_CCM = 0.1                 # decimated grid for CCM/PCMCI
TAU_MAX_PCMCI = 24           # covers the longest prescribed lag (2.3 ka)
LIB_SIZES = "50 100 200 300 400"
CCM_THRESHOLD = 0.3

METHOD_COLORS = {'Granger': CARC_COLORS['blue'],
                 'CCM': CARC_COLORS['orange'],
                 'PCMCI': CARC_COLORS['green']}


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


def _simplex_rho(vals, E):
    """Leave-one-out simplex forecast skill at embedding E (scripts/19;
    pyEDM EmbedDimension hangs on Windows, so E selection uses this
    direct KDTree equivalent)."""
    v = np.asarray(vals, float)
    n = len(v)
    if n < E + 20:
        return np.nan
    emb = np.column_stack([v[i: n - E + i] for i in range(E)])
    tgt = v[E:]
    from scipy.spatial import cKDTree
    _, idx = cKDTree(emb).query(emb, k=2)
    return float(np.corrcoef(tgt[idx[:, 1]], tgt)[0, 1])


def _decimate(x, grid, dt_out=DT_CCM):
    """Decimate a dt=0.05 series to dt_out by taking every k-th point."""
    k = int(round(dt_out / S22.DT))
    return grid[::k], x[::k]


def ccm_detect_pair(series, src, dst, E_map):
    """CCM test for the directed pair src -> dst on decimated series.
    Returns True if cross-map skill SRC:DST > DST:SRC and > threshold."""
    from pyEDM import CCM as pyCCM
    g, xs = _decimate(series[src], series['__t__'])
    _, xd = _decimate(series[dst], series['__t__'])
    sub = pd.DataFrame({'Time': g, 'SRC': xs, 'DST': xd})
    try:
        ccm = pyCCM(dataFrame=sub, E=E_map[src], Tp=0, columns='SRC',
                    target='DST', libSizes=LIB_SIZES, sample=15,
                    showPlot=False, legacy=True)
        fwd = float(ccm.iloc[-1]['SRC:DST'])
        rev = float(ccm.iloc[-1]['DST:SRC'])
    except Exception:
        return False
    return fwd > rev and fwd > CCM_THRESHOLD


def pcmci_detect_all(series):
    """One 6-variable PCMCI run on the decimated deconfounded series.
    Returns the set of detected directed edges (lagged links only)."""
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    # dt = 0.1 decimation of the native dt = 0.05 grid (design docstring)
    grid = series['__t__'][::2]
    arr = np.column_stack([np.asarray(series[v], float)[::2]
                           for v in S22.VAR_ORDER])
    df = pp.DataFrame(data=arr, datatime=grid,
                      var_names=list(S22.VAR_ORDER))
    pcmci = PCMCI(dataframe=df, cond_ind_test=ParCorr(significance='analytic'),
                  verbosity=0)
    results = pcmci.run_pcmci(tau_max=TAU_MAX_PCMCI, pc_alpha=0.05,
                              max_conds_dim=4)
    graph = results['graph']
    names = list(S22.VAR_ORDER)
    det = set()
    for j, tgt in enumerate(names):
        for i, src in enumerate(names):
            if i == j:
                continue
            if np.any(graph[i, j, 1:] == '-->'):
                det.add((src, tgt))
    return det


def realization_block(seed, params, n_pert, use_ccm, use_pcmci):
    """One realization. Returns ({pair: robustness} for the requested
    method), where robustness = detections / n_pert."""
    phi_map = params['phi'][BETA]
    beta_map = {v: BETA for v in S22.VAR_ORDER}
    sig_map, age_map = params['sigma_eps'], params['age']
    np.random.seed(seed)
    data, t = S22.gen_scenario(seed, False, phi_map, beta_map, sig_map,
                               age_map, True)
    res = S22.layer2(data, ALPHA)
    res['__t__'] = t
    E_map = {}
    for v in S22.VAR_ORDER:
        try:
            E_map[v] = max(int(np.nanargmax(
                [_simplex_rho(res[v][::2], E) for E in range(1, 7)])) + 1, 2)
        except Exception:
            E_map[v] = 3
    rng = np.random.default_rng(seed)
    counts = {(v1, v2): 0 for v1 in S22.VAR_ORDER
              for v2 in S22.VAR_ORDER if v1 != v2}
    for _ in range(n_pert):
        ps = S22.perturb_draw(res, t, 'test_joint', rng, age_map)
        if use_ccm:
            # test_joint keys ps by (driver, target) tuples with values
            # (target_series, driver_series); rebuild the per-pair view
            # expected by ccm_detect_pair (record-name keys + '__t__').
            for (src, dst) in counts:
                yp, xp = ps[(src, dst)]
                pair_series = {'__t__': t, src: xp, dst: yp}
                if ccm_detect_pair(pair_series, src, dst, E_map):
                    counts[(src, dst)] += 1
        if use_pcmci:
            ps_pcmci = S22.perturb_draw(res, t, 'test_var', rng, age_map)
            ps_pcmci['__t__'] = t
            for e in pcmci_detect_all(ps_pcmci):
                counts[e] += 1
    rob = {p: c / n_pert for p, c in counts.items()}
    return rob, E_map


def _ckpt_path(method):
    return os.path.join(OUTPUT_DIR, f'_ckpt_method_validation_{method}.json')


def _load_ckpt(method):
    rob_by_real = {}
    path = _ckpt_path(method)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as fh:
            for item in json.load(fh):
                rob_by_real[int(item['real'])] = {
                    tuple(k): float(v) for k, v in item['counts']}
    return rob_by_real


def _save_ckpt(method, rob_by_real):
    with open(_ckpt_path(method), 'w', encoding='utf-8') as fh:
        json.dump([{'real': r, 'counts': [[list(k), v] for k, v in c.items()]}
                   for r, c in sorted(rob_by_real.items())], fh)


def run_method(method, params, n_real, n_pert):
    use_ccm = method == 'CCM'
    use_pcmci = method == 'PCMCI'
    # Per-realization checkpointing: realization r uses seed 42 + r and is
    # deterministic, so after a crash every checkpointed block is skipped
    # identically and only the interrupted batch is recomputed. Only the
    # full configuration (8 x 80) checkpoints; --quick results are never
    # persisted, so a smoke run cannot pollute a later full re-run.
    full = (n_real, n_pert) == (8, 80)
    rob_by_real = _load_ckpt(method) if full else {}
    results = [(rob, None) for r in range(n_real)
               if (rob := rob_by_real.get(r)) is not None]
    todo = ([r for r in range(n_real) if r not in rob_by_real]
            if full else list(range(n_real)))
    if todo:
        outs = Parallel(n_jobs=8, verbose=1)(
            delayed(realization_block)(42 + r, params, n_pert, use_ccm,
                                       use_pcmci) for r in todo)
        if full:
            for r, (rob, _) in zip(todo, outs):
                rob_by_real[r] = rob
            _save_ckpt(method, rob_by_real)
            print(f'  [{method}] batch of {len(todo)} realizations '
                  f'checkpointed', flush=True)
        results.extend(outs)
    else:
        print(f'  [{method}] all realizations already checkpointed, '
              f'skipping', flush=True)
    rob_med = {p: [] for p in S22.TRUE_PAIRS + S22.ALL_CTL}
    for rob, _ in results:
        for p in rob_med:
            rob_med[p].append(rob.get(p, 0.0))
    rows = []
    for a, b in S22.TRUE_PAIRS:
        vals = rob_med[(a, b)]
        rows.append({'method': method, 'pair': f'{a} -> {b}',
                     'robustness': float(np.median(vals)),
                     'robustness_se': float(np.std(vals, ddof=1)),
                     'n_real': n_real, 'n_pert': n_pert})
    ctl = {p: float(np.median(rob_med[p])) for p in S22.ALL_CTL}
    rows.append({'method': method, 'pair': 'control_max',
                 'robustness': float(np.max(list(ctl.values()))),
                 'robustness_se': float(np.nan),
                 'n_real': n_real, 'n_pert': n_pert})
    rows.append({'method': method, 'pair': 'control_mean',
                 'robustness': float(np.mean(list(ctl.values()))),
                 'robustness_se': float(np.nan),
                 'n_real': n_real, 'n_pert': n_pert})
    rows.append({'method': method, 'pair': 'control_n_pass95',
                 'robustness': int(np.sum(np.array(list(ctl.values()))
                                          >= 0.95)),
                 'robustness_se': float(np.nan),
                 'n_real': n_real, 'n_pert': n_pert})
    return pd.DataFrame(rows)


def granger_canonical_rows():
    """Canonical Granger column from run22f Table 1 (20 x 200, dt=0.05)."""
    t1 = pd.read_csv(os.path.join(OUTPUT_DIR,
                                  'scenarioA_final_table1.csv'))
    full = pd.read_csv(os.path.join(
        OUTPUT_DIR, 'scenarioA_final_table1_full_beta0.25.csv'))
    ratio = {r['causal_edge']: r['lag_error_ratio']
             for _, r in t1[t1['beta'] == '0.25'].iterrows()
             if not r['causal_edge'].startswith('negative')}
    rows = []
    for _, r in full[full['prescribed_true_edge']].iterrows():
        p = r['robustness_with_error_median']
        rows.append({'method': 'Granger', 'pair': f"{r['source']} -> {r['target']}",
                     'robustness': float(p),
                     'robustness_se': float(np.sqrt(p * (1 - p) / 200)),
                     'n_real': 20, 'n_pert': 200})
    ctl = full[~full['prescribed_true_edge']]['robustness_with_error_median']
    rows.append({'method': 'Granger', 'pair': 'control_max',
                 'robustness': float(ctl.max()),
                 'robustness_se': float(np.sqrt(ctl.max() * (1 - ctl.max())
                                                / 200)),
                 'n_real': 20, 'n_pert': 200})
    rows.append({'method': 'Granger', 'pair': 'control_mean',
                 'robustness': float(ctl.mean()),
                 'robustness_se': float(np.nan), 'n_real': 20, 'n_pert': 200})
    rows.append({'method': 'Granger', 'pair': 'control_n_pass95',
                 'robustness': int((ctl >= 0.95).sum()),
                 'robustness_se': float(np.nan), 'n_real': 20, 'n_pert': 200})
    df = pd.DataFrame(rows)
    df['lag_error_ratio'] = [ratio.get(p, np.nan) for p in df['pair']]
    return df


def plot_from_csv(df):
    apply_style('paper')
    edges = [f'{a} -> {b}' for a, b in S22.TRUE_PAIRS]
    methods = ['Granger', 'CCM', 'PCMCI']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4),
                                   gridspec_kw={'width_ratios': [1.35, 1]})

    x = np.arange(len(edges))
    w = 0.26
    for k, m in enumerate(methods):
        vals = [float(df[(df['method'] == m) & (df['pair'] == e)]
                      .iloc[0]['robustness']) for e in edges]
        ses = [float(df[(df['method'] == m) & (df['pair'] == e)]
                     .iloc[0]['robustness_se']) for e in edges]
        ax1.bar(x + (k - 1) * w, np.array(vals) * 100, width=w,
                yerr=np.array(ses) * 100, capsize=3,
                color=METHOD_COLORS[m], label=f'CARC+{m}',
                error_kw={'elinewidth': 1.0, 'ecolor': '0.3'})
    ax1.axhline(95, color=CARC_COLORS['red'], lw=1.2, ls='--')
    ax1.text(2.62, 96.5, '95% honesty filter', fontsize=8,
             color=CARC_COLORS['red'], ha='right')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['NGRIP $\\to$\nEDC $\\delta$D',
                         'CO$_2$ $\\to$ CH$_4$', 'Cariaco $\\to$\nCheng'],
                        fontsize=8)
    ax1.set_ylabel('Robustness (%)')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.25, lw=0.6, axis='y')
    ax1.legend(fontsize=8, frameon=True)
    ax1.set_title('(a) True-edge robustness (Scenario A, $\\beta$=0.25)',
                  fontsize=10)

    vals = {m: float(df[(df['method'] == m) & (df['pair'] == 'control_max')]
                     .iloc[0]['robustness']) for m in methods}
    ses = {m: float(df[(df['method'] == m) & (df['pair'] == 'control_max')]
                     .iloc[0]['robustness_se']) for m in methods}
    npass = {m: int(df[(df['method'] == m)
                       & (df['pair'] == 'control_n_pass95')].iloc[0]
                     ['robustness']) for m in methods}
    ax2.bar(np.arange(3), [vals[m] * 100 for m in methods],
            yerr=[ses[m] * 100 for m in methods], capsize=3,
            color=[METHOD_COLORS[m] for m in methods],
            error_kw={'elinewidth': 1.0, 'ecolor': '0.3'})
    for k, m in enumerate(methods):
        ax2.text(k, vals[m] * 100 + 1.2, f'{vals[m]*100:.1f}%\n'
                 f'{npass[m]}/27 pass', ha='center', fontsize=8)
    ax2.axhline(5.0, color=CARC_COLORS['grey'], lw=1.0, ls='-.')
    ax2.text(2.85, 6.0, 'nominal 5%', fontsize=8,
             color=CARC_COLORS['grey'], ha='right')
    ax2.set_xticks(np.arange(3))
    ax2.set_xticklabels([f'CARC+\n{m}' for m in methods], fontsize=8)
    ax2.set_ylabel('Robustness of worst control (%)')
    ax2.set_ylim(0, 20)
    ax2.grid(True, alpha=0.25, lw=0.6, axis='y')
    ax2.set_title('(b) Negative controls (27 directed cells)', fontsize=10)

    fig.tight_layout()
    save_figure(fig, 'FigS13_Method_Comparison')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true',
                    help='smoke test: 2 realizations x 10 perturbations')
    ap.add_argument('--plot-only', action='store_true',
                    help='redraw S13 from outputs/method_validation_final.csv')
    args = ap.parse_args()

    csv = os.path.join(OUTPUT_DIR, 'method_validation_final.csv')
    if args.plot_only:
        plot_from_csv(pd.read_csv(csv))
        return

    params = load_params()
    if args.quick:
        n_real, n_pert = 2, 10
    else:
        n_real, n_pert = 8, 80

    rows = [granger_canonical_rows()]
    for m in ('CCM', 'PCMCI'):
        print(f'== {m} ({n_real} realizations x {n_pert} perturbations) ==',
              flush=True)
        rows.append(run_method(m, params, n_real, n_pert))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(csv, index=False)
    print(f'Saved {csv}', flush=True)
    print('\n== Table 2 (final) ==')
    for m in ('Granger', 'CCM', 'PCMCI'):
        sub = df[df['method'] == m]
        e = {r['pair']: r['robustness'] for _, r in sub.iterrows()}
        print(f"  {m:8s}: NGRIP->EDC {e.get('NGRIP -> EDC_dD', np.nan):.3f} | "
              f"CO2->CH4 {e.get('CO2 -> CH4', np.nan):.3f} | "
              f"Cariaco->Cheng {e.get('Cariaco -> Cheng', np.nan):.3f} | "
              f"ctl_max {e.get('control_max', np.nan):.3f} | "
              f"n_pass95 {int(e.get('control_n_pass95', -1))}")

    plot_from_csv(df)


if __name__ == '__main__':
    main()
