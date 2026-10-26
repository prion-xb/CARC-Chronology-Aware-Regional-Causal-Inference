#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
39_table2_zzf.py
----------------
Table 2 rebuild: Scenario A with CCM and PCMCI as alternative Layer-3
causal testers (paper Section 4.1, Table 2).

Paper Table 2 (robustness %, current measured values; the CCM/PCMCI
columns are reproduced by this script, the Granger column by the Table 1
driver scripts/22_scenarioA_final.py):
  edge             lambda   Granger   CCM     PCMCI
  NGRIP->EDC_dD    ~1.2      6.0      4.0     41.0
  CO2->CH4         ~0.9      6.5      0.5     42.5
  Cariaco->Cheng   ~10.0     4.5      0.0     97.5
  (controls: CCM <= 8.5%, PCMCI <= 98.0%)

The supplementary's Appendix A description of the CCM/PCMCI settings was
lost with the implementation (the dump contains no CCM/PCMCI section), so
the decision rules below are documented rebuild choices:

  * data: scripts/34 Scenario A generator + Layer 2 (keep, alpha=0.5),
    per-draw perturbation (paper 3.4 order), full span 10-58 ka;
  * CCM: numpy implementation validated against pyEDM 2.5 (skills agree
    to ~0.03 on a test pair; pyEDM itself is 12 s/call and unusable at
    200 draws x 30 pairs). E = 3 fixed (pyEDM's EmbedDimension hangs on
    this Windows build), tau = 1, libSizes [150, 450], E+1 distance-
    weighted simplex projection, self-neighbor exclusion = 1.
    Direction drv->tgt significant iff
        rho(drv | M_tgt) > 0.1                     (cross-map skill)
    and rho(drv | M_tgt) - rho(tgt | M_drv) > 0.1  (direction asymmetry)
    and rho(drv | M_tgt) rises with libSize        (convergence signature)
  * PCMCI: tigramite 5.x ParCorr, one system-wide run per draw
    (6 variables), tau_max = 5, pc_alpha = 0.05; drv->tgt significant iff
    min over tau of p_matrix[tgt, drv, tau] < 0.05
    (pc_alpha is a per-condition independence level inside PCMCI,
    not FDR control).

Robustness = fraction of the 200 draws with a significant edge; reported
as the median over 5 realizations (runtime constraint; documented).

Outputs:
  outputs/table2_zzf_ccm_pcmci.csv      per-edge robustness x tester
  outputs/table2_zzf_ccm_pcmci_full.csv all 30 directed cells

Usage:
  python scripts/39_table2_zzf.py --quick   # 1 realization x 20 draws
  python scripts/39_table2_zzf.py           # full run (5 x 200)
"""

import os, sys, argparse, warnings, importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.config import OUTPUT_DIR

_spec = importlib.util.spec_from_file_location(
    'reg34', os.path.join(os.path.dirname(__file__), '34_scenarioA_zzf.py'))
reg34 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reg34)

E_CCM = 3
LIBS_CCM = [150, 450]
SKILL_MIN = 0.1
ASYMM_MIN = 0.1
TAU_MAX = 5
PC_ALPHA = 0.05

TRUE_EDGES = [(a, b) for a, b, _ in reg34.TRUE_EDGES]
ALL_CTL = [(v1, v2) for v1 in reg34.VAR_ORDER for v2 in reg34.VAR_ORDER
           if v1 != v2 and (v1, v2) not in TRUE_EDGES]

# tigramite imports (5.x: ParCorr lives in .parcorr)
from tigramite import data_processing as tpp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI


# ---------------------------------------------------------------------------
# CCM (numpy; validated against pyEDM 2.5 -- skills agree within ~0.03)
# ---------------------------------------------------------------------------
def ccm_skill(x, y, lib_sizes, rng, E=E_CCM, tau=1, exclusion=1):
    """rho(x_hat | M_y) per libSize: cross-map skill of x from y's
    time-delay embedding (simplex projection, distance-weighted)."""
    n = len(x)
    m = n - (E - 1) * tau
    if m < 30:
        return [np.nan] * len(lib_sizes)
    if E == 1:
        emb = y[:m, None]
    else:
        emb = np.column_stack(
            [y[(E - 1 - j) * tau:(E - 1 - j) * tau + m] for j in range(E)])
    t_arr = np.arange(m)
    out = []
    for L in lib_sizes:
        lib_idx = np.sort(rng.choice(m, size=min(L, m), replace=False))
        tree = cKDTree(emb[lib_idx])
        d, nn = tree.query(emb, k=E + 3)
        j_all = lib_idx[nn[:, 1:]]              # (m, k) neighbor time indices
        d_all = d[:, 1:]
        valid = np.abs(j_all - t_arr[:, None]) > exclusion
        cum = np.cumsum(valid, axis=1)
        keep = (cum <= E + 1) & valid
        cnt = keep.sum(1)
        d0 = np.maximum(d_all[:, :1], 1e-12)
        w = np.exp(-d_all / d0)
        w = np.where(keep, w, 0.0)
        wsum = w.sum(1, keepdims=True)
        w = np.divide(w, wsum, out=np.zeros_like(w), where=wsum > 0)
        xj = np.where(j_all >= 0, x[np.clip(j_all, 0, n - 1)], 0.0)
        pred = (w * xj).sum(1)
        msk = (cnt >= E + 1) & np.isfinite(pred)
        if msk.sum() < 30:
            out.append(np.nan)
        else:
            out.append(float(np.corrcoef(x[:m][msk], pred[msk])[0, 1]))
    return out


def ccm_significant(x_drv, x_tgt, rng):
    """CCM decision for drv -> tgt (see module docstring)."""
    s_drv = ccm_skill(x_drv, x_tgt, LIBS_CCM, rng)   # rho(drv | M_tgt)
    s_tgt = ccm_skill(x_tgt, x_drv, LIBS_CCM, rng)   # rho(tgt | M_drv)
    if np.isnan(s_drv).any() or np.isnan(s_tgt).any():
        return False
    return (s_drv[-1] > SKILL_MIN
            and s_drv[-1] - s_tgt[-1] > ASYMM_MIN
            and s_drv[-1] - s_drv[0] > 0.0)


def pcmci_links(residuals):
    """One system-wide PCMCI run; returns set of significant (drv, tgt)."""
    arr = np.column_stack([residuals[v] for v in reg34.VAR_ORDER])
    try:
        dframe = tpp.DataFrame(data=arr, var_names=list(reg34.VAR_ORDER))
        pcmci = PCMCI(dataframe=dframe, cond_ind_test=ParCorr())
        out = pcmci.run_pcmci(tau_max=TAU_MAX, pc_alpha=PC_ALPHA)
    except Exception:
        return set()
    p = out['p_matrix']            # p[tgt, drv, tau]
    links = set()
    for ti, tgt in enumerate(reg34.VAR_ORDER):
        for di, drv in enumerate(reg34.VAR_ORDER):
            if drv == tgt:
                continue
            if np.nanmin(p[ti, di, :]) < 0.05:
                links.add((drv, tgt))
    return links


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def run_realization(seed, n_perturb, sigma_map, span, tester):
    counts = {(v1, v2): 0 for v1 in reg34.VAR_ORDER for v2 in reg34.VAR_ORDER
              if v1 != v2}
    data, t, Z = reg34.generate_zzf(seed, False, span=span)
    rng = np.random.default_rng(seed * 1000 + 7)
    for _ in range(n_perturb):
        raw = {v: data[v][np.argsort(t + rng.normal(0, sigma_map[v], len(t)))]
               for v in reg34.VAR_ORDER}
        ps = reg34.layer2(raw, mode='keep', alpha=reg34.ARC_ALPHA, Z=Z)
        if tester == 'CCM':
            for drv, tgt in counts:
                if ccm_significant(ps[drv], ps[tgt], rng):
                    counts[(drv, tgt)] += 1
        else:
            links = pcmci_links(ps)
            for p in links:
                counts[p] += 1
    return {p: c / n_perturb for p, c in counts.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--quick', action='store_true',
                    help='smoke test: 1 realization x 20 draws')
    ap.add_argument('--realizations', type=int, default=5)
    ap.add_argument('--perturb', type=int, default=200)
    ap.add_argument('--span', type=float, default=58.0)
    ap.add_argument('--seed-base', type=int, default=42)
    args = ap.parse_args()

    if args.quick:
        n_real, n_pert = 1, 20
    else:
        n_real, n_pert = args.realizations, args.perturb

    sigma_map = dict(reg34.S1_PARAMS['age'])
    print(f"Table 2 rebuild: span={args.span} {n_real} realizations x "
          f"{n_pert} draws, testers = CCM + PCMCI", flush=True)

    all_rows = []
    for tester in ['CCM', 'PCMCI']:
        acc = {p: [] for p in TRUE_EDGES + ALL_CTL}
        for r in range(n_real):
            rob = run_realization(args.seed_base + r, n_pert, sigma_map,
                                  args.span, tester)
            for p, v in rob.items():
                acc[p].append(v)
            print(f"  {tester}: realization {r + 1}/{n_real} done",
                  flush=True)
        for drv, tgt, lag in reg34.TRUE_EDGES:
            med = float(np.median(acc[(drv, tgt)]))
            all_rows.append({
                'tester': tester, 'causal_edge': f'{drv} -> {tgt}',
                'true_lag_ka': lag,
                'lag_error_ratio': round(lag / sigma_map[drv], 1),
                'robustness': med,
                'passes_95': med >= 0.95,
            })
        ctl = [float(np.median(acc[p])) for p in ALL_CTL]
        all_rows.append({
            'tester': tester, 'causal_edge': '12 negative controls',
            'true_lag_ka': np.nan, 'lag_error_ratio': np.nan,
            'robustness': max(ctl),
            'robustness_min': min(ctl),
            'robustness_median': float(np.median(ctl)),
            'passes_95': max(ctl) >= 0.95,
        })

    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(OUTPUT_DIR, 'table2_zzf_ccm_pcmci.csv')
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}\n")
    print(df.to_string(index=False))
    print("\nPaper Table 2 (measured): Granger 6.0/6.5/4.5, "
          "CCM 4.0/0.5/0.0, PCMCI 41.0/42.5/97.5; "
          "controls CCM <= 8.5%, PCMCI <= 98.0%.")


if __name__ == '__main__':
    main()
