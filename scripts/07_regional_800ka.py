#!/usr/bin/env python3
"""
07_regional_800ka.py
--------------------
Regional stacking and DDR (Directional Dominance Ratio) analysis for
the full 0–800 ka interval, split into early (0–400 ka) and late
(400–800 ka) halves.

Requires the full proxy database (set CARC_DATA_DIR env var or place files
in proxy_records/).
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
import warnings
warnings.filterwarnings('ignore')

from src.config import (DATA_DIR, OUTPUT_DIR, DT_LOW_RES, AGE_GRID_800,
                         N_PERTURB, ARC_ALPHA, MAX_GRANGER_LAG)
from src.io import load_all_records, standardize
from src.arc import arc_deconfound
from src.stacking import build_region_dict, stack_region

# ---------------------------------------------------------------------------
# ARC on stack (uses library arc_deconfound)
# ---------------------------------------------------------------------------
def apply_arc_to_stacks(stack, proxy_stack, alpha=0.5):
    """Apply ARC deconfounding to a regional stack (perturbation ensemble)."""
    n_pert, t_len = stack.shape
    residual = np.zeros_like(stack)
    for pert in range(n_pert):
        residual[pert] = arc_deconfound(stack[pert], proxy_stack[pert], alpha)
    return residual


# ---------------------------------------------------------------------------
# DDR analysis
# ---------------------------------------------------------------------------
def save_ddr_results(candidates, out_path):
    """Save DDR edge results to CSV for reuse by figure scripts."""
    rows = []
    for source, target, forward_rate, reverse_rate, ddr in candidates:
        rows.append({
            'source': source,
            'target': target,
            'forward_rate': forward_rate,
            'reverse_rate': reverse_rate,
            'ddr': ddr,
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nDDR results saved to {out_path} ({len(rows)} edges)")


def run_ddr(stacks, n_pert, label):
    regions = list(stacks.keys())
    results = {}
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            r1, r2 = regions[i], regions[j]
            c12, c21 = 0, 0
            s1, s2 = stacks[r1], stacks[r2]
            for pert in range(n_pert):
                df = pd.DataFrame({'A': s1[pert], 'B': s2[pert]})
                df_diff = df.diff().dropna()
                if len(df_diff) < 10: continue
                try:
                    model = VAR(df_diff)
                    lag_order = model.select_order(maxlags=MAX_GRANGER_LAG)
                    opt_lag = lag_order.aic
                    if opt_lag is None or opt_lag < 1: continue
                except:
                    continue
                try:
                    res = grangercausalitytests(df_diff[['B', 'A']], maxlag=opt_lag, verbose=False)
                    p_vals = [res[i + 1][0]['ssr_chi2test'][1] for i in range(opt_lag)]
                    if min(p_vals) < 0.05: c12 += 1
                except: pass
                try:
                    res = grangercausalitytests(df_diff[['A', 'B']], maxlag=opt_lag, verbose=False)
                    p_vals = [res[i + 1][0]['ssr_chi2test'][1] for i in range(opt_lag)]
                    if min(p_vals) < 0.05: c21 += 1
                except: pass
            results[(r1, r2)] = (c12, c21)

    MIN_DDR, MIN_DOM = 1.5, 0.8
    cand = []
    for (r1, r2), (c12, c21) in results.items():
        rate12, rate21 = c12 / n_pert, c21 / n_pert
        ddr12 = rate12 / rate21 if rate21 > 0 else np.inf
        ddr21 = rate21 / rate12 if rate12 > 0 else np.inf
        if ddr12 >= MIN_DDR and rate12 >= MIN_DOM:
            cand.append((r1, r2, rate12, rate21, ddr12))
        if ddr21 >= MIN_DDR and rate21 >= MIN_DOM:
            cand.append((r2, r1, rate21, rate12, ddr21))

    print(f"\n[{label}] Directional edges (DDR≥{MIN_DDR}, rate≥{MIN_DOM:.0%}):")
    if not cand:
        print("  None")
    else:
        cand.sort(key=lambda x: x[4], reverse=True)
        for s, t, rd, rr, ddr in cand:
            print(f"  {s} → {t}: rate={rd:.1%}, rev={rr:.1%}, DDR={ddr:.2f}")
    return cand


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    GRID = AGE_GRID_800
    DT = DT_LOW_RES

    print("Loading records (0-800 ka)...")
    records, _ = load_all_records(DATA_DIR, age_min=0.0, age_max=800.0)

    # Merge Southern Ocean records into the Antarctic composite,
    # matching the paper's six composites ("Antarctic/Southern Ocean").
    records = [(i, s, 'Antarctic' if r == 'Southern_Ocean' else r, n)
               for i, s, r, n in records]

    # Re-interpolate to DT=1.0 grid
    from scipy import interpolate
    aligned = []
    for interp_old, sigma, region, name in records:
        old_grid = np.arange(0, 800.01, 0.05)  # original DT_HIGH_RES grid
        if len(interp_old) == len(old_grid):
            f = interpolate.interp1d(old_grid, interp_old, kind='linear',
                                     bounds_error=False, fill_value='extrapolate')
            new_interp = f(GRID)
            new_interp = np.nan_to_num(new_interp, nan=np.nanmean(new_interp))
            if np.std(new_interp) > 1e-6:
                aligned.append((standardize(new_interp), sigma, region, name))

    print(f"Loaded {len(aligned)} records.")

    # Build regions
    region_dict = build_region_dict(aligned)
    print("\nRegions for stacking:")
    for r, v in region_dict.items():
        print(f"  {r}: {len(v)} records")

    # Stack
    rng = np.random.default_rng(42)
    region_stacks = {}
    for region, recs in region_dict.items():
        region_stacks[region] = stack_region(recs, GRID, N_PERTURB, rng, smooth_sigma=5.0)
        print(f"  {region} stack: {region_stacks[region].shape}")

    # ARC
    proxy_region = 'Antarctic' if 'Antarctic' in region_stacks else list(region_stacks.keys())[0]
    print(f"\nARC proxy region: {proxy_region}")
    proxy_series = region_stacks[proxy_region]

    arc_stacks = {}
    for region, stack in region_stacks.items():
        if region == proxy_region:
            arc_stacks[region] = stack.copy()
        else:
            arc_stacks[region] = apply_arc_to_stacks(stack, proxy_series, alpha=ARC_ALPHA)
        for i in range(arc_stacks[region].shape[0]):
            arc_stacks[region][i] = standardize(arc_stacks[region][i])

    # DDR: early vs late
    half = len(GRID) // 2
    early_stacks = {r: arc_stacks[r][:, :half] for r in arc_stacks}
    late_stacks = {r: arc_stacks[r][:, half:] for r in arc_stacks}

    print("\n" + "=" * 60)
    print("DDR Analysis: 0–800 ka, split early/late")
    print("=" * 60)
    early_cand = run_ddr(early_stacks, N_PERTURB, "0-400 ka")
    late_cand = run_ddr(late_stacks, N_PERTURB, "400-800 ka")

    # Save combined DDR results for figure scripts
    save_ddr_results(early_cand + late_cand,
                     os.path.join(OUTPUT_DIR, 'ddr_regional.csv'))

    # Window comparison
    print("\nWindow comparison:")
    early_dict = {(s, t): ddr for s, t, _, _, ddr in early_cand}
    late_dict = {(s, t): ddr for s, t, _, _, ddr in late_cand}
    all_pairs = set(early_dict.keys()) | set(late_dict.keys())
    for pair in all_pairs:
        e_ddr = early_dict.get(pair, 0)
        l_ddr = late_dict.get(pair, 0)
        if e_ddr > 0 and l_ddr > 0:
            print(f"  {pair[0]} → {pair[1]}: early DDR={e_ddr:.2f}, late DDR={l_ddr:.2f}")
        elif e_ddr > 0:
            print(f"  {pair[0]} → {pair[1]}: early only (DDR={e_ddr:.2f})")
        elif l_ddr > 0:
            print(f"  {pair[0]} → {pair[1]}: late only (DDR={l_ddr:.2f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
