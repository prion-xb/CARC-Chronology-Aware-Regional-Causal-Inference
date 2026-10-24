#!/usr/bin/env python3
"""
18_stacking_visualization.py
----------------------------
Visualize the six regional stacks and compare three stacking schemes on the
same age-perturbation ensemble:

  1. median  — current CARC default (outlier-robust)
  2. mean    — simple average of resampled records (equal weight per record)
  3. lr04    — LR04-faithful: raw (non-resampled) measurements pooled into
               1-kyr bins; higher-resolution records contribute more points
               per bin, so resolution weighting is emergent (the paper's
               own wording), not an explicit per-record weight
  4. weighted — inverse-variance mean, weights = 1 / sigma^2 per record

For each region and each scheme, the perturbation ensemble is reduced to a
median curve (bold) with a P5-P95 band. Member records are drawn as thin gray
lines. A comparison panel reports cross-scheme RMSE and Pearson r.

Usage:
  python scripts/18_stacking_visualization.py [10-60|0-800]
Defaults to 10-60 ka.

Outputs:
  figures/FigS_Stacking_<window>.png/.pdf
  outputs/stacking_scheme_comparison_<window>.csv
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import (DATA_DIR, OUTPUT_DIR, FIGURE_DIR, N_PERTURB,
                        AGE_GRID_10_60, AGE_GRID_800)
from src.io import load_all_records
from src.stacking import build_region_dict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAPER_REGIONS = ['North_Atlantic', 'Antarctic', 'Tropical',
                 'East_Asian_Monsoon', 'South_America', 'Other_Lakes']
REGION_LABELS = {
    'North_Atlantic': 'North Atlantic',
    'Antarctic': 'Antarctic/SO',
    'Tropical': 'Tropical',
    'East_Asian_Monsoon': 'E. Asian Monsoon',
    'South_America': 'S. Am. Monsoon',
    'Other_Lakes': 'High-Lat Lakes',
}
REGION_COLORS = {
    'North_Atlantic': '#D73027', 'Antarctic': '#2166AC', 'Tropical': '#E66101',
    'East_Asian_Monsoon': '#1B7837', 'South_America': '#762A83',
    'Other_Lakes': '#008080',
}

INK, INK_SEC, INK_MUTED = '#0b0b0b', '#52514e', '#898781'
SURFACE, GRID = '#fcfcfb', '#c3c2b7'
MEMBER = '#b9b8b0'          # thin member lines
SCHEME_STYLE = {            # median / mean / lr04 / weighted
    'median':   dict(color='#2a78d6', lw=2.2, ls='-',  zorder=6),
    'mean':     dict(color='#eb6834', lw=1.6, ls='--', zorder=5),
    'lr04':     dict(color='#eb6834', lw=1.6, ls='-',  zorder=4),
    'weighted': dict(color='#1baf7a', lw=1.6, ls=':',  zorder=3),
}


# ---------------------------------------------------------------------------
# Stacking schemes (shared perturbation stream for comparability)
# ---------------------------------------------------------------------------
def stack_schemes(region_records, grid, n_pert, smooth_sigma, rng,
                  raw_points=None, bin_width=1.0):
    """Return {scheme: np.ndarray (n_pert, len(grid))} for one region.

    All schemes share the same age perturbations per record so differences
    between schemes are attributable to the aggregation rule only.

    'lr04' pools raw measurements into bin_width bins per draw: each
    record's measurement ages are perturbed, all points are pooled and
    bin-averaged. A high-resolution record contributes more points per bin,
    so it is implicitly weighted more — no explicit per-record weight is
    ever applied (this is exactly the mechanism behind LR04's sentence
    "sites with higher resolution are more heavily weighted").
    """
    dt = grid[1] - grid[0]
    out = {s: np.zeros((n_pert, len(grid)))
           for s in ('median', 'mean', 'weighted', 'lr04')}

    sigmas = np.array([s for _, s, _ in region_records])
    w = (1.0 / sigmas ** 2)
    w = w / w.sum()

    # raw measurement points, standardized per record (as LR04 normalizes
    # each record to zero mean / unit variance before stacking)
    grid0, grid1 = grid[0], grid[-1]
    n_bins = int(np.ceil((grid1 - grid0) / bin_width)) + 1
    pts = []
    for interp, sigma, name in region_records:
        age, val = (raw_points or {}).get(name, (None, None))
        if age is None or len(age) < 10:
            age, val = grid, interp          # fallback: uniform-grid points
        m = (age >= grid0) & (age <= grid1) & np.isfinite(age) & np.isfinite(val)
        age, val = age[m].astype(float), val[m].astype(float)
        val = (val - val.mean()) / val.std()   # LR04-style normalization
        pts.append((age, val, sigma))

    for pert in range(n_pert):
        signals = []
        for interp, sigma, _ in region_records:
            age_pert = grid + rng.normal(0, sigma, len(grid))
            signals.append(interp[np.argsort(age_pert)])
        S = np.vstack(signals)
        out['median'][pert] = np.median(S, axis=0)
        out['mean'][pert] = S.mean(axis=0)
        out['weighted'][pert] = w @ S

        # LR04 pooled-points draw: perturb each measurement age, pool into
        # bins, average per bin, map back onto the fine grid
        sums = np.zeros(n_bins)
        cnt = np.zeros(n_bins)
        for age, val, sigma in pts:
            ap = age + rng.normal(0, sigma, len(age))
            b = np.clip(((ap - grid0) / bin_width).astype(int), 0, n_bins - 1)
            sums += np.bincount(b, weights=val, minlength=n_bins)
            cnt += np.bincount(b, minlength=n_bins)
        bins = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan)
        centers = grid0 + (np.arange(n_bins) + 0.5) * bin_width
        ok = np.isfinite(bins)
        out['lr04'][pert] = (np.interp(grid, centers[ok], bins[ok])
                             if ok.sum() else np.zeros_like(grid))

    from scipy.ndimage import gaussian_filter1d
    for scheme, arr in out.items():
        for i in range(n_pert):
            arr[i] = gaussian_filter1d(arr[i], sigma=smooth_sigma / dt,
                                       mode='nearest')
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def main():
    window = sys.argv[1] if len(sys.argv) > 1 else '10-60'
    if window == '10-60':
        grid, smooth = AGE_GRID_10_60, 0.5
        age_min, age_max = 10.0, 60.0
    else:
        grid, smooth = AGE_GRID_800, 5.0
        age_min, age_max = 0.0, 800.0

    print(f"Loading records ({window} ka)...")
    records, _ = load_all_records(DATA_DIR, age_min=age_min, age_max=age_max)
    records = [(i, s, 'Antarctic' if r == 'Southern_Ocean' else r, n)
               for i, s, r, n in records]
    region_dict = build_region_dict(records)
    region_dict = {r: v for r, v in region_dict.items() if r in PAPER_REGIONS}

    rng = np.random.default_rng(42)
    from src.io import load_record
    raw_points = {}
    for recs in region_dict.values():
        for _, _, name in recs:
            if name not in raw_points:
                raw_points[name] = load_record(name)

    stacks, comparison_rows = {}, []
    for region, recs in region_dict.items():
        stacks[region] = stack_schemes(recs, grid, N_PERTURB, smooth, rng,
                                       raw_points=raw_points)
        # cross-scheme comparison on median curves
        meds = {s: np.median(arr, axis=0)
                for s, arr in stacks[region].items()}
        schemes = ['median', 'mean', 'lr04', 'weighted']
        for i, s1 in enumerate(schemes):
            for s2 in schemes[i + 1:]:
                comparison_rows.append({
                    'region': region,
                    'scheme_A': s1, 'scheme_B': s2,
                    'rmse': float(np.sqrt(np.mean((meds[s1] - meds[s2]) ** 2))),
                    'pearson_r': float(np.corrcoef(meds[s1], meds[s2])[0, 1]),
                    'n_records': len(recs),
                })
        # effective number of records under 1/sigma^2 weighting
        w = 1.0 / np.array([s for _, s, _ in recs]) ** 2
        w = w / w.sum()
        comparison_rows.append({
            'region': region, 'scheme_A': 'weighted', 'scheme_B': '(eff N)',
            'rmse': float(1.0 / np.sum(w ** 2)),
            'pearson_r': np.nan, 'n_records': len(recs),
        })
        print(f"  {region}: {len(recs)} records")

    comp_df = pd.DataFrame(comparison_rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    comp_csv = os.path.join(OUTPUT_DIR, f'stacking_scheme_comparison_{window}.csv')
    comp_df.to_csv(comp_csv, index=False, float_format='%.4f')
    print(f"Saved: {comp_csv}")
    print(comp_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Figure: 6 region panels + summary column
    # ------------------------------------------------------------------
    plt.rcParams.update({'font.size': 8.5, 'font.family': 'Arial',
                         'axes.edgecolor': GRID, 'text.color': INK})
    fig = plt.figure(figsize=(11.5, 8.6), facecolor=SURFACE)
    # top=0.898: panel titles render ABOVE their axes top edge (baseline at
    # top+pad), so the top row must clear the subtitle line at y≈0.933
    gs = fig.add_gridspec(2, 3, wspace=0.16, hspace=0.30,
                          left=0.065, right=0.955, bottom=0.088, top=0.898)

    for k, region in enumerate(PAPER_REGIONS):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        ax.set_facecolor(SURFACE)
        recs = region_dict[region]
        # member records (thin gray); records have variable lengths, so
        # resample each onto the panel grid before plotting
        for interp, sigma, name in recs:
            idx = np.round(np.linspace(0, len(interp) - 1, len(grid))).astype(int)
            ax.plot(grid, interp[idx], color=MEMBER, lw=0.4, alpha=0.7,
                    zorder=1)
        # schemes: median curve + P5-P95 band for the median stack
        med = np.median(stacks[region]['median'], axis=0)
        p5, p95 = np.percentile(stacks[region]['median'], [5, 95], axis=0)
        ax.fill_between(grid, p5, p95, color='#86b6ef', alpha=0.25,
                        zorder=2, linewidth=0)
        for scheme, st in SCHEME_STYLE.items():
            ax.plot(grid, np.median(stacks[region][scheme], axis=0),
                    label=scheme, **st)
        ax.set_title(f"{REGION_LABELS[region]}  (n = {len(recs)})",
                     fontsize=9.5, color=INK, fontweight='bold', pad=4)
        ax.set_xlim(grid[0], grid[-1])
        ax.tick_params(colors=INK_MUTED, labelsize=7)
        ax.grid(True, color='#e1e0d9', lw=0.5, alpha=0.6)
        if k // 3 == 1:
            ax.set_xlabel('Age (ka)', fontsize=8, color=INK_SEC)
        if k % 3 == 0:
            ax.set_ylabel('standardized value', fontsize=8, color=INK_SEC)

    # legend (drawn once, on the figure)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([], [], color=MEMBER, lw=0.8, label='member records'),
        Patch(facecolor='#86b6ef', alpha=0.35, label='median stack P5–P95'),
        Line2D([], [], color='#2a78d6', lw=2.2, label='median stack'),
        Line2D([], [], color='#eb6834', lw=1.6, ls='--', label='mean stack (equal records)'),
        Line2D([], [], color='#eb6834', lw=1.6, ls='-', label='LR04 pooled points (1-kyr bins)'),
        Line2D([], [], color='#1baf7a', lw=1.6, ls=':', label='weighted stack (1/σ²)'),
    ]
    fig.legend(handles=handles, loc='upper right', frameon=False, fontsize=8,
               bbox_to_anchor=(0.965, 0.955), handlelength=2.0)

    # title split in two short lines: one long line at fontsize 12 would run
    # past the right canvas edge (layout QA: text_outside)
    fig.text(0.065, 0.962,
             'Regional stacking: median vs mean vs LR04 pooled vs 1/σ²-weighted',
             fontsize=12, fontweight='bold', color=INK)
    fig.text(0.065, 0.930,
             f'({window} ka, {N_PERTURB}-member age-perturbation ensemble)',
             fontsize=9, color=INK_SEC)

    # summary strip: cross-scheme RMSE per region (compact notation so the
    # six slots never overlap: m–m = median vs mean, L = LR04, w = weighted)
    ax_sum = fig.add_axes([0.065, 0.004, 0.89, 0.042])
    ax_sum.set_facecolor(SURFACE)
    ax_sum.set_xlim(-1.6, 5.5)
    ax_sum.set_ylim(0, 1)
    ax_sum.axis('off')
    ax_sum.text(-1.55, 0.55, 'RMSE vs median:',
                ha='left', va='bottom', fontsize=6.8, color=INK_MUTED)
    ax_sum.text(-1.55, 0.05, 'm–m mean · L LR04 · w 1/σ²',
                ha='left', va='bottom', fontsize=6.8, color=INK_MUTED)
    for k, region in enumerate(PAPER_REGIONS):
        sub = comp_df[comp_df['region'] == region]
        med_vs_mean = sub[(sub['scheme_A'] == 'median') &
                          (sub['scheme_B'] == 'mean')]['rmse'].values
        med_vs_lr04 = sub[(sub['scheme_A'] == 'median') &
                          (sub['scheme_B'] == 'lr04')]['rmse'].values
        med_vs_w = sub[(sub['scheme_A'] == 'median') &
                       (sub['scheme_B'] == 'weighted')]['rmse'].values
        r1 = med_vs_mean[0] if len(med_vs_mean) else np.nan
        r3 = med_vs_lr04[0] if len(med_vs_lr04) else np.nan
        r2 = med_vs_w[0] if len(med_vs_w) else np.nan
        ax_sum.text(k, 0.55, f'{REGION_LABELS[region]}',
                    ha='center', va='bottom', fontsize=7, color=INK_SEC)
        ax_sum.text(k, 0.05,
                    f'm–m {r1:.2f} · L {r3:.2f} · w {r2:.2f}',
                    ha='center', va='bottom', fontsize=7, color=INK_MUTED)

    os.makedirs(FIGURE_DIR, exist_ok=True)
    for ext in ('png', 'pdf'):
        path = os.path.join(FIGURE_DIR, f'FigS_Stacking_{window}.{ext}')
        fig.savefig(path, dpi=300, facecolor=SURFACE)
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
