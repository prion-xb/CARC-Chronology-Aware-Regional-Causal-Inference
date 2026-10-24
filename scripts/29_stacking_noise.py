#!/usr/bin/env python3
"""
29_stacking_noise.py
--------------------
Layer-1 stacking effectiveness (Fig S11): per-region noise-variance
reduction of the regional composite relative to the median member record,
on the 10-60 ka analysis window (dt = 0.05 ka grid).

  - Members: all records of the region covering 10-60 ka
    (src.io.load_all_records, classify_record grouping of src/stacking.py,
    Southern Ocean merged into the Antarctic/Southern Ocean composite,
    names mapped to the paper's six composites / Table S4).
  - Noise: AR(1) innovation standard deviation of each standardized
    member series (fit_ar1) -- the sigma_eps quantity of the framework's
    effective-SNR theory (Theorem 3).
  - Stack: pointwise median of the standardized members + Gaussian
    smoothing (sigma = 0.5 ka, matching the 10-60 ka stacks of Fig S3 /
    scripts/18) -- the Layer 1 composite; no ARC deconfounding here.
  - Theoretical floor: 1/sqrt(K) for K independent, equal-variance
    records. Spatial correlation and variance heterogeneity reduce the
    realized gain; the gap between the achieved ratio and 1/sqrt(K)
    quantifies those effects.

The 10-60 ka member counts printed below should be compared with the
paper's statement of 2-5 vs 13-25 members across regions; the full
0-800 ka counts (Table S4) are 13/18/34/6/3/6.

Output: outputs/stacking_noise_reduction.csv, FigS11_Stacking_Noise_Reduction.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from src.config import OUTPUT_DIR, DATA_DIR
from src.io import load_all_records
from src.stacking import build_region_dict, classify_record
from src.visualization import apply_style, save_figure, CARC_COLORS

# classify_record region -> paper composite name (Table S4 / Fig S3)
REGION_NAMES = {
    'North_Atlantic': 'North Atlantic',
    'Antarctic': 'Antarctic/Southern Ocean',
    'Southern_Ocean': 'Antarctic/Southern Ocean',
    'Tropical': 'Tropical',
    'East_Asian_Monsoon': 'East Asian Monsoon',
    'South_America': 'South American Monsoon',
    'Other_Lakes': 'High-latitude Lakes',
}

S4_COUNTS = {'North Atlantic': 13, 'Antarctic/Southern Ocean': 18,
             'Tropical': 34, 'East Asian Monsoon': 6,
             'South American Monsoon': 3, 'High-latitude Lakes': 6}


def fit_ar1(x):
    x = (x - x.mean()) / x.std()
    phi = float(np.linalg.lstsq(x[:-1, None], x[1:], rcond=None)[0][0])
    return float((x[1:] - phi * x[:-1]).std())


def main():
    apply_style('paper')

    print('Loading records (10-60 ka, dt=0.05)...')
    records, _ = load_all_records(DATA_DIR, age_min=10.0, age_max=60.0)

    region_dict = {}
    for interp, sigma, region, name in records:
        key = REGION_NAMES.get(region)
        if key is None:
            continue
        region_dict.setdefault(key, []).append((interp, sigma, name))

    rows = []
    for region, recs in sorted(region_dict.items(),
                               key=lambda kv: -len(kv[1])):
        K = len(recs)
        if K < 2:
            continue
        member_noise = np.array([fit_ar1(x) for x, _, _ in recs])
        stack = np.median(np.column_stack(
            [(x - x.mean()) / x.std() for x, _, _ in recs]), axis=1)
        # Gaussian smoothing as in the Fig S3 / scripts/18 10-60 ka stacks
        stack = gaussian_filter1d(stack, sigma=0.5 / 0.05, mode='nearest')
        stack_noise = fit_ar1(stack)
        med = float(np.median(member_noise))
        ratio = stack_noise / med
        theo = 1.0 / np.sqrt(K)
        rms_err = float(np.sqrt(np.mean([
            float(np.median(s)) ** 2 for _, s, _ in recs])))
        rows.append({'region': region, 'n_records_10_60': K,
                     's4_count': S4_COUNTS.get(region, np.nan),
                     'median_member_noise': med,
                     'stack_noise': stack_noise,
                     'achieved_ratio': ratio,
                     'theoretical_1_sqrt_K': theo,
                     'rms_age_error_ka': rms_err})
        print(f'  {region:28s} K={K:2d} (S4={S4_COUNTS.get(region)}) '
              f'member_med={med:.3f} stack={stack_noise:.3f} '
              f'ratio={ratio:.3f} vs 1/sqrt(K)={theo:.3f}')

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, 'stacking_noise_reduction.csv'),
              index=False)

    # --- figure ---
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(df))
    w = 0.36
    ax.bar(x - w / 2, df['achieved_ratio'], width=w,
           color=CARC_COLORS['blue'], label='achieved (stack / median member)')
    ax.bar(x + w / 2, df['theoretical_1_sqrt_K'], width=w,
           color=CARC_COLORS['orange'], alpha=0.85,
           label=r'theoretical floor  $1/\sqrt{K}$')
    for i, r in df.iterrows():
        ax.text(i - w / 2, r['achieved_ratio'] + 0.02, f"{r['achieved_ratio']:.2f}",
                ha='center', fontsize=8, color=CARC_COLORS['blue'])
        ax.text(i + w / 2, r['theoretical_1_sqrt_K'] + 0.02,
                f"{r['theoretical_1_sqrt_K']:.2f}", ha='center', fontsize=8,
                color=CARC_COLORS['orange'])
        ax.text(i, 1.06, f"K={int(r['n_records_10_60'])}", ha='center',
                fontsize=8.5, color='0.25')
    ax.axhline(1.0, color=CARC_COLORS['grey'], lw=0.8, ls=':')
    ax.text(0.02, 1.02, 'no reduction', fontsize=7.5,
            color=CARC_COLORS['grey'], va='bottom')
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace('Antarctic/Southern', 'Antarctic/\nSouthern')
                        .replace('High-latitude', 'High-lat.')
                        .replace('South American', 'S. American')
                        .replace('East Asian', 'E. Asian')
                        for r in df['region']], fontsize=8, rotation=12)
    ax.set_ylabel('Noise-variance ratio (stack / median member)')
    ax.set_ylim(0, 1.12)
    ax.grid(True, alpha=0.25, lw=0.6, axis='y')
    ax.legend(fontsize=8.5, frameon=True, loc='upper right')
    ax.set_title('Layer 1 stacking: AR(1)-innovation noise reduction, '
                 '10-60 ka', fontsize=10)

    fig.tight_layout(pad=0.6)
    save_figure(fig, 'FigS11_Stacking_Noise_Reduction')
    plt.close(fig)
    print('Saved outputs/stacking_noise_reduction.csv')


if __name__ == '__main__':
    main()
