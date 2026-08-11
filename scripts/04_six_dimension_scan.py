#!/usr/bin/env python3
"""
04_six_dimension_scan.py
------------------------
6-dimension parameter sweep: common-forcing strength (β) × age-error scale ×
N_repeats, comparing Raw Granger vs CARC Granger TPR/FPR.

Generates: figures/violin_TPR_multipanel.png, figures/violin_FPR_multipanel.png
           outputs/continuous_robustness_report.csv
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings('ignore')

from src.config import FIGURE_DIR, OUTPUT_DIR
from src.arc import arc_deconfound

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_TIME, N_REPEATS, N_PERTURB = 500, 20, 100
PHI, SIGMA_Z, SIGMA_EPS = 0.8, 1.0, 0.4
CAUSAL_STRENGTH, TRUE_LAG = 0.05, 2
BETAS = [0.0, 0.5, 1.0, 1.5, 2.0]
AGE_SCALES = [0.0, 1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Data generation & test functions
# ---------------------------------------------------------------------------
def generate_data_with_errors(n_time, phi, beta, sigma_z, sigma_eps,
                              true_lag, causal_strength, age_error_std, seed):
    np.random.seed(seed)
    z = np.zeros(n_time)
    z[0] = np.random.normal(0, sigma_z)
    for t in range(1, n_time):
        z[t] = phi * z[t - 1] + np.random.normal(0, sigma_z * np.sqrt(1 - phi ** 2))

    eps_x = np.random.normal(0, sigma_eps, n_time)
    eps_y = np.random.normal(0, sigma_eps, n_time)
    X = np.zeros(n_time); Y = np.zeros(n_time)
    X[0] = np.random.normal(0, 1); Y[0] = np.random.normal(0, 1)

    for t in range(1, n_time):
        X[t] = phi * X[t - 1] + beta * z[t] + eps_x[t]
        causal = causal_strength * X[t - true_lag] if (true_lag is not None and t > true_lag) else 0.0
        Y[t] = phi * Y[t - 1] + beta * z[t] + causal + eps_y[t]

    if age_error_std > 0:
        orig_time = np.arange(n_time)
        perturbed = orig_time + np.random.normal(0, age_error_std, size=n_time)
        idx = np.argsort(perturbed)
        fy = interp1d(perturbed[idx], Y[idx], kind='linear', fill_value='extrapolate', bounds_error=False)
        fx = interp1d(perturbed[idx], X[idx], kind='linear', fill_value='extrapolate', bounds_error=False)
        Y, X = fy(orig_time), fx(orig_time)
        valid = ~(np.isnan(Y) | np.isnan(X))
        X, Y = X[valid], Y[valid]
    return pd.Series(X), pd.Series(Y), pd.Series(z)


def granger_test_raw(Y, X, maxlag=3):
    df = pd.concat([Y, X], axis=1).dropna()
    df.columns = ['Y', 'X']
    if len(df) < maxlag + 3: return np.nan
    try:
        res = grangercausalitytests(df, maxlag, verbose=False)
        return min(res[lag][0]['ssr_ftest'][1] for lag in range(1, maxlag + 1))
    except:
        return np.nan


def carc_robustness(Y, X, proxy, n_perturb=100, error_std=2.0, noise_std=0.05, alpha=0.5):
    Y_clean = arc_deconfound(Y.values, proxy.values, alpha)
    X_clean = arc_deconfound(X.values, proxy.values, alpha)
    sig = 0
    for _ in range(n_perturb):
        n = len(Y_clean)
        orig = np.arange(n)
        perturbed = orig + np.random.normal(0, error_std, size=n)
        idx = np.argsort(perturbed)
        fy = interp1d(perturbed[idx], Y_clean[idx], kind='linear', fill_value='extrapolate', bounds_error=False)
        fx = interp1d(perturbed[idx], X_clean[idx], kind='linear', fill_value='extrapolate', bounds_error=False)
        Yp, Xp = fy(orig), fx(orig)
        valid = ~(np.isnan(Yp) | np.isnan(Xp))
        if valid.sum() < 10: continue
        Yp = Yp[valid] + np.random.normal(0, noise_std * np.std(Yp[valid]), size=valid.sum())
        Xp = Xp[valid] + np.random.normal(0, noise_std * np.std(Xp[valid]), size=valid.sum())
        p = granger_test_raw(pd.Series(Yp), pd.Series(Xp))
        if not np.isnan(p) and p < 0.05: sig += 1
    return sig / n_perturb


def raw_robustness(Y, X, n_perturb=100, noise_std=0.05):
    sig = 0
    for _ in range(n_perturb):
        Yp = Y + np.random.normal(0, noise_std * Y.std())
        Xp = X + np.random.normal(0, noise_std * X.std())
        p = granger_test_raw(Yp, Xp)
        if not np.isnan(p) and p < 0.05: sig += 1
    return sig / n_perturb


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main():
    print("Running 6-dimension parameter sweep...")
    records = []
    for beta in BETAS:
        for age_scale in AGE_SCALES:
            age_error_std = 2.0 * age_scale
            print(f"  β={beta}, age_scale={age_scale}")
            for rep in range(N_REPEATS):
                seed = rep * 100 + int(beta * 10) + int(age_scale * 100)
                X_t, Y_t, Z = generate_data_with_errors(N_TIME, PHI, beta, SIGMA_Z, SIGMA_EPS,
                                                        TRUE_LAG, CAUSAL_STRENGTH, age_error_std, seed)
                X_n, Y_n, _ = generate_data_with_errors(N_TIME, PHI, beta, SIGMA_Z, SIGMA_EPS,
                                                        None, 0.0, age_error_std, seed + 10000)
                proxy = Z.rolling(30, center=True).mean().fillna(method='bfill').fillna(method='ffill')
                raw_t = raw_robustness(Y_t, X_t, n_perturb=N_PERTURB)
                raw_n = raw_robustness(Y_n, X_n, n_perturb=N_PERTURB)
                carc_t = carc_robustness(Y_t, X_t, proxy, n_perturb=N_PERTURB)
                carc_n = carc_robustness(Y_n, X_n, proxy, n_perturb=N_PERTURB)
                records.append({'beta': beta, 'age_scale': age_scale, 'method': 'Raw_Granger', 'type': 'TPR', 'value': raw_t})
                records.append({'beta': beta, 'age_scale': age_scale, 'method': 'Raw_Granger', 'type': 'FPR', 'value': raw_n})
                records.append({'beta': beta, 'age_scale': age_scale, 'method': 'CARC_Granger', 'type': 'TPR', 'value': carc_t})
                records.append({'beta': beta, 'age_scale': age_scale, 'method': 'CARC_Granger', 'type': 'FPR', 'value': carc_n})

    df = pd.DataFrame(records)

    # Save report
    report = df.groupby(['beta', 'age_scale', 'method', 'type'])['value'].agg(['mean', 'std', 'median']).reset_index()
    report.to_csv(os.path.join(OUTPUT_DIR, 'continuous_robustness_report.csv'), index=False)
    print(f"Report saved.")

    # Plot
    sns.set_style("whitegrid")
    palette = {'Raw_Granger': 'lightcoral', 'CARC_Granger': 'steelblue'}
    thresholds = [0.60, 0.75, 0.95, 0.99]

    for metric in ['TPR', 'FPR']:
        fig, axes = plt.subplots(len(BETAS), len(AGE_SCALES),
                                 figsize=(4 * len(AGE_SCALES), 3.5 * len(BETAS)),
                                 sharex=True, sharey=True)
        if len(BETAS) == 1: axes = np.expand_dims(axes, axis=0)
        if len(AGE_SCALES) == 1: axes = np.expand_dims(axes, axis=1)

        for i, beta in enumerate(BETAS):
            for j, age in enumerate(AGE_SCALES):
                ax = axes[i, j]
                sub = df[(df['beta'] == beta) & (df['age_scale'] == age) & (df['type'] == metric)]
                for method, color in palette.items():
                    data = sub[sub['method'] == method]['value'].dropna()
                    if len(data) > 0:
                        parts = ax.violinplot(data, positions=[list(palette.keys()).index(method)],
                                              widths=0.6, showmeans=False, showmedians=False, showextrema=False)
                        for pc in parts['bodies']:
                            pc.set_facecolor(color); pc.set_alpha(0.3)
                            pc.set_edgecolor('black'); pc.set_linewidth(1)
                sns.boxplot(data=sub, x='method', y='value', palette=palette,
                            width=0.4, fliersize=0, linewidth=1, ax=ax,
                            boxprops=dict(facecolor='white', alpha=0.8, edgecolor='black'),
                            medianprops=dict(color='black', linewidth=2),
                            whiskerprops=dict(color='black'), capprops=dict(color='black'), legend=False)
                sns.stripplot(data=sub, x='method', y='value', palette=palette,
                              size=4, jitter=0.25, alpha=0.6, edgecolor='white', linewidth=0.5,
                              ax=ax, legend=False, zorder=10)
                for th in thresholds:
                    ax.axhline(th, color='gray', linestyle='--', linewidth=1, alpha=0.7)
                ax.set_title(f'β={beta}, age={age}')
                ax.set_ylim(-0.05, 1.05)
                if i == len(BETAS) - 1: ax.set_xlabel('Method')
                if j == 0: ax.set_ylabel(metric)

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, f'violin_{metric}_multipanel.png'), dpi=300)
        plt.close()

    print(f"Figures saved to {FIGURE_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
