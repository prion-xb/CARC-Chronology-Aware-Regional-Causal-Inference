#!/usr/bin/env python3
"""
03_synthetic_validation.py
--------------------------
Validate CARC on synthetic data with known ground truth.

1. FPR scan: CARC vs direct Granger as common-forcing strength β increases
2. Robustness vs age-error scaling for true causal edges
3. Generates: figures/carc_fpr_vs_beta.png, figures/carc_synthetic_validation.png
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import FIGURE_DIR, OUTPUT_DIR
from src.synthetic import generate_null_data
from src.granger import granger_test_pair, robustness
from src.arc import arc_deconfound

os.makedirs(FIGURE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Experiment 1: FPR vs common-forcing strength β
# ---------------------------------------------------------------------------
def experiment_fpr_vs_beta():
    print("=" * 60)
    print("Experiment 1: FPR vs common-forcing strength β")
    print("=" * 60)

    beta_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    n_repeats = 30
    n_perturb = 100

    direct_fpr, carc_fpr = [], []

    for beta in beta_values:
        print(f"  β = {beta}")
        direct_fp, carc_fp = 0, 0

        for rep in range(n_repeats):
            seed = rep * 100 + int(beta * 100)
            X, Y, Z = generate_null_data(n=500, phi=0.85, beta=beta,
                                         sigma_z=1.0, sigma_eps=0.8, seed=seed)
            # Smooth Z as proxy
            import pandas as pd
            proxy = pd.Series(Z).rolling(window=30, center=True).mean()
            proxy = proxy.fillna(method='bfill').fillna(method='ffill').values

            # Direct Granger
            if granger_test_pair(Y, X, maxlag=3):
                direct_fp += 1

            # CARC: ARC then robustness
            Y_clean = arc_deconfound(Y, proxy, alpha=0.5)
            X_clean = arc_deconfound(X, proxy, alpha=0.5)
            rob = robustness(Y_clean, X_clean, sigma_err=2.0, n_perturb=n_perturb)
            if rob >= 0.95:
                carc_fp += 1

        direct_fpr.append(direct_fp / n_repeats)
        carc_fpr.append(carc_fp / n_repeats)
        print(f"    Direct FPR={direct_fpr[-1]:.3f}, CARC FPR={carc_fpr[-1]:.3f}")

    # Plot
    # Save results for reuse
    fpr_df = pd.DataFrame({
        'beta': beta_values,
        'direct_fpr': direct_fpr,
        'carc_fpr': carc_fpr,
    })
    fpr_df.to_csv(os.path.join(OUTPUT_DIR, 'fpr_vs_beta.csv'), index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(beta_values, direct_fpr, 'o-', label='Direct Granger', color='#D95319', linewidth=2)
    ax.plot(beta_values, carc_fpr, 's-', label='CARC (95% honesty filter)', color='#4472C4', linewidth=2)
    ax.axhline(0.05, linestyle='--', color='gray', label='Nominal α = 5%')
    ax.set_xlabel('Common forcing strength (β)', fontsize=12)
    ax.set_ylabel('False Positive Rate', fontsize=12)
    ax.set_title('CARC vs. Direct Granger: FPR under common forcing', fontsize=14)
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'carc_fpr_vs_beta.png'), dpi=300)
    plt.close()
    print(f"Saved: carc_fpr_vs_beta.png")


# ---------------------------------------------------------------------------
# Experiment 2: Detection stability vs age-error scaling
# ---------------------------------------------------------------------------
def experiment_stability_vs_error():
    print("\n" + "=" * 60)
    print("Experiment 2: Detection stability vs age-error scaling")
    print("=" * 60)

    from src.synthetic import generate_scenario_A
    from src.config import TIER1_SIGMA, AGE_GRID_10_60, DT_HIGH_RES
    from src.arc import arc_deconfound_batch
    from scipy.ndimage import gaussian_filter1d

    AGE_GRID = AGE_GRID_10_60
    DT = DT_HIGH_RES
    TRUE_EDGES = [('NGRIP', 'EDC_dD'), ('CO2', 'CH4'), ('Cariaco', 'Cheng')]

    scales = [0.0, 0.3, 0.5, 1.0, 1.5, 2.0]
    n_perturb = 100

    results = {scale: {} for scale in scales}

    for scale in scales:
        print(f"  Error scale = {scale}")
        edge_counts = {edge: 0 for edge in TRUE_EDGES}
        n_real = 20

        for r in range(n_real):
            data, edges_cfg, t_grid = generate_scenario_A(
                seed=42 + r, include_age_errors=(scale > 0))

            # ARC preprocessing
            ngrip_trend = gaussian_filter1d(data['NGRIP'], sigma=1.0 / DT, mode='nearest')
            residuals = arc_deconfound_batch(data, ngrip_trend, alpha=0.5, keep_proxy='NGRIP')

            for pert in range(n_perturb):
                perturbed = {}
                for var in residuals:
                    sigma = TIER1_SIGMA.get(var, 1.0) * scale
                    age_pert = AGE_GRID + np.random.normal(0, sigma, len(AGE_GRID))
                    idx = np.argsort(age_pert)
                    perturbed[var] = residuals[var][idx]

                # Granger test
                from src.granger import granger_detect
                detected = granger_detect(perturbed, maxlag=5)
                for driver, target in TRUE_EDGES:
                    if (driver, target) in detected:
                        edge_counts[(driver, target)] += 1

        total = n_real * n_perturb
        for edge in TRUE_EDGES:
            results[scale][edge] = edge_counts[edge] / total
            print(f"    {edge[0]} → {edge[1]}: {results[scale][edge]:.1%}")

    # Save results for reuse
    rows = []
    for scale in scales:
        for edge in TRUE_EDGES:
            rows.append({
                'error_scale': scale,
                'driver': edge[0],
                'target': edge[1],
                'robustness': results[scale][edge],
            })
    stab_df = pd.DataFrame(rows)
    stab_df.to_csv(os.path.join(OUTPUT_DIR, 'stability_vs_error.csv'), index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for edge in TRUE_EDGES:
        rates = [results[s][edge] for s in scales]
        ax.plot(scales, rates, 'o-', label=f'{edge[0]} → {edge[1]}')
    ax.axhline(0.95, color='gray', linestyle='--', label='95% threshold')
    ax.set_xlabel('Age error scaling factor')
    ax.set_ylabel('Detection Stability')
    ax.set_title('CARC Detection Stability vs. Age Error')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'carc_synthetic_validation.png'), dpi=300)
    plt.close()
    print(f"Saved: carc_synthetic_validation.png")


if __name__ == "__main__":
    experiment_fpr_vs_beta()
    experiment_stability_vs_error()
    print("\nDone.")
