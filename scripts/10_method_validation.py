#!/usr/bin/env python3
"""
10_method_validation.py
-----------------------
Method comparison and audit:
1. ARC validation: Original Granger vs Fixed α=0.5 vs Adaptive ARC
2. CCM audit: TPR/FPR vs autocorrelation strength φ

Requires: src/validation.py (PaleoDataGenerator, CausalMethodEvaluator)

Output: figures/arc_validation.png, figures/ccm_audit_phi.png
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from src.config import FIGURE_DIR, OUTPUT_DIR
from src.validation import (PaleoDataGenerator, CausalMethodEvaluator,
                             granger_causality, ccm_causality)
from src.arc import extract_residuals_adaptive

os.makedirs(FIGURE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. ARC Validation
# ---------------------------------------------------------------------------
def validate_arc():
    print("=" * 60)
    print("ARC Validation: Original vs Fixed α vs Adaptive")
    print("=" * 60)

    gen = PaleoDataGenerator(scenario='A', random_seed=42)

    # Original Granger
    evaluator_orig = CausalMethodEvaluator(gen, granger_causality, "Granger")
    audit_orig = evaluator_orig.evaluate(n_realizations=20, use_age_errors=False)

    # Fixed α=0.5
    def granger_fixed_alpha(data, dt, **kwargs):
        proxy = data['NGRIP']
        residuals = extract_residuals_adaptive(data, proxy, alpha=0.5)
        return granger_causality(residuals, dt)

    evaluator_fixed = CausalMethodEvaluator(gen, granger_fixed_alpha, "Fixed α=0.5")
    audit_fixed = evaluator_fixed.evaluate(n_realizations=20, use_age_errors=False)

    # Adaptive ARC (median-edge-count heuristic)
    def granger_arc(data, dt, **kwargs):
        proxy = data['NGRIP']
        alphas = np.linspace(0.0, 1.0, 11)
        all_edges = []
        for alpha in alphas:
            residuals = extract_residuals_adaptive(data, proxy, alpha)
            edges = granger_causality(residuals, dt)
            all_edges.append(edges)
        edge_counts = [len(e) for e in all_edges]
        median_count = np.median(edge_counts)
        for alpha, edges in zip(alphas, all_edges):
            if len(edges) <= median_count:
                return edges
        return all_edges[-1]

    evaluator_arc = CausalMethodEvaluator(gen, granger_arc, "ARC")
    audit_arc = evaluator_arc.evaluate(n_realizations=20, use_age_errors=False)

    # Report
    print(f"{'Method':<30} {'FPR':>8} {'TPR':>8}")
    print("-" * 48)
    print(f"{'Original Granger':<30} {audit_orig['FPR']:>8.3f} {audit_orig['TPR']:>8.3f}")
    print(f"{'Fixed α=0.5':<30} {audit_fixed['FPR']:>8.3f} {audit_fixed['TPR']:>8.3f}")
    print(f"{'Adaptive ARC':<30} {audit_arc['FPR']:>8.3f} {audit_arc['TPR']:>8.3f}")

    # Save results for reuse
    arc_df = pd.DataFrame({
        'method': ['Original Granger', 'Fixed α=0.5', 'Adaptive ARC'],
        'fpr': [audit_orig['FPR'], audit_fixed['FPR'], audit_arc['FPR']],
        'tpr': [audit_orig['TPR'], audit_fixed['TPR'], audit_arc['TPR']],
    })
    arc_df.to_csv(os.path.join(OUTPUT_DIR, 'method_validation_arc.csv'),
                  index=False)

    _plot_arc(arc_df)


def _plot_arc(df):
    methods = ['Original\nGranger', 'Fixed α=0.5', 'Adaptive\nARC']
    fprs = df['fpr'].values
    tprs = df['tpr'].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(methods, fprs, color=['#d62728', '#1f77b4', '#2ca02c'])
    ax1.axhline(0.05, color='k', linestyle='--')
    ax1.set_ylabel('FPR'); ax1.set_title('False Positive Rate')
    ax2.bar(methods, tprs, color=['#d62728', '#1f77b4', '#2ca02c'])
    ax2.set_ylabel('TPR'); ax2.set_title('True Positive Rate')
    for ax in [ax1, ax2]:
        for bar in ax.patches:
            h = bar.get_height()
            # tall bars: put the value inside the bar top so it cannot reach
            # the panel title above (layout QA: text_overlap with title)
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f'{h:.3f}', ha='center',
                    va='top' if h > 0.2 else 'baseline')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'arc_validation.png'), dpi=150)
    plt.close()
    print("Saved: arc_validation.png")


# ---------------------------------------------------------------------------
# 2. CCM Audit
# ---------------------------------------------------------------------------
def audit_ccm():
    print("\n" + "=" * 60)
    print("CCM Audit: TPR/FPR vs autocorrelation φ")
    print("=" * 60)

    phi_range = np.arange(0.3, 0.95, 0.05)
    tpr_vals, fpr_vals = [], []

    for phi in phi_range:
        gen = PaleoDataGenerator(scenario='A', random_seed=42)
        gen.phi = {v: phi for v in gen.var_names}
        evaluator = CausalMethodEvaluator(gen, ccm_causality, "CCM")
        audit = evaluator.evaluate(n_realizations=20, use_age_errors=False)
        tpr_vals.append(audit['TPR'])
        fpr_vals.append(audit['FPR'])
        print(f"  φ={phi:.2f} → TPR={audit['TPR']:.3f}, FPR={audit['FPR']:.3f}")

    # Save results for reuse
    ccm_df = pd.DataFrame({
        'phi': phi_range,
        'tpr': tpr_vals,
        'fpr': fpr_vals,
    })
    ccm_df.to_csv(os.path.join(OUTPUT_DIR, 'ccm_audit_phi.csv'), index=False)

    _plot_ccm(ccm_df)


def _plot_ccm(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df['phi'], df['tpr'], 'bo-', label='TPR', linewidth=2)
    ax.plot(df['phi'], df['fpr'], 'rs-', label='FPR', linewidth=2)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Autocorrelation coefficient φ')
    ax.set_ylabel('Rate')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_title('CCM Audit: Autocorrelation degrades performance')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'ccm_audit_phi.png'), dpi=150)
    plt.close()
    print("Saved: ccm_audit_phi.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Method comparison and audit')
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip the simulations; redraw figures from the '
                             'cached CSVs in outputs/ (method_validation_arc.csv, '
                             'ccm_audit_phi.csv)')
    args = parser.parse_args()
    if args.plot_only:
        arc_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'method_validation_arc.csv'))
        ccm_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'ccm_audit_phi.csv'))
        _plot_arc(arc_df)
        _plot_ccm(ccm_df)
    else:
        validate_arc()
        audit_ccm()
    print("\nDone.")
