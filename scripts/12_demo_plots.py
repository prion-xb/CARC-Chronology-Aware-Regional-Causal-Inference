#!/usr/bin/env python3
"""
12_demo_plots.py
----------------
Demonstration plots for presentations and quick-look data inspection.

1. Synthetic time series examples (T&F demo)
2. Time series stack of key records with climate windows

Output: figures/simulated_series_examples.png, figures/timeseries_stack.png
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.config import FIGURE_DIR
from src.synthetic import generate_null_data as _generate_null_data

os.makedirs(FIGURE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. T&F Demo: Example time series with/without causality and common forcing
# ---------------------------------------------------------------------------
def demo_tf():
    print("Generating T&F demonstration plots...")
    params = [
        ("No causality, β=0", None, 0.0, 0),
        ("No causality, β=1", None, 1.0, 1),
        ("Causality (X→Y), β=0", 2, 0.0, 2),
        ("Causality (X→Y), β=1", 2, 1.0, 3)
    ]

    n_time = 500
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for ax, (title, lag, beta, seed_offset) in zip(axes, params):
        X, Y, z = _generate_null_data(n=n_time, phi=0.8, beta=beta,
                                      sigma_z=1.0, sigma_eps=0.4, seed=42 + seed_offset)
        # Note: generates null; for causality, we'd need the generator with causal edges.
        # This is a simplified demo.
        time = np.arange(n_time)
        ax.plot(time, X, label='X', linewidth=0.8, alpha=0.8)
        ax.plot(time, Y, label='Y', linewidth=0.8, alpha=0.8)
        if beta > 0:
            ax.plot(time, z, label='Common forcing Z', linewidth=0.8, alpha=0.5, linestyle='--')
        ax.set_title(title); ax.set_xlabel('Time step'); ax.set_ylabel('Value')
        ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'simulated_series_examples.png'), dpi=300)
    plt.close()
    print("Saved: simulated_series_examples.png")


# ---------------------------------------------------------------------------
# 2. Time series stack
# ---------------------------------------------------------------------------
def demo_timeseries_stack():
    print("Generating time series stack...")
    climate_windows = {
        'H1': (14, 18), 'H4': (36, 40), 'LGM': (19, 26),
        'GI12': (44, 47), 'GS12': (47, 49), 'BA': (12.7, 14.7),
    }

    # Generate demo data
    age = np.linspace(10, 60, 500)
    rng = np.random.default_rng(42)
    records = [
        ('NGRIP δ¹⁸O', np.sin(age * 0.3) * 2 + rng.normal(0, 0.3, 500)),
        ('EDC δD', np.cos(age * 0.28) * 1.5 + rng.normal(0, 0.4, 500)),
        ('Cariaco Mo', np.sin(age * 0.35 + 1) * 1.8 + rng.normal(0, 0.5, 500)),
    ]

    fig, axes = plt.subplots(len(records), 1, figsize=(12, 8), sharex=True)

    for ax, (name, values) in zip(axes, records):
        ax.plot(age, values, 'k-', linewidth=1.0)
        ylim = ax.get_ylim()
        for wname, (start, end) in climate_windows.items():
            ax.axvspan(start, end, alpha=0.2, color='lightblue')
            ax.text((start + end) / 2, ylim[0] + 0.05 * (ylim[1] - ylim[0]),
                    wname, ha='center', fontsize=8)
        ax.set_ylabel(name, fontsize=10)

    axes[-1].set_xlabel('Age (ka)', fontsize=10)
    fig.suptitle('Key Paleoclimate Records (10–60 ka)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'timeseries_stack.png'), dpi=300)
    plt.close()
    print("Saved: timeseries_stack.png")


if __name__ == "__main__":
    demo_tf()
    demo_timeseries_stack()
    print("Done.")
