#!/usr/bin/env python3
"""Elements for main-text Fig. 2 — Layer-1 regional stacking on the 10-60 ka grid.

Emits one small standalone plot per region (six files) instead of the 2x3
composite. Each element shows the member records, the median stack across the
200-realization age-perturbation ensemble, its P5-P95 band, and the six climate
windows as alternating shading.

The stacking itself is not reimplemented: it imports the same
`stack_schemes` machinery and the same seed (42) as
scripts/33_layer1_stacking_10_60.py, so the curves are identical to the
composite the paper already reports.

Output: Figures/elements/fig2_stack_<region>.{png,pdf,svg}
"""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.config import DATA_DIR, OUTPUT_DIR, AGE_GRID_10_60, N_PERTURB, CLIMATE_WINDOWS
from src.io import load_all_records
from src.stacking import build_region_dict
from src.elements import (ONE_COL, MEMBER, GRID, INK_SEC, INK_MUTED,
                          REGION_COLORS, use_element_style, save_element)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_spec = importlib.util.spec_from_file_location(
    "stackviz", os.path.join(SCRIPT_DIR, "18_stacking_visualization.py"))
stackviz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stackviz)

PAPER_REGIONS = stackviz.PAPER_REGIONS
REGION_LABELS = stackviz.REGION_LABELS

WINDOW_ORDER = ["GS-12 (stadial)", "GI-12 (interstadial)", "H4 event",
                "Last Glacial Maximum", "H1 event", "Bølling-Allerød"]
WINDOW_ABBR = {"GS-12 (stadial)": "GS-12", "GI-12 (interstadial)": "GI-12",
               "H4 event": "H4", "Last Glacial Maximum": "LGM",
               "H1 event": "H1", "Bølling-Allerød": "B-A"}
WIN_FILL = ["#ececea", "#e0e0dc"]

NOISE_REGION_MAP = {"Tropical": "Tropical",
                    "Antarctic/Southern Ocean": "Antarctic",
                    "North Atlantic": "North_Atlantic",
                    "East Asian Monsoon": "East_Asian_Monsoon",
                    "High-latitude Lakes": "Other_Lakes",
                    "South American Monsoon": "South_America"}

SLUG = {"North_Atlantic": "north_atlantic", "Antarctic": "antarctic",
        "Tropical": "tropical", "East_Asian_Monsoon": "east_asian_monsoon",
        "South_America": "south_american_monsoon", "Other_Lakes": "high_lat_lakes"}


def load_noise_ratios():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "stacking_noise_reduction.csv"))
    return {NOISE_REGION_MAP[r.region]: (r.achieved_ratio, r.theoretical_1_sqrt_K)
            for _, r in df.iterrows() if r.region in NOISE_REGION_MAP}


def draw_region(region, members, stack, noise, label_windows):
    """One region -> one standalone element."""
    grid = AGE_GRID_10_60
    color = REGION_COLORS[region]
    fig, ax = plt.subplots(figsize=ONE_COL)

    ymin, ymax = -3.6, 3.6
    for j, wname in enumerate(WINDOW_ORDER):
        w0, w1 = CLIMATE_WINDOWS[wname]
        ax.add_patch(Rectangle((w0, ymin), w1 - w0, ymax - ymin,
                               facecolor=WIN_FILL[j % 2], edgecolor="none", zorder=0))
        if label_windows:
            # B-A (12.7-14.7) and H1 (14.0-18.0) actually overlap in age, so
            # their labels are staggered onto two rows to keep them apart
            y = ymax * (0.96 if j % 2 == 0 else 0.80)
            ax.text((w0 + w1) / 2, y, WINDOW_ABBR[wname], fontsize=5.8,
                    color=INK_MUTED, ha="center", va="top", zorder=4)

    for interp, _sigma, _name in members:
        idx = np.round(np.linspace(0, len(interp) - 1, len(grid))).astype(int)
        ax.plot(grid, interp[idx], color=MEMBER, lw=0.35, alpha=0.7, zorder=1)

    med = np.median(stack, axis=0)
    p5, p95 = np.percentile(stack, [5, 95], axis=0)
    ax.fill_between(grid, p5, p95, color=color, alpha=0.22, zorder=2, lw=0)
    ax.plot(grid, med, color=color, lw=1.8, zorder=3)

    ratio, ref = noise
    ax.text(0.015, 0.045, f"noise ×{ratio:.2f}   (1/√K = {ref:.2f})",
            transform=ax.transAxes, fontsize=6.4, color=INK_SEC,
            ha="left", va="bottom", zorder=5)

    ax.set_xlim(10, 60)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Age (ka BP)")
    ax.set_ylabel("Standardized value")
    ax.set_title(f"{REGION_LABELS[region]}  (n = {len(members)})", fontsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.55, zorder=1.5)
    ax.set_axisbelow(False)
    save_element(fig, f"fig2_stack_{SLUG[region]}", subdir="fig2_layer1_stacking")


def main():
    use_element_style()
    print("Loading records (10-60 ka)...")
    records, _ = load_all_records(DATA_DIR, age_min=10.0, age_max=60.0)
    records = [(i, s, "Antarctic" if r == "Southern_Ocean" else r, n)
               for i, s, r, n in records]
    region_dict = {r: v for r, v in build_region_dict(records).items()
                   if r in PAPER_REGIONS}
    noise = load_noise_ratios()

    rng = np.random.default_rng(42)          # same seed as script 33
    for region in PAPER_REGIONS:
        recs = region_dict[region]
        stack = stackviz.stack_schemes(recs, AGE_GRID_10_60, N_PERTURB, 0.5, rng)["median"]
        # only the first element carries the window abbreviations; the rest stay
        # clean, since the labels are identical across regions
        draw_region(region, recs, stack, noise[region],
                    label_windows=(region == PAPER_REGIONS[0]))


if __name__ == "__main__":
    main()
