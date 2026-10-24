# -*- coding: utf-8 -*-
"""Step 07 - DDR landscape over ARC strength x measurement noise (Fig. 3b).

Claim: directional dominance requires both deconfounding and low
measurement noise - DDR peaks on a ridge at moderate ARC strength
(alpha = 0.5-0.75) with sigma <= 0.05, and collapses to <= 1.3
(symmetric detection) at sigma = 0.5.  The operating point
(alpha = 0.5, sigma = 0.05) sits on the ridge at DDR = 6.65, near the
ridge maximum (8.16 at alpha = 0.75) - the adopted parameters are not
cherry-picked for DDR, and the sub-threshold regime does not hinge on
the exact alpha / sigma choice.

Design:
  discrete 5 x 5 cell map (no interpolation), SEQ blue ramp capped at
  #5e97df so black cell annotations stay readable, log sigma axis
  (0.01-0.5), operating point starred (orange, markers layer), DDR = 1
  and DDR = 3 anchored on the colorbar ticks.

Data: outputs/param_sensitivity_cache.csv (canonical scripts/14
parameter-sensitivity sweep), N_perturb = 200 subset, pooled group mean
per cell (5-15 replicate runs per cell).

Convergence target (asserted): DDR(alpha=0.5, sigma=0.05) = 6.65.

Usage:
    python scripts/fignew/07_sensitivity_ddr.py
Outputs: figures_new/07_sensitivity_ddr/sensitivity_ddr.png / .svg
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.dirname(os.path.abspath(__file__)), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from style import (apply_paleo_style, save, INK, SECONDARY_INK, GRID,  # noqa: E402
                   CAT2)

OUT_DIR = os.path.join(ROOT, "figures_new", "07_sensitivity_ddr")

# SEQ blue ramp capped at #5e97df: black 9pt annotations stay readable on
# the darkest cell (contrast >= 6.9:1 against the max-DDR cell colour).
SEQ_BLUES = ["#f0f5fc", "#cfe0f4", "#9cc0ea", "#5e97df"]

RUN_ALPHA, RUN_SIGMA, RUN_DDR = 0.5, 0.05, 6.65
DDR_VMAX = 9.0                          # colorbar [1, 9], ticks 1/3/5/7/9

FIG_W, FIG_H = 178.0, 118.0             # print size, mm
MAP_X, MAP_Y, MAP_W, MAP_H = 0.27, 0.25, 0.44, 0.66
CB_X, CB_W = 0.72, 0.016


# ------------------------------------------------------------------ data ----
def load_ddr_grid():
    """DDR = tpr/max(fpr, 0.01) on the alpha x noise_std grid (N = 200).

    The cache pools the three sweep rows of scripts/14 (alpha x N,
    alpha x sigma, N x sigma); cells on a fixed line repeat, so the
    per-cell group size varies between 5 and 15.
    """
    p = pd.read_csv(os.path.join(ROOT, "outputs", "param_sensitivity_cache.csv"))
    sub = p[p["n_perturb"] == 200]
    piv = (sub.groupby(["alpha", "noise_std"], as_index=False)["robustness"]
              .mean())
    cnt = sub.groupby(["alpha", "noise_std"]).size()
    alphas = sorted(piv["alpha"].unique())
    noises = sorted(piv["noise_std"].unique())
    z = np.full((len(noises), len(alphas)), np.nan)
    counts = np.zeros_like(z, dtype=int)
    for i, ns in enumerate(noises):
        for j, a in enumerate(alphas):
            row = piv[(piv["alpha"] == a) & (piv["noise_std"] == ns)]
            if len(row):
                z[i, j] = row["robustness"].iloc[0]
            counts[i, j] = int(cnt.get((a, ns), 0))
    return np.array(alphas), np.array(noises), z, counts


def _edges(vals, log=False):
    """Cell edges as neighbour midpoints; outer edges symmetric so the
    nominal grid values sit at the cell centres (tick positions)."""
    v = np.log10(vals) if log else np.asarray(vals, dtype=float)
    inner = 0.5 * (v[:-1] + v[1:])
    lo = v[0] - 0.5 * (v[1] - v[0])
    hi = v[-1] + 0.5 * (v[-1] - v[-2])
    e = np.concatenate([[lo], inner, [hi]])
    return 10.0 ** e if log else e


# ------------------------------------------------------------------ plot ----
def mark(ax, x, y, **kw):
    """Plot a marker that lands in the markers layer, not a colour role."""
    (line,) = ax.plot([x], [y], **kw)
    line.set_gid("layer-markers")
    return line


def plot(alphas, noises, z, counts):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig = Figure(figsize=(FIG_W / 25.4, FIG_H / 25.4), facecolor="white")
    ax = fig.add_axes([MAP_X, MAP_Y, MAP_W, MAP_H])
    apply_paleo_style(ax)
    ax.grid(False)                       # the cells are their own grid
    ax.minorticks_off()

    xe = _edges(alphas)
    ye = _edges(noises, log=True)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUES)
    pc = ax.pcolormesh(xe, ye, z, cmap=cmap, vmin=1.0, vmax=DDR_VMAX,
                       edgecolors=GRID, linewidths=0.6)
    pc.set_gid("layer-data")             # GRID edges must not class it grid

    ax.set_xlim(xe[0], xe[-1])
    ax.set_xticks(alphas)
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_yscale("log")
    ax.set_ylim(ye[0], ye[-1])
    ax.set_yticks(noises)
    ax.set_yticklabels(["0.01", "0.05", "0.1", "0.2", "0.5"])
    ax.set_xlabel("ARC stripping strength $\\alpha$", fontsize=10, color=INK)
    ax.set_ylabel("measurement noise $\\sigma$", fontsize=10, color=INK)
    ax.set_title("(b) DDR landscape — ARC strength $\\alpha$ × measurement "
                 "noise $\\sigma$",
                 loc="left", fontsize=10.5, fontweight="bold", color=INK,
                 pad=6)

    # cell values at the nominal grid points (the run cell is starred)
    for i, ns in enumerate(noises):
        for j, a in enumerate(alphas):
            if a == RUN_ALPHA and ns == RUN_SIGMA:
                continue
            ax.text(a, ns, f"{z[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color=INK, zorder=3)

    mark(ax, RUN_ALPHA, RUN_SIGMA, marker="*", ms=16, mfc=CAT2, mec="white",
         mew=1.0, ls="none", zorder=4)

    cax = fig.add_axes([CB_X, MAP_Y, CB_W, MAP_H])
    cbar = fig.colorbar(pc, cax=cax, ticks=[1, 3, 5, 7, 9])
    cbar.outline.set_color(SECONDARY_INK)
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(labelsize=9, colors=INK, length=2.5, width=0.6)
    cbar.set_label("DDR", fontsize=10, color=INK)

    # key + provenance (computed from the grid so the figure cannot drift)
    run = float(z[list(noises).index(RUN_SIGMA), list(alphas).index(RUN_ALPHA)])
    fig.text(MAP_X, 0.175,
             f"operating point (star): $\\alpha$ = {RUN_ALPHA}, "
             f"$\\sigma$ = {RUN_SIGMA}, DDR = {run:.2f}",
             fontsize=9, color=INK)
    fig.text(MAP_X, 0.125,
             "DDR = 1: symmetric (non-directional) detection; "
             "$DDR \\geq 3$: strong directional dominance (paper reference).",
             fontsize=9, color=INK)
    fig.text(MAP_X, 0.088,
             f"grid = pooled group mean, $N_{{perturb}}$ = 200, "
             f"{counts.min()}-{counts.max()} realizations per cell "
             "(outputs/param_sensitivity_cache.csv).",
             fontsize=9, color=INK)

    save(fig, os.path.join(OUT_DIR, "sensitivity_ddr"))
    print_anchors(alphas, noises, z, run)


# ---------------------------------------------------------------- anchors ----
def print_anchors(alphas, noises, z, run):
    print("\n== anchors (Fig. 3b DDR grid) ==")
    print(f"  run point alpha={RUN_ALPHA}, sigma={RUN_SIGMA}: "
          f"DDR = {run:.4f} (target {RUN_DDR})")
    print(f"  grid: alpha {alphas.tolist()} x sigma {noises.tolist()}")
    for i, ns in enumerate(noises):
        print(f"    sigma={ns:4.2f}:  " + "  ".join(f"{v:5.2f}" for v in z[i]))
    imax = np.unravel_index(np.nanargmax(z), z.shape)
    print(f"  max: DDR = {z[imax]:.2f} at alpha={alphas[imax[1]]}, "
          f"sigma={noises[imax[0]]}")
    print(f"  sigma=0.5 row max: {z[-1].max():.2f} (symmetric at high noise)")
    print(f"  alpha=0 column max: {z[:, 0].max():.2f} (no deconfounding)")
    print(f"  cells with DDR >= 3: {int((z >= 3).sum())}/25")
    print("  claim: DDR peaks at moderate alpha with sigma <= 0.05 and "
          "collapses to ~1 at sigma = 0.5; the operating point sits on "
          "the ridge.")
    assert abs(run - RUN_DDR) < 0.05, "run point DDR deviates from 6.65"


def main():
    alphas, noises, z, counts = load_ddr_grid()
    plot(alphas, noises, z, counts)


if __name__ == "__main__":
    main()
