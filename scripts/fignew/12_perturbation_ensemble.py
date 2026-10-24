# -*- coding: utf-8 -*-
"""Step 12 - the age-perturbation ensemble (FigS11 material, Section 4.2).

Claim: age uncertainty repositions sharp record features by ~1 sigma with no
systematic bias - for NGRIP d18O (sigma = 0.65 ka) the detected peak/trough
positions spread over 1.35-1.75 ka (P5-P95, median 1.6 ka) across the M = 200
age-perturbation realizations, somewhat narrower than the 2 x 1.64 sigma =
2.13 ka of a sharp isolated peak because the 0.05-ka-grid series is smooth and
curvature pins local extrema.  Stacking pays for noise suppression with
feature sharpness: the North Atlantic stack of Fig. 2 keeps a single local
maximum, whose position wanders 3.8 ka (P5-P95) across draws - broad-feature
position uncertainty is set by feature width, not by sigma.

Design:
  (a) ensemble spaghetti: the M = 200 age-perturbed NGRIP realizations as
      thin blue lines + the unperturbed record in bold black + the P5-P95
      band + climate-window bands; the detected peaks/troughs marked on the
      unperturbed record (filled circles / open squares).
  (b) feature-position sensitivity: one horizontal bar per feature at its
      nominal age (y-axis, paleo-inverted), spanning the P5-P95 of the
      detected positions across draws with the mean as a tick; the +/- 1
      sigma reference band and a dashed zero line make bias and spread
      readable at a glance.  x-axis = detected position - nominal (ka).

Data: NGRIP (Erhardt et al. 2016, sigma = 0.65 ka) on the canonical 10-60 ka
grid (Delta t = 0.05 ka) loaded through step 11's canonical-80 filter; the
perturbation rule is the stackviz one (sort grid + N(0, sigma)), M = 200,
seed 42 - the same stream that feeds Fig. 2's panels.  The North Atlantic
stack contrast uses the identical stack_schemes call as step 11.

Convergence targets (asserted):
  record sigma = 0.65 ka on the 1001-point grid, M = 200 draws
  12 peaks + 13 troughs (prominence >= 2.2, spacing >= 1.5 ka) = 25 features
  per-feature P5-P95 width within [1.3, 1.8] ka, median width in [1.55, 1.65]
  |bias| <= 0.5 ka per feature, pooled median |bias| < 0.1 ka
  North Atlantic stack: exactly 1 local maximum at prominence >= 0.12

Usage:
    python scripts/fignew/12_perturbation_ensemble.py
Outputs: figures_new/12_perturbation_ensemble/perturbation_ensemble.png / .svg
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from scipy.signal import find_peaks

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.dirname(os.path.abspath(__file__)), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from style import (save, apply_paleo_style, INK, SURFACE, GRID, BAND,
                   CAT1, REGION_ORDER)  # noqa: E402
from src.config import (AGE_GRID_10_60, N_PERTURB, CLIMATE_WINDOWS)  # noqa: E402

# step 11's canonical loader + stackviz (same code path as Fig. 2)
spec = importlib.util.spec_from_file_location(
    "s11", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "11_layer1_stacking.py"))
s11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s11)

OUT_DIR = os.path.join(ROOT, "figures_new", "12_perturbation_ensemble")

# detection / matching parameters (documented in the footer)
PROM_THR = 2.2          # peak/trough prominence floor (standardized units)
SPACING_KA = 1.5        # minimum feature spacing
MATCH_WIN_KA = 1.0      # per-draw matching half-window around the nominal age
SIGMA_EXPECT = 0.65     # NGRIP 1-sigma age error (ka), asserted

FIG_W, FIG_H = 178.0, 175.0
AX_A = [0.075, 0.460, 0.900, 0.435]    # (a) spaghetti: age on x
AX_B = [0.075, 0.290, 0.900, 0.120]    # (b) position shifts: shift on x
SHIFT_LO, SHIFT_HI = -1.2, 1.2
FOOTER_Y = (0.250, 0.225, 0.200, 0.175, 0.150, 0.125, 0.100, 0.075,
            0.050, 0.025)
DRAW_ALPHA, DRAW_LW = 0.10, 0.4        # 200-line spaghetti is a cloud, not lines
NA_STACK_PROM = 0.12                   # prominence floor for the stack contrast


# ------------------------------------------------------------------ data ----
def load_ngrip():
    """NGRIP (interpolated, sigma, name) from step 11's canonical loader."""
    region_dict = s11.load_region_dict()
    hits = [r for r in region_dict["North_Atlantic"] if "GRIP" in r[2].upper()]
    assert len(hits) == 1, f"expected exactly one NGRIP record, got {len(hits)}"
    interp, sigma, name = hits[0]
    assert abs(sigma - SIGMA_EXPECT) < 1e-9, f"NGRIP sigma drifted: {sigma}"
    assert len(interp) == len(AGE_GRID_10_60) == 1001
    return interp, sigma, name


def detect_features(series, grid):
    """Major peaks and troughs of *series* (prominence/spacing-selected)."""
    dmin = int(SPACING_KA / (grid[1] - grid[0]))
    pk, _ = find_peaks(series, prominence=PROM_THR, distance=dmin)
    tr, _ = find_peaks(-series, prominence=PROM_THR, distance=dmin)
    return grid[pk], grid[tr]


def perturb_draws(interp, sigma, grid, rng, n_draws):
    """One draw per realization: sort grid + N(0, sigma) (stackviz rule)."""
    n = len(interp)
    return np.array([interp[np.argsort(grid + rng.normal(0, sigma, n))]
                     for _ in range(n_draws)])


def tracked_positions(draws, grid, nominal_ages, sign=+1):
    """Detected feature position per draw (argmax of sign*series in the
    +/- MATCH_WIN_KA window around each nominal age)."""
    out = []
    for ta in nominal_ages:
        m = (grid >= ta - MATCH_WIN_KA) & (grid <= ta + MATCH_WIN_KA)
        out.append([grid[m][np.argmax(sign * d[m])] for d in draws])
    return np.array(out).T                   # (n_draws, n_features)


def stack_contrast():
    """The Fig. 2 North Atlantic stack: how many features survive, and how
    far the surviving maximum wanders across the same M = 200 draws."""
    region_dict = s11.load_region_dict()
    recs = region_dict["North_Atlantic"]
    assert len(recs) == 13
    rng = np.random.default_rng(42)
    stacks = s11.stackviz.stack_schemes(recs, AGE_GRID_10_60, N_PERTURB,
                                        s11.SMOOTH_SIGMA, rng)["median"]
    med = np.median(stacks, axis=0)
    pk, _ = find_peaks(med, prominence=NA_STACK_PROM,
                       distance=int(1.0 / 0.05))
    # wander of the draw maximum in a fixed window around the surviving peak
    w0 = float(AGE_GRID_10_60[pk[0]]) - 4.0 if len(pk) else None
    wander = None
    if len(pk):
        w1 = float(AGE_GRID_10_60[pk[0]]) + 4.0
        m = (AGE_GRID_10_60 >= w0) & (AGE_GRID_10_60 <= w1)
        pos = np.array([AGE_GRID_10_60[m][np.argmax(d[m])] for d in stacks])
        wander = tuple(np.percentile(pos, [5, 95]))
    return len(pk), [float(AGE_GRID_10_60[p]) for p in pk], wander


# ------------------------------------------------------------------ plot ----
_marker_n = [0]


def mark(ax, x, y, marker, ms=4, mfc=INK, mec=INK, mew=1.0, zorder=4):
    """One feature symbol with a unique gid, so every marker buckets into
    the SVG layer-markers wrapper (a bare repeated gid would keep the first
    occurrence outside the wrapper - 12's own layout lesson)."""
    (mk,) = ax.plot([x], [y], marker, ms=ms, mfc=mfc, mec=mec, mew=mew,
                    color=INK, ls="none", zorder=zorder)
    mk.set_gid(f"layer-markers-{_marker_n[0]:03d}")
    _marker_n[0] += 1
    return mk


def no_text_overruns(fig):
    """Geometry QA: every text stays inside the figure (0.5% tolerance,
    measured on the real renderer - the 06/07 footer-overflow lesson)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    w, h = fig.bbox.width, fig.bbox.height
    ok = True
    texts = list(fig.texts)
    for ax in fig.axes:
        texts += list(ax.texts)
    for t in texts:
        bb = t.get_window_extent(renderer=renderer)
        if (bb.x0 < -0.005 * w or bb.x1 > 1.005 * w
                or bb.y0 < -0.005 * h or bb.y1 > 1.005 * h):
            print(f"  text overrun: {t.get_text()[:48]!r} x=[{bb.x0:.0f},"
                  f"{bb.x1:.0f}] of {w:.0f}px")
            ok = False
    return ok


def panel_a(fig, grid, draws, interp, p5, p95, pk_ages, tr_ages, ylim):
    ax = fig.add_axes(AX_A)
    apply_paleo_style(ax)
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(ylim)

    # climate windows as alternating neutral bands (behind everything)
    for j, wname in enumerate(s11.WINDOW_ORDER):
        w0, w1 = CLIMATE_WINDOWS[wname]
        ax.add_patch(Rectangle((w0, ylim[0]), w1 - w0, ylim[1] - ylim[0],
                               facecolor=s11.WIN_COL[j % 2],
                               edgecolor="none", zorder=0))

    # the ensemble: 200 thin blue draws + P5-P95 band + bold unperturbed line
    for d in draws:
        ax.plot(grid, d, color=CAT1, lw=DRAW_LW, alpha=DRAW_ALPHA, zorder=1)
    ax.fill_between(grid, p5, p95, color=BAND, alpha=0.45, zorder=2)
    ax.plot(grid, interp, color=INK, lw=1.6, zorder=3)

    # detected features on the unperturbed record
    for ta in pk_ages:
        mark(ax, ta, float(interp[np.argmin(np.abs(grid - ta))]), "o",
             ms=4, mfc=INK, mec=INK, zorder=4)
    for ta in tr_ages:
        mark(ax, ta, float(interp[np.argmin(np.abs(grid - ta))]), "s",
             ms=4.5, mfc=SURFACE, mew=1.0, zorder=4)

    ax.set_xlabel("Age (ka BP)", fontsize=10, color=INK)
    ax.set_ylabel("NGRIP δ¹⁸O (standardized)", fontsize=10, color=INK,
                  labelpad=8)


def panel_b(fig, pk_ages, tr_ages, pk_pos, tr_pos):
    ax = fig.add_axes(AX_B)
    apply_paleo_style(ax)
    ax.set_xlim(SHIFT_LO, SHIFT_HI)
    ax.set_ylim(60, 10)                     # paleo: older ages at the bottom
    ax.set_xticks([-1.0, -0.5, 0, 0.5, 1.0])

    # +/- 1 sigma reference band + zero line (annotations, not data)
    ax.axvspan(-SIGMA_EXPECT, SIGMA_EXPECT, color=BAND, alpha=0.45, zorder=0)
    (zero,) = ax.plot([0, 0], [10, 60], color=INK, lw=0.9, ls=(0, (4, 3)),
                      zorder=1.5)
    zero.set_gid("layer-annotations")

    def feature_row(age, pos):
        w5, w95 = np.percentile(pos, [5, 95])
        ax.plot([w5 - age, w95 - age], [age, age], color=INK, lw=1.6,
                solid_capstyle="butt", zorder=3)
        mark(ax, pos.mean() - age, age, "|", ms=7, mew=1.6, zorder=4)

    for age, pos in zip(pk_ages, pk_pos.T):
        feature_row(age, pos)
        mark(ax, 0, age, "o", ms=4.5, mfc=SURFACE, mew=1.0, zorder=5)
    for age, pos in zip(tr_ages, tr_pos.T):
        feature_row(age, pos)
        mark(ax, 0, age, "s", ms=4.5, mfc=SURFACE, mew=1.0, zorder=5)

    ax.set_xlabel("detected position − nominal (ka)", fontsize=10, color=INK)
    ax.set_ylabel("feature age (ka BP)", fontsize=10, color=INK, labelpad=8)


def plot(pk_ages, tr_ages, pk_pos, tr_pos, draws, interp, p5, p95,
         n_stack_pk, stack_ages, stack_wander):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig = Figure(figsize=(FIG_W / 25.4, FIG_H / 25.4), facecolor="white")

    fig.text(0.048, 0.965, "The age-perturbation ensemble — feature "
             "positions under age error", fontsize=10.5, fontweight="bold",
             color=INK)
    fig.text(0.048, 0.940, f"NGRIP δ¹⁸O (σ = {SIGMA_EXPECT:.2f} ka), "
             f"M = {N_PERTURB} realizations, seed 42, 10–60 ka grid "
             f"(Δt = 0.05 ka)", fontsize=9, color=INK)

    ylim = (float(interp.min()) - 0.35, float(interp.max()) + 0.35)
    panel_a(fig, AGE_GRID_10_60, draws, interp, p5, p95, pk_ages, tr_ages,
            ylim)
    panel_b(fig, pk_ages, tr_ages, pk_pos, tr_pos)

    # footer: symbol decode + measured numbers + provenance (each line short
    # enough to clear the right edge - the 06/07 lesson, QA-checked)
    n_pk, n_tr = len(pk_ages), len(tr_ages)
    w_all = np.concatenate([np.percentile(pk_pos, 95, axis=0)
                            - np.percentile(pk_pos, 5, axis=0),
                            np.percentile(tr_pos, 95, axis=0)
                            - np.percentile(tr_pos, 5, axis=0)])
    b_all = np.concatenate([pk_pos.mean(axis=0) - pk_ages,
                            tr_pos.mean(axis=0) - tr_ages])
    naive = 2 * 1.6448536 * SIGMA_EXPECT
    lines = [
        f"thin blue = the M = {N_PERTURB} age-perturbed draws; bold black "
        f"= the unperturbed record",
        f"● / □ = the {n_pk} peaks / {n_tr} troughs (prominence ≥ "
        f"{PROM_THR:g}, spacing ≥ {SPACING_KA:g} ka; matched ± "
        f"{MATCH_WIN_KA:g} ka)",
        "alternating bands = climate windows GS-12 / GI-12 / H4 / LGM / "
        "H1 / B-A",
        "(b): bars = P5–P95 of the detected position, ticks = mean, "
        "shading = ±1σ",
        f"position spread: P5–P95 widths {w_all.min():.2f}–"
        f"{w_all.max():.2f} ka (median {np.median(w_all):.2f})",
        f"vs 2 × 1.64σ = {naive:.2f} ka for a sharp isolated peak — "
        f"smoothing curvature pins the extrema",
        f"bias: |max| {np.abs(b_all).max():.2f} ka, pooled median "
        f"{np.median(np.abs(b_all)):.2f} ka — no systematic shift",
        f"contrast: the North Atlantic stack (Fig. 2, 13 members) keeps "
        f"{n_stack_pk} local max",
        f"(prominence ≥ {NA_STACK_PROM:g}); its position wanders "
        f"{stack_wander[0]:.1f}–{stack_wander[1]:.1f} ka (P5–P95) across "
        f"draws" if stack_wander else
        f"(prominence ≥ {NA_STACK_PROM:g}); no position is definable",
        "data: NGRIP δ¹⁸O, 10–60 ka grid (step 11 loader); perturbation = "
        "scripts/18/29, seed 42",
    ]
    assert len(lines) == len(FOOTER_Y)
    for y, line in zip(FOOTER_Y, lines):
        fig.text(0.048, y, line, fontsize=9, color=INK)

    assert no_text_overruns(fig), "some text runs off the figure edges"
    save(fig, os.path.join(OUT_DIR, "perturbation_ensemble"))
    print_anchors(pk_ages, tr_ages, pk_pos, tr_pos, draws, n_stack_pk,
                  stack_ages, stack_wander)


# ---------------------------------------------------------------- anchors ----
def print_anchors(pk_ages, tr_ages, pk_pos, tr_pos, draws, n_stack_pk,
                  stack_ages, stack_wander):
    print("\n== anchors (FigS11 perturbation ensemble) ==")
    grid = AGE_GRID_10_60
    print(f"  grid: {grid[0]:g}-{grid[-1]:g} ka, dt = {grid[1] - grid[0]:.2f}"
          f" ka ({len(grid)} points), M = {len(draws)}, seed 42, "
          f"sigma = {SIGMA_EXPECT} ka")
    print(f"  features: {len(pk_ages)} peaks + {len(tr_ages)} troughs = "
          f"{len(pk_ages) + len(tr_ages)} "
          f"(prominence >= {PROM_THR:g}, spacing >= {SPACING_KA:g} ka)")
    w_all = np.concatenate([np.percentile(pk_pos, 95, axis=0)
                            - np.percentile(pk_pos, 5, axis=0),
                            np.percentile(tr_pos, 95, axis=0)
                            - np.percentile(tr_pos, 5, axis=0)])
    b_all = np.concatenate([pk_pos.mean(axis=0) - pk_ages,
                            tr_pos.mean(axis=0) - tr_ages])
    naive = 2 * 1.6448536 * SIGMA_EXPECT
    print(f"  position spread: P5-P95 width {w_all.min():.2f}-"
          f"{w_all.max():.2f} ka, median {np.median(w_all):.2f} ka vs "
          f"sharp-peak 2 x 1.64 sigma = {naive:.2f} ka")
    print(f"  bias: |max| {np.abs(b_all).max():.2f} ka, pooled median "
          f"{np.median(np.abs(b_all)):.2f} ka (no systematic shift)")
    print(f"  North Atlantic stack contrast: {n_stack_pk} local max at "
          f"prominence >= {NA_STACK_PROM:g} "
          f"({'/'.join(f'{a:.1f}' for a in stack_ages)})"
          + (f", wander {stack_wander[0]:.1f}-{stack_wander[1]:.1f} ka "
             f"(P5-P95)" if stack_wander else ""))
    print("  claim: age error repositions sharp features by ~1 sigma with "
          "no bias; stacking buys noise suppression at the price of feature "
          "sharpness (positions then set by width, not sigma).")

    assert len(draws) == N_PERTURB and draws.shape[1] == len(grid)
    assert len(pk_ages) + len(tr_ages) == 25, "feature count drifted"
    assert (w_all > 1.3).all() and (w_all < 1.8).all(), \
        f"a position spread left [1.3, 1.8] ka: {w_all}"
    assert 1.55 < np.median(w_all) < 1.65, "median spread drifted"
    assert np.abs(b_all).max() < 0.5, "a feature bias exceeded 0.5 ka"
    assert np.median(np.abs(b_all)) < 0.1, "pooled bias drift"
    assert n_stack_pk == 1, "stack feature count drifted"


def main():
    interp, sigma, name = load_ngrip()
    grid = AGE_GRID_10_60
    pk_ages, tr_ages = detect_features(interp, grid)
    rng = np.random.default_rng(42)
    draws = perturb_draws(interp, sigma, grid, rng, N_PERTURB)
    pk_pos = tracked_positions(draws, grid, pk_ages)
    tr_pos = tracked_positions(draws, grid, tr_ages, sign=-1)
    p5, p95 = np.percentile(draws, [5, 95], axis=0)
    n_stack_pk, stack_ages, stack_wander = stack_contrast()
    plot(pk_ages, tr_ages, pk_pos, tr_pos, draws, interp, p5, p95,
         n_stack_pk, stack_ages, stack_wander)


if __name__ == "__main__":
    main()
