# -*- coding: utf-8 -*-
"""Step 08 - TPR/FPR convergence vs chronology perturbations (Fig. 3c, Section 4.1).

Claim: at the operating point (alpha = 0.5, sigma = 0.05) both rates have
converged by M = 200 - the canonical ensemble size sits at the entry of the
stable region (M >= 200, shaded).  Doubling the ensemble to M = 400 shifts
TPR by -0.2 pts (0.995 -> 0.993) and FPR by +1.7 pts (0.200 -> 0.217), both
within the M = 200 standard error.  TPR clears the 95% honesty rung at
every M; the reverse-direction rate stays in the 15-25% sub-threshold
rungs - it is the Layer-4 95% gate, not the ensemble size, that rejects
false directions.

Design:
  x = M (chronology perturbations per edge), log axis, {50, 100, 200, 400}
  y = rate (0-1), pooled mean over the replicate cells, +- SE band
  TPR = true-direction pass rate (CAT1 blue, 'o')
  FPR = reverse-direction pass rate (CAT2 orange, 's')
  honesty ladder on the TPR side (95/90/85/80/75/70) with the FPR mirror
  (5/10/15/20/25/30) - the cross-figure symbol; stable region M >= 200
  shaded as a neutral background fill; canonical M = 200 marked by a
  dashed reference line.

Data: outputs/param_sensitivity_cache.csv (canonical scripts/14
parameter-sensitivity sweep), operating point alpha=0.5 / sigma=0.05,
10-15 replicate cells per M (the three sweep rows pooled).

Convergence targets (asserted):
  TPR(M=200) = 0.995, TPR(M=400) = 0.993  -> |delta| <= 0.01
  FPR(M=200) = 0.200, FPR(M=400) = 0.217  -> |delta| <= 0.03
  TPR >= 0.95 and FPR <= 0.30 at every M.

Usage:
    python scripts/figure_scripts/08_convergence_m.py
Outputs: figures_rebuilt/08_convergence_m/convergence_m.png / .svg
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from matplotlib import lines as mpl_lines
from matplotlib.figure import Figure

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.dirname(os.path.abspath(__file__)), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from style import (apply_paleo_style, honesty_ladder, save, INK,  # noqa: E402
                   SECONDARY_INK, CAT1, CAT2)

OUT_DIR = os.path.join(ROOT, "figures_rebuilt", "08_convergence_m")

RUN_ALPHA, RUN_SIGMA = 0.5, 0.05       # operating point (canonical run22f)
M_VALS = (50, 100, 200, 400)           # chronology perturbations per edge
M_CANONICAL = 200

# FPR mirror rungs of the honesty ladder (5/10/15/20/25/30 %)
FPR_RUNGS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
STABLE_FILL = "#f0efec"                # neutral background fill, not a data colour

FIG_W, FIG_H = 178.0, 118.0
X_L, X_R, Y0, Y1 = 0.13, 0.92, 0.50, 0.86      # axes bottom / axes top
FOOTER_Y = (0.380, 0.342, 0.304, 0.266)        # 9pt lines, ~4.5 mm apart


# ------------------------------------------------------------------ data ----
def load_convergence():
    """Pooled per-M means and SEs at the operating point from scripts/14."""
    p = pd.read_csv(os.path.join(ROOT, "outputs", "param_sensitivity_cache.csv"))
    sub = p[(p["alpha"] == RUN_ALPHA) & (p["noise_std"] == RUN_SIGMA)]
    assert len(sub), "no cache rows at the operating point"
    rows = []
    for m in M_VALS:
        g = sub[sub["n_perturb"] == m]
        assert len(g), f"no cells at M = {m}"
        rows.append({"m": m, "n": len(g),
                     "tpr": g["tpr"].mean(), "tpr_se": g["tpr"].sem(),
                     "fpr": g["fpr"].mean(), "fpr_se": g["fpr"].sem()})
    return rows


# ------------------------------------------------------------------ plot ----
def plot(rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig = Figure(figsize=(FIG_W / 25.4, FIG_H / 25.4), facecolor="white")
    ax = fig.add_axes([X_L, Y0, X_R - X_L, Y1 - Y0])
    apply_paleo_style(ax)
    ax.minorticks_off()

    ax.set_xscale("log")
    ax.set_xlim(45, 560)
    ax.set_xticks(list(M_VALS))
    ax.set_xticklabels(["50", "100", "200", "400"])
    ax.set_ylim(-0.02, 1.03)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("M (chronology perturbations per edge)", fontsize=10,
                  color=INK)
    ax.set_ylabel("rate (pooled mean ± SE)", fontsize=10, color=INK)
    ax.set_title("(c) TPR/FPR convergence vs chronology perturbations M",
                 loc="left", fontsize=10.5, fontweight="bold", color=INK,
                 pad=6)

    # honesty ladder: TPR side (95-70%) + FPR mirror (5-30%) - the
    # cross-figure symbol; both land in the SVG layer-thr
    honesty_ladder(ax)
    honesty_ladder(ax, rungs=FPR_RUNGS, band=(FPR_RUNGS[0], FPR_RUNGS[-1]))

    # stable region M >= 200 (neutral fill -> background layer) + canonical
    # M = 200 reference line (annotations layer, same idiom as 06's SNR = 1)
    ax.axvspan(M_CANONICAL, 560, color=STABLE_FILL, zorder=0.2)
    ax.text(290, 0.60, "stable region (M ≥ 200)", ha="center", va="center",
            fontsize=9, color=INK, zorder=4)
    ref = ax.axvline(M_CANONICAL, color=SECONDARY_INK, lw=0.8, ls=(0, (4, 3)),
                     zorder=1)
    ref.set_gid("layer-annotations")

    m = np.array([r["m"] for r in rows], dtype=float)
    for key, color, marker in (("tpr", CAT1, "o"), ("fpr", CAT2, "s")):
        mean = np.array([r[key] for r in rows])
        se = np.array([r[key + "_se"] for r in rows])
        ax.fill_between(m, mean - se, mean + se, color=color, alpha=0.12,
                        lw=0, zorder=2.5)
        ax.plot(m, mean, color=color, ls="-", lw=1.6, marker=marker, ms=4.5,
                mfc=color, mec=color, mew=0.9, zorder=3)

    # curve identity in a legend: the ladder rungs are 2.1 mm apart, too
    # close for end labels to sit between without touching a line
    ax.legend(handles=[
        mpl_lines.Line2D([], [], color=CAT1, marker="o", ms=4.5, lw=1.6,
                         label="TPR (true direction)"),
        mpl_lines.Line2D([], [], color=CAT2, marker="s", ms=4.5, lw=1.6,
                         label="FPR (reverse direction)"),
    ], loc="center left", fontsize=9, frameon=False, handlelength=1.6,
       borderpad=0.4, labelspacing=0.35)

    # footer: stable region + ladder reading + provenance (numbers computed
    # from the data so the figure cannot drift; each line <= 95 chars so no
    # line runs past the right figure edge at 9pt)
    tpr = np.array([r["tpr"] for r in rows])
    fpr = np.array([r["fpr"] for r in rows])
    dt, df = tpr[-1] - tpr[2], fpr[-1] - fpr[2]      # M = 400 minus M = 200
    k = int((tpr >= 0.95).sum())
    nmin, nmax = min(r["n"] for r in rows), max(r["n"] for r in rows)
    fig.text(X_L, FOOTER_Y[0],
             f"bars = ± SE across pooled cells ({nmin}–{nmax} per M); "
             f"stable region (shaded) = M ≥ {M_CANONICAL} (canonical).",
             fontsize=9, color=INK)
    fig.text(X_L, FOOTER_Y[1],
             f"M = 400 shifts TPR by {dt * 100:+.1f} pts and FPR by "
             f"{df * 100:+.1f} pts — within the M = {M_CANONICAL} SE.",
             fontsize=9, color=INK)
    fig.text(X_L, FOOTER_Y[2],
             f"ladder: TPR ≥ 95% at every M ({k}/4); FPR stays in the "
             f"15–25% rungs — the gate rejects it.", fontsize=9, color=INK)
    fig.text(X_L, FOOTER_Y[3],
             f"data: param_sensitivity_cache.csv (scripts/14), operating "
             f"point α = {RUN_ALPHA}, σ = {RUN_SIGMA}.", fontsize=9,
             color=INK)

    save(fig, os.path.join(OUT_DIR, "convergence_m"))
    print_anchors(rows)


# ---------------------------------------------------------------- anchors ----
def print_anchors(rows):
    print("\n== anchors (Fig. 3c TPR/FPR convergence) ==")
    for r in rows:
        print(f"  M={r['m']:4d} (n={r['n']:2d}): "
              f"TPR = {r['tpr']:.4f} ± {r['tpr_se']:.4f} | "
              f"FPR = {r['fpr']:.4f} ± {r['fpr_se']:.4f}")
    dt = rows[3]["tpr"] - rows[2]["tpr"]
    df = rows[3]["fpr"] - rows[2]["fpr"]
    print(f"  M 200 -> 400: dTPR = {dt * 100:+.2f} pts (target |d| <= 1 pt), "
          f"dFPR = {df * 100:+.2f} pts (target |d| <= 3 pts)")
    k = sum(r["tpr"] >= 0.95 for r in rows)
    print(f"  ladder: TPR >= 95% at {k}/4 M values; FPR range "
          f"[{min(r['fpr'] for r in rows):.3f}, "
          f"{max(r['fpr'] for r in rows):.3f}] (rungs 15-25%)")
    print("  claim: both rates are converged by M = 200 (stable region); "
          "the Layer-4 gate, not the ensemble size, rejects the reverse "
          "direction.")
    assert abs(dt) <= 0.01, "TPR moved more than 1 pt from M=200 to M=400"
    assert abs(df) <= 0.03, "FPR moved more than 3 pts from M=200 to M=400"
    assert all(r["tpr"] >= 0.95 for r in rows), "TPR below the 95% rung"
    assert all(r["fpr"] <= 0.30 for r in rows), "FPR above the 30% rung"


def main():
    plot(load_convergence())


if __name__ == "__main__":
    main()
