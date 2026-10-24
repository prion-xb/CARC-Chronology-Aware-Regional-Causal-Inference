# -*- coding: utf-8 -*-
"""Step 14 - sliding-window stability (Fig. 4, Section 4.4).

Claim: the continuous window sweep confirms that honest detection is never
sustained for the NGRIP -> EDC dD pairing - across W = 2/4/8 ka windows the
smoothed TPR peaks at 77.0% / 86.3% / 97.7%, and only 6/241, 14/231, 24/211
raw windows reach the 95% rung (the W = 8 ka curve crosses it only briefly,
near 14 ka); the sub-threshold regime is insensitive to the window width.
FPR stays above the 5% mirror rung almost everywhere (smoothed minima
6.2% / 4.3% / 3.5%), and DDR >= 3 in only 28% / 29% / 27% of windows.

Design (double column, 178 x 200 mm), one metric per panel - the Layout B
reading of the paper's Fig. 4 caption:
  (a) standardized NGRIP and EDC dD series with MIS shading (warm tint for
      odd stages, cool for even; labels at the panel bottom where both
      series stay clear of them)
  (b) TPR panel: raw (faint) + smoothed (sigma = 4 window steps) curves
      for W = 2/4/8 ka (CAT1/CAT2/CAT3), honesty ladder 95-70% + band
  (c) FPR panel: mirror rungs 5-30% + band
  (d) DDR panel: display-clipped to 0-10, DDR = 3 (dashed) and DDR = 1
      (dotted) references labelled at the old-age end
  footer: colour decode, MIS decode, ladder reading, data facts, provenance

Data: outputs/sliding_window_cache.csv (scripts/15; NGRIP -> EDC_dD,
ARC alpha = 0.5, N = 100 perturbations, maxlag 3, step 0.2 ka, grid
10-60 ka at 0.05).  Cache verified bit-identical by full recomputation
on 2026-09-13.  Panel (a) series loaded through scripts/15's own
load_tier1_data, so the plot shows exactly the series the sweep used.

Convergence targets (asserted):
  683 cache rows = 241/231/211 windows at W = 2/4/8
  TPR >= 95% raw windows: 6 / 14 / 24
  smoothed TPR maxima: 0.770 / 0.863 / 0.977 (W = 8 at 14.0 ka)
  smoothed FPR minima: 0.062 / 0.043 / 0.035
  DDR >= 3 window fractions: 28% / 29% / 27%
  (a) series on the 1001-point grid, standardized (std = 1)

Usage:
    python scripts/fignew/14_sliding_window.py
Outputs: figures_new/14_sliding_window/sliding_window.png / .svg
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from scipy.ndimage import gaussian_filter1d

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.dirname(os.path.abspath(__file__)), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from style import (apply_paleo_style, honesty_ladder, save, INK,  # noqa: E402
                   SECONDARY_INK, CAT1, CAT2, CAT3)
from src.config import OUTPUT_DIR  # noqa: E402

# canonical sliding-window engine of scripts/15: its loader gives the exact
# standardized series the sweep consumed (same idiom as 05 -> scripts/22)
SPEC = importlib.util.spec_from_file_location(
    "sw15", os.path.join(ROOT, "scripts", "15_sliding_window_carc.py"))
sw15 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sw15)

OUT_DIR = os.path.join(ROOT, "figures_new", "14_sliding_window")

FIG_W, FIG_H = 178.0, 220.0
AX_L, AX_R = 0.075, 0.985
GRID_T, GRID_B = 0.955, 0.262
FOOTER_Y0 = 0.215          # first footer line, 9pt lines 0.023 apart;
FOOTER_DY = 0.023          # GRID_B leaves ~10 mm for the (d) tick labels +
                           # xlabel below the axes (measured, not guessed)

W_VALS = (2, 4, 8)         # window widths, ka
W_COLORS = {2: CAT1, 4: CAT2, 8: CAT3}
SMOOTH_SIGMA = 4           # window steps (= 0.8 ka), old Fig7's SMOOTH_SIGMA_CS
DDR_CLIP = 10.0            # display clip for the DDR panel

FPR_RUNGS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)   # mirror rungs (08 idiom)

# MIS stages on 10-60 ka: odd = warm, even = cool (paper caption "MIS shading")
MIS = [("1", 10.0, 14.0, "warm"), ("2", 14.0, 29.0, "cool"),
       ("3", 29.0, 57.0, "warm"), ("4", 57.0, 60.0, "cool")]
MIS_FILL = {"warm": "#f6efe3", "cool": "#e9eef2"}


# ------------------------------------------------------------------ data ----
def load_cache():
    """{W: DataFrame(time, tpr, fpr, ddr)} from the scripts/15 cache."""
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "sliding_window_cache.csv"))
    assert list(df.columns) == ["time", "tpr", "fpr", "ddr",
                                "window_width"], "cache schema drifted"
    assert len(df) == 683, f"expected 683 cache rows, got {len(df)}"
    res = {w: df[df.window_width == w].reset_index(drop=True) for w in W_VALS}
    assert len(res[2]) == 241 and len(res[4]) == 231 and len(res[8]) == 211, \
        "window counts drifted"
    return res


def load_series():
    """(grid, {'NGRIP': ..., 'EDC_dD': ...}) - the sweep's own inputs."""
    grid, data = sw15.load_tier1_data()
    for name in ("NGRIP", "EDC_dD"):
        s = data[name]
        assert len(s) == 1001, f"{name}: expected 1001 grid points"
        assert abs(float(s.std()) - 1.0) < 1e-6, f"{name}: not standardized"
    return grid, data


def smooth(v):
    """Gaussian smoothing over the finite part (sigma in index units)."""
    filled = np.interp(np.arange(len(v)),
                       np.where(np.isfinite(v))[0], v[np.isfinite(v)])
    return gaussian_filter1d(filled, sigma=SMOOTH_SIGMA)


# ------------------------------------------------------------------ plot ----
def draw_metric_panel(ax, df, col, ladder=None):
    """One metric panel: raw (faint) + smoothed curves for the three W."""
    if ladder is not None:
        rungs, band = ladder
        honesty_ladder(ax, rungs=rungs, band=band)
    ax.grid(True, color="#d9d8d4", linewidth=0.5, alpha=0.8, zorder=0.6)
    for w in W_VALS:
        g = df[w]
        t, v = g.time.values, g[col].values
        if col == "ddr":
            v = np.clip(v, 0, DDR_CLIP)
        sm = np.clip(smooth(v), 0, DDR_CLIP) if col == "ddr" else smooth(v)
        ax.plot(t, v, color=W_COLORS[w], lw=0.5, alpha=0.22, zorder=1.1)
        ax.plot(t, sm, color=W_COLORS[w], lw=2.0, zorder=1.4)


def panel_series(ax, grid, data):
    """(a) NGRIP + EDC dD with MIS shading; MIS labels at the panel bottom."""
    for name, lo, hi, kind in MIS:
        ax.axvspan(lo, hi, color=MIS_FILL[kind], zorder=0)
        ax.text((lo + hi) / 2, -3.45, f"MIS {name}", ha="center",
                va="bottom", fontsize=9, color=INK, zorder=2)

    ax.plot(grid, data["NGRIP"], color=INK, lw=1.3, zorder=1.2)
    ax.plot(grid, data["EDC_dD"], color=SECONDARY_INK, lw=1.3, zorder=1.3)
    ax.legend(handles=[
        Line2D([], [], color=INK, lw=1.3, label="NGRIP δ¹⁸O"),
        Line2D([], [], color=SECONDARY_INK, lw=1.3, label="EDC δD")],
        loc="upper left", frameon=False, fontsize=9)

    ax.set_xlim(10, 60)
    ax.invert_xaxis()
    ax.set_ylim(-4.0, 4.0)          # EDC peak 3.73 at 11.5 ka stays inside
    ax.set_yticks([-3, 0, 3])
    ax.tick_params(labelbottom=False)
    ax.set_ylabel("Standardized proxy value", fontsize=10, color=INK,
                  labelpad=4)


def plot(results):
    os.makedirs(OUT_DIR, exist_ok=True)
    grid, data = load_series()

    fig = Figure(figsize=(FIG_W / 25.4, FIG_H / 25.4), facecolor="white")
    gs = fig.add_gridspec(4, 1, left=AX_L, right=AX_R, top=GRID_T,
                          bottom=GRID_B, hspace=0.30,
                          height_ratios=[1.5, 1, 1, 1])

    # (a) proxy series + MIS bands
    ax = fig.add_subplot(gs[0])
    apply_paleo_style(ax)
    panel_series(ax, grid, data)
    ax.set_title("(a) Proxy records: NGRIP and EDC δD (standardized)",
                 loc="left", fontsize=10.5, fontweight="bold", color=INK)

    # (b) TPR with the honesty ladder
    ax = fig.add_subplot(gs[1])
    apply_paleo_style(ax)
    draw_metric_panel(ax, results, "tpr",
                      ladder=((0.95, 0.90, 0.85, 0.80, 0.75, 0.70),
                              (0.70, 0.95)))
    ax.set_xlim(10, 60)
    ax.invert_xaxis()
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.tick_params(labelbottom=False)
    ax.set_ylabel("TPR", fontsize=10, color=INK, labelpad=4)
    ax.set_title("(b) True-positive rate (TPR): NGRIP → EDC δD",
                 loc="left", fontsize=10.5, fontweight="bold", color=INK)

    # (c) FPR with the mirror rungs
    ax = fig.add_subplot(gs[2])
    apply_paleo_style(ax)
    draw_metric_panel(ax, results, "fpr",
                      ladder=(FPR_RUNGS, (FPR_RUNGS[0], FPR_RUNGS[-1])))
    ax.set_xlim(10, 60)
    ax.invert_xaxis()
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.tick_params(labelbottom=False)
    ax.set_ylabel("FPR", fontsize=10, color=INK, labelpad=4)
    ax.set_title("(c) False-positive rate (FPR): reverse direction",
                 loc="left", fontsize=10.5, fontweight="bold", color=INK)

    # (d) DDR with the 3 / 1 references
    ax = fig.add_subplot(gs[3])
    apply_paleo_style(ax)
    draw_metric_panel(ax, results, "ddr")
    ref3 = ax.axhline(3.0, color=INK, lw=1.0, ls=(0, (5, 3)), zorder=2)
    ref3.set_gid("layer-annotations")
    ref1 = ax.axhline(1.0, color=SECONDARY_INK, lw=0.8, ls=(0, (2, 3)),
                      zorder=2)
    ref1.set_gid("layer-annotations")
    # label positions sit in spans where all three smoothed curves avoid the
    # label zones (computed: 27.6-35.4 ka avoids 2.75-3.95; 51.8-60 ka avoids
    # 0.7-1.8) - the left margin is crossed by the W = 4 curve at 3.25
    ax.text(31.5, 3.35, "DDR = 3", ha="center", va="center", fontsize=9,
            color=INK, zorder=3)
    ax.text(58.7, 1.35, "DDR = 1", ha="left", va="center", fontsize=9,
            color=INK, zorder=3)
    ax.set_xlim(10, 60)
    ax.invert_xaxis()
    ax.set_ylim(0, DDR_CLIP)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_ylabel("DDR", fontsize=10, color=INK, labelpad=4)
    ax.set_xlabel("Age (ka BP)", fontsize=10, color=INK)
    ax.set_title("(d) Directional dominance ratio (DDR)",
                 loc="left", fontsize=10.5, fontweight="bold", color=INK)

    # footer: colour decode, MIS decode, ladder reading, data facts,
    # provenance (line lengths QA-checked)
    foot = [
        "smoothed curves (σ = 4 window steps) over raw per-window scores "
        "(faint); W = 2/4/8 ka = blue/orange/aqua",
        "panel (a): MIS bands alternate warm (stages 1, 3) / cool "
        "(stages 2, 4); series = the sweep inputs",
        "TPR: windows at/above the 95% rung — 6/241 (W = 2), 14/231 "
        "(W = 4), 24/211 (W = 8)",
        "smoothed TPR maxima 77.0% / 86.3% / 97.7%; the W = 8 curve "
        "crosses the 95% rung only briefly, near 14 ka",
        "FPR: smoothed minima 6.2% / 4.3% / 3.5%; below the 5% mirror "
        "rung briefly at ~12 ka (W = 4), 26–29 ka (W = 8)",
        "DDR: windows ≥ 3 — 28% / 29% / 27%; smoothed maxima 10.0 / "
        "10.0 / 7.6 (display clipped at 10)",
        "honesty ladder: bold rung = 95% (TPR) / 5% (FPR mirror); "
        "lighter rungs fade to 70% (TPR) / 30% (FPR)",
        "shaded zone = sub-threshold region; DDR references at 3 "
        "(dashed) and 1 (dotted)",
        "data: outputs/sliding_window_cache.csv (scripts/15, NGRIP→EDC "
        "δD, α = 0.5, N = 100, maxlag 3, step 0.2 ka)",
    ]
    for k, line in enumerate(foot):
        fig.text(0.048, FOOTER_Y0 - k * FOOTER_DY, line, fontsize=9,
                 color=INK)

    assert no_text_overruns(fig), "some text runs off the figure edges"
    save(fig, os.path.join(OUT_DIR, "sliding_window"))
    print_anchors(results, grid, data)


# ---------------------------------------------------------------- anchors ----
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


def print_anchors(results, grid, data):
    print("\n== anchors (Fig. 4 sliding-window stability) ==")
    n95 = {}
    smax, smin_f = {}, {}
    ddr3 = {}
    for w in W_VALS:
        g = results[w]
        t, tpr, fpr, ddr = (g.time.values, g.tpr.values, g.fpr.values,
                            g.ddr.values)
        n95[w] = int((tpr >= 0.95).sum())
        st = smooth(tpr)
        smax[w] = (float(st.max()), float(t[st.argmax()]))
        smin_f[w] = float(smooth(fpr).min())
        ddr3[w] = float((ddr >= 3).mean())
        print(f"  W={w}: {len(g)} windows, TPR>=95%: {n95[w]}, "
              f"smoothed TPR max {smax[w][0]:.3f} @ {smax[w][1]:.1f} ka, "
              f"smoothed FPR min {smin_f[w]:.3f}, DDR>=3 {ddr3[w]:.1%}")
    print(f"  series: grid {len(grid)} pts, NGRIP std "
          f"{data['NGRIP'].std():.3f}, EDC max {data['EDC_dD'].max():.2f} "
          f"@ 11.5 ka")

    assert n95 == {2: 6, 4: 14, 8: 24}, "raw TPR>=95% counts drifted"
    assert abs(smax[2][0] - 0.770) < 2e-3 and smax[2][1] == 11.0
    assert abs(smax[4][0] - 0.863) < 2e-3 and abs(smax[4][1] - 42.2) < 0.2
    assert abs(smax[8][0] - 0.977) < 2e-3 and abs(smax[8][1] - 14.0) < 0.2
    assert abs(smin_f[2] - 0.062) < 2e-3
    assert abs(smin_f[4] - 0.043) < 2e-3
    assert abs(smin_f[8] - 0.035) < 2e-3
    assert abs(ddr3[2] - 67 / 241) < 1e-9
    assert abs(ddr3[4] - 68 / 231) < 1e-9
    assert abs(ddr3[8] - 56 / 211) < 1e-9
    print("  claim: honest detection is never sustained; the sub-threshold "
          "regime is insensitive to the window width.")


def main():
    results = load_cache()
    plot(results)


if __name__ == "__main__":
    main()
