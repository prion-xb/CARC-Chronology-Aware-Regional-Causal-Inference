# -*- coding: utf-8 -*-
"""Step 17 - escape path (Fig. 7, Section 5.3).

Claim: compressing age errors alone cannot rescue honest detection.  At
0.1x the published 1-sigma age errors the strongest prescribed Scenario A
edge (NGRIP -> EDC dD) reaches 92.5% - still below the 95% honesty rung -
while the worst of the 27 negative controls stays at 54.5% and 0 of 27
controls pass at every scale.  The detectability phase diagram shows why:
the 95% escape boundary sits far from the real operating points, requiring
an injected causal strength of ~0.8 together with <= 0.2x age errors.

Design (double column, 178 x 192 mm), two stacked panels:
  (a) robustness (%) of the three prescribed Scenario A edges vs age-error
      scale 0.1-5x (log axis), CAT1/CAT3/CAT5 with o/s/^ markers (same
      edge colour code as 09); gray band = P5-P95 across the 27 negative
      controls, dashed gray = the worst control; honesty ladder 95-70% +
      band on the S axis; dotted vertical line at 1x = published errors;
      bold readout of the claim in the empty zone below the 70% rung
  (b) detectability phase diagram: NGRIP -> EDC dD robustness on the
      {causal strength} x {lag / error ratio} grid (SEQ ramp contourf,
      #5e97df cap), bold THR_95 contour = the empirical 95% escape
      boundary; the three edges are dotted at strength 0.4 with their
      Table 1 lag/error ratios - all in the sub-threshold regime
  footer: edge decode, band decode, claim reading, grid facts, data
  provenance

Data: outputs/age_error_scaling.csv + age_error_scaling_controls.csv
(scripts/24, run22f config, 20 realizations x 200 perturbations; the
controls file is the per-control dump added to scripts/24 on 2026-09-13,
its summary rows bit-checked against the pre-existing CSV) and
outputs/detectability_phase.csv (scripts/26, 10 realizations x 200).

Convergence targets (asserted):
  1x: edges 6.0 / 6.5 / 4.5%, control max 7.25%, 0/27 pass
  0.1x: NGRIP -> EDC dD 92.5%, control max 54.5%, 0/27 pass
  controls file: 27 curves x 6 scales, per-scale max == summary control_max
  phase grid: 25 cells, (strength 0.4, scale 1.0) = 5.75% (Table 1: 6.0%,
  n_real=10 MC noise), cells >= 95% = {(0.8, 0.2), (1.6, 0.2)} only,
  0 controls pass anywhere; Table 1 ratios 1.1 / 0.9 / 12.1

Usage:
    python scripts/figure_scripts/17_escape_path.py
Outputs: figures_rebuilt/17_escape_path/escape_path.png / .svg
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.dirname(os.path.abspath(__file__)), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from style import (apply_paleo_style, honesty_ladder, save, INK,  # noqa: E402
                   SECONDARY_INK, CAT1, CAT3, CAT5, THR_95)
from src.config import OUTPUT_DIR  # noqa: E402

OUT_DIR = os.path.join(ROOT, "figures_rebuilt", "17_escape_path")

FIG_W, FIG_H = 178.0, 192.0
AX_L, AX_R = 0.075, 0.985
GRID_T, GRID_B = 0.955, 0.285
FOOTER_Y0 = 0.24
FOOTER_DY = 0.023          # 9pt footer lines; GRID_B leaves ~8 mm for the
                           # (b) tick labels + xlabel (measured, not guessed)

EDGES = [("NGRIP -> EDC_dD", CAT1, "o", "NGRIP → EDC δD"),
         ("CO2 -> CH4", CAT3, "s", "CO$_2$ → CH$_4$"),
         ("Cariaco -> Cheng", CAT5, "^", "Cariaco → Cheng")]

SEQ_LO, SEQ_HI = "#e8f0fb", "#5e97df"     # 07's validated SEQ cap (black-text
                                          # contrast >= 6.9:1 on the cap)
SEQ_CMAP = LinearSegmentedColormap.from_list("seq", [SEQ_LO, SEQ_HI])

SCALES = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]


# ------------------------------------------------------------------ data ----
def load_scale():
    """Summary CSV: 3 edges + control max/mean/n_pass95 per scale."""
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "age_error_scaling.csv"))
    assert sorted(df.error_scale.unique()) == SCALES, "scale grid drifted"
    assert len(df) == 6 * 6, f"expected 36 summary rows, got {len(df)}"
    return df


def load_controls():
    """Per-control dump: 27 negative controls per scale (scripts/24, added
    2026-09-13 for the Fig. 7 control band)."""
    c = pd.read_csv(os.path.join(OUTPUT_DIR, "age_error_scaling_controls.csv"))
    assert len(c) == 6 * 27, f"expected 162 control rows, got {len(c)}"
    for s in SCALES:
        sub = c[c.error_scale == s]
        assert len(sub) == 27 and sub.control.nunique() == 27, \
            f"scale {s}: expected 27 distinct controls"
    return c


def load_phase():
    """Detectability grid (scripts/26): NGRIP -> EDC dD robustness over
    {causal strength} x {error scale}, with the lag/error ratio column."""
    p = pd.read_csv(os.path.join(OUTPUT_DIR, "detectability_phase.csv"))
    assert len(p) == 25, f"expected 25 grid cells, got {len(p)}"
    return p


def load_table1():
    """Table 1 (beta = 0.25): NGRIP robustness + the three lag/error ratios
    for the panel (b) operating-point dots."""
    t1 = pd.read_csv(os.path.join(OUTPUT_DIR, "scenarioA_final_table1.csv"))
    t1 = t1[t1.beta == "0.25"]
    edge_names = {e[0] for e in EDGES}
    ratios = {r.causal_edge: float(r.lag_error_ratio) for r in t1.itertuples()
              if r.causal_edge in edge_names}
    ngr = t1[t1.causal_edge.str.startswith("NGRIP")].iloc[0]
    return ratios, float(ngr.robustness_with_error)


# ------------------------------------------------------------------ plot ----
def panel_a(ax, scale, ctl):
    """(a) robustness vs age-error scale for the three edges + control band."""
    honesty_ladder(ax, rungs=(95, 90, 85, 80, 75, 70), band=(70, 95))

    p5 = [np.percentile(ctl[ctl.error_scale == s].robustness, 5) * 100
          for s in SCALES]
    p95 = [np.percentile(ctl[ctl.error_scale == s].robustness, 95) * 100
           for s in SCALES]
    band = ax.fill_between(SCALES, p5, p95, color=SECONDARY_INK, alpha=0.16,
                           lw=0, zorder=1.2)
    band.set_gid("layer-annotations")
    worst = scale[scale.pair == "control_max"].sort_values("error_scale")
    wl = ax.plot(worst.error_scale, worst.robustness * 100, color=SECONDARY_INK,
                 lw=0.9, ls=(0, (4, 3)), zorder=1.4)[0]
    wl.set_gid("layer-annotations")

    for name, c, m, _lab in EDGES:
        sub = scale[scale.pair == name].sort_values("error_scale")
        ax.plot(sub.error_scale, sub.robustness * 100, "-", color=c, lw=2.0,
                marker=m, ms=5, mfc=c, mec=c, zorder=1.6)

    v = ax.axvline(1.0, color=SECONDARY_INK, lw=0.9, ls=(0, (1, 3)), zorder=1.0)
    v.set_gid("layer-annotations")
    ax.text(1.04, 96.5, "published 1σ errors", ha="left", va="bottom",
            fontsize=9, color=INK)

    # bold readout in the empty zone between the data (< 55%) and the 70%
    # rung; the spike (92.5 at 0.1x) stays clear above it
    ax.text(5.3, 62.5, "at 0.1×: strongest edge 92.5% — still below the "
            "95% rung", ha="right", va="center", fontsize=11,
            fontweight="bold", color=INK, zorder=3)

    ax.legend(handles=[Line2D([], [], color=c, marker=m, ms=5, lw=2.0,
                              label=lab) for _, c, m, lab in EDGES],
              loc="center right", bbox_to_anchor=(0.985, 0.40), frameon=False,
              fontsize=9)

    ax.set_xscale("log")
    ax.set_xlim(0.09, 5.5)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(labelbottom=False)
    ax.set_ylabel("Robustness (%)", fontsize=10, color=INK, labelpad=4)
    ax.set_title("(a) Robustness vs age-error scale — the three Scenario A "
                 "edges", loc="left", fontsize=10.5, fontweight="bold",
                 color=INK)


def panel_b(ax, phase, ratios):
    """(b) detectability phase diagram with the 95% escape boundary."""
    piv = phase.pivot_table(index="causal_strength", columns="lag_error_ratio",
                            values="robustness_focus")
    piv = piv.sort_index(ascending=True)
    piv = piv[sorted(piv.columns, reverse=True)]     # high precision left
    X, Y = np.meshgrid(piv.columns.values, piv.index.values)
    Z = piv.values

    cs = ax.contourf(X, Y, Z, levels=[0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0],
                     cmap=SEQ_CMAP, vmin=0, vmax=1.0, extend="both", zorder=1)
    cs.set_gid("layer-data")
    cb = ax.figure.colorbar(cs, ax=ax, fraction=0.046, pad=0.03,
                            ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cb.set_label("Robustness", fontsize=10, color=INK)
    cb.ax.tick_params(labelsize=9, colors=INK)

    ax.contour(X, Y, Z, levels=[0.95], colors=THR_95, linewidths=2.2,
               linestyles=[(0, (5, 3))], zorder=2)

    for name, c, m, _lab in EDGES:
        ax.plot(ratios[name], 0.4, "o", ms=8, mfc=c, mec="white", mew=1.4,
                zorder=4)

    ax.text(4.6, 1.25, "escape region\nrobustness ≥ 95%", ha="center",
            va="center", fontsize=9, color=INK, zorder=5)
    ax.text(2.5, 0.95, "escape boundary\n(robustness = 95%)", ha="left",
            va="center", fontsize=9, color=INK, zorder=5)
    ax.text(0.42, 0.22, "sub-threshold regime", ha="center", va="center",
            fontsize=9, color=INK, zorder=5)

    ax.set_xscale("log")
    ax.set_xlim(0.2, 13)
    ax.set_yscale("log")
    ax.set_ylim(0.1, 1.6)
    ax.set_yticks([0.1, 0.2, 0.4, 0.8, 1.6])
    ax.set_xlabel("Lag / RMS age error (relative chronological precision)",
                  fontsize=10, color=INK)
    ax.set_ylabel("Injected causal strength (β)", fontsize=10, color=INK,
                  labelpad=4)
    ax.set_title("(b) Detectability phase diagram — NGRIP → EDC δD",
                 loc="left", fontsize=10.5, fontweight="bold", color=INK)


def plot(scale, ctl, phase, ratios, t1_ngr):
    os.makedirs(OUT_DIR, exist_ok=True)

    fig = Figure(figsize=(FIG_W / 25.4, FIG_H / 25.4), facecolor="white")
    gs = fig.add_gridspec(2, 1, left=AX_L, right=AX_R, top=GRID_T,
                          bottom=GRID_B, hspace=0.30,
                          height_ratios=[1.0, 1.1])

    ax = fig.add_subplot(gs[0])
    apply_paleo_style(ax)
    panel_a(ax, scale, ctl)

    ax = fig.add_subplot(gs[1])
    apply_paleo_style(ax)
    panel_b(ax, phase, ratios)

    foot = [
        "(a) three Scenario A edges (run22f config, 20 realizations × 200 "
        "perturbations each)",
        "gray band = P5–P95 across the 27 controls, dashed = worst control; "
        "0 of 27 pass the 95% rung",
        "at 0.1×: strongest edge 92.5%, worst control 54.5% — age-error "
        "compression alone cannot escape",
        "(b) NGRIP → EDC δD on a 5×5 grid (10 realizations × 200); bold "
        "contour = the 95% escape boundary",
        "dots = the three edges at β = 0.4 and their Table 1 lag/error "
        "ratios (1.1 / 0.9 / 12.1)",
        "data: outputs/age_error_scaling.csv + _controls (scripts/24) and "
        "detectability_phase.csv (scripts/26)",
    ]
    for k, line in enumerate(foot):
        assert len(line) <= 107, f"footer line {k} too long: {len(line)}"
        fig.text(0.048, FOOTER_Y0 - k * FOOTER_DY, line, fontsize=9,
                 color=INK)

    assert no_text_overruns(fig), "some text runs off the figure edges"
    save(fig, os.path.join(OUT_DIR, "escape_path"))
    print_anchors(scale, ctl, phase, ratios, t1_ngr)


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


def print_anchors(scale, ctl, phase, ratios, t1_ngr):
    print("\n== anchors (Fig. 7 escape path) ==")
    v1 = {r.pair: r.robustness for r in scale[scale.error_scale == 1.0]
          .itertuples()}
    v01 = {r.pair: r.robustness for r in scale[scale.error_scale == 0.1]
           .itertuples()}
    print(f"  1x: { {k: round(v, 4) for k, v in v1.items()} }")
    print(f"  0.1x: { {k: round(v, 4) for k, v in v01.items()} }")

    assert abs(v1["NGRIP -> EDC_dD"] - 0.06) < 1e-9, "1x NGRIP drifted"
    assert abs(v1["CO2 -> CH4"] - 0.065) < 1e-9, "1x CO2 drifted"
    assert abs(v1["Cariaco -> Cheng"] - 0.045) < 1e-9, "1x Cariaco drifted"
    assert abs(v1["control_max"] - 0.0725) < 1e-9, "1x control max drifted"
    assert int(v1["control_n_pass95"]) == 0
    assert abs(v01["NGRIP -> EDC_dD"] - 0.925) < 1e-9, "0.1x NGRIP drifted"
    assert abs(v01["control_max"] - 0.545) < 1e-9, "0.1x control max drifted"
    assert int(v01["control_n_pass95"]) == 0

    for s in SCALES:
        sub = ctl[ctl.error_scale == s]
        want = scale[(scale.error_scale == s)
                     & (scale.pair == "control_max")].iloc[0].robustness
        assert abs(sub.robustness.max() - want) < 1e-12, \
            f"scale {s}: per-control max != summary control_max"
        assert (sub.robustness >= 0.95).sum() == 0, \
            f"scale {s}: a control passes the 95% rung"
        assert int(scale[(scale.error_scale == s)
                         & (scale.pair == "control_n_pass95")]
                   .iloc[0].robustness) == 0
    q = np.percentile(ctl[ctl.error_scale == 1.0].robustness, [5, 95])
    print(f"  control band at 1x: P5 {q[0]:.3f} / P95 {q[1]:.3f} "
          f"(n = 27)")

    cell = phase[(phase.causal_strength == 0.4)
                 & (phase.error_scale == 1.0)].iloc[0]
    print(f"  phase cell (β = 0.4, 1x): {cell.robustness_focus:.4f} vs "
          f"Table 1 NGRIP {t1_ngr:.4f} (grid n_real = 10, MC noise)")
    assert abs(cell.robustness_focus - 0.0575) < 1e-9, "phase cell drifted"
    assert (phase.control_n_pass95 == 0).all(), "a control passes in the grid"
    ge95 = phase[phase.robustness_focus >= 0.95]
    got = set(zip(ge95.causal_strength, ge95.error_scale))
    assert got == {(0.8, 0.2), (1.6, 0.2)}, f"escape cells drifted: {got}"
    assert ratios == {"NGRIP -> EDC_dD": 1.1, "CO2 -> CH4": 0.9,
                      "Cariaco -> Cheng": 12.1}, f"Table 1 ratios drifted"
    print(f"  escape cells (>= 95%): {sorted(got)} | Table 1 ratios "
          f"{ratios}")
    print("  claim: age-error compression alone cannot escape - 92.5% at "
          "0.1x is still below the rung, and the boundary needs β ~ 0.8 "
          "with <= 0.2x errors.")


def main():
    scale = load_scale()
    ctl = load_controls()
    phase = load_phase()
    ratios, t1_ngr = load_table1()
    plot(scale, ctl, phase, ratios, t1_ngr)


if __name__ == "__main__":
    main()
