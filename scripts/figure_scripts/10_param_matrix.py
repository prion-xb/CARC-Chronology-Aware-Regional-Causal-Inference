# -*- coding: utf-8 -*-
"""Step 10 - parameter recommendation matrix (Fig. 3d, Section 4.1).

Claim: the recommended pipeline configuration is the objective-selected
one.  On every axis of the 144-config screening grid the recommended
option attains the lowest control FPR achievable with that option fixed,
so the recommendation follows from the grid, not from prior settings.
The paired true-edge robustness at those cells stays at 6-17% - far below
the 95% honesty rung; grid-wide, robustness >= 90% occurs only at control
FPR >= 78.8% (no differencing + min-p F), so sensitivity is available only
at the price of false-positive control, and the lexicographic objective
(min control FPR, then max robustness) by design trades TPR for FPR.

Design:
  a matrix without axes (table look): rows = the five pipeline axes,
  per-row option header band, each cell = control FPR (top, bold if it is
  the row's lowest) + true-edge robustness (bottom).  The
  objective-selected cell (run22f: alpha 0.5 / force / aic_f / test_joint
  / maxlag 50) is tinted light blue and starred.  Footer carries the data
  facts + the honesty-ladder reading (the ladder itself is not drawn:
  step 10 is not in the ladder figure list, same idiom as 08/09).

Data: outputs/scenarioA_final_grid.csv (scripts/22, 144-config screening,
4 realizations x 100 perturbations per config); the final Table 1 numbers
at the selected config come from the 20 x 200 full run in
outputs/scenarioA_final_table1_full_beta0.25.csv.

Convergence targets (asserted):
  selected row = (0.5, force, aic_f, test_joint, 50), the global
    lexicographic argmin (err_ctl_mean = 0.0593, err_ctl_pass95 = 0)
  per axis: the recommended option's best err_ctl_mean is the strict row
    minimum on all five axes
  Table 1 at beta = 0.25: 6.0 / 6.5 / 4.5 % (+- 0.5 pts), 0/30 directed
    pairs pass the 95% rung
  grid facts: 36/144 configs keep every control cell below the 95% rung
    (the selected config is one); the 19 configs with robustness >= 90%
    all have control FPR >= 0.78

Usage:
    python scripts/figure_scripts/10_param_matrix.py
Outputs: figures_rebuilt/10_param_matrix/param_matrix.png / .svg
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.dirname(os.path.abspath(__file__)), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from style import save, INK, SURFACE  # noqa: E402

OUT_DIR = os.path.join(ROOT, "figures_rebuilt", "10_param_matrix")

# pipeline axes: (row label, grid column, option display labels)
AXES = [
    ("ARC strength α", "arc_alpha", ["0", "0.5", "1"]),
    ("differencing", "diff", ["none", "adf", "force"]),
    ("lag test", "test", ["min-p F", "AIC-lag F"]),
    ("perturbation scheme", "pert",
     ["gen_only", "test_var", "test_joint", "gen_and_test"]),
    ("lag horizon", "maxlag", ["5", "50"]),
]
# display label -> grid value (grid uses snake_case keys for the test axis)
DISPLAY_TO_KEY = {"min-p F": "minp_f", "AIC-lag F": "aic_f"}
# canonical run22f (objective-selected) configuration
SELECTED = {"arc_alpha": 0.5, "diff": "force", "test": "aic_f",
            "pert": "test_joint", "maxlag": 50}

SEL_FILL = "#e3eefb"       # light blue cell tint (not a palette role colour)
CELL_EDGE = "#c9c8c3"      # neutral cell border (not a palette role colour)

FIG_W, FIG_H = 178.0, 118.0
AX_X, AX_Y, AX_W, AX_H = 0.315, 0.26, 0.64, 0.606
XLO, XHI = -1.5, 4.15      # cells occupy x in [0, 4]; labels left of 0
YTOP = 6.5                 # ylim top; row pitch = 1.3 (0.3 header + 1 cell)
FOOTER_Y = (0.225, 0.190, 0.155, 0.120, 0.085, 0.050)


# ------------------------------------------------------------------ data ----
def _opt_key(col, label):
    """Grid value for a display label; float/int columns compare by value."""
    label = DISPLAY_TO_KEY.get(label, label)
    if col == "arc_alpha":
        return float(label)
    if col == "maxlag":
        return int(label)
    return label


def _opt_mask(g, col, label):
    """Row mask for one option label (display or grid value)."""
    return g[col] == _opt_key(col, label)


def best_cell(g, col, label):
    """The grid config attaining the lowest control FPR with this option
    fixed (lexicographic tiebreak, mirroring scripts/22's sort)."""
    sub = g[_opt_mask(g, col, label)].sort_values(
        by=["err_ctl_mean", "err_ctl_max", "err_true_mean"],
        ascending=[True, True, False])
    assert len(sub), f"no grid rows for {col} = {label}"
    return sub.iloc[0]


def load_grid():
    g = pd.read_csv(os.path.join(ROOT, "outputs", "scenarioA_final_grid.csv"))
    assert len(g) == 144, "grid is not the 144-config screen"
    sel = g.iloc[0]
    assert (sel["arc_alpha"], sel["diff"], sel["test"], sel["pert"],
            int(sel["maxlag"])) == (SELECTED["arc_alpha"], SELECTED["diff"],
                                    SELECTED["test"], SELECTED["pert"],
                                    SELECTED["maxlag"]), "grid row 0 drifted"
    # the per-axis cells shown in the matrix
    cells = {}
    for name, col, opts in AXES:
        for label in opts:
            cells[(name, label)] = best_cell(g, col, label)
    return g, cells


def load_table1():
    """Final full-run numbers at the selected config (20 x 200), beta 0.25."""
    f = os.path.join(ROOT, "outputs",
                     "scenarioA_final_table1_full_beta0.25.csv")
    df = pd.read_csv(f)
    assert len(df) == 30, "Table 1 full run is not the 30 directed pairs"
    expect = {("NGRIP", "EDC_dD"): 0.060, ("CO2", "CH4"): 0.065,
              ("Cariaco", "Cheng"): 0.045}
    got = {}
    for (s, t), v in expect.items():
        row = df[(df["source"] == s) & (df["target"] == t)]
        assert len(row) == 1, f"Table 1 missing {s} -> {t}"
        got[(s, t)] = float(row["robustness_with_error_median"].iloc[0])
    return got, int(df["passes_95_with_error"].sum())


# ------------------------------------------------------------------ plot ----
def mark(ax, x, y, **kw):
    """Marker that lands in the SVG markers layer (07's idiom)."""
    (line,) = ax.plot([x], [y], **kw)
    line.set_gid("layer-markers")
    return line


def no_text_overruns(fig):
    """Geometry QA: every text stays inside the figure (0.5% tolerance).

    Measured on the real renderer, so a footer line that would run off the
    right edge (the 06/07 lesson) is caught here, not in Illustrator."""
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


def plot(g, cells, table1, n_passes):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig = Figure(figsize=(FIG_W / 25.4, FIG_H / 25.4), facecolor="white")
    ax = fig.add_axes([AX_X, AX_Y, AX_W, AX_H])
    ax.set_xlim(XLO, XHI)
    ax.set_ylim(0, YTOP)
    ax.axis("off")

    fig.text(0.04, 0.930, "(d) Parameter recommendation matrix "
             "(144-config screening grid)", fontsize=10.5,
             fontweight="bold", color=INK)
    fig.text(0.04, 0.895,
             "star = objective-selected configuration (run22f) · bold = "
             "lowest control FPR on that axis", fontsize=9, color=INK)

    for i, (name, col, opts) in enumerate(AXES):
        bottom = YTOP - 1.3 * (i + 1)          # cell bottom (top row first)
        bests = [float(cells[(name, o)]["err_ctl_mean"]) for o in opts]
        row_min = min(bests)
        ax.text(-0.06, bottom + 0.5, name, ha="right", va="center",
                fontsize=9, color=INK)
        for j, label in enumerate(opts):
            r = cells[(name, label)]
            ctl, tpr = float(r["err_ctl_mean"]), float(r["err_true_mean"])
            is_min = ctl == row_min
            # the selected config is the row minimum on every axis
            # (asserted in print_anchors); detect it by the option label
            is_sel = (label == _sel_label(col))
            ax.add_patch(Rectangle((j, bottom), 1.0, 1.0,
                                   facecolor=SEL_FILL if is_sel else SURFACE,
                                   edgecolor=CELL_EDGE, linewidth=0.6))
            ax.text(j + 0.5, bottom + 1.15, label, ha="center", va="center",
                    fontsize=9, color=INK)
            ax.text(j + 0.5, bottom + 0.66, f"{ctl * 100:.1f}%", ha="center",
                    va="center", fontsize=9, color=INK,
                    fontweight="bold" if is_min else "normal")
            ax.text(j + 0.5, bottom + 0.34, f"{tpr * 100:.1f}%", ha="center",
                    va="center", fontsize=9, color=INK)
            if is_sel:
                mark(ax, j + 0.85, bottom + 0.85, marker="*", ms=11,
                     mfc=INK, mec=INK, ls="none", zorder=4)

    # footer: cell definition + objective + ladder reading + Table 1
    # (numbers computed from the data so the figure cannot drift; each line
    # kept short enough to clear the right edge - the 06/07 lesson)
    n_clean = int((g["err_ctl_pass95"] == 0).sum())
    hi = g[g["err_true_mean"] >= 0.9]
    hi_min = float(hi["err_ctl_mean"].min())
    fig.text(0.04, FOOTER_Y[0],
             "cell top: lowest control FPR over the 144-config grid with "
             "that option fixed (bold = axis minimum);",
             fontsize=9, color=INK)
    fig.text(0.04, FOOTER_Y[1],
             "cell bottom: true-edge robustness of the same config — "
             "4 realizations × 100 perturbations per config.",
             fontsize=9, color=INK)
    fig.text(0.04, FOOTER_Y[2],
             "lexicographic objective (min control FPR, then max "
             "robustness) → α 0.5 / force / AIC-lag F / test_joint / "
             "maxlag 50 (star).", fontsize=9, color=INK)
    fig.text(0.04, FOOTER_Y[3],
             f"ladder: {n_clean}/144 configs keep every control cell below "
             f"the 95% rung (the star is one of them);",
             fontsize=9, color=INK)
    fig.text(0.04, FOOTER_Y[4],
             f"grid-wide, true-edge robustness ≥ 90% occurs only at "
             f"control FPR ≥ {hi_min * 100:.1f}%.",
             fontsize=9, color=INK)
    fig.text(0.04, FOOTER_Y[5],
             f"Table 1 at the star (20 × 200): "
             f"{table1[('NGRIP', 'EDC_dD')] * 100:.1f}/"
             f"{table1[('CO2', 'CH4')] * 100:.1f}/"
             f"{table1[('Cariaco', 'Cheng')] * 100:.1f}%, {n_passes}/27 "
             f"controls pass 95% — scenarioA_final_grid.csv (scripts/22).",
             fontsize=9, color=INK)

    assert no_text_overruns(fig), "some text runs off the figure edges"
    save(fig, os.path.join(OUT_DIR, "param_matrix"))
    print_anchors(g, cells, table1, n_passes)


def _sel_label(col):
    """Display label of the selected option on this axis."""
    val = SELECTED[col]
    if col == "arc_alpha":
        return f"{val:g}"
    for disp, key in DISPLAY_TO_KEY.items():
        if key == val:
            return disp
    return str(val)


# ---------------------------------------------------------------- anchors ----
def print_anchors(g, cells, table1, n_passes):
    print("\n== anchors (Fig. 3d parameter recommendation matrix) ==")
    sel = g.iloc[0]
    print(f"  selected (grid row 0): alpha={sel['arc_alpha']}, "
          f"diff={sel['diff']}, test={sel['test']}, pert={sel['pert']}, "
          f"maxlag={int(sel['maxlag'])} | err_ctl_mean={sel['err_ctl_mean']:.4f}, "
          f"err_ctl_max={sel['err_ctl_max']:.3f}, "
          f"err_true_mean={sel['err_true_mean']:.4f}, "
          f"pass95={sel['err_ctl_pass95']:.3f}")
    for name, col, opts in AXES:
        vals = []
        for label in opts:
            r = cells[(name, label)]
            vals.append(f"{label}={r['err_ctl_mean'] * 100:.2f}%"
                        f"/{r['err_true_mean'] * 100:.2f}%")
        print(f"  {name:20s} " + "  ".join(vals))
    rmin = {name: min(float(cells[(name, o)]["err_ctl_mean"])
                      for o in opts) for name, _, opts in AXES}
    for name, col, opts in AXES:
        sel_cell = cells[(name, _sel_label(col))]
        assert float(sel_cell["err_ctl_mean"]) == rmin[name], \
            f"{name}: selected option is not the axis minimum"
        others = [float(cells[(name, o)]["err_ctl_mean"])
                  for o in opts if o != _sel_label(col)]
        assert all(v > rmin[name] for v in others), \
            f"{name}: axis minimum is not unique"
    print(f"  per axis: selected option attains the strict minimum "
          f"control FPR on all {len(AXES)} axes")
    n_clean = int((g["err_ctl_pass95"] == 0).sum())
    hi = g[g["err_true_mean"] >= 0.9]
    hi_min = float(hi["err_ctl_mean"].min())
    print(f"  grid facts: {n_clean}/144 configs have no control cell >= "
          f"95%; robustness >= 90% in {len(hi)} configs, min control FPR "
          f"among them {hi_min:.4f}")
    none = cells[("differencing", "none")]
    adf = cells[("differencing", "adf")]
    print(f"  note: differencing 'none' best = "
          f"{none['err_ctl_mean']:.4f}/{none['err_true_mean']:.4f} "
          f"(cfg alpha={none['arc_alpha']}, test={none['test']}) equals "
          f"'adf' best {adf['err_ctl_mean']:.4f}/{adf['err_true_mean']:.4f} "
          f"(cfg alpha={adf['arc_alpha']}) - ADF flags no differencing on "
          f"ARC residuals here")
    print(f"  Table 1 at beta=0.25 (20 x 200): "
          f"NGRIP->EDC_dD {table1[('NGRIP', 'EDC_dD')] * 100:.1f}%, "
          f"CO2->CH4 {table1[('CO2', 'CH4')] * 100:.1f}%, "
          f"Cariaco->Cheng {table1[('Cariaco', 'Cheng')] * 100:.1f}%, "
          f"{n_passes}/30 pass 95%")
    print("  claim: the recommendation is the objective-selected config; "
          "on every axis it attains the lowest control FPR of the grid, "
          "and sensitivity is only available at the price of FPR control.")
    assert abs(sel["err_ctl_mean"] - 0.059259259) < 1e-9
    assert sel["err_ctl_pass95"] == 0.0
    assert n_clean == 36, "clean-config count drifted"
    assert len(hi) == 19 and hi_min >= 0.78, "high-TPR trade-off drifted"
    for s, t, v in (("NGRIP", "EDC_dD", 0.060), ("CO2", "CH4", 0.065),
                    ("Cariaco", "Cheng", 0.045)):
        assert abs(table1[(s, t)] - v) <= 0.005, f"Table 1 {s} -> {t} drifted"
    assert n_passes == 0, "Table 1 passes drifted"


def main():
    g, cells = load_grid()
    table1, n_passes = load_table1()
    plot(g, cells, table1, n_passes)


if __name__ == "__main__":
    main()
