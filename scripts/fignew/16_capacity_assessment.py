# -*- coding: utf-8 -*-
"""Step 16 - causal-inference capacity assessment (new contribution figure).

Claim: at the current data quality, what can each region pair / record
support?  Regional level: 0 of the 180 pair-windows reaches the 95% rung
and only one reaches even the 70% rung (North Atlantic -> Tropical in
H4 = 78% vs 66%, the Fig. 6 anchor).  Record level: the strongest honest
(forced-difference) pairing per record spans S_max = 0.27-0.92, and
reaching the 95% rung would need a 1.2x-6.2x compression of the age
errors (median 4.9x) even under the most favourable error-sensitivity
shape observed in Scenario A.

Design (user decisions 2026-09-13: pair x window grid for N1; best-case
NGRIP->EDC relative-gain shape, multiplicative model, for the N2
conversion; two outputs so the main-text / supplement placement stays
open, 需求总结 section 12.2):
  16a capacity_regions  (178 x 148 mm): 15 pair rows x 6 window columns.
      Each cell = one pair-window: filled dot A->B, hollow dot B->A
      (driver-region colour), vertical span = directional spread, in-cell
      ladder hairlines 95% (solid THR_95) / 70% (dashed THR_LADDER).
      Rows sorted by the pair's strongest cell; right margin column
      "max" = the pair's best window.  The strongest pair-window is drawn
      heavier.  Rung readout lines under the grid.
  16b capacity_records  (178 x 222 mm): table, 2 blocks x 40 records.
      Columns per block: region chip + record name | S_max | factor
      (gradient cell, SEQ ramp capped at #5e97df like 07).  S_max = the
      record's strongest forced-difference rate (rate_alldiff) over its
      79 partners x both roles.  factor = 1/f where S_max * g(f) = 0.95,
      g = relative robustness gain of the best-case Scenario A edge
      (NGRIP -> EDC dD, 0.06 at 1x -> 0.925 at 0.1x), log-log
      interpolation.  The reference edge itself would need >10x (its
      g(0.1) = 15.4 < 0.95/0.06 = 15.8), consistent with Fig. 7.

Convergence targets (asserted):
  16a: 15 pairs, 6 windows, 180 finite values; max = 0.78 at
       (North_Atlantic, Tropical, H4 event, A_to_B); that cell = 0.78 /
       0.66 exactly; no cell >= 0.80; first row = North_Atlantic ->
       Tropical; rung counts over the 180 values printed and pinned.
  16b: 80 records, 6,320 pairs; S_max quartiles 0.27 / 0.44 / 0.535 /
       0.66 / 0.92; top record ODP 1094 Ba/Fe = 0.92; factors
       1.2x-6.2x, median 4.9x; calibration anchors 1x = 6.0/6.5/4.5% +
       control max 7.25% + 0 controls pass, 0.1x = 92.5% / control 54.5%.

Usage:
    python scripts/fignew/16_capacity_assessment.py
Outputs: figures_new/16_capacity_assessment/capacity_regions.{png,svg},
         figures_new/16_capacity_assessment/capacity_records.{png,svg}
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

from style import (SURFACE, INK, SECONDARY_INK, GRID, THR_95, THR_LADDER,  # noqa: E402
                   REGION_ORDER, REGION_CATS, save)
from src.config import OUTPUT_DIR  # noqa: E402
from src.record_names import display_name  # noqa: E402

OUT_DIR = os.path.join(ROOT, "figures_new", "16_capacity_assessment")

FIG_W = 178.0
FIG_H_A = 148.0          # 16a: pair x window grid
FIG_H_B = 222.0          # 16b: record table

# windows in stratigraphic order (old -> young), 13's WINDOW_ORDER
WINDOW_ORDER = ['GS-12 (stadial)', 'GI-12 (interstadial)', 'H4 event',
                'Bølling-Allerød', 'Last Glacial Maximum', 'H1 event']
WINDOW_SHORT = {'GS-12 (stadial)': 'GS-12',
                'GI-12 (interstadial)': 'GI-12',
                'H4 event': 'H4',
                'Bølling-Allerød': 'B-A',
                'Last Glacial Maximum': 'LGM',
                'H1 event': 'H1'}

# short region names for pair labels (15's SHORT_REGION, further shortened
# so the longest pair label fits the label zone at 9 pt bold)
SHORT_REGION = {'North_Atlantic': 'N. Atlantic', 'Antarctic': 'Antarctic–SO',
                'Tropical': 'Tropical', 'East_Asian_Monsoon': 'E. Asian',
                'South_America': 'S. Amer.', 'Other_Lakes': 'Lakes'}

# record_profile region keys -> canonical colour slots (Southern_Ocean is
# merged into Antarctic, 15's REGION_KEY convention)
REGION_CAT_KEY = {'Southern_Ocean': 'Antarctic'}

RUNGS = (0.95, 0.90, 0.85, 0.80, 0.75, 0.70)

# SEQ ramp for the factor cells: light = easy, dark = hard; the cap
# #5e97df is 07's validated limit so black text stays readable on it
SEQ_LO = "#e8f0fb"
SEQ_HI = "#5e97df"
FACTOR_VMAX = 6.5        # gradient display range: 1x .. 6.5x


def _lerp(c0, c1, t):
    a = np.array([int(c0[i:i + 2], 16) for i in (1, 3, 5)], float)
    b = np.array([int(c1[i:i + 2], 16) for i in (1, 3, 5)], float)
    v = a + (b - a) * np.clip(t, 0.0, 1.0)
    return "#%02x%02x%02x" % tuple(np.rint(v).astype(int))


def _factor_color(fac):
    if not np.isfinite(fac):
        return SEQ_HI
    return _lerp(SEQ_LO, SEQ_HI, (fac - 1.0) / (FACTOR_VMAX - 1.0))


# ------------------------------------------------------------------ data ----
def load_data():
    """(rates16a, pairs16a, prof16b) - canonical outputs, no recompute."""
    reg = pd.read_csv(os.path.join(OUTPUT_DIR, "regional_robustness_windows.csv"))
    reg = reg[reg.window.isin(WINDOW_ORDER)]
    assert set(reg.window) == set(WINDOW_ORDER)
    pairs = list(reg[["region_A", "region_B"]].drop_duplicates().itertuples(
        index=False, name=None))
    assert len(pairs) == 15, f"expected 15 region pairs, got {len(pairs)}"
    rates = {(r.region_A, r.region_B, r.window, "A_to_B"): r.A_to_B
             for r in reg.itertuples()}
    rates.update({(r.region_A, r.region_B, r.window, "B_to_A"): r.B_to_A
                  for r in reg.itertuples()})
    assert len(rates) == 180 and all(np.isfinite(list(rates.values())))

    # --- record level: pairwise scan (15's sources) + calibration curve ---
    prof = pd.read_csv(os.path.join(OUTPUT_DIR, "pairwise_scan",
                                    "record_profile.csv"))
    assert len(prof) == 80, f"expected 80 records, got {len(prof)}"
    shards = pd.concat([pd.read_csv(f) for f in sorted(
        glob_shards())], ignore_index=True)
    assert len(shards) == 6320, "pair count drifted"
    mx_out = shards.groupby("source")["rate_alldiff"].max()
    mx_in = shards.groupby("target")["rate_alldiff"].max()
    prof["S_max"] = pd.concat([mx_out, mx_in], axis=1).max(axis=1).values
    prof["_disp"] = prof["record"].map(display_name)
    # Unicode subscripts ₂ ₃ ₄ (CO₂, CH₄, bSiO₂, CaCO₃) lack Arial glyphs;
    # mathtext keeps the academic spelling with Arial-rendered glyphs
    for sub, digit in (("₂", "2"), ("₃", "3"), ("₄", "4")):
        prof["_disp"] = prof["_disp"].str.replace(
            sub, f"$_{digit}$", regex=False)
    assert prof["_disp"].notna().all()
    cal = pd.read_csv(os.path.join(OUTPUT_DIR, "age_error_scaling.csv"))
    return rates, pairs, prof, cal


def glob_shards():
    import glob
    return glob.glob(os.path.join(OUTPUT_DIR, "pairwise_scan", "shard_*.csv"))


# ------------------------------------------------------------ conversion ----
def best_case_gain(cal):
    """(f, g) - relative robustness gain of the best-case Scenario A edge
    (NGRIP -> EDC dD) vs age-error scale, f ascending, g ascending."""
    ref = cal[cal.pair == "NGRIP -> EDC_dD"].set_index("error_scale")
    base = float(ref.loc[1.0, "robustness"])
    f = np.array([0.1, 0.2, 0.5, 1.0])
    g = np.array([float(ref.loc[s, "robustness"]) / base for s in f])
    assert np.all(np.diff(g) < 0), "gain curve not monotone"
    return f, g


def factor_to_95(S, f, g):
    """1/f* where S * g(f*) = 0.95 (log-log piecewise linear); np.inf if
    the target gain lies beyond the swept range (f* < 0.1)."""
    if S >= 0.95:
        return 1.0
    gt = 0.95 / S
    lg = np.log10(g)[::-1]       # ascending gain (g decreases with f)
    lf = np.log10(f)[::-1]       # corresponding descending scale
    if np.log10(gt) > lg[-1]:    # beyond the 0.1x swept range
        return np.inf
    idx = int(np.searchsorted(lg, np.log10(gt)))
    if idx == 0:
        return 1.0 / (10.0 ** lf[0])
    t = (np.log10(gt) - lg[idx - 1]) / (lg[idx] - lg[idx - 1])
    return 1.0 / (10.0 ** (lf[idx - 1] + t * (lf[idx] - lf[idx - 1])))


# ------------------------------------------------------------ 16a: regions --
GRID_L, GRID_R, GRID_T, GRID_B = 0.26, 0.955, 0.885, 0.335
FOOTER_DY = 0.0145        # 9 pt line pitch in figure coords


def _grid_row_center(i, nrows=15):
    row_h = (GRID_T - GRID_B) / (nrows + (nrows - 1) * 0.10)
    return GRID_T - i * 1.1 * row_h - row_h / 2


def _col_center(j):
    w = (GRID_R - GRID_L) / (6 + 5 * 0.10)
    return GRID_L + j * 1.1 * w + w / 2


def draw_cell(fig, gs, i, j, src, dst, w, rates, emph):
    ax = fig.add_subplot(gs[i, j])
    ax.set_facecolor(SURFACE)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ab = float(rates[(src, dst, w, "A_to_B")])
    ba = float(rates[(src, dst, w, "B_to_A")])
    assert np.isfinite(ab) and np.isfinite(ba)

    # in-cell ladder: 95% solid + 70% dashed (matrix rendering, 需求总结 §5)
    ax.axhline(0.95, color=THR_95, lw=0.9, zorder=1)
    ax.axhline(0.70, color=THR_LADDER, lw=0.7, ls=(0, (3, 3)), zorder=1)

    heavy = emph == (i, j)
    lw = 2.2 if heavy else 1.2
    ms = 4.5 if heavy else 3.5
    ax.plot([0.0, 0.0], [min(ab, ba), max(ab, ba)], color=SECONDARY_INK,
            lw=lw, zorder=2)
    ax.plot([-0.15], [ab], 'o', ms=ms, color=REGION_CATS[src], mec='none',
            ls='none', zorder=3)                       # filled = A -> B
    ax.plot([0.15], [ba], 'o', ms=ms, color=REGION_CATS[src], mfc='none',
            mec=REGION_CATS[src], mew=0.9, ls='none', zorder=3)  # hollow B->A
    return ax


def plot_16a(rates, pairs):
    """(a) pair x window capacity spectrum with in-cell honesty ladders."""
    # rows sorted by the pair's strongest cell (max over 12 directed values)
    rows = []
    for src, dst in pairs:
        m = max(rates[(src, dst, w, d)] for w in WINDOW_ORDER
                for d in ("A_to_B", "B_to_A"))
        rows.append((m, src, dst))
    rows.sort(key=lambda t: (-t[0], REGION_ORDER.index(t[1]),
                             REGION_ORDER.index(t[2])))
    max_vals = [m for m, _, _ in rows]
    assert max_vals[0] == max(max_vals)

    fig = Figure(figsize=(FIG_W / 25.4, FIG_H_A / 25.4), facecolor="white")
    gs = fig.add_gridspec(15, 6, left=GRID_L, right=GRID_R, top=GRID_T,
                          bottom=GRID_B, wspace=0.10, hspace=0.10)

    fig.text(0.048, 0.985, "Regional capacity spectrum — which rung each "
             "pair can support, window by window",
             fontsize=10.5, fontweight="bold", color=INK)
    fig.text(0.048, 0.958, "cell: robustness of the pair in one window — "
             "filled dot = A → B, hollow dot = B → A, span = directional "
             "spread", fontsize=9, color=INK)
    fig.text(0.048, 0.936, "hairlines = 95% (solid) / 70% (dashed) honesty "
             "rungs in every cell; the strongest pair-window is drawn "
             "heavier", fontsize=9, color=INK)

    # column headers (window names + the max column at the right edge)
    for j, w in enumerate(WINDOW_ORDER):
        fig.text(_col_center(j), 0.912, WINDOW_SHORT[w], ha="center",
                 va="center", fontsize=9, fontweight="bold", color=INK)
    fig.text(GRID_R + 0.012, 0.912, "max", ha="right", va="center",
             fontsize=9, fontweight="bold", color=INK)

    # global max cell -> drawn heavier
    mx = max(max_vals)
    emph = None
    for i, (m, src, dst) in enumerate(rows):
        for j, w in enumerate(WINDOW_ORDER):
            if rates[(src, dst, w, "A_to_B")] == mx:
                emph = (i, j)
    assert emph is not None

    cell_axes = {}
    for i, (m, src, dst) in enumerate(rows):
        for j, w in enumerate(WINDOW_ORDER):
            cell_axes[(i, j)] = draw_cell(fig, gs, i, j, src, dst, w,
                                          rates, emph)
        # row label: driver colour chip + pair name (first-column axes)
        ax0 = cell_axes[(i, 0)]
        ax0.add_patch(Rectangle((-0.315, 0.27), 0.14, 0.46,
                                transform=ax0.transAxes, clip_on=False,
                                facecolor=REGION_CATS[src],
                                edgecolor="none"))
        ax0.text(-0.10, 0.5, f"{SHORT_REGION[src]} → {SHORT_REGION[dst]}",
                 ha="right", va="center", fontsize=9, fontweight="bold",
                 color=INK, transform=ax0.transAxes)
        # right-margin max column
        fig.text(GRID_R + 0.012, _grid_row_center(i), f"{m:.2f}",
                 ha="right", va="center", fontsize=9, color=INK)

    # rung readout (the honest reading of the whole grid)
    vals = list(rates.values())
    counts = [int((np.array(vals) >= r).sum()) for r in RUNGS]
    fig.text(0.048, 0.312, "0 of 180 pair-windows reach the 95% rung — only "
             "one reaches the 70% rung", fontsize=9.5, fontweight="bold",
             color=INK)
    fig.text(0.048, 0.290, f"rung counts: ≥95% {counts[0]} · ≥90% "
             f"{counts[1]} · ≥85% {counts[2]} · ≥80% {counts[3]} · ≥75% "
             f"{counts[4]} · ≥70% {counts[5]}", fontsize=9, color=INK)

    foot = [
        "data: outputs/regional_robustness_windows.csv (scripts/09; ARC α = "
        "0.5, M = 200, seed 42)",
        "15 pairs × 6 windows × 2 directions = 180 pair-window robustness "
        "values",
        "strongest: North Atlantic → Tropical in H4 = 78% vs 66% (Fig. 6 "
        "anchor)",
        "columns in stratigraphic order: GS-12 · GI-12 · H4 · B-A · LGM · "
        "H1; rows sorted by strongest cell",
        "dots: filled = A → B, hollow = B → A; vertical span = directional "
        "spread",
        "hairlines = 95% (solid) / 70% (dashed) honesty rungs per cell; max "
        "column = the pair's best window",
    ]
    for k, line in enumerate(foot):
        assert len(line) <= 107, f"footer line too long ({len(line)})"
        fig.text(0.048, 0.272 - k * FOOTER_DY, line, fontsize=9, color=INK)

    assert no_text_overruns(fig), "some text runs off the figure edges"
    save(fig, os.path.join(OUT_DIR, "capacity_regions"))
    return rows, counts, mx


# ------------------------------------------------------------ 16b: records --
TBL_TOP, TBL_ROW, TBL_HEAD = 154.0, 3.75, 6.0     # table units = mm
TBL_H = TBL_TOP + TBL_HEAD                      # header baseline zone
BLK = ((8.0, 48.0, 51.0, 61.0, 64.0, 77.0),      # (name x0, name x1, S c,
       (89.0, 129.0, 132.0, 142.0, 145.0, 158.0))  #  factor x0, factor x1)


def _row_y(i):
    """bottom of row i, then its centre: rows i = 0..39 from the top."""
    return TBL_TOP - (i + 1) * TBL_ROW, TBL_TOP - (i + 0.5) * TBL_ROW


def plot_16b(prof, cal):
    """(b) record-level capacity table with the factor gradient."""
    f_arr, g_arr = best_case_gain(cal)
    p = prof.sort_values(["S_max", "_disp", "record"],
                         ascending=[False, True, True]).reset_index(drop=True)
    p["factor"] = [factor_to_95(s, f_arr, g_arr) for s in p.S_max]

    fig = Figure(figsize=(FIG_W / 25.4, FIG_H_B / 25.4), facecolor="white")
    fig.text(0.048, 0.982, "Record-level capacity — robustness ceiling and "
             "compression needed for the 95% rung",
             fontsize=10.5, fontweight="bold", color=INK)
    fig.text(0.048, 0.958, "S_max = the record's strongest forced-difference "
             "rate over its 79 partners × both roles", fontsize=9, color=INK)
    fig.text(0.048, 0.938, "factor = 1/f where S_max · g(f) = 0.95; g = "
             "best-case Scenario A error-sensitivity (NGRIP → EDC δD)",
             fontsize=9, color=INK)

    ax = fig.add_axes([0.045, 0.15, 0.91, 0.75])
    ax.set_xlim(0, 170)
    ax.set_ylim(0, TBL_H)
    ax.axis("off")

    # header bands
    for k, (nx0, _, sc, _, fx0, fx1) in enumerate(BLK):
        ax.text(nx0 + 1, TBL_TOP + 2, "record", ha="left", va="center",
                fontsize=9, fontweight="bold", color=INK)
        ax.text(sc + 5, TBL_TOP + 2, "S_max", ha="center", va="center",
                fontsize=9, fontweight="bold", color=INK)
        ax.text((fx0 + fx1) / 2, TBL_TOP + 2, "factor", ha="center",
                va="center", fontsize=9, fontweight="bold", color=INK)
        ax.hlines(TBL_TOP, nx0, fx1, color=GRID, lw=0.6)

    for k, blk in enumerate(BLK):
        nx0, nx1, sc, _, fx0, fx1 = blk
        sub = p.iloc[k * 40:(k + 1) * 40]
        for r, (_, row) in enumerate(sub.iterrows()):
            yb, yc = _row_y(r)
            ax.add_patch(Rectangle((nx0 + 0.5, yc - 1.25), 2.5, 2.5,
                                   facecolor=REGION_CATS[
                                       REGION_CAT_KEY.get(row.region,
                                                          row.region)],
                                   edgecolor="none"))
            ax.text(nx0 + 4.5, yc, row._disp, ha="left", va="center",
                    fontsize=9, color=INK)
            ax.text(sc + 5, yc, f"{row.S_max:.2f}", ha="center",
                    va="center", fontsize=9, color=INK)
            fac = row.factor
            ax.add_patch(Rectangle((fx0, yc - 1.4), fx1 - fx0, 2.8,
                                   facecolor=_factor_color(fac),
                                   edgecolor="none",
                                   gid=f"layer-background-cell-{k * 40 + r:03d}"))
            lab = f"{fac:.1f}×" if np.isfinite(fac) else ">10×"
            ax.text((fx0 + fx1) / 2, yc, lab, ha="center", va="center",
                    fontsize=9, color=INK)
            if r < 39:
                ax.hlines(yb, nx0, fx1, color=GRID, lw=0.4)
        ax.hlines(TBL_TOP - 40 * TBL_ROW, nx0, fx1, color=GRID, lw=0.6)

    # gradient legend (below the table, figure coords via the axes)
    x0, x1, gy = 28.0, 78.0, -5.5
    for k in range(40):
        ax.add_patch(Rectangle((x0 + k * (x1 - x0) / 40, gy), (x1 - x0) / 40,
                               1.8, facecolor=_lerp(SEQ_LO, SEQ_HI, k / 39),
                               edgecolor="none",
                               gid=f"layer-background-grad-{k:02d}",
                               clip_on=False))
    for t in range(1, 7):
        x = x0 + (t - 1) * (x1 - x0) / (FACTOR_VMAX - 1)
        ax.plot([x, x], [gy - 0.5, gy - 0.2], color=INK, lw=0.6,
                clip_on=False)
        ax.text(x, gy - 0.7, f"{t}×", ha="center", va="top", fontsize=9,
                color=INK, clip_on=False)
    ax.text(x1 + 4, gy + 0.9, "compression factor needed to reach the 95% "
            "rung", ha="left", va="center", fontsize=9, color=INK,
            clip_on=False)

    foot = [
        "data: outputs/pairwise_scan/ (scripts/19) — S_max = max forced-"
        "difference rate over 79 partners × 2 roles",
        "conversion: g(f) = best-case edge NGRIP → EDC δD (0.06 at 1× → "
        "0.925 at 0.1×), log-log",
        "assumption: all records follow this relative shape; the reference "
        "edge itself would need >10× compression",
        f"anchors: S_max quartiles 0.92 / 0.66 / 0.535 / 0.44 / 0.27; "
        f"factors {p.factor.min():.1f}×–{p.factor.max():.1f}×, median "
        f"{p.factor.median():.1f}×",
        "gradient: compression factor, light = easy → dark = hard "
        "(#5e97df cap); chips = region",
    ]
    for k, line in enumerate(foot):
        assert len(line) <= 107, f"footer line too long ({len(line)})"
        fig.text(0.048, 0.068 - k * FOOTER_DY, line, fontsize=9, color=INK)

    assert no_text_overruns(fig), "some text runs off the figure edges"
    save(fig, os.path.join(OUT_DIR, "capacity_records"))
    return p


# ---------------------------------------------------------------- QA/anchors --
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


def print_anchors(rates, pairs, rows, counts, mx, prof, cal):
    print("\n== anchors (16a capacity regions) ==")
    print(f"  pairs: {len(pairs)} | windows: {len(WINDOW_ORDER)} | values: "
          f"{len(rates)}")
    print(f"  max: {mx:.4f} | rung counts {counts} | row order (first 3): "
          f"{[f'{s}->{d}' for _, s, d in rows[:3]]}")
    print("  NA->Trop H4: %.4f / %.4f" % (
        rates[("North_Atlantic", "Tropical", "H4 event", "A_to_B")],
        rates[("North_Atlantic", "Tropical", "H4 event", "B_to_A")]))

    print("\n== anchors (16b capacity records) ==")
    print(f"  records: {len(prof)} | S_max quartiles "
          f"{np.percentile(prof.S_max, [0, 25, 50, 75, 100]).round(3)}")
    top = prof.loc[prof.S_max.idxmax()]
    print(f"  top record: {top._disp} (S_max = {top.S_max:.2f}, region "
          f"{top.region})")
    f_arr, g_arr = best_case_gain(cal)
    fac = prof.S_max.apply(lambda s: factor_to_95(s, f_arr, g_arr))
    print(f"  factors: min {fac.min():.3f} / median {fac.median():.3f} / "
          f"max {fac.max():.3f}")
    c1 = cal[cal.error_scale == 1.0].set_index("pair")["robustness"]
    c01 = cal[cal.error_scale == 0.1].set_index("pair")["robustness"]
    print(f"  calibration 1×: {c1['NGRIP -> EDC_dD']:.3f} / "
          f"{c1['CO2 -> CH4']:.3f} / {c1['Cariaco -> Cheng']:.3f}, control "
          f"max {c1['control_max']:.4f}, pass95 {int(c1['control_n_pass95'])}")
    print(f"  calibration 0.1×: NGRIP->EDC {c01['NGRIP -> EDC_dD']:.3f}, "
          f"control max {c01['control_max']:.3f}")

    # ---- asserts 16a ----
    assert len(pairs) == 15 and len(WINDOW_ORDER) == 6
    assert len(rates) == 180
    assert abs(mx - 0.78) < 1e-9, f"max drifted: {mx}"
    assert abs(rates[("North_Atlantic", "Tropical", "H4 event", "A_to_B")]
               - 0.78) < 1e-9
    assert abs(rates[("North_Atlantic", "Tropical", "H4 event", "B_to_A")]
               - 0.66) < 1e-9
    assert counts[0] == 0 and counts[3] == 0, "no cell >= 80% rung"
    assert counts[4] == 1 and counts[5] == 1, "only the 0.78 cell >= 70%"
    assert rows[0][1:] == ("North_Atlantic", "Tropical")
    # ---- asserts 16b ----
    q = np.percentile(prof.S_max, [0, 25, 50, 75, 100])
    assert np.allclose(q, [0.27, 0.44, 0.535, 0.66, 0.92], atol=1e-9), q
    assert abs(top.S_max - 0.92) < 1e-9 and top.record == \
        "ODP1094_BaFe", f"top record drifted: {top.record}"
    assert abs(fac.min() - 1.155) < 0.05, fac.min()
    assert abs(fac.median() - 4.928) < 0.05, fac.median()
    assert abs(fac.max() - 6.214) < 0.05, fac.max()
    for pair, want in (("NGRIP -> EDC_dD", 0.06), ("CO2 -> CH4", 0.065),
                       ("Cariaco -> Cheng", 0.045)):
        assert abs(c1[pair] - want) < 1e-9, (pair, c1[pair])
    assert abs(c1["control_max"] - 0.0725) < 1e-9
    assert int(c1["control_n_pass95"]) == 0
    assert abs(c01["NGRIP -> EDC_dD"] - 0.925) < 1e-9
    assert abs(c01["control_max"] - 0.545) < 1e-9
    # the reference edge itself sits beyond the swept range
    assert factor_to_95(0.06, f_arr, g_arr) == np.inf
    print("  claim: 0 of 180 pair-windows reach 95%, only one reaches 70%; "
          "records need 1.2×-6.2× age-error compression even in the best "
          "case.")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rates, pairs, prof, cal = load_data()
    rows, counts, mx = plot_16a(rates, pairs)
    plot_16b(prof, cal)
    print_anchors(rates, pairs, rows, counts, mx, prof, cal)


if __name__ == "__main__":
    main()
