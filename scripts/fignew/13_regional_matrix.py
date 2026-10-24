# -*- coding: utf-8 -*-
"""Step 13 - regional causal matrix (Fig. 6, Section 4.4).

Claim: at regional scale every directed pair-window combination stays in
the sub-threshold regime - none of the 180 directed combinations (15 pairs
x 6 windows x 2 directions) reaches the 95% honesty rung.  Stacking 80
records into six regional composites sharpens directional asymmetry (the
strongest direction, North Atlantic -> Tropical in H4, scores 78.0%
against 66.0% in reverse), but no cell meets the yellow-border criterion
(median |A->B - B->A| >= 0.3; maximum 0.145).

Design (double column, 178 x 178 mm):
  6 x 6 grid in canonical region order (rows = drivers, columns = targets).
  Each off-diagonal cell (row, col) is a mini strip-chart of the row ->
  column robustness over the six climate windows (window encoded by
  position, oldest first: GS-12, GI-12, H4, LGM, H1, B-A), with the window
  median as a thick bar, the 95%/70% honesty rungs as hairlines (matrix
  in-cell ladder per 需求总结: 95% solid + 70% dashed), and stars marking
  the 21 phase-significant pair-windows of Table 5.  Cell curves take the
  driver region's fixed colour slot (style.REGION_CATS); the diagonal
  carries the region chip + name + record count of the scripts/09 stack.
  The yellow border would flag cells with median directional asymmetry
  >= 0.3 - asserted absent (max 0.145), so it appears in the legend only.

Data: outputs/regional_robustness_windows.csv (scripts/09, M = 200 draws,
seed 42; median-scheme regional stacks, ARC alpha = 0.5 against the North
Atlantic composite, per-window pairwise Granger robustness).  Table 5
star list = scripts/37 PAPER_T5 (current manuscript, 21 pair-windows).

Convergence targets (asserted):
  CSV 90 rows / 180 directed values; every window of every pair finite
  North Atlantic -> Tropical, H4 event = 0.78 / 0.66 reverse (exact)
  max robustness overall = 0.78; 0 of 180 values >= 0.95
  per-pair median |A->B - B->A|: max = 0.145, all < 0.3 (no yellow border)
  all 21 Table 5 pair-windows map onto finite cells (21 stars)
  stack composition = canonical 80-record inventory (Table S2/S4):
  13/18/34/6/3/6 (non-inventory files 15_Ba, 15_Br, Lapa_d18O excluded)

Usage:
    python scripts/fignew/13_regional_matrix.py
Outputs: figures_new/13_regional_matrix/regional_matrix.png / .svg
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.dirname(os.path.abspath(__file__)), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from style import (save, INK, THR_95, THR_LADDER, SURFACE,  # noqa: E402
                   REGION_ORDER, REGION_CATS)
from src.config import OUTPUT_DIR, DATA_DIR  # noqa: E402
from src.io import load_all_records, filter_canonical_records  # noqa: E402
from src.stacking import build_region_dict  # noqa: E402

OUT_DIR = os.path.join(ROOT, "figures_new", "13_regional_matrix")

FIG_W, FIG_H = 178.0, 178.0
GRID_L, GRID_R, GRID_T, GRID_B = 0.150, 0.985, 0.885, 0.385
FOOTER_Y = (0.262, 0.234, 0.206, 0.178, 0.150, 0.122, 0.094)

ACCENT = "#eda100"          # yellow-border criterion (CAT4 slot; none drawn)
ASYM_MIN = 0.30             # median |A->B - B->A| to flag a cell

# windows shown in each cell, oldest first (left), as in 11's Fig. 2
WINDOW_ORDER = ['GS-12 (stadial)', 'GI-12 (interstadial)', 'H4 event',
                'Last Glacial Maximum', 'H1 event', 'Bølling-Allerød']

ROW_LABEL = {
    'North_Atlantic': 'North Atlantic',
    'Antarctic': 'Antarctic/SO',
    'Tropical': 'Tropical',
    'East_Asian_Monsoon': 'E. Asian\nMonsoon',
    'South_America': 'S. Am.\nMonsoon',
    'Other_Lakes': 'High-Lat\nLakes',
}

# Paper Table 5 (scripts/37 PAPER_T5, current manuscript): phase-significant
# (driver, target, window) rows; stars go on cell (driver -> target) at the
# window's strip position.  T5 region keys use the 6-region assignment names.
PAPER_T5 = {
    ('East_Asian_Monsoon', 'North_Atlantic', 'H1 event'),
    ('North_Atlantic', 'HighLat_Lakes', 'H1 event'),
    ('Antarctic_SouthernOcean', 'South_America', 'H1 event'),
    ('Antarctic_SouthernOcean', 'HighLat_Lakes', 'H1 event'),
    ('Tropical', 'East_Asian_Monsoon', 'H1 event'),
    ('South_America', 'Tropical', 'H1 event'),
    ('East_Asian_Monsoon', 'South_America', 'H1 event'),
    ('South_America', 'HighLat_Lakes', 'H1 event'),
    ('East_Asian_Monsoon', 'Antarctic_SouthernOcean', 'H1 event'),
    ('East_Asian_Monsoon', 'HighLat_Lakes', 'H4 event'),
    ('North_Atlantic', 'East_Asian_Monsoon', 'Bølling-Allerød'),
    ('Antarctic_SouthernOcean', 'East_Asian_Monsoon', 'Bølling-Allerød'),
    ('East_Asian_Monsoon', 'Tropical', 'Bølling-Allerød'),
    ('East_Asian_Monsoon', 'South_America', 'Bølling-Allerød'),
    ('East_Asian_Monsoon', 'HighLat_Lakes', 'Bølling-Allerød'),
    ('North_Atlantic', 'East_Asian_Monsoon', 'Last Glacial Maximum'),
    ('Tropical', 'East_Asian_Monsoon', 'Last Glacial Maximum'),
    ('East_Asian_Monsoon', 'Antarctic_SouthernOcean', 'Last Glacial Maximum'),
    ('East_Asian_Monsoon', 'South_America', 'Last Glacial Maximum'),
    ('East_Asian_Monsoon', 'HighLat_Lakes', 'Last Glacial Maximum'),
    ('South_America', 'East_Asian_Monsoon', 'GI-12 (interstadial)'),
}
T5_REGION = {'Antarctic_SouthernOcean': 'Antarctic',
             'HighLat_Lakes': 'Other_Lakes'}


# ------------------------------------------------------------------ data ----
def load_rates():
    """{(src, dst, window): robustness} for both directions, from the
    scripts/09 CSV (90 rows x 2 directions = 180 directed values)."""
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'regional_robustness_windows.csv'))
    assert list(df.columns) == ['window', 'region_A', 'region_B',
                                'A_to_B', 'B_to_A'], "CSV schema drifted"
    assert len(df) == 90, f"expected 90 pair-window rows, got {len(df)}"
    rates = {}
    for _, r in df.iterrows():
        rates[(r.region_A, r.region_B, r.window)] = float(r.A_to_B)
        rates[(r.region_B, r.region_A, r.window)] = float(r.B_to_A)
    assert len(rates) == 180, f"expected 180 directed values, got {len(rates)}"
    return df, rates


def stack_counts():
    """Records per regional composite on the 10-60 ka grid, as loaded by
    scripts/09 (canonical 80-record inventory + Southern_Ocean merged into
    Antarctic).  Printed on the diagonal because these are the stacks the
    CSV numbers come from."""
    records, _ = load_all_records(DATA_DIR, age_min=10.0, age_max=60.0)
    records = filter_canonical_records(records)
    records = [(i, s, 'Antarctic' if r == 'Southern_Ocean' else r, n)
               for i, s, r, n in records]
    rd = build_region_dict(records)
    return {r: len(rd.get(r, [])) for r in REGION_ORDER}


# ------------------------------------------------------------------ plot ----
_star_n = [0]


def star(ax, x, y):
    """One Table-5 star with a unique gid (the 12 lesson: repeated bare
    marker gids leave the first occurrence outside the layer wrapper)."""
    (mk,) = ax.plot([x], [y], '*', ms=11, mfc='white', mec=INK, mew=0.8,
                    color=INK, ls='none', zorder=4)
    mk.set_gid(f"layer-markers-{_star_n[0]:03d}")
    _star_n[0] += 1
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


def cell(fig, gs, i, j, src, dst, rates, counts, stars_placed):
    ax = fig.add_subplot(gs[i, j])
    ax.set_facecolor(SURFACE)
    ax.set_xlim(-0.5, len(WINDOW_ORDER) - 0.5)
    ax.set_ylim(-0.12, 1.12)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if i == j:                                  # diagonal: chip + name + count
        ax.add_patch(Rectangle((0.31, 0.665), 0.38, 0.05,
                               facecolor=REGION_CATS[src], edgecolor='none'))
        ax.text(0.5, 0.46, ROW_LABEL[src], ha='center', va='center',
                fontsize=9, fontweight='bold', color=INK, transform=ax.transAxes)
        ax.text(0.5, 0.08, f"n = {counts[src]} records", ha='center',
                va='center', fontsize=9, color=INK, transform=ax.transAxes)
    else:
        vals = np.array([rates[(src, dst, w)] for w in WINDOW_ORDER])
        assert np.all(np.isfinite(vals)), f"NaN in {src}->{dst}"
        med = float(np.median(vals))

        # yellow-border criterion: median |A->B - B->A| >= 0.3 (asserted
        # absent; gid pre-set so a future trigger lands in annotations)
        rev = np.array([rates[(dst, src, w)] for w in WINDOW_ORDER])
        if abs(med - float(np.median(rev))) >= ASYM_MIN:
            ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                   fill=False, edgecolor=ACCENT, lw=1.8,
                                   clip_on=False, gid='layer-annotations'))

        # in-cell ladder: 95% solid + 70% dashed (matrix rendering of the
        # honesty ladder, 需求总结 section 5)
        ax.axhline(0.95, color=THR_95, lw=0.9, zorder=1)
        ax.axhline(0.70, color=THR_LADDER, lw=0.7, ls=(0, (3, 3)), zorder=1)

        xs = np.arange(len(WINDOW_ORDER))
        ax.plot(xs, vals, color=REGION_CATS[src], lw=1.0, zorder=2)
        (dots,) = ax.plot(xs, vals, 'o', ms=3.5, mfc=REGION_CATS[src],
                          mec='none', color=REGION_CATS[src], ls='none',
                          zorder=3)
        ax.hlines(med, -0.32, len(WINDOW_ORDER) - 0.68, color=REGION_CATS[src],
                  lw=2.2, zorder=2.5, capstyle='round')

        for k, w in enumerate(WINDOW_ORDER):
            key = (src, dst, w)
            if key in stars_placed:
                star(ax, k, vals[k])

    if j == 0:                                  # row label (driver)
        ax.text(-0.08, 0.5, ROW_LABEL[src], ha='right', va='center',
                fontsize=9, fontweight='bold', color=INK,
                transform=ax.transAxes)


def plot(rates, counts, stars_placed):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig = Figure(figsize=(FIG_W / 25.4, FIG_H / 25.4), facecolor='white')
    gs = fig.add_gridspec(6, 6, left=GRID_L, right=GRID_R, top=GRID_T,
                          bottom=GRID_B, wspace=0.10, hspace=0.10)

    fig.text(0.048, 0.985, "Regional causal matrix — directional "
             "robustness (row → column)", fontsize=10.5, fontweight="bold",
             color=INK)
    fig.text(0.048, 0.958,
             "off-diagonal cell (row, col): robustness of row → column "
             "across six climate windows", fontsize=9, color=INK)
    fig.text(0.048, 0.936,
             "thick bar = window median; hairlines = 95% / 70% honesty "
             "rungs", fontsize=9, color=INK)

    for i, src in enumerate(REGION_ORDER):
        for j, dst in enumerate(REGION_ORDER):
            cell(fig, gs, i, j, src, dst, rates, counts, stars_placed)

    # column labels (targets) in the band above the grid; two-line wrap so
    # the long names never collide with the neighbouring column's label
    # (layout QA lesson)
    cell_w = (GRID_R - GRID_L) / 6.0
    for j, dst in enumerate(REGION_ORDER):
        fig.text(GRID_L + (j + 0.5) * cell_w, 0.910, ROW_LABEL[dst],
                 ha='center', va='center', fontsize=9, fontweight='bold',
                 color=INK)

    handles = [
        Line2D([], [], marker='o', ms=4, mfc=INK, mec=INK, color=INK,
               ls='-', lw=1.0, label='per-window robustness'),
        Line2D([0], [0], color=INK, lw=2.2, label='median across windows'),
        Line2D([0], [0], color=THR_95, lw=0.9, label='95% honesty rung'),
        Line2D([0], [0], color=THR_LADDER, lw=0.8, ls=(0, (3, 3)),
               label='70% lower rung'),
        Line2D([], [], marker='*', ms=11, mfc='white', mec=INK, mew=0.8,
               color=INK, ls='none',
               label='phase-significant pair–window (Table 5)'),
        Rectangle((0, 0), 1, 1, fill=False, edgecolor=ACCENT, lw=1.8,
                  label='median |A→B − B→A| ≥ 0.3 — none (max 0.145)'),
    ]
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, 0.315), ncol=3, frameon=False,
               fontsize=9, handlelength=2.0, columnspacing=2.0,
               handletextpad=0.6)

    # footer: symbol decode + data facts + provenance (each line short
    # enough to clear the right edge - QA-checked)
    fig.text(0.048, FOOTER_Y[0],
             "window order in each cell (left→right): GS-12, GI-12, H4, "
             "LGM, H1, B-A", fontsize=9, color=INK)
    fig.text(0.048, FOOTER_Y[1],
             "none of the 180 directed pair–window combinations (15 pairs "
             "× 6 windows × 2 directions) reaches 95%", fontsize=9, color=INK)
    fig.text(0.048, FOOTER_Y[2],
             "strongest direction: North Atlantic → Tropical in H4 — 78.0% "
             "vs 66.0% in reverse", fontsize=9, color=INK)
    fig.text(0.048, FOOTER_Y[3],
             "largest forward–reverse contrast: Tropical → Antarctic/SO in "
             "the LGM — 65.5% vs 2.5%", fontsize=9, color=INK)
    fig.text(0.048, FOOTER_Y[4],
             "stars = 21 phase-significant pair–windows of Table 5 "
             "(phase-lag sign consistency, scripts/37)", fontsize=9, color=INK)
    fig.text(0.048, FOOTER_Y[5],
             "data: outputs/regional_robustness_windows.csv (scripts/09, "
             "M = 200, seed 42)", fontsize=9, color=INK)
    fig.text(0.048, FOOTER_Y[6],
             "stacks = the 80-record inventory (Table S2/S4): "
             "13/18/34/6/3/6", fontsize=9, color=INK)

    assert no_text_overruns(fig), "some text runs off the figure edges"
    save(fig, os.path.join(OUT_DIR, "regional_matrix"))
    print_anchors(rates, counts, stars_placed)


# ---------------------------------------------------------------- anchors ----
def print_anchors(rates, counts, stars_placed):
    print("\n== anchors (Fig. 6 regional causal matrix) ==")
    print(f"  values: {len(rates)} directed (15 pairs x 6 windows x 2)")
    h4 = rates[('North_Atlantic', 'Tropical', 'H4 event')]
    h4r = rates[('Tropical', 'North_Atlantic', 'H4 event')]
    print(f"  North Atlantic -> Tropical, H4: {h4:.1%} vs {h4r:.1%} reverse")
    mx = max(rates.values())
    n95 = sum(v >= 0.95 for v in rates.values())
    print(f"  max robustness {mx:.1%}; values >= 95%: {n95} of {len(rates)}")
    asym = {}
    pairs = [tuple(sorted([a, b])) for a, b in
             {(r[0], r[1]) for r in rates}]
    for a, b in sorted(set(pairs)):
        m_ab = float(np.median([rates[(a, b, w)] for w in WINDOW_ORDER]))
        m_ba = float(np.median([rates[(b, a, w)] for w in WINDOW_ORDER]))
        asym[(a, b)] = abs(m_ab - m_ba)
    worst = max(asym.items(), key=lambda kv: kv[1])
    print(f"  median |A->B - B->A|: max {worst[1]:.3f} ({worst[0]}) "
          f"< {ASYM_MIN} -> no yellow border")
    print(f"  Table 5 stars placed: {len(stars_placed)} (paper: 21)")
    print(f"  stack counts (canonical 80 records): "
          + " / ".join(f"{counts[r]}" for r in REGION_ORDER)
          + f" = {sum(counts.values())} records (Table S4: 13/18/34/6/3/6)")
    assert h4 == 0.78 and h4r == 0.66, "H4 NA->Tropical anchor drifted"
    assert mx == 0.78 and n95 == 0, "sub-threshold anchor drifted"
    assert abs(worst[1] - 0.145) < 1e-9, f"max median asymmetry drifted "
    assert worst[1] < ASYM_MIN, "yellow-border criterion unexpectedly met"
    assert len(stars_placed) == 21, "Table 5 star count drifted"
    assert counts == {'North_Atlantic': 13, 'Antarctic': 18, 'Tropical': 34,
                      'East_Asian_Monsoon': 6, 'South_America': 3,
                      'Other_Lakes': 6}, "stack counts drifted"
    print("  claim: no regional pair-window reaches the 95% rung; stacking "
          "sharpens directional asymmetry but stays sub-threshold.")


def main():
    df, rates = load_rates()
    counts = stack_counts()
    stars_placed = {(T5_REGION.get(a, a), T5_REGION.get(b, b), w)
                    for a, b, w in PAPER_T5}
    for a, b, w in stars_placed:
        assert (a, b, w) in rates, f"Table 5 pair {a}->{b} {w} not in CSV"
    plot(rates, counts, stars_placed)


if __name__ == "__main__":
    main()
