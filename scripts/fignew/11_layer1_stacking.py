# -*- coding: utf-8 -*-
"""Step 11 - Layer-1 regional stacking (Fig. 2, Section 4.2).

Claim: regional stacking suppresses record-level noise.  Six regional
composites on the 10-60 ka grid: thin member lines under the bold median
stack with its P5-P95 age-perturbation band, read against the six
climate-state windows used by the downstream causality tests (GS-12,
GI-12, H4, LGM, H1, B-A).  Each panel's upper right compares the measured
noise reduction (stack / median-member AR(1) innovation sigma) with the
1/sqrt(K) floor for K independent equal-variance records; the gap
quantifies spatial correlation and variance heterogeneity.

Design (double column, 178 x 176 mm):
  2 x 3 panels in canonical region order (scripts/08/09/17/18); region
  colours are the fixed slots of style.REGION_CATS (CAT1..CAT6 in region
  order), so adjacent panels are validated CVD pairs.  Alternating
  neutral bands mark the six climate windows; a shared strip under the
  panels decodes them.  The noise pair lives in the title block
  (right-aligned above the axes), so no text ever covers data lines.

Data: canonical Layer-1 machinery of scripts/18 (stackviz median scheme,
seed 42, M = N_PERTURB = 200, Gaussian low-pass sigma = 0.5 ka, shared
perturbation stream), the same call scripts/33 makes; records filtered to
the paper's 80-record inventory (data/region_assignment_80_paper.csv) so
member counts = Table S4 (the 3 non-inventory files in proxy_records/,
incl. Lapa_d18O, are excluded).  Noise numbers from
outputs/stacking_noise_reduction.csv (scripts/29), re-derived here from
the data files and asserted against the CSV.

Convergence targets (asserted):
  grid 10-60 ka, dt = 0.05 ka (1001 points), M = 200, seed 42
  region counts = Table S4: 13 / 18 / 34 / 6 / 3 / 6 (80 records)
  per-region noise ratio matches scripts/29 (member AR(1) innovation
  median + smoothed median stack re-derived from the data, |dev| < 1e-9);
  e.g. North Atlantic 0.53 vs 1/sqrt(13) = 0.28, E. Asian Monsoon
  0.07 vs 0.41

Usage:
    python scripts/fignew/11_layer1_stacking.py
Outputs: figures_new/11_layer1_stacking/layer1_stacking.png / .svg
"""
from __future__ import annotations

import importlib.util
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

from style import (apply_paleo_style, save, INK, GRID, BAND,  # noqa: E402
                   REGION_ORDER, REGION_CATS)
from src.config import (DATA_DIR, OUTPUT_DIR, AGE_GRID_10_60,  # noqa: E402
                        N_PERTURB, CLIMATE_WINDOWS)
from src.io import load_all_records  # noqa: E402
from src.stacking import build_region_dict  # noqa: E402

# canonical stacking machinery (median scheme) of scripts/18, as reused by
# scripts/33 and by 04's Layer-1 signature panel
SPEC = importlib.util.spec_from_file_location(
    "stackviz", os.path.join(ROOT, "scripts", "18_stacking_visualization.py"))
stackviz = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stackviz)

OUT_DIR = os.path.join(ROOT, "figures_new", "11_layer1_stacking")

SMOOTH_SIGMA = 0.5                     # ka, Gaussian low-pass (scripts/18/33)
MEMBER = "#b9b8b0"                     # thin member lines (scripts/33)
WIN_COL = ["#ecece8", "#e2e2dc"]       # alternating climate-window neutrals
SWATCH_EDGE = "#c9c8c3"     # strip swatch border: neutral, not GRID, so the
                            # swatches stay background and do not hijack layer-grid

# windows shown on the 10-60 ka grid, oldest first (left on the inverted
# age axis), as in scripts/33
WINDOW_ORDER = ['GS-12 (stadial)', 'GI-12 (interstadial)', 'H4 event',
                'Last Glacial Maximum', 'H1 event', 'Bølling-Allerød']
WINDOW_ABBR = {'GS-12 (stadial)': 'GS-12', 'GI-12 (interstadial)': 'GI-12',
               'H4 event': 'H4', 'Last Glacial Maximum': 'LGM',
               'H1 event': 'H1', 'Bølling-Allerød': 'B-A'}

# noise CSV region name -> pipeline region key (scripts/33)
NOISE_REGION_MAP = {'Tropical': 'Tropical',
                    'Antarctic/Southern Ocean': 'Antarctic',
                    'North Atlantic': 'North_Atlantic',
                    'East Asian Monsoon': 'East_Asian_Monsoon',
                    'High-latitude Lakes': 'Other_Lakes',
                    'South American Monsoon': 'South_America'}

# canonical member counts, 10-60 ka = Table S4 (scripts/29)
S4_COUNTS = {'North_Atlantic': 13, 'Antarctic': 18, 'Tropical': 34,
             'East_Asian_Monsoon': 6, 'South_America': 3,
             'Other_Lakes': 6}

FIG_W, FIG_H = 178.0, 176.0
GRID_L, GRID_R, GRID_T, GRID_B = 0.048, 0.975, 0.855, 0.250
WSPACE, HSPACE = 0.12, 0.40
FOOTER_Y = (0.150, 0.123, 0.096, 0.069, 0.042, 0.015)
STRIP_Y, STRIP_H = 0.180, 0.032


# ------------------------------------------------------------------ data ----
def fit_ar1(x):
    """AR(1) innovation sigma of a standardized series (scripts/29)."""
    x = (x - x.mean()) / x.std()
    phi = float(np.linalg.lstsq(x[:-1, None], x[1:], rcond=None)[0][0])
    return float((x[1:] - phi * x[:-1]).std())


def load_canonical_stems():
    """The paper's 80-record inventory: data/region_assignment_80_paper.csv
    (76 clean keys covering 80 repository file stems; 4 keys split into 2
    files each).  proxy_records/ holds 83 files - the 3 extras (15_Ba,
    15_Br, Lapa_d18O) are not in the paper and must not enter Fig. 2."""
    df = pd.read_csv(os.path.join(ROOT, "data",
                                  "region_assignment_80_paper.csv"),
                     encoding="utf-8-sig")
    stems = set()
    for v in df["S2_File_IDs"]:
        for s in str(v).split(";"):
            s = s.strip()
            if s:
                stems.add(s)
    assert len(stems) == 80, f"canonical inventory = {len(stems)} stems"
    return stems


def load_region_dict():
    """Records grouped by region on the 10-60 ka grid (scripts/33 call).

    Filtered to the paper's 80-record inventory, so the panel member
    counts reproduce Table S4 (13/18/34/6/3/6) instead of picking up the
    3 non-inventory files in proxy_records/.  Region grouping follows
    classify_record (src/stacking.py) as in scripts/18/29/33; note the
    paper's assignment CSV places U1391 in Tropical whereas Table S4 and
    the Layer-1 stacks count it under North Atlantic."""
    records, _ = load_all_records(DATA_DIR, age_min=10.0, age_max=60.0)
    stems = load_canonical_stems()
    # one repository filename carries a trailing space before its
    # extension ('32_Bereiter et al.-2015-GRL-revised CO2 .xlsx') - strip
    # on both sides of the comparison
    records = [(i, s, r, n.strip()) for i, s, r, n in records
               if n.strip() in stems]
    assert len(records) == 80, f"canonical 10-60 ka load = {len(records)}"
    records = [(i, s, 'Antarctic' if r == 'Southern_Ocean' else r, n)
               for i, s, r, n in records]
    region_dict = build_region_dict(records)
    region_dict = {r: v for r, v in region_dict.items() if r in REGION_ORDER}
    for r in REGION_ORDER:
        assert r in region_dict, f"region {r} missing on the 10-60 ka grid"
    return region_dict


def load_noise_ratios(region_dict):
    """{region_key: (achieved_ratio, theoretical_1_sqrt_K)} from scripts/29.

    Re-derives the CSV numbers from the current data files (same code
    path as scripts/29: fit_ar1 member median + smoothed median stack)
    and asserts agreement, so the annotation cannot drift from the data.
    """
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'stacking_noise_reduction.csv'))
    assert list(df.columns) == ['region', 'n_records_10_60', 's4_count',
                                'median_member_noise', 'stack_noise',
                                'achieved_ratio', 'theoretical_1_sqrt_K',
                                'rms_age_error_ka'], "noise CSV schema drifted"
    out = {}
    for _, row in df.iterrows():
        key = NOISE_REGION_MAP[row.region]
        assert key in REGION_ORDER, f"noise CSV region {row.region} unknown"
        assert int(row.n_records_10_60) == len(region_dict[key]), \
            f"{key}: CSV count {int(row.n_records_10_60)} != "
        assert int(row.s4_count) == S4_COUNTS[key], \
            f"{key}: S4 count drifted"
        # re-derive from data (29's exact definition)
        recs = region_dict[key]
        member_noise = np.array([fit_ar1(x) for x, _, _ in recs])
        med = float(np.median(member_noise))
        assert abs(med - float(row.median_member_noise)) < 1e-9, \
            f"{key}: median member noise drifted from scripts/29"
        stack = np.median(np.column_stack(
            [(x - x.mean()) / x.std() for x, _, _ in recs]), axis=1)
        from scipy.ndimage import gaussian_filter1d
        stack = gaussian_filter1d(stack, sigma=SMOOTH_SIGMA / 0.05,
                                  mode='nearest')
        stack_noise = float(fit_ar1(stack))
        assert abs(stack_noise - float(row.stack_noise)) < 1e-9, \
            f"{key}: stack noise drifted from scripts/29"
        out[key] = (float(row.achieved_ratio), float(row.theoretical_1_sqrt_K))
    assert set(out) == set(REGION_ORDER), "noise CSV does not cover 6 regions"
    return out


def build_stacks(region_dict):
    """Per-region median-scheme perturbation ensemble (seed 42)."""
    rng = np.random.default_rng(42)
    stacks = {}
    for region in REGION_ORDER:
        recs = region_dict[region]
        schemes = stackviz.stack_schemes(recs, AGE_GRID_10_60, N_PERTURB,
                                         SMOOTH_SIGMA, rng)
        stacks[region] = schemes['median']
        assert stacks[region].shape == (N_PERTURB, len(AGE_GRID_10_60)), \
            f"{region}: stack shape drifted"
    return stacks


# ------------------------------------------------------------------ plot ----
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


def panel(fig, gs, k, region, recs, stacks, noise):
    grid = AGE_GRID_10_60
    ax = fig.add_subplot(gs[k // 3, k % 3])
    apply_paleo_style(ax)
    ax.minorticks_off()

    # climate windows as alternating neutral bands (behind everything)
    for j, wname in enumerate(WINDOW_ORDER):
        w0, w1 = CLIMATE_WINDOWS[wname]
        ax.add_patch(Rectangle((w0, -3.6), w1 - w0, 7.2,
                               facecolor=WIN_COL[j % 2],
                               edgecolor="none", zorder=0))
    # grid over the window bands, under the data (33's z-order)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.8, zorder=0.5)

    # member records (thin); records have variable lengths, so resample
    # each onto the panel grid before plotting (scripts/18/33 idiom)
    for interp, sigma, name in recs:
        idx = np.round(np.linspace(0, len(interp) - 1, len(grid))).astype(int)
        ax.plot(grid, interp[idx], color=MEMBER, lw=0.4, alpha=0.7, zorder=1)

    # median stack + P5-P95 perturbation band (Layer-1 stack symbol)
    med = np.median(stacks, axis=0)
    p5, p95 = np.percentile(stacks, [5, 95], axis=0)
    ax.fill_between(grid, p5, p95, color=BAND, alpha=0.45, lw=0, zorder=2)
    ax.plot(grid, med, color=REGION_CATS[region], lw=2.2, zorder=3)

    # title block: centred title; the noise comparison right-aligned just
    # below it, above the axes so it never covers data lines
    ax.set_title(f"{stackviz.REGION_LABELS[region]}  (n = {len(recs)})",
                 fontsize=10.5, fontweight="bold", color=INK, pad=18)
    ratio, theo = noise[region]
    ax.text(0.995, 1.036, f"noise ×{ratio:.2f} vs 1/√K {theo:.2f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color=INK)

    ax.set_xlim(10, 60)
    ax.invert_xaxis()
    ax.set_ylim(-3.6, 3.6)
    if k // 3 == 1:
        ax.set_xlabel("Age (ka BP)", fontsize=10, color=INK)
    else:
        ax.tick_params(labelbottom=False)
    if k % 3 == 0:
        ax.set_ylabel("standardized value", fontsize=10, color=INK,
                      labelpad=8)
    else:
        ax.tick_params(labelleft=False)


def window_strip(fig):
    """Shared decode strip for the six climate-window bands."""
    lax = fig.add_axes([GRID_L, STRIP_Y, GRID_R - GRID_L, STRIP_H])
    lax.axis("off")
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)
    lax.text(0.0, 0.5, "climate windows:", fontsize=9, color=INK,
             ha="left", va="center")
    x = 0.175
    for j, wname in enumerate(WINDOW_ORDER):
        lax.add_patch(Rectangle((x, 0.30), 0.030, 0.40,
                                facecolor=WIN_COL[j % 2],
                                edgecolor=SWATCH_EDGE, lw=0.5))
        lax.text(x + 0.040, 0.5, WINDOW_ABBR[wname], fontsize=9, color=INK,
                 ha="left", va="center")
        x += 0.135


def plot(region_dict, stacks, noise):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig = Figure(figsize=(FIG_W / 25.4, FIG_H / 25.4), facecolor="white")
    gs = fig.add_gridspec(2, 3, left=GRID_L, right=GRID_R, top=GRID_T,
                          bottom=GRID_B, wspace=WSPACE, hspace=HSPACE)

    fig.text(0.048, 0.973, "Layer 1 — regional stacking on the 10–60 ka "
             "grid", fontsize=10.5, fontweight="bold", color=INK)
    fig.text(0.048, 0.949,
             "median across M = 200 age-perturbation realizations "
             "(Gaussian low-pass σ = 0.5 ka)", fontsize=9, color=INK)

    for k, region in enumerate(REGION_ORDER):
        panel(fig, gs, k, region, region_dict[region], stacks[region],
              noise)

    window_strip(fig)

    # footer: symbol decode + data facts + provenance (each line short
    # enough to clear the right edge - the 06/07 lesson, QA-checked)
    n = {r: len(region_dict[r]) for r in REGION_ORDER}
    fig.text(0.048, FOOTER_Y[0],
             "thin = member records (standardized); bold = median of the "
             "M = 200 age-perturbation realizations",
             fontsize=9, color=INK)
    fig.text(0.048, FOOTER_Y[1],
             "with Gaussian smoothing σ = 0.5 ka; shaded = P5–P95 band",
             fontsize=9, color=INK)
    fig.text(0.048, FOOTER_Y[2],
             "alternating bands = climate windows GS-12 / GI-12 / H4 / "
             "LGM / H1 / B-A; upper right: noise × vs 1/√K",
             fontsize=9, color=INK)
    fig.text(0.048, FOOTER_Y[3],
             "noise × = stack / median-member AR(1) innovation σ; 1/√K = "
             "independent equal-variance floor",
             fontsize=9, color=INK)
    fig.text(0.048, FOOTER_Y[4],
             f"members (Table S4): {n['North_Atlantic']} / {n['Antarctic']} "
             f"/ {n['Tropical']} / {n['East_Asian_Monsoon']} / "
             f"{n['South_America']} / {n['Other_Lakes']} = "
             f"{sum(n.values())} on the 10–60 ka grid (Δt = 0.05 ka)",
             fontsize=9, color=INK)
    fig.text(0.048, FOOTER_Y[5],
             "data: scripts/18 stackviz (seed 42) + "
             "outputs/stacking_noise_reduction.csv (scripts/29)",
             fontsize=9, color=INK)

    assert no_text_overruns(fig), "some text runs off the figure edges"
    save(fig, os.path.join(OUT_DIR, "layer1_stacking"))
    print_anchors(region_dict, stacks, noise)


# ---------------------------------------------------------------- anchors ----
def print_anchors(region_dict, stacks, noise):
    print("\n== anchors (Fig. 2 Layer-1 regional stacking) ==")
    grid = AGE_GRID_10_60
    print(f"  grid: {grid[0]:g}-{grid[-1]:g} ka, dt = "
          f"{grid[1] - grid[0]:.2f} ka ({len(grid)} points), "
          f"M = {N_PERTURB}, seed 42, sigma_smooth = {SMOOTH_SIGMA} ka")
    print(f"  stacks: {stacks[REGION_ORDER[0]].shape[0]} realizations x "
          f"{stacks[REGION_ORDER[0]].shape[1]} points per region")
    total = sum(len(region_dict[r]) for r in REGION_ORDER)
    for r in REGION_ORDER:
        ratio, theo = noise[r]
        med = np.median(stacks[r], axis=0)
        p5, p95 = np.percentile(stacks[r], [5, 95], axis=0)
        print(f"  {r:20s} K={len(region_dict[r]):2d}  noise ratio "
              f"{ratio:.3f} vs 1/sqrt(K) {theo:.3f}  "
              f"stack span [{np.min(p5):+.2f}, {np.max(p95):+.2f}]")
    assert total == 80, f"total records {total} != 80"
    assert len(grid) == 1001 and abs(grid[1] - grid[0] - 0.05) < 1e-12
    for r in REGION_ORDER:
        assert len(region_dict[r]) == S4_COUNTS[r], f"{r}: S4 count drifted"
    print(f"  counts: {total}/80 records match Table S4 "
          f"(13/18/34/6/3/6); noise CSV re-derived from data, |dev| < 1e-9")
    print("  claim: regional stacking suppresses record-level noise; the "
          "gap to the 1/√K floor quantifies spatial correlation and "
          "variance heterogeneity.")


def main():
    region_dict = load_region_dict()
    noise = load_noise_ratios(region_dict)
    stacks = build_stacks(region_dict)
    plot(region_dict, stacks, noise)


if __name__ == "__main__":
    main()
