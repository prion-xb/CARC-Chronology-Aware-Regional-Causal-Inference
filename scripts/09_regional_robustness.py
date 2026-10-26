#!/usr/bin/env python3
"""
09_regional_robustness.py
-------------------------
Directional robustness analysis for regional pairs across climate windows
(paper Section 4.3), computed from on-the-fly regional stacks.

Pipeline per the paper:
  1. Load all records (10–60 ka), merge Southern Ocean into the Antarctic
     composite ("Antarctic/Southern Ocean"), and stack the six regional
     composites with a 200-member age-perturbation ensemble.
  2. ARC-deconfound every composite against the North Atlantic stack
     (α=0.5), which carries the NGRIP common-forcing signal.
  3. For each climate window and each region pair, compute directional
     robustness via age-perturbed pairwise Granger tests.
  4. Error-scaling: shrink the pairing's chronology errors until the
     95% threshold is crossed.

Output: figures/FigS_Directional_Robustness.png
        outputs/regional_robustness_windows.csv
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from src.config import (DATA_DIR, OUTPUT_DIR, FIGURE_DIR, N_PERTURB,
                         ARC_ALPHA, MAX_GRANGER_LAG, NOISE_STD,
                         AGE_GRID_10_60, CLIMATE_WINDOWS, STABILITY_THRESHOLD)
from src.io import load_all_records, filter_canonical_records
from src.stacking import build_region_dict, stack_region
from src.arc import arc_deconfound
from src.granger import age_perturb_pair
from src.visualization import (apply_style, CARC_COLORS, CARC_COLORS_LIGHT,
                                add_panel_label, add_minor_ticks, save_figure)

import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


# ---------------------------------------------------------------------------
# Layer 3 (per the paper): ADF stationarity check with first-differencing,
# then Granger F-test across lags 1..maxlag.
# ---------------------------------------------------------------------------
def granger_adf(Y, X, maxlag=MAX_GRANGER_LAG, alpha=0.05):
    """Test whether X Granger-causes Y, differencing series whose ADF test
    does not reject a unit root (paper Layer 3)."""
    Y, X = np.asarray(Y, float), np.asarray(X, float)
    try:
        Y = np.diff(Y) if adfuller(Y, autolag='AIC')[1] > 0.05 else Y
        X = np.diff(X) if adfuller(X, autolag='AIC')[1] > 0.05 else X
    except Exception:
        return False
    n = min(len(Y), len(X))
    Y, X = Y[:n], X[:n]
    if n < maxlag + 3:
        return False
    try:
        res = grangercausalitytests(pd.DataFrame({'Y': Y, 'X': X}), maxlag, verbose=False)
        pvals = [res[lag][0]['ssr_ftest'][1] for lag in range(1, maxlag + 1)]
        return min(pvals) < alpha
    except Exception:
        return False


def robustness_adf(Y, X, sigma_err, n_perturb=N_PERTURB, maxlag=MAX_GRANGER_LAG,
                   alpha=0.05, noise_std=NOISE_STD, rng=None, dt=1.0):
    """CARC robustness with the paper's Layer-3 Granger variant.

    ``sigma_err`` is in physical ka; ``dt`` is the grid spacing in ka
    (0.05 for the 10–60 ka grid).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    sig = 0
    for _ in range(n_perturb):
        Yp, Xp = age_perturb_pair(Y, X, sigma_err, rng, dt=dt)
        if noise_std > 0:
            Yp = Yp + rng.normal(0, noise_std * np.std(Yp), size=len(Yp))
            Xp = Xp + rng.normal(0, noise_std * np.std(Xp), size=len(Xp))
        if granger_adf(Yp, Xp, maxlag, alpha):
            sig += 1
    return sig / n_perturb

# The paper's six regional composites (classify_record region names)
PAPER_REGIONS = ['North_Atlantic', 'Antarctic', 'Tropical',
                 'East_Asian_Monsoon', 'South_America', 'Other_Lakes']

REGION_LABELS = {
    'North_Atlantic': 'North Atlantic',
    'Antarctic': 'Antarctic/SO',
    'Tropical': 'Tropical',
    'East_Asian_Monsoon': 'E. Asian Monsoon',
    'South_America': 'S. Am. Monsoon',
    'Other_Lakes': 'High-Lat Lakes',
}


def run_analysis():
    print("Loading records (10-60 ka)...")
    records, grid = load_all_records(DATA_DIR, age_min=10.0, age_max=60.0)

    # Restrict to the paper's 80-record inventory (Table S2/S4); the data
    # directory's extra files (MD04-2881_Ba_128-760ka, MD04-2881_Br_128-760ka,
    # Lapa_d18O) are not in the paper.
    records = filter_canonical_records(records)

    # Merge Southern Ocean into Antarctic — paper's "Antarctic/Southern Ocean"
    records = [(i, s, 'Antarctic' if r == 'Southern_Ocean' else r, n)
               for i, s, r, n in records]

    region_dict = build_region_dict(records)
    region_dict = {r: v for r, v in region_dict.items() if r in PAPER_REGIONS}
    print("\nRegional composites:")
    for r in PAPER_REGIONS:
        if r in region_dict:
            print(f"  {REGION_LABELS.get(r, r)}: {len(region_dict[r])} records")

    # --- Layer 1: age-perturbation stacking ---
    rng = np.random.default_rng(42)
    stacks, stack_sigmas = {}, {}
    for region, recs in region_dict.items():
        stacks[region] = stack_region(recs, grid, N_PERTURB, rng, smooth_sigma=0.5)
        # Stack-level error: median record sigma / sqrt(N) (error reduction
        # under independent stacking; matches the paper's ~0.45 ka RMS for
        # the SAM-North Atlantic pairing)
        med_sigma = float(np.median([s for _, s, _ in recs]))
        stack_sigmas[region] = med_sigma / np.sqrt(len(recs))
        print(f"  stack {REGION_LABELS.get(region, region)}: "
              f"1σ = {stack_sigmas[region]:.3f} ka "
              f"(median {med_sigma:.2f} / √{len(recs)})")

    # --- Layer 2: ARC with the North Atlantic composite as common-forcing proxy ---
    proxy_stack = stacks['North_Atlantic']
    arc = {}
    for region, stack in stacks.items():
        if region == 'North_Atlantic':
            arc[region] = stack.copy()
        else:
            arc[region] = np.stack([
                arc_deconfound(stack[p], proxy_stack[p], ARC_ALPHA)
                for p in range(N_PERTURB)])
        for p in range(N_PERTURB):
            mu, sd = arc[region][p].mean(), arc[region][p].std()
            arc[region][p] = (arc[region][p] - mu) / sd
    print(f"\nARC proxy: North Atlantic (α={ARC_ALPHA})")

    # --- Layer 3+4: windowed directional robustness ---
    pairs = [(a, b) for i, a in enumerate(PAPER_REGIONS)
             for b in PAPER_REGIONS[i + 1:]]
    pairs = [(a, b) for a, b in pairs if a in arc and b in arc]

    all_results = {}
    for wlabel, (w0, w1) in CLIMATE_WINDOWS.items():
        idx0, idx1 = np.searchsorted(grid, w0), np.searchsorted(grid, w1)
        if idx1 - idx0 < MAX_GRANGER_LAG + 3:
            continue
        all_results[wlabel] = {}
        for a, b in pairs:
            sig_err = np.sqrt((stack_sigmas[a] ** 2 + stack_sigmas[b] ** 2) / 2)
            ya = np.median(arc[a], axis=0)[idx0:idx1]
            yb = np.median(arc[b], axis=0)[idx0:idx1]
            if len(ya) < MAX_GRANGER_LAG + 3:
                continue
            r_ab = robustness_adf(yb, ya, sig_err, n_perturb=N_PERTURB,
                                  maxlag=MAX_GRANGER_LAG, noise_std=NOISE_STD,
                                  dt=grid[1] - grid[0])
            r_ba = robustness_adf(ya, yb, sig_err, n_perturb=N_PERTURB,
                                  maxlag=MAX_GRANGER_LAG, noise_std=NOISE_STD,
                                  dt=grid[1] - grid[0])
            all_results[wlabel][(a, b)] = (r_ab, r_ba)

    # --- Report ---
    print("\n" + "=" * 78)
    print("Directional robustness by window and region pair (A→B / B→A)")
    print("=" * 78)
    rows = []
    for wlabel in CLIMATE_WINDOWS:
        if wlabel not in all_results:
            continue
        print(f"\n{wlabel}:")
        for a, b in pairs:
            if (a, b) not in all_results[wlabel]:
                continue
            r_ab, r_ba = all_results[wlabel][(a, b)]
            flag = "  <-- bidirectional ≥95%" if (r_ab >= STABILITY_THRESHOLD and
                                                  r_ba >= STABILITY_THRESHOLD) else ""
            print(f"  {REGION_LABELS.get(a, a):<16} ↔ {REGION_LABELS.get(b, b):<16} "
                  f"{r_ab:6.1%} / {r_ba:6.1%}{flag}")
            rows.append({'window': wlabel, 'region_A': a, 'region_B': b,
                         'A_to_B': r_ab, 'B_to_A': r_ba})

    res_df = pd.DataFrame(rows)
    out_csv = os.path.join(OUTPUT_DIR, 'regional_robustness_windows.csv')
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # --- Error-scaling for the strongest single-direction pairing ---
    best = None
    for wlabel in all_results:
        for (a, b), (r_ab, r_ba) in all_results[wlabel].items():
            for src, dst, r in [(a, b, r_ab), (b, a, r_ba)]:
                if best is None or r > best[0]:
                    best = (r, wlabel, src, dst)
    r_max, w_max, s_best, t_best = best
    print(f"\nStrongest single-direction robustness: {REGION_LABELS.get(s_best, s_best)} → "
          f"{REGION_LABELS.get(t_best, t_best)} = {r_max:.1%} ({w_max})")

    # Error-scaling targets the paper's key pairing: SAM → North Atlantic in H1
    # (pairs are stored as ('North_Atlantic', 'South_America'); [1] = SAM → NA)
    s_best, t_best, w_max = 'South_America', 'North_Atlantic', 'H1 event'
    r_max = all_results[w_max][(t_best, s_best)][1]
    print(f"Error-scaling target (paper): {REGION_LABELS.get(s_best, s_best)} → "
          f"{REGION_LABELS.get(t_best, t_best)} = {r_max:.1%} ({w_max})")

    sig_best = np.sqrt((stack_sigmas[s_best] ** 2 + stack_sigmas[t_best] ** 2) / 2)
    w0, w1 = CLIMATE_WINDOWS[w_max]
    idx0, idx1 = np.searchsorted(grid, w0), np.searchsorted(grid, w1)
    ys = np.median(arc[s_best], axis=0)[idx0:idx1]
    yt = np.median(arc[t_best], axis=0)[idx0:idx1]

    print("\nError-scaling (fraction of present 1σ errors → robustness):")
    crossing = None
    for scale in [1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.2]:
        r = robustness_adf(yt, ys, sig_best * scale, n_perturb=N_PERTURB,
                           maxlag=MAX_GRANGER_LAG, noise_std=NOISE_STD,
                           dt=grid[1] - grid[0])
        print(f"  {scale:.2f} → {r:.1%}")
        if crossing is None and r >= STABILITY_THRESHOLD:
            crossing = scale
    if crossing is not None:
        print(f"  95% threshold crossed at ~{crossing:.0%} of present errors")
    else:
        print("  95% threshold not reached at any tested scale")

    return all_results, pairs


def load_plot_data():
    """Rebuild results from the saved CSV (fast, no compute)."""
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'regional_robustness_windows.csv'))
    results = {}
    pairs = []
    for _, row in df.iterrows():
        pair = (row['region_A'], row['region_B'])
        if pair not in pairs:
            pairs.append(pair)
        results.setdefault(row['window'], {})[pair] = (row['A_to_B'], row['B_to_A'])
    return results, pairs


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plot-only", action="store_true",
                    help="rebuild figure from outputs/regional_robustness_windows.csv")
    ap.add_argument("--csv-only", action="store_true",
                    help="run the analysis and save the CSV, skip the figure")
    args = ap.parse_args()

    if args.plot_only:
        print("Plot-only mode: loading saved table...")
        all_results, pairs = load_plot_data()
    else:
        all_results, pairs = run_analysis()

    # --- Figure ---
    if not args.csv_only:
        plot_pairs(results=all_results, pairs=pairs, labels=REGION_LABELS)

    print("\nDone.")


# ---------------------------------------------------------------------------
# Plotting: bar chart + asymmetry scatter for the three data-rich windows
# ---------------------------------------------------------------------------
def _place_labels(ax, points, names, fontsize=8, obstacles=()):
    """Place text labels next to scatter points without collisions.

    Greedy with two refinements: labels in the densest region are placed
    first (so they claim the best offsets), then colliding labels re-try
    their offset ladder in a few repair passes. Deterministic. ``obstacles``
    are display-space Bboxes (e.g. an in-axes legend) labels must also avoid.
    """
    canvas = ax.figure.canvas
    renderer = canvas.get_renderer()
    canvas.draw()                        # once, so transforms are current
    ax_bbox = ax.get_window_extent()
    xspan = ax.get_xlim()[1] - ax.get_xlim()[0]
    yspan = ax.get_ylim()[1] - ax.get_ylim()[0]
    dpi = ax.figure.dpi
    # candidate offsets in display px, spiralling outward ring by ring so a
    # tight cluster fans its labels into the surrounding whitespace (the six
    # points near the bottom-left corner exhaust a hand-picked ladder's right
    # and upward reach, which maxed at +44/+26 px)
    offsets = []
    for r in (10, 22, 34, 46, 58):
        for k in range(8):
            th = (k + 0.5) * np.pi / 4 + np.pi / 8
            offsets.append((r * np.cos(th), r * np.sin(th)))
    offsets = [tuple(round(float(v)) for v in o) for o in offsets]

    def _try(lab, x, y, placed, skip=-1, pad=1):
        """Return bbox of first collision-free offset, else None.

        ``pad`` lets repair passes push labels slightly into the axes margin
        (up to pad px) when the in-axes ladder is exhausted by a tight cluster.
        """
        fb = lab.figure.bbox
        # restore on failure: the caller of a failed repair pass keeps the
        # old bbox, so the label must not be left stranded on the last
        # candidate it tried (the labels below the axes bottom were exactly
        # that — _try set the last offset, failed, and never moved back)
        orig_pos = lab.get_position()
        for dx, dy in offsets:
            # points -> data units via the axes extent
            ox = x + dx / 72.0 * dpi * xspan / ax_bbox.width
            oy = y + dy / 72.0 * dpi * yspan / ax_bbox.height
            lab.set_position((ox, oy))
            bb = lab.get_window_extent(renderer)
            if (bb.x0 < ax_bbox.x0 - pad or bb.y0 < ax_bbox.y0 - pad
                    or bb.x1 > ax_bbox.x1 + pad or bb.y1 > ax_bbox.y1 + pad):
                continue
            # also never past the canvas (a tight bottom cluster plus the
            # margin allowance could push labels off the figure entirely)
            if (bb.x0 < fb.x0 + 1 or bb.y0 < fb.y0 + 1
                    or bb.x1 > fb.x1 - 1 or bb.y1 > fb.y1 - 1):
                continue
            if all(_overlap(bb, p) == 0 for k, p in enumerate(placed)
                   if k != skip and p is not None):
                return bb
        lab.set_position(orig_pos)
        return None

    labels = []
    for (x, y), name in zip(points, names):
        lab = ax.text(x, y, name, fontsize=fontsize, ha='left', va='bottom',
                      bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                                edgecolor='#cccccc', alpha=0.9))
        labels.append((lab, x, y))

    # place densest region first
    cx = float(np.mean([p[0] for p in points]))
    cy = float(np.mean([p[1] for p in points]))
    order = sorted(range(len(labels)),
                   key=lambda i: (points[i][0] - cx) ** 2 + (points[i][1] - cy) ** 2)
    placed = [None] * len(labels)
    for ob in obstacles:
        placed.append(ob)               # obstacles never move / never skipped
    for i in order:
        lab, x, y = labels[i]
        bb = _try(lab, x, y, placed)
        if bb is None:
            lab.set_position((x, y))     # last resort: sit on the point
            bb = lab.get_window_extent(renderer)
        placed[i] = bb

    # repair passes: colliding labels re-try their ladder, with a growing
    # allowance to spill into the axes margin when a cluster is tight
    for pad in (1, 8, 15):
        moved = False
        for i in range(len(labels)):
            if all(_overlap(placed[i], p) == 0
                   for k, p in enumerate(placed) if k != i and p is not None):
                continue
            lab, x, y = labels[i]
            bb = _try(lab, x, y, placed, skip=i, pad=pad)
            if bb is not None:
                placed[i] = bb
                moved = True
        if not moved:
            break


def _overlap(b1, b2):
    x0 = max(b1.x0, b2.x0)
    y0 = max(b1.y0, b2.y0)
    x1 = min(b1.x1, b2.x1)
    y1 = min(b1.y1, b2.y1)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def plot_pairs(results, pairs, labels):
    apply_style("paper", font_scale=1.0)  # print size: base ~8 pt
    data_windows = ['H1 event', 'H4 event', 'Last Glacial Maximum']

    # Bar chart: all pairs in the data-rich windows
    pair_rows = []
    for wl in data_windows:
        if wl not in results:
            continue
        for (a, b) in pairs:
            if (a, b) not in results[wl]:
                continue
            r_ab, r_ba = results[wl][(a, b)]
            pair_rows.append((f"{labels.get(a, a)} vs {labels.get(b, b)}",
                              r_ab, r_ba, wl))

    fig = plt.figure(figsize=(7.0, 5.5))  # 178 mm wide, same aspect
    ax1 = fig.add_subplot(2, 1, 1)
    x = np.arange(len(pair_rows))
    width = 0.32
    rob_ab = [r[1] for r in pair_rows]
    rob_ba = [r[2] for r in pair_rows]
    ax1.bar(x - width / 2, rob_ab, width, label='A→B',
            color=CARC_COLORS["red"], alpha=0.8, edgecolor="#333333", linewidth=0.5)
    ax1.bar(x + width / 2, rob_ba, width, label='B→A',
            color=CARC_COLORS["blue"], alpha=0.8, edgecolor="#333333", linewidth=0.5)
    ax1.set_ylabel('Robustness Score', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{r[0]}\n({r[3]})" for r in pair_rows],
                        rotation=45, ha='right', fontsize=7)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=10, framealpha=0.8)
    ax1.grid(axis='y', linestyle='dotted', alpha=0.25)
    ax1.axhline(0.95, color='grey', linestyle='--', alpha=0.5, linewidth=1.2)
    add_minor_ticks(ax1)
    ax1.set_title('Directional Robustness by Regional Pair (H1/H4/LGM)',
                  fontweight='bold', fontsize=10)
    add_panel_label(ax1, '(a)')

    # Scatter: asymmetry
    ax2 = fig.add_subplot(2, 1, 2)
    win_colors = {'H1 event': CARC_COLORS["purple"],
                  'H4 event': CARC_COLORS["orange"],
                  'Last Glacial Maximum': CARC_COLORS["teal"]}
    for wl in data_windows:
        m = [i for i, r in enumerate(pair_rows) if r[3] == wl]
        ax2.scatter([rob_ab[i] for i in m], [rob_ba[i] for i in m],
                    c=win_colors[wl], s=100, alpha=0.8, zorder=5,
                    edgecolors="white", linewidth=1, label=wl)
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1.2, alpha=0.7, label='y=x (symmetry)')
    ax2.set_xlabel('Robustness (A→B)', fontsize=10)
    ax2.set_ylabel('Robustness (B→A)', fontsize=10)
    ax2.set_xlim(0, 1.05)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle='dotted', alpha=0.25)
    # no set_aspect('equal'): a square axes inside the wide panel shrank to
    # ~320 px, piling the labelled pairs into an unresolvable corner cluster
    # and forcing centred titles into the '(b)' panel label (layout QA)
    add_minor_ticks(ax2)
    ax2.set_title('Directional Asymmetry Analysis',
                  fontweight='bold', fontsize=10)
    ax2.text(0.985, 1.02, 'labels: pairs with |A→B − B→A| ≥ 0.08',
             transform=ax2.transAxes, ha='right', va='bottom',
             fontsize=7, color='#888888')
    add_panel_label(ax2, '(b)')
    # label only the asymmetric pairs (|A→B − B→A| ≥ 0.08) — points hugging
    # the diagonal are symmetric by construction; naming all 45 would turn
    # the dense upper corner into an unreadable label pile. Short codes
    # (full names in panel a): NA, SO, TR, EA, SA, HL.
    REGION_CODE = {'North_Atlantic': 'NA', 'Antarctic': 'SO', 'Tropical': 'TR',
                   'East_Asian_Monsoon': 'EA', 'South_America': 'SA',
                   'Other_Lakes': 'HL'}
    asym_idx = [i for i in range(len(pair_rows))
                if abs(rob_ab[i] - rob_ba[i]) >= 0.08]
    # pair_rows stores only the display name; rebuild codes from the window
    # table instead of parsing names
    name_to_code = {}
    for (a, b) in pairs:
        name_to_code[f"{labels.get(a, a)} vs {labels.get(b, b)}"] = \
            f"{REGION_CODE.get(a, a)}–{REGION_CODE.get(b, b)}"
    code_labels = [name_to_code.get(pair_rows[i][0],
                                    pair_rows[i][0].replace(' ', ''))
                   for i in asym_idx]
    # suptitle inside the reserved top strip (tight_layout rect)
    fig.suptitle("CARC Directional Robustness — Regional Causal Asymmetry",
                 fontsize=14, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    # legend and label placement AFTER tight_layout: both are measured in
    # display pixels, so the layout must be final first (running the placer
    # pre-layout left labels colliding after the axes moved)
    leg = ax2.legend(fontsize=10, framealpha=0.8)
    fig.canvas.draw()
    leg_bb = leg.get_window_extent(fig.canvas.get_renderer())
    _place_labels(ax2, [(rob_ab[i], rob_ba[i]) for i in asym_idx],
                  code_labels, fontsize=8, obstacles=[leg_bb])
    save_figure(fig, "FigS_Directional_Robustness", fmts=("pdf", "png"))
    plt.close()
    print("Figure saved.")


if __name__ == "__main__":
    main()
