#!/usr/bin/env python3
"""Elements for supplementary Figs. S2–S5 (renumbered supplement).

The submitted composites packed several messages onto one canvas and the
co-author flagged overlapping text. Each message here becomes its own small
element, sized to be legible at single-column width, for the user to lay out
in Illustrator. Elements carry no panel letters and no suptitle.

  Fig. S2 (Record strip)  -> one element per region (7): that region's
       records as horizontal bars of best outgoing robustness, with honesty
       status (● 95% / ◌ 80% / ○ none; * no coordinates) and defect letters
       (n/a/r = recovered by dropping noise/age-error/ARC, t = trend leak).
       Data: outputs/pairwise_scan/ via load_atlas_data() in
       scripts/20_map_atlas.py (same CSVs the published strip reads).

  Fig. S3 (Conditional causality) -> one grouped-bar element per classical
       pair of interest (3): A→B vs B→A stability across the six climate
       windows. Data: outputs/conditional_causality_allpairs.csv via
       load_plot_data() in scripts/06_conditional_causality.py.

  Fig. S4 (Six-record pairwise matrix) -> the 6x6 directed-robustness
       matrix, one element (line + median per cell, 95%/80% thresholds,
       stars for the classical pairs, accent border for asymmetric cells).
       NOTE: load_plot_data() in script 06 builds var_names from record_A
       only, so Cheng2016 (present only as record_B) silently drops out and
       the published matrix rendered 5x5. This element rebuilds var_names
       from the CSV's full pair set, i.e. the six-record reference set.

  Fig. S5 (Regional directional robustness) -> one grouped-bar element per
       data-rich window (H1, H4, LGM; 15 pairs, A→B vs B→A) plus the
       asymmetry scatter (A→B vs B→A with the asymmetric pairs labelled) as
       its own element. Data: outputs/regional_robustness_windows.csv via
       load_plot_data() in scripts/09_regional_robustness.py.

Counting note (data trap): per-record counters in record_profile.csv count
every directed edge at BOTH endpoints, so summing them over records
double-counts by exactly 2x (e.g. n_trend_leak95 sums to 4,486 while the
per-pair table holds 2,243). All edge totals here are taken from the
per-pair shard tables.

Output: Figures/elements/figS2_record_strip/,
        Figures/elements/figS3_conditional_causality/,
        Figures/elements/figS4_pairwise_matrix/,
        Figures/elements/figS5_directional_robustness/
"""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from src.record_names import display_name
from src.elements import (ONE_COL_TALL, INK, INK_SEC, INK_MUTED, GRID,
                          use_element_style, save_element, safe_label, tidy)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPT_DIR, file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Original generating scripts, loaded so the numbers come from the same
# loaders the published figures used (never hard-coded).
atlas = _load("map_atlas", "20_map_atlas.py")
cond = _load("conditional_causality", "06_conditional_causality.py")
reg = _load("regional_robustness", "09_regional_robustness.py")

SUB2 = "figS2_record_strip"
SUB3 = "figS3_conditional_causality"
SUB4 = "figS4_pairwise_matrix"
SUB5 = "figS5_directional_robustness"

# Palette from the source figures (blue family for layer recovery, orange
# for trend leak, red/blue for the two directions).
BLUE, BLUE_MID, BLUE_PALE = atlas.BLUE, "#6da7ec", "#9ec5f4"
ORANGE = atlas.ORANGE
RED = cond.CARC_COLORS["red"]
PAIR_BLUE = cond.CARC_COLORS["blue"]

DEFECTS = [("n", "recov_nonoise", BLUE), ("a", "recov_noage", BLUE_MID),
           ("r", "recov_noarc", BLUE_PALE), ("t", "n_trend_leak95", ORANGE)]

# ---------------------------------------------------------------------------
# Fig. S2 — record strip, one element per region
# ---------------------------------------------------------------------------
REGION_ORDER = ["North_Atlantic", "Antarctic", "Southern_Ocean", "Tropical",
                "East_Asian_Monsoon", "South_America", "Other_Lakes"]
REGION_SLUG = {"North_Atlantic": "north_atlantic", "Antarctic": "antarctic",
               "Southern_Ocean": "southern_ocean", "Tropical": "tropical",
               "East_Asian_Monsoon": "east_asian_monsoon",
               "South_America": "south_america", "Other_Lakes": "high_lat_lakes"}


def el_region_strip(prof, region):
    sub = prof[prof["region"] == region].sort_values("max_rate_out",
                                                     ascending=False)
    n = len(sub)
    # margins scale with height so the title (top) and xlabel (bottom) never
    # clip, even for the 3-row South America element
    h = max(2.0, 0.72 + n * 0.25)
    fig = plt.figure(figsize=(5.7, h))
    bottom = max(0.10, 0.36 / h)
    top = 1 - 0.42 / h
    gs = fig.add_gridspec(1, 2, width_ratios=[2.15, 1.0], wspace=0.03,
                          left=0.015, right=0.985, top=top, bottom=bottom)
    ax_lab = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])
    y = np.arange(n)[::-1]
    ax_bar.barh(y, sub["max_rate_out"], height=0.62, color=sub["region_hue"],
                edgecolor="none", alpha=0.85)
    for yi, (_, r) in zip(y, sub.iterrows()):
        mark = "●" if r["any_pass95"] else ("◌" if r["any_pass80"] else "○")
        star = "*" if pd.isna(r["lat"]) else ""
        lab = f"{mark}{star} {safe_label(display_name(r['record']))}"
        ax_lab.text(1.0, yi, lab, ha="right", va="center", fontsize=6.8,
                    color=INK_SEC)
        x_def = max(r["max_rate_out"], 0) + 0.015
        for c, col, colr in DEFECTS:
            if r[col] > 0:
                ax_bar.text(x_def, yi, c, ha="left", va="center", fontsize=6.8,
                            fontweight="bold", color=colr)
                x_def += 0.045
    ax_lab.set_xlim(0, 1)
    ax_lab.set_ylim(-0.5, n - 0.5)
    ax_lab.axis("off")
    ax_bar.set_xlim(0, 1.20)
    ax_bar.set_ylim(-0.5, n - 0.5)
    ax_bar.set_yticks([])
    ax_bar.axvline(atlas.STABILITY_THRESHOLD, color=GRID, lw=0.8, ls=(0, (5, 3)))
    ax_bar.axvline(atlas.THRESH_80, color=GRID, lw=0.6, ls=(0, (1, 4)))
    ax_bar.set_xticks(np.arange(0, 1.01, 0.25))
    ax_bar.set_xlabel("best outgoing robustness", fontsize=7.5)
    tidy(ax_bar, grid_axis="x")
    fig.text(0.5, 0.985, f"{atlas.REGION_LABEL[region]} ({n})\n"
             "● 95% · ◌ 80% · ○ none · n/a/r = noise/age/ARC recovered · "
             "t = trend leak", ha="center", va="top", fontsize=7.0, color=INK)
    save_element(fig, f"figS2{chr(ord('a') + REGION_ORDER.index(region))}_"
                      f"{REGION_SLUG[region]}", subdir=SUB2, tight=False)


# ---------------------------------------------------------------------------
# Fig. S3 — conditional causality, one grouped-bar element per classical pair
# ---------------------------------------------------------------------------
PAIR_SLUGS = [("NGRIP", "EDC_dD", "figS3a_ngrip_edc_dd"),
              ("EDC_dD", "EDC_CO2", "figS3b_edc_dd_edc_co2"),
              ("Cariaco", "Cheng2016", "figS3c_cariaco_cheng2016")]


def el_cond_pair(results, v1, v2, slug):
    windows = list(cond.CLIMATE_WINDOWS.keys())
    r12 = [results[w][(v1, v2)]["v1→v2"] for w in windows]
    r21 = [results[w][(v1, v2)]["v2→v1"] for w in windows]
    fig, ax = plt.subplots(figsize=(4.6, 3.55))
    fig.subplots_adjust(left=0.115, right=0.975, top=0.88, bottom=0.245)
    x = np.arange(len(windows))
    w = 0.32
    ax.bar(x - w / 2, r12, w, color=RED, alpha=0.88,
           label=f"{display_name(v1)}→{display_name(v2)}")
    ax.bar(x + w / 2, r21, w, color=PAIR_BLUE, alpha=0.88,
           label=f"{display_name(v2)}→{display_name(v1)}")
    ax.axhline(0.95, color=INK_MUTED, lw=0.8, ls=(0, (5, 3)))
    ax.text(len(windows) - 0.5, 0.952, "95% stability", fontsize=6.2,
            color=INK_SEC, ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(windows, rotation=45, ha="right", fontsize=6.0)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Directional stability")
    ax.set_title(f"{display_name(v1)} ↔ {display_name(v2)}")
    ax.legend(loc="upper left", fontsize=6.4)
    tidy(ax, grid_axis="y")
    save_element(fig, slug, subdir=SUB3, tight=False)


# ---------------------------------------------------------------------------
# Fig. S4 — six-record pairwise robustness matrix (single element)
# ---------------------------------------------------------------------------
SIX_RECORDS = ["NGRIP", "EDC_dD", "EDC_CO2", "EDC_CH4", "Cariaco", "Cheng2016"]
ASYM_MIN = 0.30


def el_matrix(results):
    present = set()
    for wres in results.values():
        for (a, b) in wres:
            present |= {a, b}
    var_names = [v for v in SIX_RECORDS if v in present]
    # chronological window order, same as the published matrix's WINDOW_ORDER
    window_order = sorted(cond.CLIMATE_WINDOWS,
                          key=lambda k: cond.CLIMATE_WINDOWS[k][0])

    rates_rm = {}
    for label, res in results.items():
        for (v1, v2), r in res.items():
            rates_rm[(v1, v2, label)] = r["v1→v2"]
            rates_rm[(v2, v1, label)] = r["v2→v1"]

    fig = plt.figure(figsize=(6.3, 6.9))
    n = len(var_names)
    gs = fig.add_gridspec(n, n, wspace=0.10, hspace=0.16,
                          left=0.16, right=0.985, top=0.80, bottom=0.115)

    for i, src in enumerate(var_names):
        for j, dst in enumerate(var_names):
            ax = fig.add_subplot(gs[i, j])
            ax.set_xlim(-0.5, len(window_order) - 0.5)
            ax.set_ylim(-0.06, 1.06)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if i == j:  # diagonal: record name
                ax.set_facecolor("none")
                ax.text(0.5, 0.5, safe_label(display_name(src)), ha="center",
                        va="center", fontsize=6.8, fontweight="bold", color=INK,
                        transform=ax.transAxes)
                continue

            vals = np.array([rates_rm.get((src, dst, w), np.nan)
                             for w in window_order])
            med = np.nanmedian(vals)
            rev = np.array([rates_rm.get((dst, src, w), np.nan)
                            for w in window_order])
            if abs(np.nanmedian(vals) - np.nanmedian(rev)) >= ASYM_MIN:
                ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                       fill=False, edgecolor=atlas.ACCENT,
                                       linewidth=1.8, clip_on=False))

            ax.axhline(0.95, color=GRID, linewidth=0.7, ls=(0, (5, 3)), zorder=1)
            ax.axhline(0.80, color=GRID, linewidth=0.5, ls=(0, (1, 4)), zorder=1)
            xs = np.arange(len(window_order))
            ok = np.isfinite(vals)
            ax.plot(xs[ok], vals[ok], color=BLUE, linewidth=1.2, zorder=2,
                    alpha=0.85)
            ax.scatter(xs[ok], vals[ok], s=15, c=BLUE, alpha=0.85,
                       edgecolors="none", zorder=3)
            if (src, dst) in cond.PAIRS_OF_INTEREST or \
                    (dst, src) in cond.PAIRS_OF_INTEREST:
                for k, w in enumerate(window_order):
                    if np.isfinite(vals[k]):
                        ax.scatter([k], [vals[k]], marker="*", s=55, c="white",
                                   edgecolors=INK, linewidths=0.8, zorder=4)
            ax.plot([-0.35, len(window_order) - 0.65], [med, med],
                    color=atlas.BLUE_HI, linewidth=2.2, zorder=5,
                    solid_capstyle="round")

            if j == 0:
                ax.text(-0.50, 0.5, safe_label(display_name(src)), ha="right",
                        va="center", fontsize=6.8, color=INK,
                        transform=ax.transAxes)
            if i == 0:
                ax.text(0.5, 1.12, safe_label(display_name(dst)), ha="center",
                        va="bottom", fontsize=6.8, color=INK,
                        transform=ax.transAxes)

    fig.text(0.5, 0.985, "Six-record pairwise matrix — cell = row → col "
                         "robustness across BA, H1, LGM, H4, GI-12, GS-12",
             fontsize=7.2, color=INK, ha="center")

    handles = [
        Line2D([], [], marker="o", linestyle="-", ms=4.5, mfc=BLUE, mec="none",
               color=BLUE, linewidth=1.2,
               label="per-window robustness\n(chronological order)"),
        Line2D([0], [0], color=atlas.BLUE_HI, linewidth=2.2,
               label="median across windows"),
        Line2D([0], [0], color=GRID, linewidth=1, linestyle=(0, (5, 3)),
               label="95% honesty threshold"),
        Line2D([0], [0], color=GRID, linewidth=1, linestyle=(0, (1, 4)),
               label="80% threshold (relaxed)"),
        Line2D([], [], marker="*", linestyle="none", ms=9, mfc="white",
               mec=INK, mew=0.8, label="classical pair of interest"),
        Rectangle((0, 0), 1, 1, fill=False, edgecolor=atlas.ACCENT,
                  linewidth=1.8, label="asymmetric pair\n(Δmedian ≥ 0.30)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=6.4, bbox_to_anchor=(0.5, 0.012), handlelength=1.3,
               columnspacing=1.4, handletextpad=0.4)

    print(f"S4: matrix records {var_names}, windows {window_order}")
    for i, src in enumerate(var_names):
        for j, dst in enumerate(var_names):
            if i != j:
                vals = np.array([rates_rm.get((src, dst, w), np.nan)
                                 for w in window_order])
                rev = np.array([rates_rm.get((dst, src, w), np.nan)
                                for w in window_order])
                if abs(np.nanmedian(vals) - np.nanmedian(rev)) >= ASYM_MIN:
                    print(f"S4: asymmetric {src}->{dst} "
                          f"med {np.nanmedian(vals):.2f} vs "
                          f"{np.nanmedian(rev):.2f}")
    save_element(fig, "figS4a_six_record_matrix", subdir=SUB4, tight=False)


# ---------------------------------------------------------------------------
# Fig. S5 — regional directional robustness (bars per window + scatter)
# ---------------------------------------------------------------------------
DATA_WINDOWS = ["H1 event", "H4 event", "Last Glacial Maximum"]
# Short codes from the published figure's asymmetry panel
REGION_CODE = {"North_Atlantic": "NA", "Antarctic": "SO", "Tropical": "TR",
               "East_Asian_Monsoon": "EA", "South_America": "SA",
               "Other_Lakes": "HL"}


def build_pair_rows(results, pairs):
    rows = []
    for wl in DATA_WINDOWS:
        if wl not in results:
            continue
        for (a, b) in pairs:
            if (a, b) not in results[wl]:
                continue
            r_ab, r_ba = results[wl][(a, b)]
            rows.append((f"{REGION_CODE.get(a, a)}–{REGION_CODE.get(b, b)}",
                         r_ab, r_ba, wl))
    return rows


def el_win_bars(rows, window, slug):
    sub = [r for r in rows if r[3] == window]
    x = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(5.9, 2.75))
    w = 0.34
    ax.bar(x - w / 2, [r[1] for r in sub], w, color=RED, alpha=0.88, label="A→B")
    ax.bar(x + w / 2, [r[2] for r in sub], w, color=PAIR_BLUE, alpha=0.88,
           label="B→A")
    ax.axhline(0.95, color=INK_MUTED, lw=0.8, ls=(0, (5, 3)))
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in sub], fontsize=6.5)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Robustness")
    ax.set_title(f"Directional robustness — {window}")
    ax.legend(loc="upper left", fontsize=6.4)
    tidy(ax, grid_axis="y")
    save_element(fig, slug, subdir=SUB5)


def el_asym_scatter(rows):
    win_colors = {"H1 event": reg.CARC_COLORS["purple"],
                  "H4 event": reg.CARC_COLORS["orange"],
                  "Last Glacial Maximum": reg.CARC_COLORS["teal"]}
    fig, ax = plt.subplots(figsize=(4.1, 4.1))
    for wl in win_colors:
        m = [i for i, r in enumerate(rows) if r[3] == wl]
        ax.scatter([rows[i][1] for i in m], [rows[i][2] for i in m],
                   c=win_colors[wl], s=60, alpha=0.85, edgecolors="white",
                   linewidth=0.8, zorder=5, label=wl)
    ax.plot([0, 1], [0, 1], color=INK_SEC, lw=0.9, ls=(0, (4, 2)),
            label="y = x (symmetry)")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_xlabel("Robustness (A→B)")
    ax.set_ylabel("Robustness (B→A)")
    ax.set_aspect("equal")
    ax.set_title("Directional asymmetry (|A→B − B→A| ≥ 0.08 labelled)")
    leg = ax.legend(loc="upper left", fontsize=6.2)
    fig.canvas.draw()
    leg_bb = leg.get_window_extent(fig.canvas.get_renderer())
    asym = [(i, rows[i]) for i in range(len(rows))
            if abs(rows[i][1] - rows[i][2]) >= 0.08]
    reg._place_labels(ax, [(r[1], r[2]) for _, r in asym],
                      [r[0] for _, r in asym], fontsize=6.6, obstacles=[leg_bb])
    tidy(ax, grid_axis="both")
    save_element(fig, "figS5d_asymmetry_scatter", subdir=SUB5)


# ---------------------------------------------------------------------------
def main():
    use_element_style()
    prof, pairs, pcm, ccm = atlas.load_atlas_data()
    cond_res, _ = cond.load_plot_data()
    reg_res, reg_pairs = reg.load_plot_data()

    # ---- sanity numbers, all from the per-pair table (see data trap above)
    full = pairs["pass95_full"].astype(bool)
    n95, n80 = int(full.sum()), int(pairs["pass80_full"].astype(bool).sum())
    recov = [int((~full & pairs[c].astype(bool)).sum())
             for c in ("pass95_nonoise", "pass95_noage", "pass95_noarc")]
    leak = int((full & ~pairs["pass95_alldiff"].astype(bool)).sum())
    alldiff95 = int(pairs["pass95_alldiff"].astype(bool).sum())
    print(f"S2: {len(prof)} records; regions "
          f"{dict(prof['region'].value_counts())}")
    print(f"S2: per-pair edges: {n95} at 95%, {n80} at 80%; "
          f"recovered noise/age/ARC = {recov}; trend leak {leak} "
          f"(profile sum {int(prof['n_trend_leak95'].sum())} = 2x); "
          f"alldiff survivors {alldiff95}")

    for region in REGION_ORDER:
        el_region_strip(prof, region)

    # ---- S3 sanity: rates for the classical pairs, window by window
    windows = list(cond.CLIMATE_WINDOWS.keys())
    for (v1, v2, slug) in PAIR_SLUGS:
        r12 = [cond_res[w][(v1, v2)]["v1→v2"] for w in windows]
        r21 = [cond_res[w][(v1, v2)]["v2→v1"] for w in windows]
        print(f"S3: {v1}↔{v2} A→B "
              + " ".join(f"{w}:{a:.2f}" for w, a in zip(windows, r12))
              + " | B→A "
              + " ".join(f"{w}:{b:.2f}" for w, b in zip(windows, r21)))
        el_cond_pair(cond_res, v1, v2, slug)

    el_matrix(cond_res)

    # ---- S5 sanity: per-window extremes + strongest single direction
    rows = build_pair_rows(reg_res, reg_pairs)
    for wl in DATA_WINDOWS:
        sub = [r for r in rows if r[3] == wl]
        bi = sum(1 for r in sub if r[1] >= 0.95 and r[2] >= 0.95)
        print(f"S5: {wl}: max A→B {max(r[1] for r in sub):.2f}, "
              f"max B→A {max(r[2] for r in sub):.2f}, "
              f"bidirectional ≥95% {bi}/15")
    best = max(rows, key=lambda r: max(r[1], r[2]))
    print(f"S5: strongest single direction "
          f"{best[0]} = {max(best[1], best[2]):.2f} ({best[3]})")
    for wl, slug in (("H1 event", "figS5a_bars_h1"),
                     ("H4 event", "figS5b_bars_h4"),
                     ("Last Glacial Maximum", "figS5c_bars_lgm")):
        el_win_bars(rows, wl, slug)
    el_asym_scatter(rows)


if __name__ == "__main__":
    main()
    from src.elements import report
    report()
