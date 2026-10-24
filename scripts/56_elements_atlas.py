#!/usr/bin/env python3
"""Elements for supplementary Fig. S1 — the record-level scrutinization atlas.

The submitted Fig. S1 packed seven axes onto one canvas: a Robinson map, four
bar blocks, and an 80-row label column. At page size the inner labels ran into
each other, which is the defect the co-author flagged. Here each block becomes
its own small plot with room for its own labels.

Data comes from the same CSVs the composite reads, through `load_atlas_data()`
in scripts/20_map_atlas.py, so the numbers match the published figure.

Two counting notes, because the composite got one of them wrong:
  * `n_trend_leak95` in the per-record profile counts each leaky edge at BOTH
    endpoints, so summing it over records double-counts (4,486 = 2 x 2,243).
    Every edge count here is taken from the per-pair table instead.
  * `max_rate_out` saturates: 75 of 80 records sit at >= 0.95 and the median is
    exactly 1.00, so ranking records by it is a 75-way tie. Records are ranked
    by `n_out_pass95` (how many of the 79 partners the record leads) instead.

Output: Figures/elements/figS1_record_atlas/*.{png,pdf,svg}
"""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.record_names import display_name
from src.elements import (ONE_COL, ONE_COL_TALL, TWO_COL, INK, INK_SEC, INK_MUTED,
                          use_element_style, save_element, safe_label, tidy)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "map_atlas", os.path.join(SCRIPT_DIR, "20_map_atlas.py"))
atlas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(atlas)

SUB = "figS1_record_atlas"
BLUE, BLUE_HI, ORANGE = atlas.BLUE, atlas.BLUE_HI, atlas.ORANGE
PALE = "#9ec5f4"


def el_map(prof):
    """Global distribution; marker area scales with outgoing honest edges."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # taller than TWO_COL so the two legends get a row each below the map
    fig = plt.figure(figsize=(TWO_COL[0], TWO_COL[1] + 0.85))
    ax = fig.add_subplot(111, projection=ccrs.Robinson())
    ax.add_feature(cfeature.LAND, facecolor="#e6e4da", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#f4f6f7", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor="#a9a89f", linewidth=0.35, zorder=1)
    ax.gridlines(draw_labels=False, linewidth=0.25, color="#d8d7cf", zorder=1)

    # Size on n_out_pass95 (0-60 across the network), not on max_rate_out:
    # 75 of 80 records max out at >= 0.95, so that would draw 80 equal dots.
    s_min, s_max, n_ref = 10.0, 110.0, 60.0
    size = lambda d: s_min + (s_max - s_min) * (d["n_out_pass95"] / n_ref)
    ax.scatter(prof["lon"], prof["lat"], s=size(prof), c=prof["region_hue"],
               alpha=0.82, transform=ccrs.PlateCarree(),
               edgecolors="white", linewidths=0.4, zorder=3)

    order = prof.groupby("region_label").size().sort_values(ascending=False)
    hue = dict(zip(prof["region_label"], prof["region_hue"]))
    region_handles = [Patch(facecolor=hue[r], alpha=0.82, label=f"{r} ({n})")
                      for r, n in order.items()]
    leg1 = ax.legend(handles=region_handles, loc="upper center", ncol=4, frameon=False,
                     fontsize=6.4, bbox_to_anchor=(0.5, -0.02), handlelength=1.1,
                     columnspacing=1.4, handletextpad=0.5)
    ax.add_artist(leg1)

    # second row, well clear of the region legend above it
    size_handles = [Line2D([], [], marker="o", ls="none", mfc=INK_MUTED, mec="white",
                           mew=0.4, ms=np.sqrt(s_min + (s_max - s_min) * (v / n_ref)),
                           label=str(v)) for v in (10, 30, 60)]
    ax.legend(handles=size_handles, loc="upper center", frameon=False, fontsize=6.4,
              bbox_to_anchor=(0.5, -0.24), ncol=3, handlelength=1.0,
              columnspacing=1.6, handletextpad=0.5,
              title="Outgoing edges passing at 95%", title_fontsize=6.4)
    save_element(fig, "figS1a_map", subdir=SUB, tight=False)


def el_layer_recovery(pairs):
    """Edges that fail with the full stack but pass once one layer is removed."""
    full = pairs["pass95_full"].astype(bool)
    layers = [("pass95_nonoise", "Observational noise"),
              ("pass95_noage", "Age error"),
              ("pass95_noarc", "ARC deconfounding")]
    # per-pair, so each edge counts once
    vals = [int((~full & pairs[c].astype(bool)).sum()) for c, _ in layers]

    fig, ax = plt.subplots(figsize=ONE_COL)
    y = np.arange(len(layers))[::-1]
    ax.barh(y, vals, height=0.55, color=[PALE, BLUE, BLUE_HI], edgecolor="none")
    for yi, v in zip(y, vals):
        ax.text(v, yi, f"  {v}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in layers])
    ax.set_xlim(0, max(vals) * 1.20)
    ax.set_xlabel("Edges recovered when the layer is removed")
    tidy(ax, grid_axis="x")
    save_element(fig, "figS1b_layer_recovery", subdir=SUB)


def el_trend_leak(pairs):
    """Raw passes vs passes that survive forced differencing, at both thresholds."""
    raw = [int(pairs["pass95_full"].astype(bool).sum()),
           int(pairs["pass80_full"].astype(bool).sum())]
    kept = [int(pairs["pass95_alldiff"].astype(bool).sum()),
            int(pairs["pass80_alldiff"].astype(bool).sum())]

    fig, ax = plt.subplots(figsize=ONE_COL)
    x = np.arange(2)
    w = 0.34
    b1 = ax.bar(x - w / 2, raw, width=w, color=BLUE_HI, edgecolor="none",
                label="As tested")
    b2 = ax.bar(x + w / 2, kept, width=w, color=ORANGE, edgecolor="none",
                label="After forced differencing")
    for bars, vals in ((b1, raw), (b2, kept)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max(raw) * 0.015, f"{v}",
                    ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(["95% honesty", "80% honesty"])
    ax.set_ylim(0, max(raw) * 1.20)
    ax.set_ylabel("Directed edges (of 6,320)")
    ax.legend(loc="upper center", ncol=1, fontsize=6.8, bbox_to_anchor=(0.72, 1.02))
    tidy(ax, grid_axis="y")
    save_element(fig, "figS1c_trend_leak", subdir=SUB)


def el_rate_distribution(pairs):
    """Where the 6,320 pairs actually sit on the honesty scale."""
    fig, ax = plt.subplots(figsize=ONE_COL)
    ax.hist(pairs["rate_full"], bins=np.linspace(0, 1, 41), color=BLUE,
            edgecolor="white", linewidth=0.3)
    for thr, lab in ((0.80, "80%"), (0.95, "95%")):
        ax.axvline(thr, color=INK_MUTED, lw=0.8, ls=(0, (4, 2.5)))
        ax.text(thr, ax.get_ylim()[1] * 0.97, f" {lab}", fontsize=6.4,
                color=INK_SEC, ha="left", va="top")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Robustness rate across the 200-realization ensemble")
    ax.set_ylabel("Directed pairs")
    tidy(ax, grid_axis="y")
    save_element(fig, "figS1d_rate_distribution", subdir=SUB)


def el_method_agreement(pairs, pcm, ccm):
    """Each tester as a retention rate, since the three saw different pair sets."""
    n_gr = int(pairs["pass95_full"].astype(bool).sum())
    n_pcm = int(pcm["pcmci_lagged_tau_ge1"].astype(bool).sum())
    ok = ccm.dropna(subset=["ccm_fwd", "ccm_rev"])
    n_ccm = int((ok["ccm_fwd"] > ok["ccm_rev"]).sum())
    counts = [(n_gr, len(pairs)), (n_pcm, len(pcm)), (n_ccm, len(ok))]
    rates = [100.0 * a / b for a, b in counts]

    fig, ax = plt.subplots(figsize=ONE_COL)
    bars = ax.bar([0, 1, 2], rates, width=0.5, color=[BLUE_HI, BLUE, PALE],
                  edgecolor="none")
    for b, r, (a, n) in zip(bars, rates, counts):
        ax.text(b.get_x() + b.get_width() / 2, r, f"{r:.0f}%",
                ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.text(b.get_x() + b.get_width() / 2, 1.5, f"{a}/{n}",
                ha="center", va="bottom", fontsize=6.2, color="white")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Granger\n(95% honest)", "PCMCI\n(lagged link)",
                        "CCM\n(fwd > rev)"], fontsize=7)
    ax.set_ylim(0, max(rates) * 1.22)
    ax.set_ylabel("Directed links retained (%)")
    tidy(ax, grid_axis="y")
    save_element(fig, "figS1e_method_agreement", subdir=SUB)


def el_top_records(prof, n=15):
    """Records leading the most partners, with room for the names."""
    top = prof.sort_values("n_out_pass95", ascending=False).head(n)
    fig, ax = plt.subplots(figsize=(4.6, ONE_COL_TALL[1]))
    y = np.arange(len(top))[::-1]
    ax.barh(y, top["n_out_pass95"], height=0.62, color=list(top["region_hue"]),
            edgecolor="none", alpha=0.88)
    for yi, v in zip(y, top["n_out_pass95"]):
        ax.text(v, yi, f"  {int(v)}", va="center", fontsize=6.4, color=INK_SEC)
    ax.set_yticks(y)
    ax.set_yticklabels([safe_label(display_name(r)) for r in top["record"]],
                       fontsize=6.6)
    ax.set_xlim(0, top["n_out_pass95"].max() * 1.12)
    ax.set_ylim(-0.7, len(top) - 0.3)
    ax.set_xlabel("Partners led at the 95% honesty threshold (of 79)")
    tidy(ax, grid_axis="x")
    save_element(fig, "figS1f_top_records", subdir=SUB)


def el_region_counts(prof):
    """How unevenly the 80 records are distributed across regions."""
    counts = prof.groupby("region_label").size().sort_values()
    hue = dict(zip(prof["region_label"], prof["region_hue"]))
    fig, ax = plt.subplots(figsize=ONE_COL)
    y = np.arange(len(counts))
    ax.barh(y, counts.values, height=0.6,
            color=[hue[r] for r in counts.index], edgecolor="none", alpha=0.88)
    for yi, v in zip(y, counts.values):
        ax.text(v, yi, f"  {v}", va="center", fontsize=7, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(list(counts.index))
    ax.set_xlim(0, counts.max() * 1.16)
    ax.set_xlabel("Records (80 total)")
    tidy(ax, grid_axis="x")
    save_element(fig, "figS1g_region_counts", subdir=SUB)


def main():
    use_element_style()
    prof, pairs, pcm, ccm = atlas.load_atlas_data()
    print(f"loaded {len(prof)} records, {len(pairs)} pair rows")
    el_map(prof)
    el_layer_recovery(pairs)
    el_trend_leak(pairs)
    el_rate_distribution(pairs)
    el_method_agreement(pairs, pcm, ccm)
    el_top_records(prof)
    el_region_counts(prof)


if __name__ == "__main__":
    main()
    from src.elements import report
    report()
