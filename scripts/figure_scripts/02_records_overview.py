# -*- coding: utf-8 -*-
"""Step 02 - the raw material: all 80 records as a contact sheet.

One mini-panel per record, grouped by region (canonical 80-record set from
data/region_assignment_80_paper.csv). Each panel shows the record over its
own age range; a small colored square encodes Tier (sequential blue ramp:
dark = Tier 1). No y-axes, no ticks - this is an overview to *see* the data,
not to read values.

Age axis inverted throughout (older ages to the left).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import Figure, apply_paleo_style, save, INK, SECONDARY_INK, GRID, SURFACE  # noqa: E402
from data_io import read_record  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from record_names import display_name  # noqa: E402

OUT = os.path.join(ROOT, "figures_rebuilt", "02_records_overview")

# sequential blue ramp (dark -> light) for Tier 1 -> 3
TIER_COLORS = {"Tier 1": "#184f95", "Tier 2": "#3987e5", "Tier 3": "#9ec5f4"}

PANEL_W, PANEL_H = 25.0, 13.0      # mm
BANNER_H = 5.0                     # mm, region banner band
NCOLS = 6
MARGIN_L, MARGIN_R = 8.0, 4.0      # mm
MARGIN_T, MARGIN_B = 8.0, 6.0      # mm
GAP = 2.5                          # mm between panels


def main():
    assign = pd.read_csv(os.path.join(ROOT, "data", "region_assignment_80_paper.csv"))
    order = assign["Region"].drop_duplicates().tolist()

    n_panel_rows = sum(-(-len(assign[assign.Region == r]) // NCOLS) for r in order)
    n_bands = n_panel_rows + len(order)  # panel rows + region banners
    W_mm = MARGIN_L + NCOLS * PANEL_W + (NCOLS - 1) * GAP + MARGIN_R
    H_mm = MARGIN_T + n_bands * PANEL_H + (n_bands - 1) * GAP + MARGIN_B
    W_f, H_f = W_mm / 25.4, H_mm / 25.4

    fig = Figure(figsize=(W_f, H_f), facecolor="white")

    # top legend: tier swatches (real rectangles, text stays ink)
    lx = MARGIN_L / W_mm
    ly = (H_mm - MARGIN_T / 2 - 2.5) / H_mm
    fig.text(lx, ly, "80 records by region", fontsize=9, color=INK, va="center")
    for i, (tier, c) in enumerate(TIER_COLORS.items()):
        tx = lx + (34 + i * 15) / W_mm
        fig.patches.append(Rectangle((tx - 0.006, ly - 0.008), 0.010, 0.016,
                                     transform=fig.transFigure,
                                     facecolor=c, edgecolor="none"))
        fig.text(tx + 0.008, ly, tier.replace("Tier ", "T"), fontsize=7.5,
                 color=INK, va="center")

    row = 0  # running band index (banners + panel rows share the grid)

    def band_top_mm():
        return H_mm - MARGIN_T - row * (PANEL_H + GAP)

    def place_panel_row():
        nonlocal row
        axes = []
        y_top = band_top_mm()
        for c in range(NCOLS):
            x0 = (MARGIN_L + c * (PANEL_W + GAP)) / W_mm
            ax = fig.add_axes([x0, (y_top - PANEL_H) / H_mm,
                               PANEL_W / W_mm, PANEL_H / H_mm])
            apply_paleo_style(ax)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor(SURFACE)
            axes.append(ax)
        row += 1
        return axes

    def place_banner(text):
        nonlocal row
        y_top = band_top_mm()
        fig.text(MARGIN_L / W_mm, (y_top - BANNER_H * 0.62) / H_mm, text,
                 fontsize=9, fontweight="bold", color=INK, va="center")
        row += 1

    missing = []
    for region in order:
        sub = assign[assign.Region == region]
        nrows = -(-len(sub) // NCOLS)
        place_banner(region.replace("_", " "))
        for r_idx in range(nrows):
            axes = place_panel_row()
            for (_, r), ax in zip(sub.iloc[r_idx * NCOLS:(r_idx + 1) * NCOLS].iterrows(),
                                  axes):
                rid = r["Record"]
                file_id = str(r["S2_File_IDs"]).split(";")[0]  # multi-file rows: primary file
                name = display_name(rid)
                if len(name) > 15:
                    name = name[:14] + "…"  # keep clear of the tier square (right)
                try:
                    rec = read_record(file_id)
                    a, v = rec["age_ka"].values, rec["value"].values
                    o = np.argsort(a)
                    ax.plot(a[o], v[o], color=INK, linewidth=0.35, zorder=2)
                    ax.invert_xaxis()
                    ax.set_xlim(a.max(), a.min())
                    lo, hi = np.percentile(v, [0.5, 99.5])  # trim 1% outliers
                    pad = 0.08 * max(hi - lo, 1e-9)
                    ax.set_ylim(lo - pad, hi + pad)
                except FileNotFoundError:
                    missing.append(file_id)
                    ax.text(0.5, 0.5, "file?", fontsize=6.5, color=INK,
                            ha="center", va="center", transform=ax.transAxes)
                ax.set_title(name, fontsize=6.5, loc="left",
                             pad=1.2, color=INK)
                ax.plot([0.985], [1.07], marker="s", ms=3.2,
                        color=TIER_COLORS.get(r["Tier"], GRID),
                        transform=ax.transAxes, clip_on=False, zorder=5)

    if missing:
        print("MISSING FILES:", missing)

    os.makedirs(OUT, exist_ok=True)
    save(fig, os.path.join(OUT, "records_by_region"), dpi=200)
    print(f"grid: {n_bands} bands, page {W_mm:.0f}x{H_mm:.0f} mm")


if __name__ == "__main__":
    main()
