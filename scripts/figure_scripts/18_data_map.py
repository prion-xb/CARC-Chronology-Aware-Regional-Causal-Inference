# -*- coding: utf-8 -*-
"""Step 18 - global data map (Fig. S9).

Claim: the 80-record inventory spans six regions with quality tiers
7 / 48 / 25; every marker is one of the paper's 80 records.

Encoding (simple, restrained - no connector-label spaghetti of the old
Fig. 1 map): marker fill = quality tier (02's blue ramp, dark = Tier 1);
marker ring = region colour (style.REGION_CATS, the fixed cross-figure
slots); co-located records at one site are jittered deterministically
(<= 1.2 deg) so every record stays visible.  Region names sit at the
cluster centroids; the scattered High-Latitude Lakes group has no
geographic label (legend chip only).

Usage:
    python scripts/figure_scripts/18_data_map.py
"""
from __future__ import annotations

import os
import sys
import zlib

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import matplotlib.path as mpath

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.dirname(os.path.abspath(__file__)), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import cartopy.crs as ccrs            # noqa: E402
import cartopy.feature as cfeature    # noqa: E402

from style import (apply_paleo_style, save, INK, SECONDARY_INK, GRID,  # noqa: E402
                   REGION_CATS, REGION_ORDER, SURFACE)

OUT_DIR = os.path.join(ROOT, "figures_rebuilt", "18_data_map")

FIG_W, FIG_H = 178.0, 106.0          # print size, mm

# tier ramp of 02_records_overview (dark = Tier 1); cross-figure symbol
TIER_COLORS = {"Tier 1": "#184f95", "Tier 2": "#3987e5", "Tier 3": "#9ec5f4"}
TIER_ORDER = ["Tier 1", "Tier 2", "Tier 3"]

# map neutrals: land/ocean/coast stay in the grey family (non-text only)
OCEAN_C = "#edf1f5"
LAND_C = "#f4f1ea"
COAST_C = "#c9c8c3"

# short display names for the anchor-site labels
SITE_LABELS = {
    "NGRIP_d18O": "NGRIP",
    "EDC_dD": "EDC",
    "DomeFuji_dD": "Dome Fuji",
    "WAIS_CH4": "WAIS",
    "Cariaco_Mo": "Cariaco",
    "Cheng2016_d18O": "Cheng 2016",
    "ODP983_IRD": "ODP 983",
    "Baikal_BioSilica": "Baikal",
}
SITE_OFFSET = {           # (dlon, dlat, ha, va) in degrees
    "NGRIP_d18O": (-4, -6, "right", "top"),
    "EDC_dD": (6, -5, "left", "bottom"),
    "DomeFuji_dD": (7, 3, "left", "bottom"),
    "WAIS_CH4": (0, 5, "center", "bottom"),
    "Cariaco_Mo": (5, 4, "left", "bottom"),
    "Cheng2016_d18O": (-5, 2, "right", "bottom"),
    "ODP983_IRD": (5, 2, "left", "bottom"),
    "Baikal_BioSilica": (4, 3, "left", "bottom"),
}

# region-name anchors (black text, white halo), in (lon, lat) - placed in
# empty sea areas clear of the member dots
REGION_LABEL_POS = {
    "North_Atlantic": (-32, 64),
    "Antarctic": (55, -62),
    "Tropical": (60, -20),
    "East_Asian_Monsoon": (107, 44),
    "South_America": (-36, -34),
}
REGION_LABEL_TXT = {
    "North_Atlantic": "North Atlantic",
    "Antarctic": "Antarctic & Southern Ocean",
    "Tropical": "Tropical",
    "East_Asian_Monsoon": "East Asian Monsoon",
    "South_America": "South America",
}

JITTER_MAX = 1.2                     # deg, co-located records only


# ------------------------------------------------------------------ data ----
def load_inventory():
    """80 canonical stems -> (record key, tier, region, lat, lon)."""
    from src.geolocation import RECORD_COORDS
    df = pd.read_csv(os.path.join(ROOT, "data",
                                  "region_assignment_80_paper.csv"),
                     encoding="utf-8-sig")
    rows = []
    for _, r in df.iterrows():
        key, tier, region = r["Record"], r["Tier"], r["Region"]
        lat, lon = RECORD_COORDS[key]
        for s in str(r["S2_File_IDs"]).split(";"):
            s = s.strip()
            if s:
                rows.append((s, key, tier, region, lat, lon))
    inv = pd.DataFrame(rows, columns=["stem", "key", "tier", "region",
                                      "lat", "lon"])
    return inv


def jitter(inv):
    """Deterministic jitter for co-located stems (same lat/lon site).

    Each site keeps its first record at the true coordinates; further
    records move <= JITTER_MAX deg along a per-stem angle (crc32 seed,
    stable across runs)."""
    jl, jn = inv["lat"].copy(), inv["lon"].copy()
    site_count = inv.groupby(["lat", "lon"]).cumcount().values
    for i, k in enumerate(site_count):
        if k == 0:
            continue
        rng = np.random.default_rng(zlib.crc32(inv["stem"].iloc[i].encode())
                                    % (2 ** 32))
        ang = rng.uniform(0, 2 * np.pi)
        rad = JITTER_MAX * (0.45 + 0.55 * (k / max(site_count.max(), 1)))
        jl.iloc[i] += rad * np.sin(ang)
        jn.iloc[i] += rad * np.cos(ang)
    inv["plat"], inv["plon"] = jl, jn
    return inv


# ------------------------------------------------------------------ plot ----
def draw_map(ax, inv):
    ax.set_extent([-180, 180, -84, 84], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_C, edgecolor="none")
    ax.add_feature(cfeature.LAND, facecolor=LAND_C, edgecolor="none")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor=COAST_C)
    ax.set_facecolor(OCEAN_C)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color=GRID,
                      alpha=0.8, linestyle="-")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 9, "color": INK}
    gl.ylabel_style = {"size": 9, "color": INK}
    gl.xlocator = _fixed_loc(np.arange(-180, 181, 30))
    gl.ylocator = _fixed_loc(np.arange(-90, 91, 30))

    # region labels: black text with a white halo (text never grey)
    for reg, (lon, lat) in REGION_LABEL_POS.items():
        ax.text(lon, lat, REGION_LABEL_TXT[reg], ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=INK, zorder=8,
                transform=ccrs.PlateCarree(),
                path_effects=[_halo()])

    # markers: tier fill + region ring (PathCollection -> markers layer)
    for tier in TIER_ORDER:
        sub = inv[inv["tier"] == tier]
        if sub.empty:
            continue
        ax.scatter(sub["plon"], sub["plat"], s=30,
                   c=TIER_COLORS[tier],
                   edgecolors=[REGION_CATS[_region_key(r)]
                               for r in sub["region"]],
                   linewidths=1.1, zorder=5, transform=ccrs.PlateCarree())

    # anchor site labels (black, haloed)
    for key, name in SITE_LABELS.items():
        sub = inv[inv["key"] == key]
        if sub.empty:
            continue
        lon0, lat0 = sub["lon"].iloc[0], sub["lat"].iloc[0]
        dlon, dlat, ha, va = SITE_OFFSET[key]
        ax.text(lon0 + dlon, lat0 + dlat, name, ha=ha, va=va, fontsize=9,
                color=INK, zorder=8, transform=ccrs.PlateCarree(),
                path_effects=[_halo()])


def _fixed_loc(vals):
    import matplotlib.ticker as mticker
    return mticker.FixedLocator(vals)


def _halo():
    from matplotlib.patheffects import withStroke
    return withStroke(linewidth=2.6, foreground="white")


def _region_key(region):
    """CSV region key -> style.REGION_ORDER key."""
    return {"Antarctic_SouthernOcean": "Antarctic",
            "HighLat_Lakes": "Other_Lakes"}.get(region, region)


# ------------------------------------------------------------- furniture ----
def add_legend_strip(fig, inv):
    """Top strip: title first, then tier swatches with counts + ring."""
    counts = inv["tier"].value_counts()
    fig.text(0.02, 0.972, "Fig. S9 — global distribution of the 80-record "
             "inventory", fontsize=10.5, fontweight="bold", color=INK,
             va="center")
    x = 0.02
    fig.text(x, 0.930, "Fill: quality tier —", fontsize=9, color=INK,
             va="center")
    x += 0.112
    for tier in TIER_ORDER:
        fig.patches.append(Rectangle((x, 0.915), 0.014, 0.014,
                                 facecolor=TIER_COLORS[tier],
                                 edgecolor=INK, linewidth=0.5,
                                 transform=fig.transFigure))
        fig.text(x + 0.017, 0.922, f"{tier} ({counts.get(tier, 0)})",
                 fontsize=9, color=INK, va="center")
        x += 0.017 + 0.012 + 0.078
    fig.text(x + 0.006, 0.930, "Ring: region", fontsize=9, color=INK,
             va="center")


def add_region_chips(fig):
    """Bottom strip: six region colour chips (fixed REGION_CATS slots)."""
    names = {"North_Atlantic": "North Atlantic", "Antarctic": "Antarctic",
             "Tropical": "Tropical", "East_Asian_Monsoon": "E. Asian Monsoon",
             "South_America": "South America", "Other_Lakes": "Lakes"}
    x = 0.02
    for reg in REGION_ORDER:
        fig.patches.append(Rectangle((x, 0.092), 0.011, 0.011,
                                 facecolor=REGION_CATS[reg],
                                 edgecolor=INK, linewidth=0.5,
                                 transform=fig.transFigure))
        fig.text(x + 0.014, 0.097, names[reg], fontsize=9, color=INK,
                 va="center")
        x += 0.014 + 0.012 + 0.013 + 0.006 * len(names[reg])
    fig.text(0.02, 0.066,
             "data: data/region_assignment_80_paper.csv (Table S2 "
             "inventory) + src/geolocation.py coordinates",
             fontsize=9, color=INK, va="center")
    fig.text(0.02, 0.041,
             "tiers 7 / 48 / 25; records per region: NA 13 · Antarctic 18 · "
             "Tropical 34 · EAM 6 · S. America 3 · Lakes 6",
             fontsize=9, color=INK, va="center")
    fig.text(0.02, 0.016,
             "co-located records jittered ≤ 1.2°; ring = region (fixed "
             "colour slots), fill = tier (02 ramp)",
             fontsize=9, color=INK, va="center")


def no_text_overruns(fig):
    """Geometry QA: every text stays inside the figure (0.5% tolerance)."""
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


# ------------------------------------------------------------------ main ----
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    inv = load_inventory()
    jitter(inv)

    # anchors
    assert len(inv) == 80, f"inventory = {len(inv)} stems"
    counts = inv["tier"].value_counts().to_dict()
    assert counts == {"Tier 1": 7, "Tier 2": 48, "Tier 3": 25}, counts
    regc = inv["region"].value_counts().to_dict()
    assert regc == {"Tropical": 34, "Antarctic_SouthernOcean": 18,
                    "North_Atlantic": 13, "East_Asian_Monsoon": 6,
                    "HighLat_Lakes": 6, "South_America": 3}, regc
    assert inv[["lat", "lon"]].notna().all().all(), "missing coordinates"
    n_sites = inv.groupby(["lat", "lon"]).ngroups
    n_jit = int(inv.groupby(["lat", "lon"]).size().gt(1).sum())
    print(f"anchors: 80 stems / {n_sites} sites, tiers {counts}, "
          f"regions {regc}")
    print(f"co-located sites: {n_jit} (jittered ≤ {JITTER_MAX} deg), "
          f"lat range {inv['lat'].min():.1f}..{inv['lat'].max():.1f}, "
          f"lon range {inv['lon'].min():.1f}..{inv['lon'].max():.1f}")

    fig = Figure(figsize=(FIG_W / 25.4, FIG_H / 25.4), facecolor="white")
    ax = fig.add_axes([0.015, 0.145, 0.97, 0.775],
                      projection=ccrs.PlateCarree())
    draw_map(ax, inv)
    add_legend_strip(fig, inv)
    add_region_chips(fig)
    assert no_text_overruns(fig), "text overruns the figure"

    save(fig, os.path.join(OUT_DIR, "data_map"))
    print("claim: 80 records, six regions, tiers 7/48/25 - every marker "
          "is one of the paper's 80 records")


if __name__ == "__main__":
    main()
