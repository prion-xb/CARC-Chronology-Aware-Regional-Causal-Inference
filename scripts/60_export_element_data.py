#!/usr/bin/env python3
"""Export the plotted data behind every element as tidy CSV.

The CARC pipeline is Python, so the numbers have to come from Python. The
*drawing*, though, is being ported to R and MATLAB. This script is the contract
between the two halves: it writes one tidy CSV per element into
`Figures/elements/data/`, and the R and MATLAB scripts plot from those files
without recomputing anything.

That keeps a single source of truth. If the pipeline is rerun, this script is
rerun, and all three languages move together.

Output: Figures/elements/data/*.csv
"""
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from src.config import (DATA_DIR, OUTPUT_DIR, AGE_GRID_10_60, N_PERTURB,
                        CLIMATE_WINDOWS)
from src.io import load_all_records
from src.stacking import build_region_dict
from src.record_names import display_name

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_OUT = os.path.join(ROOT, "Figures", "elements", "data")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPT_DIR, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write(df, name):
    os.makedirs(DATA_OUT, exist_ok=True)
    path = os.path.join(DATA_OUT, f"{name}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  data: {name}.csv  ({len(df)} rows, {len(df.columns)} cols)")


# --------------------------------------------------------------------------
# Fig. 2 - Layer-1 regional stacking
# --------------------------------------------------------------------------

def export_fig2():
    stackviz = _load("stackviz", "18_stacking_visualization.py")
    elstack = _load("elstack", "54_elements_stacking.py")

    records, _ = load_all_records(DATA_DIR, age_min=10.0, age_max=60.0)
    records = [(i, s, "Antarctic" if r == "Southern_Ocean" else r, n)
               for i, s, r, n in records]
    region_dict = {r: v for r, v in build_region_dict(records).items()
                   if r in stackviz.PAPER_REGIONS}
    noise = elstack.load_noise_ratios()

    grid = AGE_GRID_10_60
    rng = np.random.default_rng(42)          # same seed as scripts 33 and 54
    stack_rows, member_rows, meta_rows = [], [], []

    for region in stackviz.PAPER_REGIONS:
        recs = region_dict[region]
        stack = stackviz.stack_schemes(recs, grid, N_PERTURB, 0.5, rng)["median"]
        med = np.median(stack, axis=0)
        p5, p95 = np.percentile(stack, [5, 95], axis=0)
        stack_rows.append(pd.DataFrame({"region": region, "age_ka": grid,
                                        "median": med, "p5": p5, "p95": p95}))
        for interp, _sigma, rec_name in recs:
            idx = np.round(np.linspace(0, len(interp) - 1, len(grid))).astype(int)
            member_rows.append(pd.DataFrame({"region": region,
                                             "record": display_name(rec_name),
                                             "age_ka": grid,
                                             "value": interp[idx]}))
        ratio, ref = noise[region]
        meta_rows.append({"region": region,
                          "region_label": stackviz.REGION_LABELS[region],
                          "n_members": len(recs),
                          "noise_ratio": ratio,
                          "theoretical_1_over_sqrt_K": ref,
                          "slug": elstack.SLUG[region]})

    write(pd.concat(stack_rows, ignore_index=True), "fig2_stack")
    write(pd.concat(member_rows, ignore_index=True), "fig2_members")
    write(pd.DataFrame(meta_rows), "fig2_region_meta")
    write(pd.DataFrame([{"window": w, "abbr": elstack.WINDOW_ABBR[w],
                         "start_ka": CLIMATE_WINDOWS[w][0],
                         "end_ka": CLIMATE_WINDOWS[w][1]}
                        for w in elstack.WINDOW_ORDER]), "fig2_windows")


# --------------------------------------------------------------------------
# Fig. S1 - record scrutinization atlas
# --------------------------------------------------------------------------

def _robinson(lon, lat):
    """Project lon/lat to Robinson metres.

    Done here rather than in R and MATLAB so all three languages draw the same
    projection without needing rnaturalearth or the Mapping Toolbox.
    """
    import cartopy.crs as ccrs
    xyz = ccrs.Robinson().transform_points(ccrs.PlateCarree(),
                                           np.asarray(lon, dtype=float),
                                           np.asarray(lat, dtype=float))
    return xyz[:, 0], xyz[:, 1]


def export_figS1():
    atlas = _load("map_atlas", "20_map_atlas.py")
    prof, pairs, pcm, ccm = atlas.load_atlas_data()

    x, y = _robinson(prof["lon"], prof["lat"])
    write(pd.DataFrame({
        "record": [display_name(r) for r in prof["record"]],
        "record_id": prof["record"],
        "lat": prof["lat"], "lon": prof["lon"],
        "robin_x": x, "robin_y": y,
        "region": prof["region"], "region_label": prof["region_label"],
        "region_hue": prof["region_hue"],
        "n_out_pass95": prof["n_out_pass95"],
        "n_in_pass95": prof["n_in_pass95"],
        "max_rate_out": prof["max_rate_out"],
    }), "figS1a_map")

    # per-pair, so each directed edge is counted once (the per-record profile
    # counts every edge at both endpoints and doubles these totals)
    full = pairs["pass95_full"].astype(bool)
    write(pd.DataFrame([
        {"layer_removed": "Observational noise", "column": "pass95_nonoise",
         "edges_recovered": int((~full & pairs["pass95_nonoise"].astype(bool)).sum())},
        {"layer_removed": "Age error", "column": "pass95_noage",
         "edges_recovered": int((~full & pairs["pass95_noage"].astype(bool)).sum())},
        {"layer_removed": "ARC deconfounding", "column": "pass95_noarc",
         "edges_recovered": int((~full & pairs["pass95_noarc"].astype(bool)).sum())},
    ]), "figS1b_layer_recovery")

    write(pd.DataFrame([
        {"threshold": "95% honesty", "as_tested": int(full.sum()),
         "after_forced_differencing": int(pairs["pass95_alldiff"].astype(bool).sum())},
        {"threshold": "80% honesty",
         "as_tested": int(pairs["pass80_full"].astype(bool).sum()),
         "after_forced_differencing": int(pairs["pass80_alldiff"].astype(bool).sum())},
    ]), "figS1c_trend_leak")

    write(pd.DataFrame({"rate_full": pairs["rate_full"]}), "figS1d_rate_distribution")

    ok = ccm.dropna(subset=["ccm_fwd", "ccm_rev"])
    write(pd.DataFrame([
        {"method": "Granger (95% honest)", "retained": int(full.sum()),
         "evaluated": len(pairs)},
        {"method": "PCMCI (lagged link)",
         "retained": int(pcm["pcmci_lagged_tau_ge1"].astype(bool).sum()),
         "evaluated": len(pcm)},
        {"method": "CCM (fwd > rev)",
         "retained": int((ok["ccm_fwd"] > ok["ccm_rev"]).sum()), "evaluated": len(ok)},
    ]), "figS1e_method_agreement")

    top = prof.sort_values("n_out_pass95", ascending=False).head(15)
    write(pd.DataFrame({"record": [display_name(r) for r in top["record"]],
                        "region_label": top["region_label"],
                        "region_hue": top["region_hue"],
                        "n_out_pass95": top["n_out_pass95"]}), "figS1f_top_records")

    counts = prof.groupby("region_label").size().sort_values()
    hue = dict(zip(prof["region_label"], prof["region_hue"]))
    write(pd.DataFrame({"region_label": counts.index, "n_records": counts.values,
                        "region_hue": [hue[r] for r in counts.index]}),
          "figS1g_region_counts")


# --------------------------------------------------------------------------
# Coastline, so the map does not depend on rnaturalearth / Mapping Toolbox
# --------------------------------------------------------------------------

def export_coastline():
    """Dump Natural Earth coastlines as polylines, in lon/lat and Robinson."""
    import cartopy.feature as cfeature
    rows = []
    for gid, geom in enumerate(cfeature.COASTLINE.geometries()):
        parts = geom.geoms if hasattr(geom, "geoms") else [geom]
        for pid, part in enumerate(parts):
            xy = np.asarray(part.coords)
            if len(xy) < 2:
                continue
            x, y = _robinson(xy[:, 0], xy[:, 1])
            rows.append(pd.DataFrame({"line_id": f"{gid}_{pid}",
                                      "lon": xy[:, 0], "lat": xy[:, 1],
                                      "robin_x": x, "robin_y": y}))
    write(pd.concat(rows, ignore_index=True), "coastline_110m")

    # graticule outline, so R and MATLAB can draw the map frame
    lon = np.linspace(-180, 180, 361)
    lat = np.linspace(-90, 90, 181)
    edge_lon = np.concatenate([lon, np.full_like(lat, 180.0), lon[::-1],
                               np.full_like(lat, -180.0)])
    edge_lat = np.concatenate([np.full_like(lon, -90.0), lat,
                               np.full_like(lon, 90.0), lat[::-1]])
    ex, ey = _robinson(edge_lon, edge_lat)
    write(pd.DataFrame({"lon": edge_lon, "lat": edge_lat,
                        "robin_x": ex, "robin_y": ey}), "map_frame")


def export_style():
    """Element style constants, so R and MATLAB match the Python output."""
    from src import elements as el
    style = {
        "sizes_inches": {"ONE_COL": el.ONE_COL, "ONE_COL_TALL": el.ONE_COL_TALL,
                         "TWO_COL": el.TWO_COL, "SQUARE": el.SQUARE},
        "colors": {"ink": el.INK, "ink_sec": el.INK_SEC, "ink_muted": el.INK_MUTED,
                   "grid": el.GRID, "surface": el.SURFACE, "member": el.MEMBER},
        "region_colors": el.REGION_COLORS,
        "tier_colors": {str(k): v for k, v in el.TIER_COLORS.items()},
        "font": {"family": "Arial", "base_pt": 8, "axis_label_pt": 8.5,
                 "tick_pt": 7.5, "legend_pt": 7},
        "dpi": 300,
    }
    os.makedirs(DATA_OUT, exist_ok=True)
    path = os.path.join(DATA_OUT, "style.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(style, fh, indent=2)
    print(f"  data: style.json")


if __name__ == "__main__":
    print("Exporting element data...")
    export_style()
    export_coastline()
    export_figS1()
    export_fig2()
    print(f"\nWritten to {DATA_OUT}")
