# -*- coding: utf-8 -*-
"""Step 19 - record quality tiers (Fig. S10).

Claim: the 80-record inventory carries quality tiers 7 / 48 / 25, spread
over record span, median sampling resolution and age-model error.

Design (simple, restrained): two scatter panels sharing the resolution
axis - (a) span x resolution, (b) age-model sigma x resolution, both log
axes.  Marker fill = quality tier (02's blue ramp, dark = Tier 1; same
encoding as Fig. S9).  No per-record labels - Table S2 carries the names.

Span and resolution are computed here from the raw record files (the
10-60 ka analyses re-grid everything, so the trimmed span would be a
degenerate ~50 ka for all records); sigma is the per-record age-model
error the pipeline perturbs with (SPECIAL_SIGMA / DEFAULT_SIGMA via
src.io.load_all_records).

Usage:
    python scripts/fignew/19_quality_tiers.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.dirname(os.path.abspath(__file__)), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from style import (apply_paleo_style, save, INK, GRID,  # noqa: E402
                   SECONDARY_INK, SURFACE)

OUT_DIR = os.path.join(ROOT, "figures_new", "19_quality_tiers")

FIG_W, FIG_H = 178.0, 112.0          # print size, mm

# tier ramp of 02_records_overview (dark = Tier 1); cross-figure symbol
TIER_COLORS = {"Tier 1": "#184f95", "Tier 2": "#3987e5", "Tier 3": "#9ec5f4"}
TIER_ORDER = ["Tier 1", "Tier 2", "Tier 3"]

RES_TICKS = [0.01, 0.05, 0.1, 0.5, 2.0]
SPAN_TICKS = [100, 500, 1000, 5000]
SPAN_LABELS = ["100", "500", "1,000", "5,000"]
SIG_TICKS = [0.2, 0.5, 1, 2]
SIG_LABELS = ["0.2", "0.5", "1", "2"]


# ------------------------------------------------------------------ data ----
def load_metrics():
    """80 canonical stems -> (span_ka, res_ka, sigma_ka, tier).

    span/res computed from the raw record files (untrimmed age columns);
    sigma from the canonical 10-60 ka loader (the perturbation sigma);
    tier from the paper inventory (region_assignment_80_paper.csv).
    """
    from src.config import DATA_DIR
    from src.io import (read_record_df, identify_columns, load_all_records,
                        filter_canonical_records)

    inv = pd.read_csv(os.path.join(ROOT, "data",
                                   "region_assignment_80_paper.csv"),
                      encoding="utf-8-sig")
    stems, tiers = [], []
    for _, r in inv.iterrows():
        for s in str(r["S2_File_IDs"]).split(";"):
            s = s.strip()
            if s:
                stems.append(s)
                tiers.append(r["Tier"])

    # stripped stem -> path
    fmap = {os.path.splitext(fn)[0].strip(): os.path.join(DATA_DIR, fn)
            for fn in os.listdir(DATA_DIR)
            if fn.endswith((".csv", ".xlsx"))}

    records, _ = load_all_records(DATA_DIR, age_min=10.0, age_max=60.0)
    canon = filter_canonical_records(records)
    sigmas = {n.strip(): sig for _, sig, _, n in canon}
    assert len(sigmas) == 80, f"sigma map = {len(sigmas)} records"

    rows = []
    for stem, tier in zip(stems, tiers):
        df = read_record_df(fmap[stem])
        ac, vc = identify_columns(df)
        age = df[ac].dropna().values.astype(float)
        age = np.sort(age)
        rows.append({"stem": stem, "tier": tier,
                     "span": float(age.max() - age.min()),
                     "res": float(np.median(np.diff(age))),
                     "sigma": sigmas[stem]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ plot ----
def panel(ax, df, xcol, xticks, xlabels, xlabel):
    """One scatter panel: y = resolution (shared log axis)."""
    for tier in ["Tier 3", "Tier 2", "Tier 1"]:     # T1 dots on top
        sub = df[df["tier"] == tier]
        if sub.empty:
            continue
        ax.scatter(sub[xcol], sub["res"], s=26, c=TIER_COLORS[tier],
                   edgecolors="white", linewidths=0.4, zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=9, color=INK)
    ax.set_yticks(RES_TICKS)
    ax.set_yticklabels([f"{t:g}" for t in RES_TICKS], fontsize=9, color=INK)
    ax.set_xlabel(xlabel, fontsize=10, color=INK)
    ax.set_ylabel("median resolution (ka)", fontsize=10, color=INK)


def add_title(fig):
    fig.text(0.02, 0.972, "Fig. S10 — quality tiers across the 80-record "
             "inventory", fontsize=10.5, fontweight="bold", color=INK,
             va="center")


def add_legend(fig, df):
    from matplotlib.patches import Rectangle
    counts = df["tier"].value_counts()
    x = 0.02
    fig.text(x, 0.934, "Fill:", fontsize=9, color=INK, va="center")
    x += 0.032
    for tier in TIER_ORDER:
        fig.patches.append(Rectangle((x, 0.927), 0.010, 0.010,
                                     facecolor=TIER_COLORS[tier],
                                     edgecolor=INK, linewidth=0.5,
                                     transform=fig.transFigure))
        fig.text(x + 0.013, 0.934, f"{tier} ({counts.get(tier, 0)})",
                 fontsize=9, color=INK, va="center")
        x += 0.013 + 0.068


def add_footer(fig):
    lines = [
        "span and resolution computed from the raw record files",
        "σ = age-model error used by the perturbation machinery "
        "(SPECIAL_SIGMA / DEFAULT_SIGMA)",
        "tier rule (scripts/01): T1 = composite ≥ 5 + event data + "
        "published σ; T2 = composite ≥ 3 + event data; T3 otherwise",
        "inventory: data/region_assignment_80_paper.csv (Table S2, "
        "80 records); tier counts 7 / 48 / 25",
    ]
    for y, s in zip([0.272, 0.238, 0.204, 0.170], lines):
        fig.text(0.02, y, s, fontsize=9, color=INK, va="center")


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
    df = load_metrics()

    # anchors
    assert len(df) == 80, f"metrics = {len(df)} records"
    counts = df["tier"].value_counts().to_dict()
    assert counts == {"Tier 1": 7, "Tier 2": 48, "Tier 3": 25}, counts
    assert df["sigma"].notna().all(), "missing sigma"
    ngr = df[df["stem"] == "NGRIP_d18O"].iloc[0]
    assert abs(ngr["sigma"] - 0.65) < 1e-9, f"NGRIP sigma = {ngr['sigma']}"
    print(f"anchors: 80 records, tiers {counts}")
    print(f"span: min {df['span'].min():.1f} / P50 {df['span'].median():.1f}"
          f" / max {df['span'].max():.1f} ka")
    print(f"resolution: min {df['res'].min():.3f} / P50 "
          f"{df['res'].median():.3f} / max {df['res'].max():.3f} ka")
    print(f"sigma: min {df['sigma'].min():.2f} / P50 "
          f"{df['sigma'].median():.2f} / max {df['sigma'].max():.2f} ka; "
          f"NGRIP = {ngr['sigma']:.2f}")

    fig = Figure(figsize=(FIG_W / 25.4, FIG_H / 25.4), facecolor="white")
    axa = fig.add_axes([0.07, 0.34, 0.41, 0.55])
    apply_paleo_style(axa)
    axb = fig.add_axes([0.55, 0.34, 0.41, 0.55])
    apply_paleo_style(axb)

    panel(axa, df, "span", SPAN_TICKS, SPAN_LABELS, "record span (ka)")
    panel(axb, df, "sigma", SIG_TICKS, SIG_LABELS,
          "age-model error σ (ka)")
    axa.set_title("(a) span × resolution", loc="left", fontsize=10.5,
                  fontweight="bold", color=INK, pad=6)
    axb.set_title("(b) age error × resolution", loc="left", fontsize=10.5,
                  fontweight="bold", color=INK, pad=6)
    axb.set_ylabel("")                      # shared y axis, left panel only
    axb.tick_params(axis="y", labelleft=False)

    add_title(fig)
    add_legend(fig, df)
    add_footer(fig)
    assert no_text_overruns(fig), "text overruns the figure"

    save(fig, os.path.join(OUT_DIR, "quality_tiers"))
    print("claim: quality tiers 7 / 48 / 25 spread over span, resolution "
          "and age-model error")


if __name__ == "__main__":
    main()
