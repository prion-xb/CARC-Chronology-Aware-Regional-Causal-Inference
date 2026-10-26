# -*- coding: utf-8 -*-
"""Step 15 - record-level attribution scrutiny (Fig. 5, Section 4.3).

Claim: none of the 6,320 directed record pairs survives the honest
(forced-difference) check - 0 pass the 95% rung, and only 4/8/21/36/50
pass the 90/85/80/75/70% rungs (strongest pair 92%, still sub-threshold).
Every one of the raw passes (mean 56.1 per record; all 80 records pass in
at least one pairing) collapses under forced differencing: raw passes are
trend leakage - shared low-frequency forcing surviving the ADF gate, not
record-specific signal.  The ablation ladder locates the binding
constraints: removing observational noise recovers 1.6 edges per record,
age error 13.3, deconfounding 19.2.

Design (double column, 178 x 225 mm), four panels - per the v2 plan and the
user's layout decisions (2026-09-13: named-record labels only; CCM panel
dropped, its 0.52/0.53 symmetry kept as a footer fact):
  (a) per-record raw 95%-honesty passes (out + in roles), dot plot, 80
      records sorted by region then passes; tier-coloured dots
      (dark = Tier 1, 02's ramp); region bands with short-name labels;
      the big "0 of 6,320" block states the honesty-check result
  (b) regional means, three bar series per region (top to bottom: raw
      95% passes - all trend leakage - / recovered -age / recovered -ARC);
      bars keep the fixed region colours of (a)
  (c) ablation ladder: mean edges recovered per record when one layer
      (observational noise / age error / ARC deconfounding) is removed
  (d) honesty ladder readout: directed pairs passing the forced-difference
      check at each rung, 95% -> 70% (the sub-threshold band as backdrop)
  footer: provenance, rung counts, ablation, named records, CCM symmetry

Data: outputs/pairwise_scan/ (scripts/19_full_pairwise_scan.py - the
canonical all-pairs scan, 6,320 directed pairs x 200 chronology
perturbations, five variants full/nonoise/noage/noarc/alldiff):
record_profile.csv (per-record aggregates), shard_*.csv (pair-level rates
for the ladder readout), ccm_pairs.csv (CCM on the 400 Granger survivors).
Record labels come from src/record_names.py display_name() (FILE_ID_
OVERRIDES fix the U1302-U1303 Ca/Sr vs Si/Sr and U1385 Ca/Ti vs Zr/Sr
mislabels; the stale outputs/table_s6_scrutiny_profile.csv still carries
the old names - regenerate only the figure, not Table S6).

Convergence targets (asserted):
  6,320 pairs; raw 95% passes 2,243 (35.5%), 80% passes 2,735
  honest (alldiff) rung counts 0/4/8/21/36/50 at 95/90/85/80/75/70%
  strongest honest pair 92%; mean 56.1 passes/record (all 80 records
  pass >= 1 pairing, 75/80 as driver); ablation means 1.6/13.3/19.2
  CCM 400 pairs: 0.52 fwd / 0.53 rev
  named: Junin GI 87, U1302-U1303 d18O 86, Dome Fuji dD 84 (leak);
         Ca/Sr 31, alkenone SST 29, Si/Ti 27 (-age); IPWP OHC 52 (-ARC)

Usage:
    python scripts/fignew/15_attribution_scrutiny.py
Outputs: figures_new/15_attribution_scrutiny/attribution_scrutiny.png / .svg
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

from style import (apply_paleo_style, save, INK, GRID, THR_BAND,  # noqa: E402
                   REGION_ORDER, REGION_CATS)
from src.config import OUTPUT_DIR  # noqa: E402
from src.record_names import display_name  # noqa: E402

SCAN_DIR = os.path.join(OUTPUT_DIR, "pairwise_scan")
OUT_DIR = os.path.join(ROOT, "figures_new", "15_attribution_scrutiny")

FIG_W, FIG_H = 178.0, 225.0
AX_L, AX_R = 0.07, 0.985
FOOTER_Y0 = 0.108          # first footer line; 9pt lines 0.0145 apart,
FOOTER_DY = 0.0145         # leaving the bottom-row xlabels their zone

# the six paper regions (Southern_Ocean merged into Antarctic, as in 41/§4.3)
REGION6_ORDER = ["North Atlantic", "Antarctic/Southern Ocean", "Tropical",
                 "East Asian Monsoon", "South American Monsoon",
                 "High-Latitude Lakes"]
REGION6_CATS = dict(zip(REGION6_ORDER, [REGION_CATS[k] for k in REGION_ORDER]))
SHORT_REGION = {"North Atlantic": "N. Atlantic",
                "Antarctic/Southern Ocean": "Antarctic–SO",
                "Tropical": "Tropical",
                "East Asian Monsoon": "E. Asian Mon.",
                "South American Monsoon": "S. Amer. Mon.",
                "High-Latitude Lakes": "Lakes"}
REGION_KEY = {
    "North_Atlantic": "North Atlantic",
    "Antarctic": "Antarctic/Southern Ocean",
    "Southern_Ocean": "Antarctic/Southern Ocean",
    "Tropical": "Tropical", "Tropical_Marine": "Tropical",
    "East_Asian_Monsoon": "East Asian Monsoon",
    "South_America": "South American Monsoon",
    "South_American_Monsoon": "South American Monsoon",
    "Other_Lakes": "High-Latitude Lakes", "Lake": "High-Latitude Lakes",
}

# tier ramp of 02_records_overview (dark = Tier 1); cross-figure symbol
TIER_COLORS = {"Tier 1": "#184f95", "Tier 2": "#3987e5", "Tier 3": "#9ec5f4"}
RUNGS = (0.95, 0.90, 0.85, 0.80, 0.75, 0.70)

# §4.3 named records: stem -> (expected value, profile column)
NAMED = {
    "Junin_GI": (87, "total"),
    "U1302-3_d18O": (86, "total"),
    "DomeFuji_dD": (84, "total"),
    "U1302-3_CaSr": (31, "recov_noage"),
    "MD01-2443_UK37_SST": (29, "recov_noage"),
    "Elgygytgyn_SiTi": (27, "recov_noage"),
    "IPWP_OHC": (52, "recov_noarc"),
}


# ------------------------------------------------------------------ data ----
def load_data():
    """(prof, shards, ccm) - the canonical scan outputs."""
    prof = pd.read_csv(os.path.join(SCAN_DIR, "record_profile.csv"))
    assert len(prof) == 80, f"expected 80 records, got {len(prof)}"
    prof["total"] = prof["n_out_pass95"] + prof["n_in_pass95"]
    prof["region6"] = prof["region"].map(REGION_KEY)
    assert prof["region6"].notna().all(), "region map incomplete"
    tier = pd.read_csv(os.path.join(OUTPUT_DIR, "data_tiers_final.csv"))
    prof = prof.merge(tier[["Record", "Tier"]], left_on="record",
                      right_on="Record", how="left")
    assert prof["Tier"].notna().all(), "tier merge incomplete"
    prof["_disp"] = prof["record"].map(display_name)
    prof["_rk"] = prof["region6"].map(REGION6_ORDER.index)

    shards = pd.concat([pd.read_csv(f) for f in sorted(
        glob_scan_shards())], ignore_index=True)
    ccm = pd.read_csv(os.path.join(SCAN_DIR, "ccm_pairs.csv"))
    return prof, shards, ccm


def glob_scan_shards():
    import glob
    return glob.glob(os.path.join(SCAN_DIR, "shard_*.csv"))


def rung_counts(shards, col):
    """[n pairs with rate >= rung] for 95..70%."""
    return [int((shards[col] >= r).sum()) for r in RUNGS]


# ------------------------------------------------------------------ plot ----
def panel_a(ax, prof):
    """(a) 80-record dot strip: raw 95% passes, tier colours, region bands."""
    p = prof.sort_values(["_rk", "total", "_disp"],
                         ascending=[True, False, True]).reset_index(drop=True)
    y = 79 - np.arange(len(p))          # row 0 (list head) at the top
    blocks = {}
    for reg in REGION6_ORDER:
        m = p.region6 == reg
        if m.any():
            i0, i1 = np.where(m)[0][[0, -1]]
            blocks[reg] = (79 - i0, 79 - i1)     # (top, bottom) in y coords
            span = ax.axhspan(79 - i1 - 0.5, 79 - i0 + 0.5,
                              color=REGION6_CATS[reg], alpha=0.16, zorder=0)
            span.set_gid(f"layer-background-{reg.replace(' ', '_')}")
            ax.text(1.2, 79 - i0 + 0.42, SHORT_REGION[reg], fontsize=8.5,
                    fontweight="bold", color=INK, va="center", zorder=4)
            if reg != REGION6_ORDER[-1]:
                ax.axhline(79 - i1 - 0.5, color=GRID, lw=0.6, zorder=1.5)
    # dots: one colour per tier (PathCollection -> markers layer)
    for tier, col in TIER_COLORS.items():
        m = p.Tier == tier
        ax.scatter(p.total[m], y[m], s=12, color=col, zorder=2,
                   edgecolors="none")
    # §4.3 named records: name next to the dot (the dot carries the count)
    for stem in NAMED:
        i = p.index[p.record == stem]
        assert len(i) == 1, f"named record {stem} not unique"
        ax.text(p.total[i[0]] - 1.2, y[i[0]], display_name(stem),
                ha="right", va="center", fontsize=9, color=INK, zorder=4)
    # big "0 of 6,320" honesty block (left strip is empty: min passes = 24)
    ax.text(9, 46.3, "0", ha="center", va="center", fontsize=26,
            fontweight="bold", color=INK, zorder=4)
    ax.text(9, 41.7, "of 6,320", ha="center", va="center", fontsize=10,
            fontweight="bold", color=INK, zorder=4)
    ax.text(9, 39.3, "pairs pass the", ha="center", va="center", fontsize=9,
            color=INK, zorder=4)
    ax.text(9, 36.9, "honesty check", ha="center", va="center", fontsize=9,
            color=INK, zorder=4)
    # tier decode (bottom-left; dots on those rows sit at x >= 28)
    for k, (tier, col) in enumerate(TIER_COLORS.items()):
        ty = 0.15 + 1.7 * k
        ax.scatter([1.5], [ty], s=12, color=col, edgecolors="none", zorder=4)
        ax.text(3.0, ty, tier, ha="left", va="center", fontsize=8, color=INK,
                zorder=4)
    ax.grid(False, axis="y")
    ax.grid(True, axis="x", color=GRID, linewidth=0.5, alpha=0.8, zorder=0.6)
    ax.set_axisbelow(False)
    ax.set_ylim(-0.8, 79.8)
    ax.set_yticks([])
    ax.set_xlim(0, 90)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xlabel("Raw 95%-honesty passes per record (out + in roles)",
                  fontsize=10, color=INK)
    ax.set_title("(a) Raw 95%-honesty passes per record, grouped by region "
                 "(tier-coloured dots)",
                 loc="left", fontsize=10.5, fontweight="bold", color=INK)


def panel_b(ax, prof):
    """(b) regional means: raw / recovered -age / recovered -ARC bars."""
    g = (prof.groupby("region6")
         .agg(n=("record", "size"),
              raw=("total", "mean"),
              rec_age=("recov_noage", "mean"),
              rec_arc=("recov_noarc", "mean"))
         .reindex(REGION6_ORDER))
    y = np.arange(6)[::-1]
    for i, key in enumerate(("raw", "rec_age", "rec_arc")):
        off = (1 - i) * 0.21
        ax.barh(y + off, g[key], height=0.17,
                color=[REGION6_CATS[r] for r in REGION6_ORDER], zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{SHORT_REGION[r]}\n(n = {int(g.loc[r, 'n'])})"
                        for r in REGION6_ORDER], fontsize=9, color=INK)
    ax.set_ylim(-0.55, 5.85)
    ax.text(0, 5.5, "honest = 0 in every region", fontsize=9, color=INK,
            va="center", zorder=4)
    ax.grid(False, axis="y")
    ax.set_xlim(0, 65)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xlabel("Mean directed edges per record", fontsize=10, color=INK)
    ax.set_title("(b) Regional means\nbars: raw · −age · −ARC",
                 loc="left", fontsize=10.5, fontweight="bold", color=INK)


def panel_c(ax, prof):
    """(c) ablation ladder: mean edges recovered when one layer is removed."""
    means = [float(prof["recov_nonoise"].mean()),
             float(prof["recov_noage"].mean()),
             float(prof["recov_noarc"].mean())]
    labs = ["Observational\nnoise", "Age\nerror", "ARC\ndeconfounding"]
    y = np.arange(3)
    ax.barh(y, means, height=0.55, color="#c9c8c3", edgecolor="none",
            zorder=3)
    for yi, m in zip(y, means):
        ax.text(m + 0.7, yi, f"{m:.1f}", ha="left", va="center", fontsize=9,
                fontweight="bold", color=INK, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=9, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, 2.6)
    ax.grid(False, axis="y")
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xlabel("Mean edges recovered per record\nwhen one layer is "
                  "removed", fontsize=9, color=INK)
    ax.set_title("(c) Ablation ladder", loc="left", fontsize=10.5,
                 fontweight="bold", color=INK)


def panel_d(ax, counts):
    """(d) honesty ladder readout: forced-difference passes per rung."""
    band = ax.axhspan(-0.5, 5.5, color=THR_BAND, zorder=0)
    band.set_gid("layer-thr-readout")
    y = np.arange(6)[::-1]
    ax.barh(y, counts, height=0.55, color=INK, zorder=3)
    for yi, c in zip(y, counts):
        if c == 0:
            ax.text(0.7, yi, "0", ha="left", va="center", fontsize=9.5,
                    fontweight="bold", color=INK, zorder=4)
        else:
            ax.text(c + 1.3, yi, str(c), ha="left", va="center", fontsize=9,
                    color=INK, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{int(r * 100)}%" for r in RUNGS], fontsize=9,
                       color=INK)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_ylim(-0.6, 5.6)
    ax.grid(False, axis="y")
    ax.set_xlim(0, 55)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xlabel("Directed pairs passing (honest variant)", fontsize=10,
                  color=INK)
    ax.set_title("(d) Honesty ladder readout", loc="left", fontsize=10.5,
                 fontweight="bold", color=INK)


def plot(prof, shards, ccm):
    os.makedirs(OUT_DIR, exist_ok=True)
    counts = rung_counts(shards, "rate_alldiff")

    fig = Figure(figsize=(FIG_W / 25.4, FIG_H / 25.4), facecolor="white")
    ax = fig.add_axes([AX_L, 0.415, AX_R - AX_L, 0.535])
    apply_paleo_style(ax)
    panel_a(ax, prof)

    ax = fig.add_axes([0.07, 0.152, 0.265, 0.22])
    apply_paleo_style(ax)
    panel_b(ax, prof)

    ax = fig.add_axes([0.372, 0.152, 0.25, 0.22])
    apply_paleo_style(ax)
    panel_c(ax, prof)

    ax = fig.add_axes([0.66, 0.152, 0.325, 0.22])
    apply_paleo_style(ax)
    panel_d(ax, counts)

    foot = [
        "data: outputs/pairwise_scan/ (scripts/19) — 6,320 pairs (80×79), "
        "200 perturbations, ADF gate, ARC α = 0.5",
        "raw: 80/80 records pass ≥1 pairing (mean 56.1; 75/80 as driver); "
        "forced-difference honesty: 0/6,320",
        "raw passes = trend leakage: shared low-frequency forcing survives "
        "the ADF gate, not record-specific signal",
        "honest counts per rung: 95% 0 · 90% 4 · 85% 8 · 80% 21 · 75% 36 · "
        "70% 50; strongest honest pair 92%",
        "ablation: −noise 1.6 · −age 13.3 · −ARC 19.2 mean recovered "
        "edges/record; age error and deconfounding bind",
        "pure-leak: Lake Junín GI 87, U1302-U1303 δ¹⁸O 86, Dome Fuji δD 84 "
        "(labels in panel a)",
        "dating targets: Ca/Sr 31, alkenone SST 29, Si/Ti 27 (−age); "
        "IPWP OHC-stack 52 (−ARC) — Table S6",
        "CCM (400 Granger survivors): 0.52 fwd vs 0.53 rev — symmetric: "
        "common forcing, not directional dominance",
    ]
    for k, line in enumerate(foot):
        fig.text(0.048, FOOTER_Y0 - k * FOOTER_DY, line, fontsize=9,
                 color=INK)

    assert no_text_overruns(fig), "some text runs off the figure edges"
    save(fig, os.path.join(OUT_DIR, "attribution_scrutiny"))
    print_anchors(prof, shards, ccm, counts)


# ---------------------------------------------------------------- anchors ----
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


def print_anchors(prof, shards, ccm, counts):
    print("\n== anchors (Fig. 5 attribution scrutiny) ==")
    full = rung_counts(shards, "rate_full")
    print(f"  pairs: {len(shards)} | full rungs {full} | "
          f"honest rungs {counts} | strongest honest "
          f"{shards.rate_alldiff.max():.2f}")
    print(f"  records: {len(prof)} | tiers "
          f"{prof.Tier.value_counts().to_dict()} | regions "
          f"{prof.region6.value_counts().reindex(REGION6_ORDER).to_dict()}")
    print(f"  passes/record: mean {prof.total.mean():.3f}, all-pass "
          f"{(prof.total > 0).sum()}/80, driver-pass "
          f"{(prof.n_out_pass95 > 0).sum()}/80 | leak == raw everywhere: "
          f"{(prof.n_trend_leak95 == prof.total).all()}")
    print(f"  ablation: {prof.recov_nonoise.mean():.1f} / "
          f"{prof.recov_noage.mean():.1f} / {prof.recov_noarc.mean():.1f}")
    print(f"  CCM: {len(ccm)} pairs, fwd {ccm.ccm_fwd.mean():.3f} / "
          f"rev {ccm.ccm_rev.mean():.3f}")
    for stem, (want, col) in NAMED.items():
        row = prof[prof.record == stem]
        got = float(row["total"].iloc[0] if col == "total"
                    else row[col].iloc[0])
        print(f"  named: {display_name(stem):<28} {col} = {got:.0f} "
              f"(want {want})")

    assert len(shards) == 6320, "pair count drifted"
    assert full == [2243, 2468, 2621, 2735, 2866, 2971], "full rungs drifted"
    assert counts == [0, 4, 8, 21, 36, 50], "honest rungs drifted"
    assert abs(float(shards.rate_alldiff.max()) - 0.92) < 1e-9
    assert len(prof) == 80
    assert prof.Tier.value_counts().to_dict() == {"Tier 1": 7, "Tier 2": 48,
                                                  "Tier 3": 25}
    assert prof.region6.value_counts().reindex(REGION6_ORDER).tolist() == \
        [13, 18, 34, 6, 3, 6], "regional record counts drifted"
    assert abs(prof.total.mean() - 56.075) < 1e-6
    assert (prof.total > 0).sum() == 80
    assert (prof.n_out_pass95 > 0).sum() == 75
    assert (prof.n_trend_leak95 == prof.total).all(), "leak != raw somewhere"
    assert abs(prof.recov_nonoise.mean() - 1.55) < 1e-9
    assert abs(prof.recov_noage.mean() - 13.3) < 1e-9
    assert abs(prof.recov_noarc.mean() - 19.25) < 1e-9
    assert len(ccm) == 400
    assert abs(ccm.ccm_fwd.mean() - 0.521) < 2e-3
    assert abs(ccm.ccm_rev.mean() - 0.526) < 2e-3
    for stem, (want, col) in NAMED.items():
        row = prof[prof.record == stem]
        assert len(row) == 1, f"named record {stem} missing"
        got = float(row["total"].iloc[0] if col == "total"
                    else row[col].iloc[0])
        assert abs(got - want) < 1e-9, f"{stem}: {got} != {want}"
    # the FILE_ID_OVERRIDES fix (Table S6 csv still has the old names)
    assert display_name("5_U1302-U1303_Si_Sr") == "U1302-U1303 Si/Sr"
    assert display_name("9_U1385-Ca-Ti") == "U1385 Ca/Ti"
    print("  claim: 0/6,320 honest passes at 95%; every raw pass is trend "
          "leakage; age error and deconfounding are the binding "
          "constraints.")


def main():
    prof, shards, ccm = load_data()
    plot(prof, shards, ccm)


if __name__ == "__main__":
    main()
