# -*- coding: utf-8 -*-
"""Step 09 - conditional robustness across common-forcing strength (Fig. 3d, Section 4.1).

Claim: deconfounding holds across forcing strengths.  On null pairs the
direct Granger test's FPR rises from 0.10 to 1.00 as the common-forcing
loading beta goes 0 -> 1, while CARC's FPR stays at <= 0.10 (flat).  At
the three implemented Scenario A betas (0.15 / 0.25 / 0.35) the true-edge
robustness stays below the 7% cap (max 6.5%) and 0/30 directed pairs pass
the 95% honesty rung - the pipeline is conditionally robust.

Design:
  (a) FPR vs beta (0 -> 1), 30 null pairs per beta.  Direct Granger
      (CAT2 orange, 'o') climbs 0.10 -> 1.00; CARC (CAT1 blue, 's') stays
      <= 0.10.  Dashed reference = nominal 5%; shaded band = the
      implemented beta sweep {0.15, 0.25, 0.35} with the canonical
      beta = 0.25 dashed reference line.
  (b) true-edge robustness medians at beta = 0.15 / 0.25 / 0.35, one line
      per true edge (CAT1/CAT3/CAT5).  Dashed reference = 7% cap; the
      95% honesty rung is off-scale, so the ladder reading is an
      annotation ("0/30 pairs pass the 95% rung").

Data: outputs/fpr_vs_beta.csv (scripts/03, 30 repeats of the null pair,
n = 500, phi = 0.85, direct Granger maxlag 3; CARC = ARC alpha = 0.5 +
robustness sigma_err = 2.0, n_perturb = 100, 95% gate) and
outputs/scenarioA_final_table1_full_beta{0.15,0.25,0.35}.csv
(scripts/22, run22f pipeline).

Convergence targets (asserted):
  direct FPR: 0.10 at beta = 0, 1.00 at beta = 1, non-decreasing from 0.1
  CARC FPR: max <= 0.10
  true edges at beta = 0.25: 0.060 / 0.065 / 0.045 (= Table 1) +- 0.005
  robustness max over the 9 beta x edge cells < 0.07; 0/30 passes per beta

Usage:
    python scripts/fignew/09_beta_sweep.py
Outputs: figures_new/09_beta_sweep/beta_sweep.png / .svg
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from matplotlib import lines as mpl_lines
from matplotlib.figure import Figure

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.dirname(os.path.abspath(__file__)), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from style import (apply_paleo_style, save, INK,  # noqa: E402
                   SECONDARY_INK, CAT1, CAT2, CAT3, CAT5)

OUT_DIR = os.path.join(ROOT, "figures_new", "09_beta_sweep")

BETA_SWEEP = (0.15, 0.25, 0.35)        # implemented Scenario A betas
BETA_CANONICAL = 0.25
ROB_CAP = 0.07                         # claimed true-edge robustness cap
NOMINAL = 0.05                         # nominal direct-test FPR

EDGE_ORDER = ["NGRIP -> EDC_dD", "CO2 -> CH4", "Cariaco -> Cheng"]
EDGE_COLORS = {"NGRIP -> EDC_dD": CAT1, "CO2 -> CH4": CAT3,
               "Cariaco -> Cheng": CAT5}
EDGE_MARKERS = {"NGRIP -> EDC_dD": "o", "CO2 -> CH4": "s",
                "Cariaco -> Cheng": "^"}
SWEEP_FILL = "#f0efec"                 # neutral fill, not a data colour

FIG_W, FIG_H = 178.0, 118.0
Y0, Y1 = 0.50, 0.86                    # axes bottom / axes top
AX_A = [0.115, Y0, 0.365, Y1 - Y0]     # panel (a)
AX_B = [0.585, Y0, 0.320, Y1 - Y0]     # panel (b)
FOOTER_Y = (0.380, 0.342, 0.304, 0.266)


# ------------------------------------------------------------------ data ----
def load_fpr_beta():
    """Direct vs CARC FPR on null pairs from scripts/03."""
    p = pd.read_csv(os.path.join(ROOT, "outputs", "fpr_vs_beta.csv"))
    assert list(p.columns) == ["beta", "direct_fpr", "carc_fpr"]
    assert p["beta"].iloc[0] == 0.0 and p["beta"].iloc[-1] == 1.0
    return p


def load_edge_robustness():
    """True-edge robustness medians per beta from scripts/22 (run22f)."""
    rows = []
    for b in BETA_SWEEP:
        f = os.path.join(ROOT, "outputs",
                         f"scenarioA_final_table1_full_beta{b:.2f}.csv")
        df = pd.read_csv(f)
        assert int(df["passes_95_with_error"].sum()) == 0, f"beta {b}: passes"
        for e in EDGE_ORDER:
            s, t = e.split(" -> ")
            g = df[(df["source"] == s) & (df["target"] == t)]
            assert len(g) == 1, f"beta {b}, edge {e}"
            rows.append({"beta": b, "edge": e,
                         "rob": float(g["robustness_with_error_median"]
                                      .iloc[0])})
    return rows


# ------------------------------------------------------------------ plot ----
def plot(fpr, rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig = Figure(figsize=(FIG_W / 25.4, FIG_H / 25.4), facecolor="white")
    axa = fig.add_axes(AX_A)
    axb = fig.add_axes(AX_B)
    for ax in (axa, axb):
        apply_paleo_style(ax)
        ax.minorticks_off()

    # ---- panel (a): FPR vs common-forcing strength beta
    axa.set_xlim(-0.04, 1.04)
    axa.set_xticks(np.arange(0, 1.01, 0.2))
    axa.set_ylim(-0.04, 1.04)
    axa.set_yticks(np.arange(0, 1.01, 0.2))
    axa.set_xlabel("common-forcing strength β", fontsize=10, color=INK)
    axa.set_ylabel("FPR", fontsize=10, color=INK, labelpad=8)
    axa.set_title("(a) FPR vs common-forcing strength", loc="left",
                  fontsize=10.5, fontweight="bold", color=INK, pad=6)

    # implemented beta window (neutral fill -> background layer) +
    # canonical beta = 0.25 reference (annotations layer, 06/08 idiom)
    axa.axvspan(BETA_SWEEP[0], BETA_SWEEP[-1], color=SWEEP_FILL, zorder=0.2)
    ref = axa.axvline(BETA_CANONICAL, color=SECONDARY_INK, lw=0.8,
                      ls=(0, (4, 3)), zorder=1)
    ref.set_gid("layer-annotations")
    nom = axa.axhline(NOMINAL, color=SECONDARY_INK, lw=0.8, ls=(0, (4, 3)),
                      zorder=1)
    nom.set_gid("layer-annotations")

    axa.plot(fpr["beta"], fpr["direct_fpr"], color=CAT2, ls="-", lw=1.6,
             marker="o", ms=4.5, mfc=CAT2, mec=CAT2, mew=0.9, zorder=3)
    axa.plot(fpr["beta"], fpr["carc_fpr"], color=CAT1, ls="-", lw=1.6,
             marker="s", ms=4.5, mfc=CAT1, mec=CAT1, mew=0.9, zorder=3)

    # center-right: the only empty corner of panel (a) - the steep direct
    # curve fills the top strip and the CARC line hugs the bottom
    axa.legend(handles=[
        mpl_lines.Line2D([], [], color=CAT2, marker="o", ms=4.5, lw=1.6,
                         label="Direct Granger"),
        mpl_lines.Line2D([], [], color=CAT1, marker="s", ms=4.5, lw=1.6,
                         label="CARC (95% gate)"),
    ], loc="center right", fontsize=9, frameon=False, handlelength=1.6,
       borderpad=0.4, labelspacing=0.35)

    # ---- panel (b): true-edge robustness at the implemented betas
    # top zone above the cap line holds the edge legend (its bottom row
    # needs descender clearance from the dashed 7% cap line)
    axb.set_xlim(0.10, 0.44)
    axb.set_xticks(list(BETA_SWEEP))
    axb.set_ylim(-0.005, 0.125)
    axb.set_yticks(np.arange(0, 0.121, 0.02))
    axb.set_xlabel("β", fontsize=10, color=INK)
    axb.set_ylabel("true-edge robustness", fontsize=10, color=INK, labelpad=8)
    axb.set_title("(b) true-edge robustness vs β (Scenario A)", loc="left",
                  fontsize=10.5, fontweight="bold", color=INK, pad=6)

    # 7% cap reference (the 95% honesty rung is off this axis; the ladder
    # reading lives in the footer, same idiom as 08).  Label sits right of
    # the beta = 0.35 markers, baseline above the line so the 'p' descender
    # clears it.
    cap = axb.axhline(ROB_CAP, color=SECONDARY_INK, lw=0.8, ls=(0, (4, 3)),
                      zorder=1)
    cap.set_gid("layer-annotations")
    axb.text(0.432, 0.0795, "7% cap", ha="right", va="bottom", fontsize=9,
             color=INK, zorder=4)

    for e in EDGE_ORDER:
        sub = [r for r in rows if r["edge"] == e]
        x = np.array([r["beta"] for r in sub])
        y = np.array([r["rob"] for r in sub])
        c, m = EDGE_COLORS[e], EDGE_MARKERS[e]
        axb.plot(x, y, color=c, ls="-", lw=1.4, marker=m, ms=4.5, mfc=c,
                 mec=c, mew=0.9, zorder=3)

    axb.legend(handles=[
        mpl_lines.Line2D([], [], color=EDGE_COLORS[e], marker=EDGE_MARKERS[e],
                         ms=4.5, lw=1.4, label=e.replace(" -> ", " → "))
        for e in EDGE_ORDER
    ], loc="upper left", fontsize=9, frameon=False, handlelength=1.6,
       borderpad=0.4, labelspacing=0.35)

    # footer: data facts + ladder reading + provenance (each line short
    # enough to stay inside the figure at 9pt)
    fig.text(0.115, FOOTER_Y[0],
             "(a) n = 30 null pairs per β: Direct Granger vs CARC = ARC + "
             "95% robustness gate.", fontsize=9, color=INK)
    fig.text(0.115, FOOTER_Y[1],
             "direct FPR 0.10→1.00 as β 0→1 (dashed = nominal 5%); "
             "CARC ≤ 0.10 — deconfounding holds.", fontsize=9, color=INK)
    fig.text(0.115, FOOTER_Y[2],
             "(b) β = 0.15/0.25/0.35: robustness ≤ 6.5% < 7% cap; 0/30 "
             "pairs pass the 95% rung.", fontsize=9, color=INK)
    fig.text(0.115, FOOTER_Y[3],
             "data: fpr_vs_beta.csv (scripts/03); table1_full_beta*.csv "
             "(scripts/22).", fontsize=9, color=INK)

    save(fig, os.path.join(OUT_DIR, "beta_sweep"))
    print_anchors(fpr, rows)


# ---------------------------------------------------------------- anchors ----
def print_anchors(fpr, rows):
    print("\n== anchors (Fig. 3d conditional robustness) ==")
    d0, d1 = fpr["direct_fpr"].iloc[0], fpr["direct_fpr"].iloc[-1]
    c_max = fpr["carc_fpr"].max()
    print(f"  direct FPR: {d0:.2f} (beta 0) -> {d1:.2f} (beta 1); "
          f"monotone from beta 0.1: "
          f"{bool(np.all(np.diff(fpr['direct_fpr'].values[1:]) >= -1e-12))}")
    print(f"  CARC FPR: max {c_max:.2f} (target <= 0.10)")
    rob_max = max(r["rob"] for r in rows)
    print(f"  true-edge robustness max over 9 cells: {rob_max * 100:.2f}% "
          f"(target < 7%)")
    for b in BETA_SWEEP:
        v = [f"{r['rob'] * 100:.2f}" for r in rows if r["beta"] == b]
        print(f"  beta = {b}: " + " / ".join(v) + " %  (0/30 pass 95%)")
    print("  claim: CARC FPR stays <= 0.10 as direct FPR 0.10 -> 1.00; "
          "robustness < 7% at every implemented beta.")
    assert d0 == 0.1 and d1 == 1.0, "direct FPR endpoints drifted"
    assert np.all(np.diff(fpr["direct_fpr"].values[1:]) >= -1e-12), \
        "direct FPR not monotone from beta 0.1"
    assert c_max <= 0.10 + 1e-12, "CARC FPR above the claimed cap"
    assert rob_max < ROB_CAP, "true-edge robustness hit the 7% cap"
    canon = {r["edge"]: r["rob"] for r in rows if r["beta"] == BETA_CANONICAL}
    expect = {"NGRIP -> EDC_dD": 0.060, "CO2 -> CH4": 0.065,
              "Cariaco -> Cheng": 0.045}
    for e, v in expect.items():
        assert abs(canon[e] - v) <= 0.005, f"{e} drifted from Table 1"


def main():
    plot(load_fpr_beta(), load_edge_robustness())


if __name__ == "__main__":
    main()
