# -*- coding: utf-8 -*-
"""Step 05 - Scenario A semi-synthetic example series (Section 4.1).

Claim: the Scenario A benchmark is six standardized records on a shared
10-60 ka grid - a common forcing Z (unobserved) loads on every record,
three prescribed edges couple driver -> target pairs, and the remaining
27 directed cells are controls with no edge (one shown).

Layout (double column, 178 x 200 mm):
  (a)    common forcing Z - AR(1) + 40-ka orbital + 25 D-O pulses
  (b-g)  three prescribed-edge rows, driver left / target right, coloured
         arrow + lag label in the gutter
  (h-i)  one control example (NGRIP | Cariaco): dashed grey connector,
         "no edge" - their shared D-O structure comes from Z alone
  bottom notes: beta, edge strength, parameter source, seed, control count

Convergence targets (all printed and checked at run time):
  generator parameters = Table S1 (fitted phi 0.710/0.948/0.993/0.987/
  0.926/0.933; calibrated phi at beta=0.25 0.639/0.852/0.936/0.910/0.835/
  0.841; innovation sd 0.702/0.299/0.071/0.111/0.377/0.358; age error
  0.65/0.83/1.65/1.66/0.23/0.14 ka);
  beta = 0.25 (primary value), edge strength 0.4, lags +0.8/+1.5/+2.3 ka,
  seed 42, grid 10-60 ka at 0.05 ka (1001 points), 27 controls (1 shown).

The series come from the canonical generator (scripts/22 gen_scenario,
parameter CSV outputs/scenarioA_final_params.csv); the displayed Z is
verified bit-identical to the generator's internal Z by reproducing NGRIP
from the same seed.

Usage:
    python scripts/fignew/05_scenarioA_series.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.dirname(os.path.abspath(__file__)), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from style import (apply_paleo_style, save, INK, CAT1, CAT2, CAT3)  # noqa: E402

OUT_DIR = os.path.join(ROOT, "figures_new", "05_scenarioA_series")

# canonical generator (scripts/22_scenarioA_final.py, module name kept local)
SPEC = importlib.util.spec_from_file_location(
    "scenario_a", os.path.join(ROOT, "scripts", "22_scenarioA_final.py"))
scenario_a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scenario_a)

SEED = 42          # first realization of the canonical run (seeds 42..61)
BETA = 0.25        # primary common-forcing loading (Table S1 note)

# Table S1 as printed in the paper (column order = scenario_a.VAR_ORDER)
TABLE_S1 = {
    "fitted_phi": (0.710, 0.948, 0.993, 0.987, 0.926, 0.933),
    "gen_phi": (0.639, 0.852, 0.936, 0.910, 0.835, 0.841),
    "sigma_eps": (0.702, 0.299, 0.071, 0.111, 0.377, 0.358),
    "age_err": (0.65, 0.83, 1.65, 1.66, 0.23, 0.14),
}
DISPLAY = {"NGRIP": "NGRIP", "EDC_dD": "EDC δD", "CO2": "EDC CO₂",
           "CH4": "EDC CH₄", "Cariaco": "Cariaco", "Cheng": "Cheng"}
# (driver, target, gutter lag label, arrow colour)
EDGE_ROWS = [("NGRIP", "EDC_dD", "+0.8 ka", CAT1),
             ("CO2", "CH4", "+1.5 ka", CAT2),
             ("Cariaco", "Cheng", "+2.3 ka", CAT3)]
CONTROL_ROW = ("NGRIP", "Cariaco")

CONN = "#8a8985"    # grey is allowed on non-text elements only

# ---------------------------------------------------------------- layout ----
FIG_W, FIG_H = 178.0, 200.0              # print size, mm
X_L, X_R = 0.065, 0.935                  # panel block edges
GUT_X1, GUT_X2 = 0.470, 0.530            # arrow span in the gutter
ROW_W = (X_R - X_L - 0.090) / 2.0
ROWS_Y = [(0.690, 0.160), (0.500, 0.160), (0.310, 0.160), (0.120, 0.160)]
Z_RECT = [X_L, 0.880, X_R - X_L, 0.075]


# ------------------------------------------------------------------ data ----
def _gen_z(t, seed):
    """Replica of the Z block of scripts/22 gen_scenario (its lines 226-237)
    so the displayed forcing is exactly the generator's own; bit-identity is
    asserted below by reproducing NGRIP from the same seed."""
    np.random.seed(seed)
    n = len(t)
    Z = np.zeros(n)
    Z[0] = np.random.normal(0, 0.8)
    for i in range(1, n):
        Z[i] = 0.6 * Z[i - 1] + np.random.normal(0, 0.8 * np.sqrt(1 - 0.36))
    Z += 0.5 * np.sin(2 * np.pi * (t - 10.0) / 40.0)
    positions = np.random.uniform(10.0 + 2 * 0.4, 60.0 - 2 * 0.4, 25)
    sigma_p = 0.4 / (2 * np.sqrt(2 * np.log(2)))
    for pos in positions:
        amp = 1.0 * np.exp(-(pos - 10.0) / 1.2)
        Z += amp * np.exp(-0.5 * ((t - pos) / sigma_p) ** 2)
    return (Z - Z.mean()) / Z.std()


def _reproduce_ngrip(t, phi_map, beta_map, sig_map, seed):
    """Z + NGRIP from the raw stream of *seed*; gen_scenario applies the
    causal edges only after all six records, and NGRIP is never a target,
    so equality with the generator output proves the Z streams match."""
    Z = _gen_z(t, seed)
    x = np.zeros(len(t))
    x[0] = np.random.normal(0, 1)
    for i in range(1, len(t)):
        x[i] = (phi_map["NGRIP"] * x[i - 1] + beta_map["NGRIP"] * Z[i]
                + np.random.normal(0, sig_map["NGRIP"]))
    return (x - x.mean()) / x.std(), Z


# --------------------------------------------------------------- panels ----
def add_panel(fig, rect, title, loc="left", bottom=False):
    """One series strip in the common paleo style; only the bottom row
    carries x tick labels (shared 10-ka grid, gridlines serve the rest)."""
    ax = fig.add_axes(rect)
    apply_paleo_style(ax)
    ax.set_xlim(10.0, 60.0)
    ax.invert_xaxis()
    ax.set_xticks(np.arange(10, 61, 10))
    ax.set_yticks([-2, 0, 2])
    if bottom:
        ax.set_xlabel("age (ka)", fontsize=10)
    else:
        ax.tick_params(labelbottom=False)
    ax.set_title(title, loc=loc, fontsize=10.5, fontweight="bold",
                 color=INK, pad=2)
    return ax


def set_ylim_pair(ax_l, ax_r, a, b):
    """One shared y-range per row so the two panels stay comparable."""
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    span = hi - lo
    for ax in (ax_l, ax_r):
        ax.set_ylim(lo - 0.25 * span, hi + 0.25 * span)


def gutter_arrow(fig, yc, color):
    """Prescribed edge: arrow from the driver panel to the target panel."""
    a = FancyArrowPatch((GUT_X1, yc), (GUT_X2, yc), arrowstyle="-|>",
                        mutation_scale=11, color=color, lw=1.1,
                        transform=fig.transFigure)
    fig.add_artist(a)


def gutter_connector(fig, yc):
    """Control: dashed grey link only - no edge is prescribed here."""
    line = Line2D([GUT_X1, GUT_X2], [yc, yc], color=CONN, lw=1.0,
                  ls=(0, (4, 2)), transform=fig.transFigure)
    line.set_gid("layer-annotations")
    fig.add_artist(line)


# ------------------------------------------------------------------- main ---
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- 1. Table S1 reconciliation (paper numbers vs parameter CSV) -------
    p = pd.read_csv(os.path.join(ROOT, "outputs",
                                 "scenarioA_final_params.csv")).set_index("record")
    names = scenario_a.VAR_ORDER
    csv = {
        "fitted_phi": [float(p.loc[n, "fitted_phi"]) for n in names],
        "gen_phi": [float(p.loc[n, "gen_phi_beta_0.25"]) for n in names],
        "sigma_eps": [float(p.loc[n, "sigma_eps"]) for n in names],
        "age_err": [float(p.loc[n, "age_err_ka"]) for n in names],
    }
    print("Table S1 reconciliation (paper vs scenarioA_final_params.csv):")
    for key in TABLE_S1:
        ok = all(abs(a - b) < 0.001 for a, b in zip(TABLE_S1[key], csv[key]))
        print(f"  {key:11s} paper {['%.3f' % v for v in TABLE_S1[key]]}")
        print(f"  {'':11s} csv   {['%.3f' % v for v in csv[key]]}  "
              f"{'OK' if ok else 'MISMATCH'}")
        assert ok, f"{key} deviates from Table S1"

    # ---- 2. canonical generation + bit-identity check ----------------------
    phi_map = {n: csv["gen_phi"][i] for i, n in enumerate(names)}
    sig_map = {n: csv["sigma_eps"][i] for i, n in enumerate(names)}
    age_map = {n: csv["age_err"][i] for i, n in enumerate(names)}
    beta_map = {n: BETA for n in names}

    data, t = scenario_a.gen_scenario(SEED, False, phi_map, beta_map,
                                      sig_map, age_map, use_Z=True)
    x_ngrip, Z = _reproduce_ngrip(t, phi_map, beta_map, sig_map, SEED)
    assert np.array_equal(x_ngrip, data["NGRIP"]), "Z stream mismatch"
    print(f"bit-check: displayed Z reproduces NGRIP from seed {SEED} "
          f"(== generator's internal Z)")
    print(f"series: {len(t)} points, 10-60 ka at "
          f"{scenario_a.DT:.2f} ka, beta = {BETA}, edge strength "
          f"{scenario_a.CAUSAL_STRENGTH}, edges "
          f"{', '.join(f'{a}->{b} +{l} ka' for a, b, l in scenario_a.TRUE_EDGES)}")

    # ---- 3. figure ---------------------------------------------------------
    fig = Figure(figsize=(FIG_W / 25.4, FIG_H / 25.4), facecolor="white")

    ax_z = add_panel(fig, Z_RECT, "(a) Common forcing Z (unobserved)",
                     loc="center")
    ax_z.plot(t, Z, color=INK, lw=0.8)
    lo, hi = Z.min(), Z.max()
    ax_z.set_ylim(lo - 0.25 * (hi - lo), hi + 0.25 * (hi - lo))

    for k, (drv, tgt, lag, color) in enumerate(EDGE_ROWS):
        y0, h = ROWS_Y[k]
        ax_l = add_panel(fig, [X_L, y0, ROW_W, h],
                         f"({chr(98 + 2 * k)}) {DISPLAY[drv]} (driver)")
        ax_r = add_panel(fig, [X_R - ROW_W, y0, ROW_W, h],
                         f"({chr(99 + 2 * k)}) {DISPLAY[tgt]} (target)")
        ax_l.plot(t, data[drv], color=INK, lw=0.8)
        ax_r.plot(t, data[tgt], color=INK, lw=0.8)
        set_ylim_pair(ax_l, ax_r, data[drv], data[tgt])
        yc = y0 + h / 2.0
        gutter_arrow(fig, yc, color)
        fig.text(0.5 * (GUT_X1 + GUT_X2), yc + 0.007, lag, ha="center",
                 fontsize=9, color=INK)

    y0, h = ROWS_Y[3]
    ax_l = add_panel(fig, [X_L, y0, ROW_W, h], "(h) NGRIP (control)")
    ax_r = add_panel(fig, [X_R - ROW_W, y0, ROW_W, h],
                     "(i) Cariaco (control)")
    ax_l.plot(t, data["NGRIP"], color=INK, lw=0.8)
    ax_r.plot(t, data["Cariaco"], color=INK, lw=0.8)
    set_ylim_pair(ax_l, ax_r, data["NGRIP"], data["Cariaco"])
    yc = y0 + h / 2.0
    gutter_connector(fig, yc)
    fig.text(0.5 * (GUT_X1 + GUT_X2), yc + 0.008, "no edge", ha="center",
             fontsize=9, color=INK)
    fig.text(0.5 * (GUT_X1 + GUT_X2), yc - 0.018, "(1 of 27)", ha="center",
             fontsize=9, color=INK)

    fig.text(0.010, 0.55, "standardized (z)", rotation=90, va="center",
             fontsize=10, color=INK)
    fig.text(X_L, 0.086, "Z = AR(1) + 40-ka orbital + 25 D–O pulses; "
             "β = 0.25 (all records); edge strength 0.4", ha="left",
             va="top", fontsize=9, color=INK)
    fig.text(X_L, 0.068, "Parameters = Table S1; seed 42, 10–60 ka at "
             "0.05 ka; 1 of 27 controls shown", ha="left", va="top",
             fontsize=9, color=INK)

    save(fig, os.path.join(OUT_DIR, "scenarioA_series"))
    print("claim: six records share an unobserved forcing Z; three "
          "prescribed edges (arrows) vs one control row (no edge)")


if __name__ == "__main__":
    main()
