# -*- coding: utf-8 -*-
"""Teaching demo: what does "regional stacking" mean?

North Atlantic region, three records as example. Two panels:
  (a) raw overlay - each record lives on its own time grid and its own unit
      scale, so "averaging them" is meaningless as-is;
  (b) the two standard preprocessing steps that make combination possible:
      interpolate each record to a common grid, standardize (z-score), then
      take the per-time median across records (+ P5-P95 spread).

This demonstrates the CONCEPT. Your assignment (Problem 2) is to argue for
your own choices and to carry the age errors through properly.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from style import Figure, apply_paleo_style, save, INK, CAT1, CAT2, CAT3, SECONDARY_INK  # noqa: E402
from data_io import read_record  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, "figures_new", "demos")

# three North Atlantic records, chosen to span very different ages & units
DEMO_RECORDS = [
    ("Erhardt_NGRIP_aerosol_d18O_step1", "NGRIP δ¹⁸O (‰)", CAT1),
    ("2_ODP983-IRD", "ODP 983 IRD (%)", CAT2),
    ("6_U1308-Si-Sr=0-800kyr", "U1308 Si/Sr", CAT3),
]


def main():
    fig = Figure(figsize=(89 / 25.4, 100 / 25.4), facecolor="white")

    # ---- panel (a): raw overlay ----------------------------------------
    ax = fig.add_subplot(211)
    apply_paleo_style(ax)
    for file_id, label, color in DEMO_RECORDS:
        rec = read_record(file_id)
        a, v = rec["age_ka"].values, rec["value"].values
        o = np.argsort(a)
        ax.plot(a[o], v[o], color=color, linewidth=0.6, label=label)
    ax.invert_xaxis()
    ax.set_ylabel("raw value (own units)", fontsize=7.5)
    ax.set_title("(a) raw records — different grids, different units",
                 fontsize=8, loc="left")
    # <= 4 series: direct labels instead of a legend box
    ax.legend(loc="upper right", fontsize=5.5, frameon=False)

    # ---- panel (b): common grid, z-score, median + spread --------------
    ax2 = fig.add_subplot(212)
    apply_paleo_style(ax2)

    grid = np.arange(0, 800, 0.5)  # common time grid, 0.5 ka step
    zmat = np.full((len(DEMO_RECORDS), len(grid)), np.nan)
    for i, (file_id, label, color) in enumerate(DEMO_RECORDS):
        rec = read_record(file_id)
        v = np.interp(grid, rec["age_ka"].values, rec["value"].values,
                      left=np.nan, right=np.nan)
        m = np.nanmean(v); s = np.nanstd(v)
        zmat[i] = (v - m) / s if s > 0 else np.nan
        ax2.plot(grid, zmat[i], color=color, linewidth=0.5, alpha=0.75,
                 label=label + " (z-scored)")

    with np.errstate(invalid="ignore"):
        med = np.nanmedian(zmat, axis=0)   # per-time median where >=1 record
        p5 = np.nanpercentile(zmat, 5, axis=0)
        p95 = np.nanpercentile(zmat, 95, axis=0)
    ax2.fill_between(grid, p5, p95, color=INK, alpha=0.10, linewidth=0)
    ax2.plot(grid, med, color=INK, linewidth=1.0, label="median stack")
    ax2.invert_xaxis()
    ax2.set_xlabel("Age (ka BP)", fontsize=7.5)
    ax2.set_ylabel("z-score", fontsize=7.5)
    ax2.set_title("(b) common grid + z-score + median (band = P5–P95)",
                  fontsize=8, loc="left")
    ax2.legend(loc="upper right", fontsize=5.5, frameon=False)

    fig.tight_layout(pad=0.8)
    os.makedirs(OUT, exist_ok=True)
    save(fig, os.path.join(OUT, "demo_stacking"), dpi=200)
    print("wrote demo_stacking")


if __name__ == "__main__":
    main()
