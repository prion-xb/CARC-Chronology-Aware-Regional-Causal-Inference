# -*- coding: utf-8 -*-
"""Step 01 - the atomic unit of every future figure.

One proxy record, plotted with its 1-sigma age-model error envelope.
This is the building block the whole paper stands on: every CARC analysis
starts from (age, value) pairs plus an error on the ages.

Usage:
    python scripts/figure_scripts/01_basic_record.py               # NGRIP d18O
    python scripts/figure_scripts/01_basic_record.py Lapa_d18O     # any record ID
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import setup_axes, save, INK, BAND, BAND_EDGE  # noqa: E402
from data_io import read_record, read_error, error_on_ages  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "figures_rebuilt", "01_basic_record")


def main(record_id: str = "NGRIP_d18O"):
    rec = read_record(record_id)
    err = read_error(record_id)

    ages = rec["age_ka"].values
    vals = rec["value"].values
    sigma = error_on_ages(err, ages) if err is not None else np.zeros_like(ages)

    # sort by age so the line connects in time order
    order = np.argsort(ages)
    ages, vals, sigma = ages[order], vals[order], sigma[order]

    fig, ax = setup_axes(figsize_mm=(89.0, 56.0))

    # 1-sigma envelope first (below the line), then the line on top
    ax.fill_between(ages, vals - sigma, vals + sigma,
                    color=BAND, edgecolor="none", alpha=0.7, zorder=1)
    ax.plot(ages, vals, color=INK, linewidth=0.8, zorder=2)

    if not err_is_scalar(err):
        ax.plot(ages, vals + sigma, color=BAND_EDGE, linewidth=0.4,
                alpha=0.9, zorder=1.5)
        ax.plot(ages, vals - sigma, color=BAND_EDGE, linewidth=0.4,
                alpha=0.9, zorder=1.5)

    # paleoclimate convention: older ages to the left
    ax.invert_xaxis()

    ax.set_xlabel("Age (ka BP)", fontsize=10)
    ax.set_ylabel("δ¹⁸O (‰)", fontsize=10)

    # single series -> no legend box; the title names it
    sigma_txt = f"±1σ age uncertainty" if err is not None else "no age-error file"
    ax.set_title(f"NGRIP δ¹⁸O   ({sigma_txt})", fontsize=10.5, loc="left",
                 color=INK, pad=6)

    fig.tight_layout(pad=0.6)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, record_id.split("_err")[0])
    save(fig, out)
    print(f"record: {len(ages)} points, "
          f"age {ages.min():.1f}-{ages.max():.1f} ka, "
          f"σ = {np.unique(np.round(sigma, 3))[:5]} ka")


def err_is_scalar(err) -> bool:
    return err is None or np.allclose(err["value"], err["value"].iloc[0])


if __name__ == "__main__":
    rid = sys.argv[1] if len(sys.argv) > 1 else "NGRIP_d18O"
    main(rid)
