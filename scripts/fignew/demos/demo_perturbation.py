# -*- coding: utf-8 -*-
"""Teaching demo: age-model errors fabricate false leads and lags.

Introduction trap 2 (paper para 22): published 1-sigma age errors within
event windows range from 0.14 to 1.66 ka, "often comparable to or larger
than the lags of interest" - yet most published analyses adopt a single
best age scale and never propagate the error.

The demo makes the trap concrete with real data.  Two records A and B are
synchronous BY CONSTRUCTION (B = A on the true age scale).  Each published
age scale is then one draw from a smooth age-model error field with
marginal 1-sigma = 0.65 ka - NGRIP's published value, inside the cited
0.14-1.66 ka range.  On the two published scales:

  * the cross-correlation peak moves from 0 to a spurious tau* = -0.35 ka
    (B appears to lead A by 0.35 ka, comparable to the 1-sigma errors),
  * the Granger test finds a highly significant edge where none exists
    (F = 6.4, p ~ 1e-9) while the reverse direction is far from
    significance (F = 1.0, p ~ 0.4).

Every number printed to stdout is the convergence target; the script
asserts them so the demo cannot drift from the paper's citations.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                      # demo_granger
sys.path.insert(0, os.path.join(_HERE, ".."))  # style, data_io
from data_io import read_record  # noqa: E402
from demo_granger import granger_f  # noqa: E402
from style import (Figure, apply_paleo_style, save,  # noqa: E402
                   INK, SECONDARY_INK, CAT1, BAND)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, "figures_new", "demos")

# ------------------------------------------------------------- parameters ---
RECORD_ID = "NGRIP_d18O"
GRID = np.arange(10, 60.001, 0.05)   # canonical 10-60 ka grid @ 0.05 ka
LOWPASS_KA = 0.5                     # Layer-1 event-scale filter
SIGMA_KA = 0.65                      # marginal 1-sigma of each age model
KERN_KA = 3.0                        # coherence scale of the error field
SEED = 33
P_LAGS = 10                          # Granger lags (0.5 ka of history)
MAX_LAG_KA = 3.0                     # panel (b) x-range


def age_model_warp(rng, n, sigma_ka=SIGMA_KA, kern_ka=KERN_KA):
    """One draw from a smooth age-model error field.

    White noise low-passed by a Gaussian of width kern_ka, normalized to
    marginal std sigma_ka: a coherent, per-age error like the output of a
    Bayesian age-depth model, not pointwise independent jitter.
    """
    dt = GRID[1] - GRID[0]
    e = rng.normal(0, 1, n)
    w = gaussian_filter1d(e, sigma=kern_ka / dt, mode="reflect")
    w = w - w.mean()
    return w * (sigma_ka / w.std())


def crosscorr(x, y, lags):
    """r(tau) = corr(x(t+tau), y(t)) for tau in lags (index units)."""
    n = len(x)
    return np.array([np.corrcoef(x[l:], y[:n - l])[0, 1] if l >= 0
                     else np.corrcoef(x[:n + l], y[-l:])[0, 1] for l in lags])


def no_text_overruns(fig):
    """Every text stays inside the figure (0.5% tolerance), measured on the
    real renderer (11's lesson: footer lines can overflow the right edge)."""
    renderer = fig.canvas.get_renderer()
    w, h = fig.bbox.width, fig.bbox.height
    ok = True
    texts = list(fig.texts)
    for ax in fig.axes:
        texts += list(ax.texts)
    for t in texts:
        bb = t.get_window_extent(renderer=renderer)
        if (bb.x0 < -0.005 * w or bb.x1 > 1.005 * w
                or bb.y0 < -0.005 * h or bb.y1 > 1.005 * h):
            print(f"  text overrun: {t.get_text()[:48]!r} "
                  f"x=[{bb.x0:.0f},{bb.x1:.0f}] of {w:.0f}px")
            ok = False
    return ok


def main():
    # ---- true record: NGRIP d18O, event-scale low-pass, standardized -------
    rec = read_record(RECORD_ID)
    ages = rec["age_ka"].values
    vals = rec["value"].values
    order = np.argsort(ages)
    ages, vals = ages[order], vals[order]
    a_true = np.interp(GRID, ages, vals)
    a_true = gaussian_filter1d(a_true, sigma=LOWPASS_KA / (GRID[1] - GRID[0]),
                               mode="reflect")
    a_true = (a_true - a_true.mean()) / a_true.std()
    n = len(GRID)

    # ---- published scales: one draw per record from its age model ---------
    rng = np.random.default_rng(SEED)
    w_a = age_model_warp(rng, n)
    w_b = age_model_warp(rng, n)
    assert np.all(np.diff(GRID - w_a) > 0), "age scale A is not monotonic"
    assert np.all(np.diff(GRID - w_b) > 0), "age scale B is not monotonic"
    a_pub = np.interp(GRID - w_a, GRID, a_true)
    b_pub = np.interp(GRID - w_b, GRID, a_true)

    # ---- apparent lead/lag: cross-correlation of the two pairs ------------
    lags = np.arange(-MAX_LAG_KA / 0.05, MAX_LAG_KA / 0.05 + 1, dtype=int)
    tau = lags * 0.05
    r_true = crosscorr(a_true, a_true, lags)     # peak exactly at 0
    r_pub = crosscorr(a_pub, b_pub, lags)        # peak at the spurious lag
    i_star = int(np.argmax(r_pub))
    tau_star = tau[i_star]

    # ---- Granger F-test on the published pair -----------------------------
    f_ab, p_ab, _, _ = granger_f(a_pub, b_pub, p=P_LAGS)
    f_ba, p_ba, _, _ = granger_f(b_pub, a_pub, p=P_LAGS)

    # ---- convergence targets (must stay true) -----------------------------
    assert np.allclose([w_a.std(), w_b.std()], SIGMA_KA, atol=1e-9)
    assert n == 1001, n
    assert abs(tau_star - (-0.35)) < 0.05, tau_star
    assert abs(r_true[60] - 1.0) < 1e-12 and abs(tau[60]) < 1e-12
    assert abs(r_pub[i_star] - 0.90) < 0.01, r_pub[i_star]
    assert abs(f_ab - 6.4) < 0.1 and p_ab < 1e-8, (f_ab, p_ab)
    assert abs(f_ba - 1.0) < 0.1 and p_ba > 0.1, (f_ba, p_ba)
    assert 0.14 - 1e-9 <= SIGMA_KA <= 1.66 + 1e-9, "sigma outside cited range"

    print("demo_perturbation - Introduction trap 2: "
          "age errors fabricate false leads/lags")
    print(f"  truth: B = A on true ages (synchronous); apparent lead = 0 ka")
    print(f"  published pair: apparent lead tau* = {tau_star:+.2f} ka "
          f"(B appears to lead A by {-tau_star:.2f} ka)")
    print(f"  peak correlation: {r_true[60]:.3f} (true) -> "
          f"{r_pub[i_star]:.3f} (published)")
    print(f"  Granger (p = {P_LAGS} lags): A->B F = {f_ab:.1f}, "
          f"p = {p_ab:.1e}   [false edge]")
    print(f"                         B->A F = {f_ba:.1f}, "
          f"p = {p_ba:.1e}   [null direction]")
    print(f"  age-model error: smooth field, marginal 1 sigma = "
          f"{SIGMA_KA:.2f} ka (NGRIP, published; Introduction cites "
          f"0.14-1.66 ka)")
    print(f"  data: NGRIP d18O, {GRID.min():.0f}-{GRID.max():.0f} ka @ "
          f"{GRID[1] - GRID[0]:.2f} ka, {n} points, {LOWPASS_KA:.1f} ka "
          f"low-pass, seed {SEED}")

    # ---- figure ------------------------------------------------------------
    fig = Figure(figsize=(178 / 25.4, 76 / 25.4), facecolor="white")
    gs = fig.add_gridspec(1, 2, left=0.075, right=0.975, top=0.86,
                          bottom=0.215, wspace=0.30)

    # (a) the two published scales: same signal, apparent offset
    ax1 = fig.add_subplot(gs[0, 0])
    apply_paleo_style(ax1)
    ax1.plot(GRID, a_pub, color=INK, linewidth=0.8, label="A, published scale")
    ax1.plot(GRID, b_pub, color=CAT1, linewidth=0.8, label="B, published scale")
    ax1.invert_xaxis()
    ax1.set_xlim(60, 10)
    ax1.set_ylim(-3.6, 3.6)
    ax1.set_xlabel("Age (ka BP)", fontsize=10)
    ax1.set_ylabel("normalized δ¹⁸O", fontsize=10)
    ax1.set_title("(a) synchronous pair, published age scales",
                  fontsize=10.5, loc="left", color=INK, pad=6)
    ax1.legend(fontsize=9, frameon=False, loc="upper right")

    # local apparent lead at the event where the two curves separate most
    sep = np.abs(a_pub - b_pub)
    t_star = int(np.argmax(sep))
    v_mid = 0.5 * (a_pub[t_star] + b_pub[t_star])
    k = np.arange(max(0, t_star - 20), min(n, t_star + 21))
    cross_a = k[np.argmin(np.abs(a_pub[k] - v_mid))]
    cross_b = k[np.argmin(np.abs(b_pub[k] - v_mid))]
    age_a = GRID[cross_a]
    age_b = GRID[cross_b]
    ax1.annotate("", xy=(age_b, v_mid), xytext=(age_a, v_mid),
                 arrowprops=dict(arrowstyle="<->", color=SECONDARY_INK,
                                 lw=0.8, shrinkA=2, shrinkB=2))

    # (b) cross-correlation: peak 0 -> spurious tau*
    ax2 = fig.add_subplot(gs[0, 1])
    apply_paleo_style(ax2)
    ax2.axvspan(-SIGMA_KA, SIGMA_KA, color=BAND, alpha=0.5, zorder=0)
    ax2.axvline(0, color=SECONDARY_INK, linewidth=0.7, ls=(0, (3, 3)),
                zorder=0.6)
    ax2.plot(tau, r_true, color=INK, linewidth=0.9,
             label="true pair (r(0) = 1)")
    ax2.plot(tau, r_pub, color=CAT1, linewidth=0.9, ls=(0, (6, 3)),
             label="published pair")
    ax2.axvline(tau_star, color=SECONDARY_INK, linewidth=0.7, ls=(0, (3, 3)),
                zorder=0.6)
    ax2.plot([tau_star], [r_pub[i_star]], marker="o", markersize=4,
             mfc=CAT1, mec="none", gid="layer-markers-peak", zorder=3)
    ax2.annotate(f"$\\tau^*$ = {tau_star:+.2f} ka ≈ 0.5σ",
                 xy=(tau_star, r_pub[i_star]),
                 xytext=(0.25, 0.045), textcoords="offset points",
                 ha="left", fontsize=9, color=INK)
    ax2.set_xlim(-MAX_LAG_KA, MAX_LAG_KA)
    ax2.set_ylim(0.30, 1.06)
    ax2.set_xlabel("apparent lag τ (ka)", fontsize=10)
    ax2.set_ylabel("correlation r", fontsize=10)
    ax2.set_title("(b) cross-correlation: false peak −0.35 ka",
                  fontsize=10.5, loc="left", color=INK, pad=6)
    ax2.legend(fontsize=9, frameon=False, loc="upper right")
    ax2.annotate("±1σ age error (0.65 ka)", xy=(-1.05, 1.035), fontsize=9,
                 color=INK)

    # ---- footer ------------------------------------------------------------
    f1 = ("Synchronous by construction: B = A on true ages. Each published "
          "scale = one draw from a smooth age-model")
    f2 = (f"error field with marginal 1σ = {SIGMA_KA:.2f} ka (NGRIP, "
          f"published; event-scale records span 0.14–1.66 ka).")
    f3 = (f"NGRIP δ¹⁸O, 10–60 ka @ 0.05 ka, {LOWPASS_KA:.1f} ka low-pass, "
          f"z-scored. Granger (p = {P_LAGS} lags): A→B F = {f_ab:.1f}, "
          f"p ≈ 1e-9")
    f4 = (f"— a false edge; B→A F = {f_ba:.1f}, p ≈ 0.4. Cross-correlation: "
          f"peak 0 → τ* = {tau_star:+.2f} ka (B appears to lead A).")
    for j, line in enumerate((f1, f2, f3, f4)):
        assert len(line) <= 107, (len(line), line)
        fig.text(0.075, 0.185 - 0.047 * j, line, fontsize=9, ha="left",
                 va="top", color=INK)

    # attach a real renderer (also required by the scripts/qa_figures hook)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(fig).draw()
    assert no_text_overruns(fig), "some text runs off the figure edges"

    os.makedirs(OUT, exist_ok=True)
    save(fig, os.path.join(OUT, "demo_perturbation"), dpi=200)
    print("  footer lines:", [len(x) for x in (f1, f2, f3, f4)])


if __name__ == "__main__":
    main()
