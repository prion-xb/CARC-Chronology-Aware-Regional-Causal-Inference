# -*- coding: utf-8 -*-
"""Step 04 - CARC framework overview (Fig. 1).

Claim: the four theorems map one-to-one onto the four pipeline layers
(dashed connectors), and each layer has a signature drawn from canonical
pipeline data - nothing hand-drawn.

Layout (double column, 178 mm):
  left   - four theorem cards (diagnosis): formula + statement; Theorem 2/3
           carry the analytic micro-panels of step 03 (E[F] vs beta, FPR
           lower bound vs n).
  right  - four layer cards (treatment), each with a signature panel:
           L1  stacking band    - North Atlantic median stack + P5-P95 band,
                                  members from the canonical stacking
                                  machinery (scripts/18 stackviz: median
                                  scheme, seed 42, M = 200, 10-60 ka grid).
           L2  AR-alpha curve   - residual AR(1) coefficient vs ARC strength
                                  (outputs/arc_alpha_sweep.csv, observable
                                  NGRIP deconfounding proxy: nearly flat,
                                  adopted alpha = 0.5).
           L3  DDR grid         - directional dominance ratio tpr/max(fpr,
                                  0.01) over the alpha x sigma_noise grid
                                  (outputs/param_sensitivity_cache.csv,
                                  n_perturb = 200 slice, groupby-mean as in
                                  scripts/14; run point alpha=0.5,
                                  sigma=0.05 starred).
           L4  honest-S bars    - distribution of the differenced robustness
                                  rate across all 6320 directed pairs
                                  (outputs/pairwise_scan/shard_*.csv):
                                  0/6320 reach 95% while 2243 raw passes
                                  exist - the honesty filter's signature.
  bottom - the three requirements Theorems 1-4 impose (paper Section 2),
           and the S >= 95% decision node.

Final state (2026-09-13 refresh, parts synced to the finished stage figures):
  L1 mini-panel carries the climate-window bands and the +/-3.6 z-range of
  11 (Fig. 2), and the North Atlantic load is filtered to the paper's
  80-record inventory (13 NA records, Table S2/S4); L3 adopts 07's 4-stop
  SEQ ramp anchored to DDR 1-9, the log noise axis and cell values (Fig. 3b
  colour semantics); L4 overlays the 95-70% honesty ladder (cross-figure
  symbol) and the strongest honest pair (92%).  Theorem 1 / 4 cards are
  formula + statement only (step 03 ships analytic micro-panels for
  Theorems 2-3 only); no numeric panel is invented here.

Theorem 3 is drawn WITHOUT the square root (bound = 1 - Phi(z - lambda_eff/2),
paper eq. 5): the sqrt(lambda_eff/2) form seen in an old Fig. 1 caption does
not match eq. 5 nor the 34.4% anchor and must not enter figures.

Usage:
    python scripts/fignew/04_layer_signatures.py
"""
from __future__ import annotations

import glob
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.dirname(os.path.abspath(__file__)), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from style import (apply_paleo_style, save, honesty_ladder, INK,  # noqa: E402
                   SECONDARY_INK, GRID, BAND, THR_95, CAT1, CAT2, SURFACE)
from src.config import (DATA_DIR, AGE_GRID_10_60, N_PERTURB,  # noqa: E402
                        CLIMATE_WINDOWS)
from src.io import load_all_records, filter_canonical_records  # noqa: E402
from src.stacking import build_region_dict  # noqa: E402

OUT_DIR = os.path.join(ROOT, "figures_new", "04_layer_signatures")

# canonical stacking machinery (median scheme) of scripts/18 (as reused by 33)
SPEC = importlib.util.spec_from_file_location(
    "stackviz", os.path.join(ROOT, "scripts", "18_stacking_visualization.py"))
stackviz = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stackviz)

# ------------------------------------------------------- neutral greys -----
# connectors / schematic outlines: grey is allowed on non-text elements only
CONN = "#8a8985"
MEMBER = "#b9b8b0"          # thin record lines (as in scripts/33)
BAR = "#b9b8b0"             # histogram bars
WIN_COL = ["#ecece8", "#e2e2dc"]       # alternating climate-window neutrals
# windows shown on the 10-60 ka grid, oldest first (left on the inverted
# age axis), as in 11
WINDOW_ORDER = ['GS-12 (stadial)', 'GI-12 (interstadial)', 'H4 event',
                'Last Glacial Maximum', 'H1 event', 'Bølling-Allerød']
# 4-stop ramp capped at #5e97df so black 9pt annotations stay readable on
# the darkest cell (07); DDR anchored 1-9 to keep Fig. 3b colour semantics
SEQ_BLUES = ["#f0f5fc", "#cfe0f4", "#9cc0ea", "#5e97df"]
DDR_VMIN, DDR_VMAX = 1.0, 9.0

# manuscript "typical paleoclimate parameters" (identical to step 03)
PHI = 0.55
GAMMA_Z = 0.6
SIGMA_Z2 = 1.0
SIGMA_EPS2 = 0.5
N_CANON = 100
P, Q = 1, 1


def lam_eff(n, beta):
    """Effective forcing coupling (paper eq. 4)."""
    return (n * beta ** 4 * GAMMA_Z ** 2
            / ((1.0 - PHI ** 2) * SIGMA_EPS2
               * (beta ** 2 * SIGMA_Z2 + SIGMA_EPS2)))


def e_f(n, beta):
    """Theorem 2: expected F under H0."""
    return 1.0 + lam_eff(n, beta) * (n - P - Q - 1) / (n * Q)


def fpr_bound(n, beta):
    """Theorem 3 lower bound, 1 - Phi(z_0.95 - lambda_eff/2) - NO root."""
    return 1.0 - norm.cdf(norm.ppf(0.95) - lam_eff(n, beta) / 2.0)


# ------------------------------------------------------------------ data ----
def load_na_stack():
    """North Atlantic Layer-1 stack via the canonical machinery (seed 42).

    Filtered to the paper's 80-record inventory (Table S2/S4, as 11/13);
    the North Atlantic region must then hold exactly 13 records.
    """
    records, _ = load_all_records(DATA_DIR, age_min=10.0, age_max=60.0)
    records = [(i, s, "Antarctic" if r == "Southern_Ocean" else r, n)
               for i, s, r, n in records]
    records = filter_canonical_records(records)
    region_dict = build_region_dict(records)
    na = region_dict["North_Atlantic"]
    assert len(na) == 13, f"canonical North Atlantic = {len(na)} records"
    rng = np.random.default_rng(42)
    schemes = stackviz.stack_schemes(na, AGE_GRID_10_60, N_PERTURB, 0.5, rng)
    return na, schemes["median"]


def load_ddr_grid():
    """DDR = tpr/max(fpr, 0.01) on the alpha x sigma_noise grid (N = 200)."""
    p = pd.read_csv(os.path.join(ROOT, "outputs", "param_sensitivity_cache.csv"))
    sub = p[p["n_perturb"] == N_PERTURB]
    piv = (sub.groupby(["alpha", "noise_std"], as_index=False)["robustness"]
              .mean())
    alphas = sorted(piv["alpha"].unique())
    noises = sorted(piv["noise_std"].unique())
    z = np.full((len(noises), len(alphas)), np.nan)
    for i, ns in enumerate(noises):
        for j, a in enumerate(alphas):
            row = piv[(piv["alpha"] == a) & (piv["noise_std"] == ns)]
            if len(row):
                z[i, j] = row["robustness"].iloc[0]
    return np.array(alphas), np.array(noises), z


def load_honest_s():
    """Differenced robustness rate of every directed pair (6320)."""
    shards = glob.glob(os.path.join(ROOT, "outputs", "pairwise_scan",
                                    "shard_*.csv"))
    return pd.concat([pd.read_csv(f) for f in shards], ignore_index=True)


# ---------------------------------------------------------------- layout ----
FIG_W, FIG_H = 178.0, 210.0          # print size, mm
ROWS_TOP = [0.958 - k * 0.194 for k in range(4)]
CARD_H = 0.186
LEFT_X, RIGHT_X, CARD_W = 0.018, 0.534, 0.448
HEADER_L = "Theoretical diagnosis (§2)"
HEADER_R = "CARC pipeline (§3)"

THEOREM_TITLES = [
    "Theorem 1 — residual correlation",
    "Theorem 2 — F-statistic inflation",
    "Theorem 3 — FPR lower bound",
    "Theorem 4 — nonlinearity is conservative",
]
THEOREM_FORMULAS = [
    r"$\mathrm{Cov}(X_{t-1},\,u_t) \neq 0$",
    r"$E[F] = 1 + \lambda_{eff}\,\frac{n-p-q-1}{n\,q}$",
    r"$\mathrm{FPR} \geq 1 - \Phi\left(z_{1-\alpha} - \frac{\lambda_{eff}}{2}\right)$",
    r"$|f'| \geq |a|,\ \ |g'| \geq |b|$",
]
THEOREM_NOTES = [
    "Under $H_0$, shared persistent forcing leaks into\n"
    "restricted-model residuals: Cov(X_{t−1}, u_t) ≠ 0.\n"
    "Systematic bias — not sampling noise.",
    None,   # carries the 03(a) micro-panel
    None,   # carries the 03(b) micro-panel
    "For Lipschitz-monotone forcing, |f′| ≥ |a|, |g′| ≥ |b|:\n"
    "the linear FPR bound is conservative — steep nonlinear\n"
    "threshold responses can only raise the true rate.",
]

LAYER_TITLES = [
    "Layer 1 — chronology-aware stacking",
    "Layer 2 — adaptive conservative deconfounding",
    "Layer 3 — pairwise causal test",
    "Layer 4 — honesty filter",
]
LAYER_FORMULAS = [
    r"$R^{(m)} = \mathrm{median}_{region}(X \cdot \Pi^{(m)}),\ \ m = 1..M$",
    r"$\tilde{R} = R - \alpha\,\hat{\Theta}\,P_{forcing},\ \ \alpha = 0.5$",
    r"$F_{i\to j} = \frac{(\mathrm{SSR}_R - \mathrm{SSR}_U)/q}{\mathrm{SSR}_U/(n-p-q-1)}$",
    r"$S_{i\to j} = \frac{1}{M}\sum_{m=1}^{M}\mathbf{1}\left(F^{(m)}_{i\to j} > F_{crit}\right)$",
]


def add_card(fig, x, y_top, dashed):
    """New card axes; returns (ax, border patch)."""
    ax = fig.add_axes([x, y_top - CARD_H, CARD_W, CARD_H])
    ax.set_axis_off()
    border = Rectangle((0, 0), 1, 1, transform=ax.transAxes, zorder=0,
                       facecolor=SURFACE, lw=0.9,
                       edgecolor=CONN if dashed else INK,
                       linestyle=(0, (3, 2)) if dashed else "solid")
    ax.add_patch(border)
    return ax, border


def signature_panel(card_ax):
    """Inset axes inside a card, common paleo style."""
    ax = card_ax.inset_axes([0.05, 0.045, 0.90, 0.60])
    apply_paleo_style(ax)
    return ax


def mark(ax, x, y, **kw):
    """Plot a marker that lands in the markers layer, not a colour role."""
    (line,) = ax.plot([x], [y], **kw)
    line.set_gid("layer-markers")
    return line


# ---------------------------------------------------------------- panels ----
def panel_theorem2(ax):
    """03(a): E[F] vs beta at n = 100, canonical parameters."""
    betas = np.linspace(0.05, 0.5, 200)
    ax.plot(betas, e_f(N_CANON, betas), "-", color=CAT1, lw=1.2)
    ax.axhline(1.0, color=SECONDARY_INK, lw=0.8, ls=(0, (3, 3)))
    ax.text(0.06, 1.4, "E[F] = 1", fontsize=9, color=INK)
    mark(ax, 0.35, e_f(N_CANON, 0.35), marker="o", ms=5.0, mfc="none",
         mec=INK, mew=1.1)
    ax.annotate(r"$\beta$ = 0.35 $\rightarrow$ E[F] = 3.4",
                xy=(0.35, e_f(N_CANON, 0.35)), xytext=(0.295, 6.4),
                ha="right", fontsize=9, color=INK)
    ax.set_xlabel("forcing loading β", fontsize=10)
    ax.set_ylabel(r"expected F under $H_0$", fontsize=10)
    ax.set_ylim(0.8, 9.5)


def panel_theorem3(ax):
    """03(b): FPR lower bound vs n (no legend, nominal 5% line kept)."""
    ns = np.linspace(50, 2000, 200)
    ax.plot(ns, fpr_bound(ns, 0.35), "-", color=CAT1, lw=1.2)
    ax.axhline(0.05, color=THR_95, lw=1.2, ls=(0, (4, 2)))
    ax.text(60, 0.10, "nominal 5%", fontsize=9, color=INK)
    bound = fpr_bound(N_CANON, 0.35)
    mark(ax, N_CANON, bound, marker="o", ms=5.0, mfc="none", mec=INK,
         mew=1.1)
    ax.annotate("34% at n = 100", xy=(N_CANON, bound),
                xytext=(160, 0.44), fontsize=9, color=INK)
    ax.set_xlabel("record length n", fontsize=10)
    ax.set_ylabel("false-positive rate", fontsize=10)
    ax.set_ylim(-0.02, 1.05)


def panel_layer1(ax, na_records, stacks):
    """L1 signature: climate windows + members + median + P5-P95 (NA)."""
    grid = AGE_GRID_10_60
    for j, wname in enumerate(WINDOW_ORDER):
        w0, w1 = CLIMATE_WINDOWS[wname]
        ax.add_patch(Rectangle((w0, -3.6), w1 - w0, 7.2,
                               facecolor=WIN_COL[j % 2],
                               edgecolor="none", zorder=0))
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.8, zorder=0.5)
    for interp, sigma, name in na_records:
        idx = np.round(np.linspace(0, len(interp) - 1, len(grid))).astype(int)
        ax.plot(grid, interp[idx], color=MEMBER, lw=0.5, alpha=0.8, zorder=1)
    med = np.median(stacks, axis=0)
    p5, p95 = np.percentile(stacks, [5, 95], axis=0)
    ax.fill_between(grid, p5, p95, color=BAND, alpha=0.45, lw=0, zorder=2)
    ax.plot(grid, med, color=CAT1, lw=2.0, zorder=3)
    ax.set_ylim(-3.6, 3.6)               # 11's fixed z-range (members clip)
    ax.set_xlim(10.0, 60.0)
    ax.invert_xaxis()
    ax.set_xlabel("age (ka)", fontsize=10)
    ax.set_ylabel("stack (z)", fontsize=10)


def panel_layer2(ax):
    """L2 signature: residual AR(1) vs ARC strength (observable proxy)."""
    df = pd.read_csv(os.path.join(ROOT, "outputs", "arc_alpha_sweep.csv"))
    ax.plot(df["alpha"], df["ar_coeff_mean"], "-", color=INK, lw=1.4)
    v = float(df.loc[df["alpha"] == 0.5, "ar_coeff_mean"].iloc[0])
    mark(ax, 0.5, v, marker="o", ms=5.5, mfc="none", mec=INK, mew=1.2)
    ax.annotate("adopted α = 0.5", xy=(0.5, v), xytext=(0.56, 0.9025),
                fontsize=9, color=INK)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("ARC strength α", fontsize=10)
    ax.set_ylabel("residual AR(1)", fontsize=10)
    ax.set_ylim(0.888, 0.916)


def _edges(vals, log=False):
    """Cell edges as neighbour midpoints (07 idiom)."""
    v = np.log10(vals) if log else np.asarray(vals, dtype=float)
    inner = 0.5 * (v[:-1] + v[1:])
    lo = v[0] - 0.5 * (v[1] - v[0])
    hi = v[-1] + 0.5 * (v[-1] - v[-2])
    e = np.concatenate([[lo], inner, [hi]])
    return 10.0 ** e if log else e


def panel_layer3(ax, alphas, noises, z):
    """L3 signature: DDR grid alpha x sigma_noise (Fig. 3b idiom).

    4-stop SEQ ramp anchored to DDR 1-9 as in 07, log noise axis, cell
    values, run point starred; the run-point readout sits in a strip above
    the grid (the card has no figure footer).
    """
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUES)
    xe = _edges(alphas)
    ye = _edges(noises, log=True)
    ax.grid(False)                       # the cells are their own grid
    pc = ax.pcolormesh(xe, ye, z, cmap=cmap, shading="flat",
                       vmin=DDR_VMIN, vmax=DDR_VMAX,
                       edgecolors=GRID, linewidths=0.5)
    pc.set_gid("layer-data")             # GRID edges must not class it grid
    ax.set_xlim(xe[0], xe[-1])
    ax.set_xticks(alphas)
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_yscale("log")
    ax.set_ylim(ye[0], 1.3)              # headroom for the run-point strip
    ax.set_yticks(noises)
    ax.set_yticklabels(["0.01", "0.05", "0.1", "0.2", "0.5"])
    run = float(z[list(noises).index(0.05), list(alphas).index(0.5)])
    for i, ns in enumerate(noises):
        for j, a in enumerate(alphas):
            if a == 0.5 and ns == 0.05:  # run cell: starred, no label
                continue
            ax.text(a, ns, f"{z[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color=INK, zorder=3)
    mark(ax, 0.5, 0.05, marker="*", ms=15, mfc=CAT2, mec="white",
         mew=1.0, ls="none", zorder=4)
    ax.text(0.02, 1.15,
            f"run point (α = 0.5, σ = 0.05): DDR = {run:.2f}",
            ha="left", va="center", fontsize=9, color=INK)
    ax.set_xlabel("ARC strength α", fontsize=10)
    ax.set_ylabel("obs. noise σ", fontsize=10)


def panel_layer4(ax, sh):
    """L4 signature: honest-S distribution + 95-70% honesty ladder."""
    ax.hist(sh["rate_alldiff"], bins=np.arange(0, 1.001, 0.1),
            color=BAR, edgecolor="white", lw=0.4, zorder=1)
    rungs = honesty_ladder(ax, axis="x")
    for r in rungs:
        r.set_zorder(2.5)                # rungs read above the bars
    mx = float(sh["rate_alldiff"].max())
    assert abs(mx - 0.92) < 0.005, f"strongest honest pair = {mx:.3f}"
    n_hon = int((sh["rate_alldiff"] >= 0.95).sum())
    assert n_hon == 0
    ax.text(0.965, 0.90, f"{n_hon}/6320 ≥ 95%", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=INK)
    ax.text(0.965, 0.76, f"strongest {mx:.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=INK)
    ax.set_xlabel("honest S (differenced)", fontsize=10)
    ax.set_ylabel("pairs", fontsize=10)
    ax.set_xlim(0, 1)


# -------------------------------------------------------------- decision ----
def add_decision_node(fig):
    """S >= 95% split at the bottom right; grey schematic, no colours."""
    cy = 0.086
    diamond = Polygon([(0.758, 0.130), (0.820, cy), (0.758, 0.042),
                       (0.696, cy)], closed=True, facecolor="none",
                      edgecolor=INK, lw=0.8, transform=fig.transFigure)
    diamond.set_gid("layer-annotations")
    fig.add_artist(diamond)
    fig.text(0.758, cy, "S ≥ 95%?", transform=fig.transFigure,
             ha="center", va="center", fontsize=9, color=INK)

    box_l = Rectangle((0.540, 0.030), 0.122, 0.112, facecolor="none",
                      edgecolor=CONN, lw=0.8, transform=fig.transFigure)
    box_l.set_gid("layer-annotations")
    fig.add_artist(box_l)
    fig.text(0.601, 0.118, "Honestly", transform=fig.transFigure,
             ha="center", va="top", fontsize=9, color=INK)
    fig.text(0.601, 0.086, "inferable\nS ≥ 0.95", transform=fig.transFigure,
             ha="center", va="center", fontsize=9, color=INK)

    box_r = Rectangle((0.828, 0.030), 0.150, 0.112, facecolor="none",
                      edgecolor=CONN, lw=0.8, transform=fig.transFigure)
    box_r.set_gid("layer-annotations")
    fig.add_artist(box_r)
    fig.text(0.903, 0.118, "Sub-threshold", transform=fig.transFigure,
             ha="center", va="top", fontsize=9, color=INK)
    fig.text(0.903, 0.086, "regime\nS < 0.95", transform=fig.transFigure,
             ha="center", va="center", fontsize=9, color=INK)

    for x0, x1 in ((0.696, 0.660), (0.820, 0.828)):
        a = FancyArrowPatch((x0, cy), (x1, cy), arrowstyle="-|>",
                            mutation_scale=10, color=CONN, lw=0.9,
                            transform=fig.transFigure)
        a.set_gid("layer-annotations")
        fig.add_artist(a)
    down = FancyArrowPatch((0.758, 0.190), (0.758, 0.130), arrowstyle="-|>",
                           mutation_scale=10, color=CONN, lw=0.9,
                           transform=fig.transFigure)
    down.set_gid("layer-annotations")
    fig.add_artist(down)


def add_bottom_notes(fig):
    """The three requirements of Section 2 (diagnosis -> treatment)."""
    lines = [
        "Three requirements follow from Theorems 1–4:",
        "(i)  compress " + r"$\lambda_{eff}$" + " → Layers 1–2",
        "(ii)  propagate age uncertainty → Layers 1 + 3",
        "(iii)  exclude non-robust directions → Layer 4",
        (r"$\lambda_{eff} = \frac{n\,\beta^4\,\gamma_Z^2}"
         r"{(1-\phi^2)\,\sigma_{\varepsilon}^2\,"
         r"(\beta^2\,\sigma_Z^2 + \sigma_{\varepsilon}^2)}$"),
    ]
    for y, s in zip([0.150, 0.133, 0.116, 0.099, 0.082], lines):
        fig.text(0.028, y, s, transform=fig.transFigure, ha="left", va="top",
                 fontsize=9, color=INK)


# ------------------------------------------------------------------- main ---
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # anchor check (canonical, printed for the record)
    lam = lam_eff(N_CANON, 0.35)
    bound = fpr_bound(N_CANON, 0.35)
    print(f"anchor: lambda_eff(n=100, beta=0.35) = {lam:.3f}, "
          f"E[F] = {e_f(N_CANON, 0.35):.2f}, "
          f"FPR lower bound = {bound:.3f} (manuscript: ~35%)")

    na_records, na_stacks = load_na_stack()
    print(f"layer1: North Atlantic stacking - {len(na_records)} records, "
          f"M = {N_PERTURB}, grid 10-60 ka (median scheme, seed 42)")

    alphas, noises, ddr = load_ddr_grid()
    run = ddr[list(noises).index(0.05), list(alphas).index(0.5)]
    print(f"layer3: DDR grid {alphas.tolist()} x {noises.tolist()}, "
          f"run point (alpha=0.5, sigma=0.05) DDR = {run:.2f}, "
          f"range {np.nanmin(ddr):.2f}-{np.nanmax(ddr):.2f}")

    sh = load_honest_s()
    n_raw = int((sh["rate_full"] >= 0.95).sum())
    n_hon = int((sh["rate_alldiff"] >= 0.95).sum())
    counts = [int((sh["rate_alldiff"] >= r).sum())
              for r in (0.95, 0.90, 0.85, 0.80, 0.75, 0.70)]
    mx = float(sh["rate_alldiff"].max())
    assert abs(mx - 0.92) < 0.005, f"strongest honest pair = {mx:.3f}"
    assert counts == [0, 4, 8, 21, 36, 50], counts
    print(f"layer4: {len(sh)} directed pairs, honest (differenced) "
          f"S >= 95%: {n_hon}/6320 (raw: {n_raw}/6320), "
          f"rung counts 95-70%: {counts}, strongest honest pair {mx:.2f}")

    fig = Figure(figsize=(FIG_W / 25.4, FIG_H / 25.4), facecolor="white")
    fig.text(0.242, 0.986, HEADER_L, transform=fig.transFigure, ha="center",
             va="top", fontsize=10.5, fontweight="bold", color=INK)
    fig.text(0.758, 0.986, HEADER_R, transform=fig.transFigure, ha="center",
             va="top", fontsize=10.5, fontweight="bold", color=INK)

    # theorem cards (left)
    for k in range(4):
        y_top = ROWS_TOP[k]
        ax, border = add_card(fig, LEFT_X, y_top, dashed=True)
        border.set_gid("layer-frame")
        ax.text(0.5, 0.935, THEOREM_TITLES[k], transform=ax.transAxes,
                ha="center", va="top", fontsize=10.5, fontweight="bold",
                color=INK)
        ax.text(0.5, 0.83, THEOREM_FORMULAS[k], transform=ax.transAxes,
                ha="center", va="center", fontsize=9.5, color=INK)
        if k == 1:
            panel_theorem2(signature_panel(ax))
        elif k == 2:
            panel_theorem3(signature_panel(ax))
        else:
            ax.text(0.5, 0.48, THEOREM_NOTES[k], transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color=INK,
                    linespacing=1.5)

    # layer cards (right)
    for k in range(4):
        y_top = ROWS_TOP[k]
        ax, border = add_card(fig, RIGHT_X, y_top, dashed=False)
        border.set_gid("layer-frame")
        ax.text(0.5, 0.935, LAYER_TITLES[k], transform=ax.transAxes,
                ha="center", va="top", fontsize=10.5, fontweight="bold",
                color=INK)
        ax.text(0.5, 0.83, LAYER_FORMULAS[k], transform=ax.transAxes,
                ha="center", va="center", fontsize=9.5, color=INK)
        if k == 0:
            panel_layer1(signature_panel(ax), na_records, na_stacks)
        elif k == 1:
            panel_layer2(signature_panel(ax))
        elif k == 2:
            panel_layer3(signature_panel(ax), alphas, noises, ddr)
        else:
            panel_layer4(signature_panel(ax), sh)

        conn = FancyArrowPatch((0.468, y_top - 0.093), (0.532, y_top - 0.093),
                               arrowstyle="-|>", mutation_scale=10,
                               color=CONN, lw=0.9, linestyle=(0, (4, 2)),
                               transform=fig.transFigure)
        conn.set_gid("layer-annotations")
        fig.add_artist(conn)

    add_decision_node(fig)
    add_bottom_notes(fig)

    save(fig, os.path.join(OUT_DIR, "layer_signatures"))
    print("claim: four theorems map one-to-one onto four pipeline layers; "
          "every signature panel is drawn from canonical pipeline data")


if __name__ == "__main__":
    main()
