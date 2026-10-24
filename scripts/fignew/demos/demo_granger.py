# -*- coding: utf-8 -*-
"""Teaching demo: what a Granger causality F-test actually does.

Builds two tiny synthetic series where X truly drives Y with a 2-step lag,
then runs the test BY HAND (two OLS fits) so every quantity is visible:

  restricted   Y(t) = c + sum_i a_i Y(t-i) + e
  unrestricted Y(t) = c + sum_i a_i Y(t-i) + sum_i b_i X(t-i) + e

  F = ((SSR_R - SSR_U) / p) / (SSR_U / (n - 2p - 1))
  p-value = survival of F(p, n-2p-1)

If adding X's past shrinks the residuals "enough", X is said to Granger-cause
Y. Note what this actually means: *X's past improves the prediction of Y* -
a forecasting statement, which is exactly why Problem 3(2) and Problem 4 can
fool it. No library needed; this is ~10 lines of linear algebra.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from style import Figure, apply_paleo_style, save, INK, CAT1, CAT2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, "figures_new", "demos")

rng = np.random.default_rng(7)


def make_series(n=400, lag=2):
    """X: AR(1) noise.  Y: driven by X shifted by `lag` steps, plus noise."""
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + rng.normal(0, 1)
    y = np.zeros(n)
    for t in range(n):
        y[t] = 0.4 * (x[t - lag] if t - lag >= 0 else 0.0) \
               + 0.5 * (y[t - 1] if t >= 1 else 0.0) + rng.normal(0, 0.3)
    return x, y


def lag_matrix(z, p):
    """Design matrix of p lags of z: rows = t, cols = [z(t-1), ..., z(t-p)]."""
    n = len(z)
    Z = np.full((n, p), np.nan)
    for k in range(1, p + 1):
        Z[k:, k - 1] = z[:-k]
    return Z


def granger_f(x, y, p=3):
    """Hand-rolled Granger test of x -> y. Returns (F, p_value, ssr_r, ssr_u)."""
    n = len(y)
    Y = y[p:]
    Xr = lag_matrix(y, p)[p:]                  # restricted: only y's past
    Xu = np.column_stack([lag_matrix(y, p), lag_matrix(x, p)])[p:]  # + x's past
    for Z in (Xr, Xu):
        assert not np.isnan(Z).any()
    ssr_r = ((Y - Xr @ np.linalg.lstsq(Xr, Y, rcond=None)[0]) ** 2).sum()
    ssr_u = ((Y - Xu @ np.linalg.lstsq(Xu, Y, rcond=None)[0]) ** 2).sum()
    f = ((ssr_r - ssr_u) / p) / (ssr_u / (n - 2 * p - 1))
    pval = stats.f.sf(f, p, n - 2 * p - 1)
    return f, pval, ssr_r, ssr_u


def main():
    x, y = make_series()

    for name, f, pv, ssr_r, ssr_u in (
        ("X -> Y (true edge)", *granger_f(x, y)),
        ("Y -> X (wrong direction)", *granger_f(y, x)),
    ):
        print(f"{name}:  F = {f:8.2f}   p = {pv:.2e}   "
              f"SSR_R = {ssr_r:8.2f}   SSR_U = {ssr_u:8.2f}")

    fig = Figure(figsize=(89 / 25.4, 44 / 25.4), facecolor="white")
    ax = fig.add_subplot(111)
    apply_paleo_style(ax)
    t = np.arange(400) * 0.05  # pretend dt = 0.05 ka
    ax.plot(t, x, color=CAT1, linewidth=0.7, label="X")
    ax.plot(t, y, color=CAT2, linewidth=0.7, label="Y (lags X by 2 steps)")
    ax.set_xlabel("time", fontsize=7.5)
    ax.set_ylabel("value", fontsize=7.5)
    ax.set_title("synthetic pair: X drives Y", fontsize=8, loc="left")
    ax.legend(loc="upper right", fontsize=5.5, frameon=False)
    fig.tight_layout(pad=0.6)
    os.makedirs(OUT, exist_ok=True)
    save(fig, os.path.join(OUT, "demo_granger"), dpi=200)


if __name__ == "__main__":
    main()
