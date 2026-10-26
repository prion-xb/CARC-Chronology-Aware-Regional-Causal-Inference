# -*- coding: utf-8 -*-
"""Teaching demo: why strip the common forcing P at all?

Generative model (paper eq. 1-2, NO direct causality between X and Y):
    P_t  = 0.98 P_{t-1} + eta        (persistent unobserved forcing)
    X_t  = 0.5 X_{t-1}  + 0.6 P_t + noise
    Y_t  = 0.5 Y_{t-1}  + 0.8 P_t + noise
X and Y are independent except for their shared response to P.

The demo runs the hand-rolled Granger F-test (demo_granger.granger_f) on:
  (a) raw series                -> inflated F (the >= 35% FPR regime);
      note the direction is decided by the loading ratio b/a, not by any
      causal structure
  (b) residuals, alpha = 0.5    -> the paper's conservative strip: reduces
      the inflation, keeps the series' AR structure
  (c) residuals, alpha = 1      -> full strip of the CONCURRENT proxy: helps,
      but does NOT clean, because each record's response to P is spread over
      lags through its own AR dynamics
  (d) oracle strip: regress on P and 20 of its lags (knowing the true P,
      which a real analyst never does) -> F ~ 1: the mechanism, exposed.

The arc of the demo is the arc of the paper: stripping compresses the
effective loading lambda_eff and lowers the FPR bound, but with an imperfect
proxy it can never reach the nominal level - which is exactly why the
honesty filter (Layer 4) exists.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo_granger import granger_f  # reuse the hand-rolled F test  # noqa: E402
from style import Figure, apply_paleo_style, save, CAT1, CAT2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, "figures_rebuilt", "demos")

rng = np.random.default_rng(3)


def deconfound(x, p, alpha):
    """Paper's ARC: strip alpha fraction of the OLS fit of x on concurrent p."""
    X = np.column_stack([np.ones(len(p)), p])
    beta = np.linalg.lstsq(X, x, rcond=None)[0]
    y_hat = X @ beta
    return x - alpha * y_hat + alpha * np.mean(y_hat)


def oracle_strip(x, p, nlags=20):
    """What an omniscient analyst could do: strip P AND its recent past."""
    n = len(p)
    P = np.column_stack([np.ones(n)] +
                        [np.r_[np.zeros(k), p[:n - k]] for k in range(nlags + 1)])
    beta = np.linalg.lstsq(P, x, rcond=None)[0]
    return x - P @ beta


def main():
    n = 600
    p = np.zeros(n)
    for t in range(1, n):
        p[t] = 0.98 * p[t - 1] + rng.normal(0, 0.05)
    x = np.zeros(n); y = np.zeros(n)
    for t in range(n):
        x[t] = 0.5 * (x[t - 1] if t else 0) + 0.6 * p[t] + rng.normal(0, 0.2)
        y[t] = 0.5 * (y[t - 1] if t else 0) + 0.8 * p[t] + rng.normal(0, 0.2)

    print(f"{'version':<34}{'F(X->Y)':>9}{'p(X->Y)':>12}{'F(Y->X)':>9}{'p(Y->X)':>12}")
    for name, xx, yy in (
        ("raw (no deconfounding)", x, y),
        ("alpha = 0.5 (paper's choice)", deconfound(x, p, 0.5), deconfound(y, p, 0.5)),
        ("alpha = 1 (full concurrent strip)", deconfound(x, p, 1.0), deconfound(y, p, 1.0)),
        ("oracle strip (P + 20 lags)", oracle_strip(x, p), oracle_strip(y, p)),
    ):
        f1, p1, _, _ = granger_f(xx, yy)
        f2, p2, _, _ = granger_f(yy, xx)
        print(f"{name:<34}{f1:9.2f}{p1:12.2e}{f2:9.2f}{p2:12.2e}")

    xr, yr = oracle_strip(x, p), oracle_strip(y, p)

    fig = Figure(figsize=(89 / 25.4, 88 / 25.4), facecolor="white")
    ax = fig.add_subplot(211); apply_paleo_style(ax)
    t = np.arange(n) * 0.05
    ax.plot(t, x, color=CAT1, linewidth=0.7, label="X (raw)")
    ax.plot(t, y, color=CAT2, linewidth=0.7, label="Y (raw)")
    ax.plot(t, p, color="#9a9a94", linewidth=0.9, ls="--", label="unobserved P")
    ax.set_title("(a) both respond to P — Granger sees a 'causal' edge", fontsize=8, loc="left")
    ax.legend(fontsize=5.5, frameon=False, loc="upper right")
    ax.set_ylabel("value", fontsize=7.5)

    ax2 = fig.add_subplot(212); apply_paleo_style(ax2)
    ax2.plot(t, xr, color=CAT1, linewidth=0.6, label="X after oracle strip")
    ax2.plot(t, yr, color=CAT2, linewidth=0.6, label="Y after oracle strip")
    ax2.set_title("(b) forcing removed — what remains is unrelated noise", fontsize=8, loc="left")
    ax2.legend(fontsize=5.5, frameon=False, loc="upper right")
    ax2.set_xlabel("time (ka)", fontsize=7.5)
    ax2.set_ylabel("residual", fontsize=7.5)

    fig.tight_layout(pad=0.8)
    os.makedirs(OUT, exist_ok=True)
    save(fig, os.path.join(OUT, "demo_confound"), dpi=200)


if __name__ == "__main__":
    main()
