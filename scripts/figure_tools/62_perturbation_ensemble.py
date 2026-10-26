#!/usr/bin/env python3
"""Retain the per-draw detail that the robustness score throws away.

`src.granger.robustness` and `scripts/19_full_pairwise_scan.py` collapse the
200-member age-perturbation ensemble into a single pass rate, which is all the
honesty filter needs but hides *why* the rate is what it is. This script reruns
the same canonical real-record path and keeps, for every draw:

  - the minimum Granger p-value and the lag that attained it
  - the lead-lag estimated by cross-correlation
  - a subset of the perturbed series themselves

so the ensemble can be plotted rather than summarised. Nothing about the test
changes: ARC alpha=0.5 against a 1-ka Gaussian-smoothed NGRIP proxy, ADF-
conditional differencing, min-p F engine at MAX_GRANGER_LAG, joint pair
perturbation at the RMS sigma, 5% observational noise, M = N_PERTURB.

Three pairs are run, ordered by combined age uncertainty, so the ensemble can
be read as a sigma ladder rather than a single anecdote.

Output: outputs/perturbation_ensemble_{draws,series,meta}.csv
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from statsmodels.tsa.stattools import adfuller

from src.config import (DATA_DIR, OUTPUT_DIR, N_PERTURB, ARC_ALPHA, NOISE_STD,
                        MAX_GRANGER_LAG, STABILITY_THRESHOLD)
from src.io import load_all_records
from src.arc import arc_deconfound_batch
from src.granger_fast import granger_min_p_fast

RNG_BASE = 20260826
N_SERIES_DRAWS = 25          # perturbed realizations kept for the spaghetti panel
XCORR_MAX_KA = 3.0           # lead-lag search half-window

# driver -> target, with display labels. Ordered by combined sigma.
PAIRS = [
    ("16_MD2622ODP1002-Mo", "14_cheng2016composite-noaa",
     "Cariaco Mo", "Chinese speleothem δ¹⁸O"),
    ("Erhardt_NGRIP_aerosol_d18O_step1", "32_Bereiter et al.-2015-GRL-revised CO2 ",
     "NGRIP δ¹⁸O", "EDC CO₂"),
    ("2_ODP983-IRD", "3_Mcmanus-ODP980-d18O",
     "ODP 983 IRD", "ODP 980 δ¹⁸O"),
]


def prepare():
    """Records, ARC variants and ADF flags -- identical to scripts/19."""
    records, grid = load_all_records(DATA_DIR, age_min=10.0, age_max=60.0)
    names = [n for _, _, _, n in records]
    raw = {n: np.asarray(v, float) for v, _, _, n in records}
    sigmas = {n: s for _, s, _, n in records}

    ngrip = [n for n in names if "GRIP" in n.upper()][0]
    dt = float(grid[1] - grid[0])
    proxy = gaussian_filter1d(raw[ngrip], sigma=1.0 / dt, mode="nearest")
    arc = arc_deconfound_batch(raw, proxy, alpha=ARC_ALPHA, keep_proxy=ngrip)

    flags = {}
    for n, v in arc.items():
        try:
            flags[n] = adfuller(v, autolag="AIC")[1] > 0.05
        except Exception:
            flags[n] = False
    return arc, sigmas, flags, grid, dt


def _xcorr_lag_ka(Y, X, dt, max_ka=XCORR_MAX_KA):
    """Lead-lag of X relative to Y, by peak absolute cross-correlation."""
    y = (Y - Y.mean()) / (Y.std() + 1e-12)
    x = (X - X.mean()) / (X.std() + 1e-12)
    n = len(y)
    max_lag = int(round(max_ka / dt))
    lags = np.arange(-max_lag, max_lag + 1)
    cc = np.empty(len(lags))
    for i, L in enumerate(lags):
        if L < 0:
            cc[i] = np.dot(y[-L:], x[:n + L]) / (n + L)
        elif L > 0:
            cc[i] = np.dot(y[:n - L], x[L:]) / (n - L)
        else:
            cc[i] = np.dot(y, x) / n
    return float(lags[np.argmax(np.abs(cc))] * dt)


def run_pair(driver, target, arc, sigmas, flags, dt, n_perturb):
    """One directed edge: driver -> target, keeping every draw.

    Y is the target and X the driver, so `granger_min_p_fast(Y, X)` -- which
    tests whether X Granger-causes Y -- reads in the same direction as the
    label. (scripts/19 passes these the other way round; see MANIFEST note.)
    """
    Y0, X0 = arc[target], arc[driver]
    dY, dX = flags[target], flags[driver]
    sig_ka = float(np.sqrt((sigmas[driver] ** 2 + sigmas[target] ** 2) / 2))
    sig_idx = sig_ka / dt
    n = len(Y0)
    rng = np.random.default_rng(RNG_BASE)

    draws, series = [], []
    for k in range(n_perturb):
        idx = np.argsort(np.arange(n) + rng.normal(0, sig_idx, n))
        Yp, Xp = Y0[idx], X0[idx]
        Yp = Yp + rng.normal(0, NOISE_STD * np.std(Yp), n)
        Xp = Xp + rng.normal(0, NOISE_STD * np.std(Xp), n)

        if k < N_SERIES_DRAWS:
            series.append(pd.DataFrame({"draw": k, "target": Yp, "driver": Xp}))

        def test(Y, X, dy, dx):
            Yt = np.diff(Y) if dy else Y
            Xt = np.diff(X) if dx else X
            m = min(len(Yt), len(Xt))
            return granger_min_p_fast(Yt[:m], Xt[:m], MAX_GRANGER_LAG), Yt[:m], Xt[:m]

        # canonical (ADF-conditional differencing), and the two ablations that
        # say where the canonical answer comes from
        (sig, p_min, best_lag), Yt, Xt = test(Yp, Xp, dY, dX)
        (sig_ad, p_ad, lag_ad), _, _ = test(Yp, Xp, True, True)     # forced diff
        (sig_na, p_na, lag_na), _, _ = test(Y0 + (Yp - Y0[idx]),    # noise only,
                                            X0 + (Xp - X0[idx]),   # no age shift
                                            dY, dX)

        draws.append({"draw": k, "p_min": p_min, "best_lag": best_lag,
                      "best_lag_ka": best_lag * dt, "significant": bool(sig),
                      "p_min_alldiff": p_ad, "significant_alldiff": bool(sig_ad),
                      "best_lag_ka_alldiff": lag_ad * dt,
                      "p_min_noage": p_na, "significant_noage": bool(sig_na),
                      "xcorr_lag_ka": _xcorr_lag_ka(Yt, Xt, dt)})

    return pd.DataFrame(draws), pd.concat(series, ignore_index=True), sig_ka


def main():
    arc, sigmas, flags, grid, dt = prepare()
    print(f"grid {grid[0]:.0f}-{grid[-1]:.0f} ka, dt={dt:.3f}, "
          f"M={N_PERTURB}, maxlag={MAX_GRANGER_LAG}, noise={NOISE_STD}")

    draw_rows, series_rows, meta_rows = [], [], []
    for driver, target, dlab, tlab in PAIRS:
        for who in (driver, target):
            if who not in arc:
                raise SystemExit(f"record not loaded: {who!r}")

        d, s, sig_ka = run_pair(driver, target, arc, sigmas, flags, dt, N_PERTURB)
        pair_id = f"{dlab} → {tlab}"
        rate = float(d["significant"].mean())

        d.insert(0, "pair", pair_id)
        draw_rows.append(d)

        # long form, one row per (draw, role, age) -- what the spaghetti needs
        for role, col in (("driver", "driver"), ("target", "target")):
            series_rows.append(pd.DataFrame({
                "pair": pair_id, "role": role, "draw": s["draw"].values,
                "age_ka": np.tile(grid, N_SERIES_DRAWS)[:len(s)],
                "value": s[col].values}))
        for role, name in (("driver", driver), ("target", target)):
            series_rows.append(pd.DataFrame({
                "pair": pair_id, "role": role, "draw": -1,
                "age_ka": grid, "value": arc[name]}))

        meta_rows.append({
            "pair": pair_id, "driver": driver, "target": target,
            "driver_label": dlab, "target_label": tlab,
            "sigma_driver_ka": sigmas[driver], "sigma_target_ka": sigmas[target],
            "sigma_rms_ka": sig_ka, "n_draws": N_PERTURB,
            "robustness": rate, "passes_95": rate >= STABILITY_THRESHOLD,
            "robustness_alldiff": float(d["significant_alldiff"].mean()),
            "robustness_noage": float(d["significant_noage"].mean()),
            "diff_driver": bool(flags[driver]), "diff_target": bool(flags[target]),
            "median_p": float(d["p_min"].median()),
            "median_p_alldiff": float(d["p_min_alldiff"].median()),
            "iqr_lag_ka": float(d["xcorr_lag_ka"].quantile(.75)
                                - d["xcorr_lag_ka"].quantile(.25)),
        })
        print(f"  {pair_id:44s} sigma_rms={sig_ka:.2f} ka  "
              f"robustness={rate:.3f}  alldiff={d['significant_alldiff'].mean():.3f}  "
              f"noage={d['significant_noage'].mean():.3f}  "
              f"median p={d['p_min'].median():.4f}")

    for df, name in ((pd.concat(draw_rows, ignore_index=True), "draws"),
                     (pd.concat(series_rows, ignore_index=True), "series"),
                     (pd.DataFrame(meta_rows), "meta")):
        path = os.path.join(OUTPUT_DIR, f"perturbation_ensemble_{name}.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  wrote {os.path.basename(path)}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
