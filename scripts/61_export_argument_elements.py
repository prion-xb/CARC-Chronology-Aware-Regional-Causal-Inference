#!/usr/bin/env python3
"""Export element data for the new "win the argument" figures (Fig A-F).

Same contract as scripts/60: Python computes, the drawing languages plot.
Every CSV here lands in Figures/elements/data/ and is consumed by the
MATLAB scripts in matlab/*.m (the repo's R port has no local R installation
to verify against, so these elements are MATLAB-only for now).

Sources are the canonical outputs in outputs/, plus the per-draw ensembles
written by scripts/62_perturbation_ensemble.py (copied, not recomputed).

Fig A - perturbation ensemble:     the one plot the paper cannot do without
Fig B - theory vs simulation:      Theorem 3 bound, direct vs CARC FPR
Fig C - null calibration:          true edges vs 27 controls, ROC sweeps
Fig D - funnel + trend leak:       where the 6,320 edges die
Fig E - detectability escape:      robustness vs age-error scale
Fig F - config selection parity:   the 144-config grid, Pareto front
Supp  - phase-time heatmap, sliding TPR/FPR/DDR, effective n, CCM audit,
        stacking noise
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import glob

import numpy as np
import pandas as pd
from scipy.stats import norm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
OUT = os.path.join(ROOT, "outputs")
DATA_OUT = os.path.join(ROOT, "Figures", "elements", "data")

Z_ALPHA = norm.ppf(0.95)


def write(df, name):
    os.makedirs(DATA_OUT, exist_ok=True)
    path = os.path.join(DATA_OUT, f"{name}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  data: {name}.csv  ({len(df)} rows, {len(df.columns)} cols)")


def load_shards():
    shards = glob.glob(os.path.join(OUT, "pairwise_scan", "shard_*.csv"))
    return pd.concat([pd.read_csv(p) for p in shards], ignore_index=True)


# ---------------------------------------------------------------------------
# Fig A - perturbation ensemble (from scripts/62)
# ---------------------------------------------------------------------------

def export_figA():
    for name in ("draws", "series", "meta"):
        src = os.path.join(OUT, f"perturbation_ensemble_{name}.csv")
        df = pd.read_csv(src)
        df.to_csv(os.path.join(DATA_OUT, f"figA_{name}.csv"),
                  index=False, encoding="utf-8-sig")
        print(f"  data: figA_{name}.csv  ({len(df)} rows)")


# ---------------------------------------------------------------------------
# Fig B - theory vs simulation
# ---------------------------------------------------------------------------

def _fpr_lb(beta, gamma_Z=0.6, phi=0.55, n=100, sigma_eps2=0.5):
    """Theorem 3 bound under script 30's canonical parameters (eqs. 4-5)."""
    lam = (n * beta ** 4 * gamma_Z ** 2
           / ((1.0 - phi ** 2) * sigma_eps2
              * (beta ** 2 + sigma_eps2)))
    return 1.0 - norm.cdf(Z_ALPHA - lam / 2.0)


def export_figB():
    fpr = pd.read_csv(os.path.join(OUT, "fpr_vs_beta.csv"))
    write(fpr.rename(columns={"direct_fpr": "fpr_direct",
                              "carc_fpr": "fpr_carc"}), "figB_fpr_vs_beta")

    # theory curve, same coordinate convention as figB_fpr_vs_beta
    beta = np.linspace(0, 1.0, 101)
    write(pd.DataFrame({"beta": beta,
                        "fpr_lb_canon": _fpr_lb(beta)}), "figB_theory_curve")

    cr = pd.read_csv(os.path.join(OUT, "continuous_robustness_report.csv"))
    write(cr[["beta", "age_scale", "method", "type", "mean", "std"]],
          "figB_continuous_robustness")

    mv = pd.read_csv(os.path.join(OUT, "method_validation_arc.csv"))
    write(mv, "figB_method_points")


# ---------------------------------------------------------------------------
# Fig C - null calibration (scenario A true edges vs 27 negative controls)
# ---------------------------------------------------------------------------

def export_figC():
    # Canonical run22f results, beta = 0.25 (the paper's primary scenario).
    # The numbers are the FULL 20x200 run (scenarioA_final_table1.csv), NOT
    # the screening grid row: the grid's 4x100 screening gives different cell
    # values (e.g. control max 0.15) and disagrees with paper Table 1
    # (6.0/6.5/4.5% true edges, control max 7.25%, 0/27 pass). Per-cell
    # robustness for the canonical config lives in
    # scenarioA_final_table1_full_beta0.25.csv.
    t1 = pd.read_csv(os.path.join(OUT, "scenarioA_final_table1.csv"))
    t1 = t1[t1.beta.astype(str) == "0.25"]
    edges = t1[t1.causal_edge != "negative controls (27 directed cells)"]
    ctl = t1[t1.causal_edge == "negative controls (27 directed cells)"].iloc[0]
    write(pd.DataFrame({
        "edge": ["NGRIP → EDC δD", "CO₂ → CH₄", "Cariaco → Cheng"],
        "robustness": edges.robustness_with_error.to_list(),
        "no_error": edges.robustness_no_error.to_list(),
        "passes_95": edges.passes_95.to_list(),
    }), "figC_true_edges")
    write(pd.DataFrame([{
        "n_control": 27, "control_mean": 0.059, "control_max": 0.0725,
        "n_ctl_pass95": int(ctl.n_ctl_pass95),
        "true_mean": edges.robustness_with_error.mean(),
        "noerr_true_mean": edges.robustness_no_error.mean(),
        "beta": 0.25,
    }]), "figC_control_summary")

    sh = load_shards()

    # Real-record S distribution vs the honesty threshold (an FPR staircase,
    # since no ground-truth labels exist for the 6,320 directed pairs)
    thresh = np.arange(0.0, 1.001, 0.05)
    write(pd.DataFrame({"threshold": thresh,
                        "fpr_real": [(sh.rate_full >= t).mean()
                                     for t in thresh],
                        "n_pairs": len(sh)}), "figC_real_fpr_sweep")

    # robustness (S) distribution for the real pairs, with the canonical
    # true-edge values exported separately for rug ticks
    bins = np.arange(0, 1.01, 0.1)
    h_real, _ = np.histogram(sh.rate_full.to_numpy(), bins=bins)
    write(pd.DataFrame({"rate_bin": (bins[:-1] + bins[1:]) / 2,
                        "count_real": h_real, "n_real": len(sh)}),
          "figC_S_distribution")


# ---------------------------------------------------------------------------
# Fig D - funnel and trend leak
# ---------------------------------------------------------------------------

def export_figD():
    sh = load_shards()
    full = sh.pass95_full.astype(bool)
    # Strict funnel: the ablation columns are computed on separate draw
    # subsets, so they are attribution bars, not a nesting sequence. The
    # funnel itself is the honest chain: tested -> pass 95% -> what
    # survives once shared-trend edges are gated out.
    counts = [
        {"stage": "1. Pairs tested", "n": len(sh)},
        {"stage": "2. Full pipeline\n(95% honesty)", "n": int(full.sum())},
        {"stage": "3. After trend-leak\ngate (alldiff)", "n": int(sh.pass95_alldiff.sum())},
    ]
    write(pd.DataFrame(counts), "figD_funnel_stages")

    # attribution: edges passing when one perturbation layer is removed
    write(pd.DataFrame([
        {"variant": "Full pipeline", "n_pass95": int(full.sum()),
         "n_pass80": int(sh.pass80_full.astype(bool).sum())},
        {"variant": "No observation noise", "n_pass95": int(sh.pass95_nonoise.sum()),
         "n_pass80": int(sh.pass80_nonoise.astype(bool).sum())},
        {"variant": "No age perturbation", "n_pass95": int(sh.pass95_noage.sum()),
         "n_pass80": int(sh.pass80_noage.astype(bool).sum())},
        {"variant": "No ARC deconfounding", "n_pass95": int(sh.pass95_noarc.sum()),
         "n_pass80": int(sh.pass80_noarc.astype(bool).sum())},
        {"variant": "Forced differencing", "n_pass95": int(sh.pass95_alldiff.sum()),
         "n_pass80": int(sh.pass80_alldiff.astype(bool).sum())},
    ]), "figD_attribution")

    write(pd.DataFrame([
        {"threshold": "95%", "as_tested": int(full.sum()),
         "forced_differencing": int(sh.pass95_alldiff.astype(bool).sum())},
        {"threshold": "80%", "as_tested": int(sh.pass80_full.astype(bool).sum()),
         "forced_differencing": int(sh.pass80_alldiff.astype(bool).sum())},
    ]), "figD_trend_leak")

    # per-pair robustness with and without the trend-leak gate
    meta = pd.read_csv(os.path.join(OUT, "perturbation_ensemble_meta.csv"))
    write(meta[["pair", "sigma_rms_ka", "robustness", "robustness_alldiff",
                "passes_95"]], "figD_pairs_vs_alldiff")


# ---------------------------------------------------------------------------
# Fig E - detectability escape (sparse grid, honest about it)
# ---------------------------------------------------------------------------

def export_figE():
    ph = pd.read_csv(os.path.join(OUT, "detectability_phase.csv"))
    write(ph, "figE_detectability")

    meta = pd.read_csv(os.path.join(OUT, "perturbation_ensemble_meta.csv"))
    write(meta[["pair", "sigma_rms_ka", "robustness", "robustness_alldiff",
                "passes_95"]], "figE_real_pairs")


# ---------------------------------------------------------------------------
# Fig F - config selection parity (144-config grid)
# ---------------------------------------------------------------------------

def export_figF():
    grid = pd.read_csv(os.path.join(OUT, "scenarioA_final_grid.csv"))
    # pandas may infer int dtype for arc_alpha (e.g. only 0/1 present in a
    # column view); coerce so the == 0.5 comparison holds
    grid["arc_alpha"] = grid["arc_alpha"].astype(float)
    grid["maxlag"] = grid["maxlag"].astype(int)
    sel = (grid.arc_alpha == 0.5) & (grid["diff"] == "force") & \
          (grid["test"] == "aic_f") & (grid.pert == "test_joint") & \
          (grid.maxlag == 50)
    grid = grid.copy()
    grid["selected"] = sel
    # Pareto frontier over (err_ctl_mean, err_true_mean): minimise both
    x = grid.err_ctl_mean.to_numpy()
    y = grid.err_true_mean.to_numpy()
    frontier = np.zeros(len(grid), dtype=bool)
    for i in range(len(grid)):
        frontier[i] = not np.any((x < x[i] - 1e-12) & (y <= y[i] + 1e-12))
    grid["pareto"] = frontier
    write(grid[["config", "arc_alpha", "diff", "test", "pert", "maxlag",
                "err_true_mean", "err_ctl_mean", "err_ctl_max",
                "noerr_true_mean", "selected", "pareto"]], "figF_grid")


# ---------------------------------------------------------------------------
# Supplementary elements
# ---------------------------------------------------------------------------

def export_supp():
    ph = pd.read_csv(os.path.join(OUT, "phase_zzf_sliding.csv"))
    keep = ph.groupby(["region_A", "region_B"]).size().nlargest(3).index
    ph2 = ph[ph.set_index(["region_A", "region_B"]).index.isin(keep)]
    write(ph2, "figS6_phase_heatmap")
    # pure-numeric matrix + row labels, so the MATLAB side never has to
    # split string pairs (its CSV string handling is fragile)
    mat = ph2.pivot_table(index=["region_A", "region_B"],
                          columns="window_start",
                          values="max_directional_consistency")
    t = sorted(mat.columns)
    cols = {"time_ka": t}
    for k in range(len(mat)):
        cols[f"pair_{k}"] = mat.iloc[k].values
    write(pd.DataFrame(cols), "figS6_phase_matrix")
    write(pd.DataFrame({"pair_label": [f"{a} / {b}"
                                       for a, b in mat.index]}),
          "figS6_phase_labels")

    sw = pd.read_csv(os.path.join(OUT, "sliding_window_cache.csv"))
    write(sw, "figS6_sliding_window")

    eff = pd.read_csv(os.path.join(OUT, "eff_sample_size_by_window.csv"))
    write(eff, "figS6_eff_n")

    ccm = pd.read_csv(os.path.join(OUT, "ccm_audit_phi.csv"))
    write(ccm, "figS6_ccm_audit")

    sn = pd.read_csv(os.path.join(OUT, "stacking_noise_reduction.csv"))
    write(sn, "figS6_stacking_noise")


if __name__ == "__main__":
    print("Exporting new element data...")
    export_figA()
    export_figB()
    export_figC()
    export_figD()
    export_figE()
    export_figF()
    export_supp()
    print(f"\nWritten to {DATA_OUT}")
