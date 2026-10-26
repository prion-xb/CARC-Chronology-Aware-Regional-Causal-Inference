# -*- coding: utf-8 -*-
"""Step 06 - True-positive rate vs SNR for the three Layer-3 testers
(Fig. 3a, Section 4.1).

Claim: Granger rises monotonically with SNR; CCM stays near zero (the
linear benchmark defeats cross-mapping); PCMCI plateaus near its SNR-1
level and declines at the strongest coupling (Cariaco -> Cheng falls
from 99% to 64%).  No 1x curve reaches even the 70% rung - the 95%
honesty bar is out of reach at realistic chronology errors; the
error-free ceiling (dashed) shows that the obstacle is data quality,
not the tester.  At SNR 1, PCMCI's Cariaco -> Cheng pass (97.5%) is
matched by a 98.0% control.

Design:
  x = SNR = coupling strength / canonical 0.4, grid {0.5, 1, 1.5, 2, 3, 5}
      (paper Section 5.3: SNR grows from 0.5 to 5.0)
  y = TPR, mean over the three prescribed edges, min-max spread shaded;
      honesty ladder background (95/90/85/80/75/70), 95% = honesty
      threshold
  solid = 1x chronology errors, dashed = error-free ceiling
  at SNR = 1 the solid curves must reproduce the paper's method table:
      Granger 6.0/6.5/4.5%, ctl max 7.25%  (Table 1 - bit-identical,
      same run22f machinery and seeds)
      CCM 4.0/0.5/0.0%, PCMCI 41.0/42.5/97.5% (Table 2; the paper
      medians are over 5 realizations - the sweep prints the 5-real
      subset and the 20-real median side by side)

Machinery per method, exactly as the paper's tables:
  Granger: scripts/22 gen_scenario + realization_block, run22f config
           (alpha=0.5, force, aic_f, test_joint, maxlag=50), 20 x 200,
           seeds 42+r, beta=0.25, params from scenarioA_final_params.csv
  CCM/PCMCI: scripts/34 submitted generator + scripts/39 numpy-CCM / tigramite
           PCMCI decision rules, 20 x 200, span 10-58 ka, per-draw
           perturbation (paper Section 3.4 order)
Strength is swept by setting the generator module attribute
CAUSAL_STRENGTH inside each joblib cell (read at call time), so the
canonical scripts stay untouched.  Each cell writes an atomic checkpoint
(outputs/06_ckpt_<method>_<c>.json), so a background run survives
interruption; reruns skip finished cells.

Usage:
    python scripts/figure_scripts/06_validation_tpr.py --quick      # smoke
    python scripts/figure_scripts/06_validation_tpr.py              # full sweep
    python scripts/figure_scripts/06_validation_tpr.py --force      # recompute all
    python scripts/figure_scripts/06_validation_tpr.py --plot-only  # redraw figure
Outputs: outputs/validation_tpr_sweep.csv (+ outputs/06_ckpt_*.json),
figures_rebuilt/06_validation_tpr/validation_tpr.png / .svg
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.dirname(os.path.abspath(__file__)), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from style import (apply_paleo_style, honesty_ladder, save, INK,
                   SECONDARY_INK, CAT1, CAT2, CAT3)  # noqa: E402

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

OUT_DIR = os.path.join(ROOT, "figures_rebuilt", "06_validation_tpr")
OUT_CSV = os.path.join(ROOT, "outputs", "validation_tpr_sweep.csv")

CANONICAL_C = 0.4
SNR_GRID = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)
STRENGTHS = tuple(round(s * CANONICAL_C, 2) for s in SNR_GRID)
BETA = 0.25
SPAN_SUBMITTED = 58.0
CFG = (0.5, "force", "aic_f", "test_joint", 50)      # canonical run22f

METHODS = ("Granger", "CCM", "PCMCI")
METHOD_COLORS = {"Granger": CAT1, "CCM": CAT3, "PCMCI": CAT2}
METHOD_MARKERS = {"Granger": "o", "CCM": "s", "PCMCI": "^"}

# paper anchors: Table 1 beta=0.25 (Granger) and Table 2 (CCM/PCMCI);
# verified against outputs/scenarioA_final_table1_full_beta0.25.csv and
# outputs/table2_ccm_pcmci.csv
PAPER_ERR = {
    "Granger": {("NGRIP", "EDC_dD"): 0.060, ("CO2", "CH4"): 0.065,
                ("Cariaco", "Cheng"): 0.045},
    "CCM": {("NGRIP", "EDC_dD"): 0.040, ("CO2", "CH4"): 0.005,
            ("Cariaco", "Cheng"): 0.000},
    "PCMCI": {("NGRIP", "EDC_dD"): 0.410, ("CO2", "CH4"): 0.425,
              ("Cariaco", "Cheng"): 0.975},
}
PAPER_NOERR = {"Granger": {("NGRIP", "EDC_dD"): 1.0, ("CO2", "CH4"): 1.0,
                           ("Cariaco", "Cheng"): 0.5}}
PAPER_CTL = {"Granger": 0.0725, "CCM": 0.085, "PCMCI": 0.98}

EDGE_ORDER = ["NGRIP -> EDC_dD", "CO2 -> CH4", "Cariaco -> Cheng"]


# ---------------------------------------------------------------- loading ----
def _load(name, relpath):
    """Fresh module instance (not registered in sys.modules): each joblib
    worker holds its own copy, so a CAUSAL_STRENGTH monkeypatch set inside
    a cell can never leak or be lost to a worker re-import."""
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(ROOT, "scripts", relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _s22_params(S22):
    p = pd.read_csv(os.path.join(ROOT, "outputs",
                                 "scenarioA_final_params.csv")).set_index("record")
    names = S22.VAR_ORDER
    return ({n: float(p.loc[n, "gen_phi_beta_0.25"]) for n in names},
            {n: BETA for n in names},
            {n: float(p.loc[n, "sigma_eps"]) for n in names},
            {n: float(p.loc[n, "age_err_ka"]) for n in names})


def _ckpt(method, c):
    return os.path.join(ROOT, "outputs",
                        f"06_ckpt_{method}_{str(c).replace('.', '_')}.json")


# ------------------------------------------------------------------- cells ----
def granger_cell(c, n_real, n_pert):
    """One strength cell: canonical run22f machinery.  The module is loaded
    in this process so S22.CAUSAL_STRENGTH = c reaches gen_scenario's
    call-time read; realizations run serially (an inner Parallel would
    re-import a fresh module with the default strength)."""
    S22 = _load("s22", "22_scenarioA_final.py")
    S22.CAUSAL_STRENGTH = float(c)
    phi_map, beta_map, sig_map, age_map = _s22_params(S22)
    pairs = S22.TRUE_PAIRS + S22.ALL_CTL
    err = {p: [] for p in pairs}
    noerr_hits = {p: 0 for p in pairs}
    for r in range(n_real):
        noerr_det, counts = S22.realization_block(
            CFG, 42 + r, phi_map, beta_map, sig_map, age_map, True, n_pert)
        for p in pairs:
            noerr_hits[p] += int(p in noerr_det)
            err[p].append(counts[p] / n_pert)
    return {"method": "Granger", "strength": c,
            "err": {f"{a} -> {b}": v for (a, b), v in err.items()},
            "noerr_hits": {f"{a} -> {b}": v for (a, b), v in noerr_hits.items()}}


def submitted_cell(c, tester, n_real, n_pert):
    """One strength cell for CCM/PCMCI: 39's rebuild machinery, exactly the
    paper Table 2 procedure (per-draw perturbation of the raw series, then
    Layer 2 keep).  Imported lazily so tigramite never enters the pickle
    of a Granger cell; 39 keeps its own internal 34, so the strength must
    be set on reg39.reg34."""
    reg39 = _load("reg39", "39_table2_ccm_pcmci.py")
    reg39.reg34.CAUSAL_STRENGTH = float(c)
    reg34 = reg39.reg34
    sigma_map = dict(reg34.S1_PARAMS["age"])
    pairs = reg39.TRUE_EDGES + reg39.ALL_CTL
    err = {p: [] for p in pairs}
    noerr_hits = {p: 0 for p in pairs}
    for r in range(n_real):
        seed = 42 + r
        rob = reg39.run_realization(seed, n_pert, sigma_map, SPAN_SUBMITTED, tester)
        for p in pairs:
            err[p].append(rob[p])
        # error-free ceiling: same pipeline, no per-draw perturbation
        data, t, Z = reg34.generate_submitted(seed, False, span=SPAN_SUBMITTED)
        ps = reg34.layer2(data, mode="keep", alpha=reg34.ARC_ALPHA, Z=Z)
        rng = np.random.default_rng(seed * 1000 + 7)
        if tester == "CCM":
            for drv, tgt in pairs:
                noerr_hits[(drv, tgt)] += int(
                    reg39.ccm_significant(ps[drv], ps[tgt], rng))
        else:
            for p in reg39.pcmci_links(ps):
                noerr_hits[p] += 1
    return {"method": tester, "strength": c,
            "err": {f"{a} -> {b}": v for (a, b), v in err.items()},
            "noerr_hits": {f"{a} -> {b}": v for (a, b), v in noerr_hits.items()}}


def _cell(method, c, n_real, n_pert, persist):
    print(f"[cell] {method} strength={c} start", flush=True)
    out = (granger_cell(c, n_real, n_pert) if method == "Granger"
           else submitted_cell(c, method, n_real, n_pert))
    if persist:
        tmp = _ckpt(method, c) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(out, fh)
        os.replace(tmp, _ckpt(method, c))          # atomic
    print(f"[cell] {method} strength={c} done", flush=True)
    return out


# -------------------------------------------------------------- assembly ----
def build_rows(cells):
    """Cells (per strength x method) -> long rows.  Per-cell TPR is the
    median over realizations (paper Table 2 wording); the plotted point is
    the mean over the three prescribed edges, spread = edge min-max."""
    rows = []
    for cell in cells:
        method = cell["method"]
        ctl_keys = [k for k in cell["err"] if k not in EDGE_ORDER]
        n_real = len(cell["err"][EDGE_ORDER[0]])
        for edge in EDGE_ORDER:
            rows.append({"method": method, "style": "err",
                         "strength": cell["strength"], "edge": edge,
                         "robustness": float(np.median(cell["err"][edge])),
                         "n_real": n_real})
            rows.append({"method": method, "style": "noerr",
                         "strength": cell["strength"], "edge": edge,
                         "robustness": cell["noerr_hits"][edge] / n_real,
                         "n_real": n_real})
        rows.append({"method": method, "style": "err",
                     "strength": cell["strength"], "edge": "control_max",
                     "robustness": float(np.max(
                         [np.median(cell["err"][k]) for k in ctl_keys])),
                     "n_real": n_real})
        rows.append({"method": method, "style": "noerr",
                     "strength": cell["strength"], "edge": "control_max",
                     "robustness": max(
                         cell["noerr_hits"][k] / n_real for k in ctl_keys),
                     "n_real": n_real})
    return pd.DataFrame(rows)


def assemble_csv():
    cells = []
    for method in METHODS:
        for c in STRENGTHS:
            with open(_ckpt(method, c), encoding="utf-8") as fh:
                cells.append(json.load(fh))
    df = build_rows(cells)
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")
    return df


# ---------------------------------------------------------------- anchors ----
def print_anchors(df, strict=True, quick_note=""):
    """Reconcile the SNR = 1 column against the paper's Table 1 / Table 2.
    Granger is asserted bit-identical (same machinery and seeds); CCM/PCMCI
    print the 20-real median next to the 5-real subset (the paper's exact
    program and seeds, asserted) plus the paper value."""
    tag = " [provisional]" if quick_note else ""
    print(f"\n== anchors at SNR = 1 (strength 0.4){quick_note} ==")
    for (drv, tgt), want in PAPER_ERR["Granger"].items():
        got = _csv_val(df, "Granger", "err", 0.4, f"{drv} -> {tgt}")
        ok = abs(got - want) < 5e-4
        print(f"  Granger 1x    {drv} -> {tgt}: sweep {got:.4f} | "
              f"Table 1 {want:.3f}  {'OK' if ok else 'MISMATCH'}{tag}")
        if strict:
            assert ok, "Granger 1x anchor deviates from Table 1"
    for (drv, tgt), want in PAPER_NOERR["Granger"].items():
        got = _csv_val(df, "Granger", "noerr", 0.4, f"{drv} -> {tgt}")
        ok = abs(got - want) < 1e-12
        print(f"  Granger noerr {drv} -> {tgt}: sweep {got:.4f} | "
              f"Table 1 {want:.2f}  {'OK' if ok else 'MISMATCH'}{tag}")
        if strict:
            assert ok, "Granger no-error anchor deviates from Table 1"
    got = _csv_val(df, "Granger", "err", 0.4, "control_max")
    ok = abs(got - PAPER_CTL["Granger"]) < 5e-4
    print(f"  Granger control_max: sweep {got:.4f} | Table 1 "
          f"{PAPER_CTL['Granger']:.4f}  {'OK' if ok else 'MISMATCH'}{tag}")
    if strict:
        assert ok, "Granger control_max anchor deviates from Table 1"
    for m in ("CCM", "PCMCI"):
        for (drv, tgt), want in PAPER_ERR[m].items():
            got20 = _csv_val(df, m, "err", 0.4, f"{drv} -> {tgt}")
            got5 = _subset_median(m, f"{drv} -> {tgt}", k=5)
            ok5 = abs(got5 - want) < 5e-4
            flag = "OK" if ok5 else "MISMATCH"
            print(f"  {m:6s} 1x   {drv} -> {tgt}: 20real {got20:.3f} | "
                  f"5real {got5:.3f} | paper(5real) {want:.3f}  {flag}{tag}")
            if strict:
                assert ok5, f"{m} 5-realization anchor deviates from Table 2"
        got = _csv_val(df, m, "err", 0.4, "control_max")
        warn = "" if abs(got - PAPER_CTL[m]) < 0.1 else \
            f"  (paper {PAPER_CTL[m]:.3f})"
        print(f"  {m:6s} control_max: 20real {got:.3f}{warn}{tag}")


def _csv_val(df, method, style, strength, edge):
    sub = df[(df["method"] == method) & (df["style"] == style)
             & (abs(df["strength"] - strength) < 1e-9)
             & (df["edge"] == edge)]
    return float(sub.iloc[0]["robustness"])


def _subset_median(method, edge, k):
    """Median over the first k realizations (seeds 42..42+k-1) - the paper
    Table 2 medians are over 5 realizations (seeds 42..46)."""
    path = _ckpt(method, 0.4)
    if not os.path.exists(path):
        return float("nan")
    with open(path, encoding="utf-8") as fh:
        cell = json.load(fh)
    return float(np.median(cell["err"][edge][:k]))


# ------------------------------------------------------------------ plot ----
FIG_W, FIG_H = 178.0, 96.0
X_L, X_R, Y0, Y1 = 0.075, 0.940, 0.16, 0.82


def _tpr_series(df, method, style):
    sub = df[(df["method"] == method) & (df["style"] == style)
             & (df["edge"] != "control_max")]
    g = sub.groupby("strength")["robustness"]
    return g.mean(), g.min(), g.max()


def plot(df):
    from matplotlib.figure import Figure

    os.makedirs(OUT_DIR, exist_ok=True)
    fig = Figure(figsize=(FIG_W / 25.4, FIG_H / 25.4), facecolor="white")
    ax = fig.add_axes([X_L, Y0, X_R - X_L, Y1 - Y0])
    apply_paleo_style(ax)
    ax.set_xlim(0.35, 5.95)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(list(SNR_GRID))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("SNR (coupling strength / canonical 0.4)", fontsize=10,
                  color=INK)
    ax.set_ylabel("true-positive rate (mean of 3 prescribed edges)",
                  fontsize=10, color=INK)
    ax.set_title("(a) Detection rate vs SNR - three Layer-3 testers",
                 loc="left", fontsize=10.5, fontweight="bold", color=INK,
                 pad=6)
    honesty_ladder(ax)

    # SNR = 1 reference (canonical operating point)
    ref = ax.axvline(1.0, color=SECONDARY_INK, lw=0.8, ls=(0, (4, 3)),
                     zorder=1)
    ref.set_gid("layer-annotations")

    ends = []
    for method in METHODS:
        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]
        for style, ls, mfc in (("err", "-", color),
                               ("noerr", (0, (5, 3)), "none")):
            mean, lo, hi = _tpr_series(df, method, style)
            x = mean.index.values / CANONICAL_C
            ax.plot(x, mean.values, color=color, ls=ls, lw=1.6, marker=marker,
                    ms=4.5, mfc=mfc, mec=color, mew=0.9, zorder=3)
            ax.fill_between(x, lo.values, hi.values, color=color, alpha=0.10,
                            lw=0, zorder=2.5)
            if style == "err":
                ends.append((method, x[-1], float(mean.values[-1])))

    # method labels at the solid-curve ends (nudged apart on overlap,
    # lifted clear of the CCM floor)
    placed = []
    for method, xe, ye in sorted(ends, key=lambda t: t[2]):
        for xp, yp in placed:
            if abs(xp - (xe + 0.12)) < 1.6 and abs(yp - ye) < 0.085:
                ye = yp - 0.085 if ye >= yp else yp + 0.085
        ye = max(ye, 0.13)
        ax.text(xe + 0.12, ye, method, fontsize=9, color=INK, va="center",
                zorder=4)
        placed.append((xe + 0.12, ye))

    # footer: style key, controls at SNR 1 (computed from the sweep, so the
    # figure can never drift from the data), ladder readings per method
    ctl = {m: _csv_val(df, m, "err", 0.4, "control_max") for m in METHODS}
    ctl_line = (f"controls at SNR 1 (1x): Granger <= {ctl['Granger'] * 100:.2f}%, "
                f"CCM <= {ctl['CCM'] * 100:.1f}%, PCMCI <= {ctl['PCMCI'] * 100:.1f}%")
    fig.text(X_L, 0.118, "solid = 1x chronology errors, dashed = error-free "
             "ceiling;  " + ctl_line, fontsize=9, color=INK)
    for i, method in enumerate(METHODS):
        fig.text(X_L, 0.075 - 0.028 * i, f"{method}: "
                 f"{_ladder_reading(df, method)}", fontsize=9, color=INK)

    save(fig, os.path.join(OUT_DIR, "validation_tpr"))
    print_claim(df)


def _ladder_reading(df, method):
    """'k/6 @95%, k/6 @70%' idiom per style (requirements summary, Section 5)."""
    bits = []
    for style, tag in (("err", "1x"), ("noerr", "no errors")):
        mean, _, _ = _tpr_series(df, method, style)
        n95 = int((mean >= 0.95).sum())
        n70 = int((mean >= 0.70).sum())
        bits.append(f"{n95}/6 >= 95%, {n70}/6 >= 70% ({tag})")
    return " | ".join(bits)


def print_claim(df):
    print("\n== claim ==")
    for m in METHODS:
        mean, _, _ = _tpr_series(df, m, "err")
        print(f"  {m:6s} 1x: {mean.iloc[0]:.3f} (SNR 0.5) -> "
              f"{mean.iloc[-1]:.3f} (SNR 5); max {mean.max():.3f}; "
              f">=70%: {int((mean >= 0.70).sum())}/6, "
              f">=95%: {int((mean >= 0.95).sum())}/6")
    cc_lo = _csv_val(df, "PCMCI", "err", STRENGTHS[0], "Cariaco -> Cheng")
    cc_hi = _csv_val(df, "PCMCI", "err", STRENGTHS[-1], "Cariaco -> Cheng")
    pcmci_ctl = _csv_val(df, "PCMCI", "err", 0.4, "control_max")
    print(f"  PCMCI Cariaco -> Cheng: {cc_lo:.3f} (SNR 0.5) -> {cc_hi:.3f} "
          f"(SNR 5); control_max at SNR 1: {pcmci_ctl:.3f} (paper note: 0.98)")
    print("  claim: Granger rises monotonically with SNR; CCM stays near "
          "zero (the linear benchmark defeats cross-mapping - Table 2: "
          "every true edge and control below 8.5%); PCMCI plateaus at its "
          "SNR-1 level and declines at the strongest coupling "
          "(Cariaco -> Cheng falls); no 1x curve reaches even the 70% "
          "rung, so the 95% honesty bar is out of reach at realistic "
          "chronology errors - the error-free ceilings (dashed) show the "
          "obstacle is data quality.")


# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 2 realizations x 20 perturbations")
    ap.add_argument("--force", action="store_true",
                    help="recompute all cells (default skips checkpoints)")
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw the figure from the sweep CSV")
    args = ap.parse_args()

    if args.plot_only:
        plot(pd.read_csv(OUT_CSV))
        return

    if args.quick:
        n_real, n_pert, persist = 2, 20, False
    else:
        n_real, n_pert, persist = 20, 200, True

    cells = [(m, c) for m in METHODS for c in STRENGTHS]
    todo = [(m, c) for m, c in cells
            if args.force or not os.path.exists(_ckpt(m, c))]
    print(f"cells: {len(cells)} total, {len(todo)} to run "
          f"({n_real} realizations x {n_pert} perturbations)", flush=True)

    if todo:
        from joblib import Parallel, delayed
        outs = Parallel(n_jobs=6, verbose=5)(
            delayed(_cell)(m, c, n_real, n_pert, persist) for m, c in todo)
    else:
        outs = None

    if args.quick:
        df = build_rows(outs)
        print_anchors(df, strict=False,
                      quick_note=f" (quick: {n_real} realizations x {n_pert})")
        print("quick mode: no checkpoints / CSV / figure written")
        return

    df = assemble_csv()
    print_anchors(df)
    plot(df)


if __name__ == "__main__":
    main()
