# CARC Reproduction Guide

This repository is the reproduction code package for the CARC paper:
`scripts/` (numbered analysis drivers 01–45, plus the `figure_scripts/` figure
rebuild series), `src/` (pipeline and testers), `proxy_records/` +
`proxy_errors/` + `data/` (data required to run), and `Tables/` (paper-table
xlsx references — git-ignored in the public repository while the manuscript is
under review; shipped inside the submission package). Run-time outputs are
written to `outputs/` and `figures/` (created automatically, not tracked).

> The manuscript docx files and the original literature archive are not
> distributed with the code; `scripts/figure_tools/` (figure/QA utilities) and
> `scripts/manuscript_tools/` (manuscript assembly tools) are workflow aids
> outside the reproduction map below.

## Version note (read first)

Numbers in earlier paper drafts were produced directly by the canonical series
(`22`, `06`, …); the numbers in the final submission were produced by the
rebuild series `34–39`. Audit decision of 2026-10-24: **all submitted values
are reproducible** — e.g. the Table 4 value 54.0 comes directly from the
existing `06_conditional_causality.py`. The table below lists both regimes
side by side; the submitted numbers are based on the rebuild series.

## Paper product → script map

| Paper product | Earlier numbers (canonical series) | Submitted numbers (rebuild series 34–39) | On-disk output |
| :--- | :--- | :--- | :--- |
| Table 1 (Scenario A) | `scripts/22_scenarioA_final.py` (run22f: α=0.5/force/aic_f/test_joint/maxlag=50; β=0.25 row = 6.0/6.5/4.5) | `scripts/34_scenarioA_submitted.py` (targets 70/60/96) | `outputs/scenarioA_final_table1.csv` |
| Table 2 (CCM/PCMCI) | — | `scripts/39_table2_ccm_pcmci.py` | `outputs/table2_ccm_pcmci.csv` |
| Table 4 (six windows) | `scripts/06_conditional_causality.py` (**lagonly** regime; 54.0 comes directly from this) | `scripts/35_table4_submitted.py` (`--all-pairs` → Table S4 180 cells) | `outputs/conditional_causality_allpairs.csv` / `outputs/table4_submitted*.csv` |
| Table 5 (phase lags) | `scripts/08_regional_phase_lag.py` | `scripts/37_phase_submitted.py` | — |
| §4.3 regional pairs | `scripts/09_regional_robustness.py` → `scripts/17_regional_pairwise_matrix.py` (**min-p** regime) | `scripts/36_regional_submitted.py` | `outputs/regional_robustness_windows.csv` |
| Table S2/S4 (record accounting) | `scripts/01_preprocess_data.py` + `data/data_tiers_final.csv` | — | `data/` |
| Table S5 (full pairwise scan + ablations) | `scripts/19_full_pairwise_scan.py` (sharded, resumable) | — | `outputs/pairwise_scan/` |
| Table S6 / Fig. 5 (scrutiny) | `scripts/41_scrutinization_figure.py` | — | `outputs/table_s6_scrutiny_profile.csv` |
| Table S7 (theoretical FPR profile + α×τ gate) | `scripts/42_theory_fpr_table.py` | — | `outputs/tableS7_fpr_profile.csv` / `outputs/tableS7_binomial_gate.csv` |
| β = 0.25 anchor inversion (App. S4.1) | `scripts/43_pairwise_beta_inversion.py` | — | `outputs/beta_inversion_pairs.csv` |
| Theorem 2 numerical-example MC (§2.2) | `scripts/44_theorem2_montecarlo.py` | — | `outputs/theorem2_mc_verification.csv` |
| Fig. 2(a) MC FPR curve | `scripts/45_fig2a_fpr_sweep.py` | — | `outputs/fig2a_mc_sweep.csv` |
| §5.3 error compression | `scripts/24_age_error_scaling.py` | — | `outputs/age_error_scaling*.csv` |
| Sliding-window Fig. 7 | `scripts/15_sliding_window_carc.py` | — | — |
| Scenario B | (removed from the paper; script kept for reference) | `scripts/38_scenarioB.py` | `outputs/scenarioB_tableS6.csv` |
| Publication figures | `scripts/13_publication_figures.py` | — | `figures/` |
| Figure rebuild series | `scripts/figure_scripts/01–19` | — | `figures_rebuilt/` (git-ignored; scripts kept) |

## Quick start

```bash
# Environment: requirements.txt / environment.yml (repository root)
pip install -r requirements.txt

python scripts/22_scenarioA_final.py --quick   # smoke run; drop --quick for the full 20×200 (heavy)
```

Every script docstring documents its `--quick`/`--probe` smoke switches and
its output CSV paths.

## Pitfalls before a full rerun (established findings)

1. **Two distinct test regimes**: Table 4 = lagonly (single F test at the
   AIC-selected lag); §4.3 regional = min-p (scan over lags 1..L). Mixing them
   changes pass rates by ~4×.
2. **pyEDM 2.5 (Windows)**: requires `legacy=True` and column-name
   sanitisation; a hanging `EmbedDimension` → fix the CCM embedding
   dimension to E = 3.
3. **tigramite 5.x**: import `ParCorr` from `tigramite.pcmci.parcorr`
   (the 4.x import fails).
4. **21 vs 34/22 numbers are not directly comparable**: the three drivers
   inject age perturbations in different orders (at generation time /
   test_joint / --perdraw).
5. **Before regenerating with `13_publication_figures.py`**: the Fig S10
   in-figure numbers are known to be stale (9/58/30 vs canonical 7/48/25) and
   must be fixed before submission — do not casually overwrite finalised
   figures.
6. `00_stacked_regions_800ka/*.csv` — old orphan stacks (original units),
   do not use.
7. All data paths resolve from `src/config.py` (repository-root relative);
   overrides are available via `CARC_DATA_DIR` / `CARC_ERROR_DIR` /
   `CARC_OUTPUT_DIR` / `CARC_FIGURE_DIR` / `CARC_CONFIG_OVERRIDE`.
