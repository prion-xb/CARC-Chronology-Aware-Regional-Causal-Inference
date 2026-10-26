# CARC: Chronology-Aware Regional Causal Inference on Paleoclimate Event Scales

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)

CARC detects causal lead–lag relationships among paleoclimate proxy records
while accounting for two failure modes: **age-model uncertainty** (chronology
errors can create spurious leads/lags) and **common-forcing confounding**
(shared external drivers such as North Atlantic climate can make independent
records look causally linked).

Edges are accepted only if they pass the **honesty filter**: after ARC
deconfounding and a ≥95% pass rate across the age-perturbation ensemble.

## Reproducing the paper results

Every table in the paper is produced by a script in `scripts/` (the numbered
01–45 series; `scripts/figure_tools/` and `scripts/manuscript_tools/` hold
auxiliary figure/manuscript tooling outside the reproduction map). The full
product-to-script map, run instructions, and known caveats are in
[README_复现指南.md](README_复现指南.md). Summary:

| Paper result | Script | On-disk output (after a run) |
| :--- | :--- | :--- |
| Table 1 (Scenario A, 20×200) | `scripts/22_scenarioA_final.py` | `outputs/scenarioA_final_table1.csv` |
| Table 2 (CCM / PCMCI) | `scripts/39_table2_ccm_pcmci.py` | `outputs/table2_ccm_pcmci.csv` |
| Table 4 / Table S4 (six climate windows, 180 directed cells) | `scripts/06_conditional_causality.py` | `outputs/conditional_causality_allpairs.csv` |
| Table 5 (regional phase lags) | `scripts/08_regional_phase_lag.py` | — |
| Table S5 (80-record pairwise scan + ablations) | `scripts/19_full_pairwise_scan.py` | `outputs/pairwise_scan/` |
| Table S7 (theoretical FPR profile + α×τ gate) | `scripts/42_theory_fpr_table.py` | `outputs/tableS7_fpr_profile.csv`, `outputs/tableS7_binomial_gate.csv` |
| β = 0.25 anchor inversion (App. S4.1) | `scripts/43_pairwise_beta_inversion.py` | `outputs/beta_inversion_pairs.csv` |
| Theorem 2 numerical-example MC (Sec. 2.2) | `scripts/44_theorem2_montecarlo.py` | `outputs/theorem2_mc_verification.csv` |
| Fig. 2(a) MC FPR curve | `scripts/45_fig2a_fpr_sweep.py` | `outputs/fig2a_mc_sweep.csv` |
| §4.3 regional pairs | `scripts/09_regional_robustness.py` → `scripts/17_regional_pairwise_matrix.py` | `outputs/regional_robustness_windows.csv` |
| Table S2 / S4 record accounting | `scripts/01_preprocess_data.py` + `data/data_tiers_final.csv` | `data/` |

All drivers have a `--quick` smoke mode (see each script's docstring). Full
runs are deterministic (fixed seeds) but heavy; e.g. the Table 1 driver's
full run is 20 realizations × 200 perturbations over a 144-config screening
grid (checkpointed, resumable). Script outputs land in `outputs/`; the
paper-table xlsx versions live in `Tables/` (both are git-ignored while the
manuscript is under review — see `.gitignore`).

```bash
pip install -r requirements.txt
python main.py --demo              # architecture demo, no data needed

cd scripts
python 22_scenarioA_final.py --quick     # Table 1 smoke (2×40)
python 39_table2_ccm_pcmci.py --quick          # Table 2 smoke (1×20)
```

Table 2 additionally requires `tigramite>=5.0` (PCMCI, `ParCorr`). The CCM
column uses a numpy implementation validated against pyEDM 2.5; pyEDM itself
is optional (Windows note: `legacy=True`).

## Quick Start

```bash
pip install -r requirements.txt
python main.py --demo
```

```bash
python main.py                       # CARC pipeline (needs data in proxy_records/)
python main.py --method granger      # Standard Granger instead of CARC
```

## Project Structure

```
├── main.py                 # Entry point (demo + full pipeline)
├── proxy_records/          # 83 proxy record data files (80 CSV + 3 XLSX)
├── proxy_errors/           # Age-model error files per record
├── data/                   # Tier assignments, quality scores, orbital forcing
├── scripts/                # Numbered analysis drivers 01–45 + fignew/ figure rebuilds
│   ├── figure_tools/       # Figure-port / QA utilities (outside the reproduction map)
│   └── manuscript_tools/   # Docx assembly tools (require the manuscript, not shipped)
├── Tables/                 # Paper tables (xlsx; git-ignored until publication, shipped in the submission package)
├── outputs/                # Reference outputs of the paper runs (git-ignored)
├── tests/                  # Unit tests
├── src/
│   ├── records.py          # PaleoRecord ABC + archive-type subclasses
│   ├── detectors.py        # CausalDetector ABC + StandardGranger + CARC
│   ├── factory.py          # RecordFactory + DetectorFactory
│   ├── pipeline.py         # CausalPipeline
│   ├── config.py           # Central configuration (paths, sigmas, windows)
│   ├── io.py               # Data loading utilities
│   ├── arc.py              # Adaptive Residual Causality deconfounding
│   ├── granger.py          # Granger tests + age-perturbation utilities
│   ├── granger_fast.py     # Block-Gram fast variant (used where noted)
│   ├── stacking.py         # Regional ensemble stacking
│   ├── synthetic.py        # Synthetic data generation (Scenario A)
│   ├── paleo_generator.py  # Class-based paleoclimate data generator
│   ├── validation.py       # CCM / PCMCI comparison testers
│   ├── visualization.py    # Publication-quality plotting utilities
│   ├── geolocation.py      # Record coordinates and regional metadata
│   └── ...
├── README_复现指南.md      # Full table → script reproduction map (中文)
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## Requirements

- Python ≥ 3.9
- numpy, scipy, pandas, statsmodels, scikit-learn
- tigramite ≥ 5.0 (Table 2 / PCMCI only)
- See `requirements.txt` for the full list

## Tests

```bash
python -m pytest tests/ -v
```

## Citation

If you use this code in your research, please cite the accompanying paper
(see [CITATION.cff](CITATION.cff)):

> Ruan, X., Wang, H., & Zhang, Z. CARC: Chronology-Aware Regional Causal
> Inference on Paleoclimate Event Scales. *Earth Science Frontiers*
> (submitted).

## License

MIT — see [LICENSE](LICENSE).
