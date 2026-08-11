# CARC — Chronology-Aware Robustness Causality

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)

An **object-oriented framework** for detecting causal lead–lag relationships
among time-series variables, with built-in handling of time-axis uncertainty
and common-forcing confounding.

## Architecture

The project is built around two class hierarchies with **Factory**, **Strategy**,
and **Template Method** design patterns:

```
PaleoRecord (ABC)                    CausalDetector (ABC)
├── IceCoreRecord                    ├── StandardGrangerDetector
├── MarineRecord                     └── CARCDetector
├── SpeleothemRecord
└── TerrestrialRecord

RecordFactory    DetectorFactory    CausalPipeline
```

## Quick Start

```bash
pip install -r requirements.txt
python main.py --demo
```

The `--demo` flag runs a self-contained architecture demonstration using synthetic
data — no external data files needed.

## CLI Usage

```bash
python main.py                       # CARC pipeline (needs data in proxy_records/)
python main.py --demo                # OOP architecture demo (no data needed)
python main.py --method granger      # Use standard Granger instead of CARC
```

## Project Structure

```
├── main.py                 # Entry point (demo + full pipeline)
├── proxy_records/          # 91 proxy record CSV/XLSX files
├── proxy_errors/           # Age-model error files per record
├── data/                   # Data quality scores, orbital-forcing data
├── scripts/                # 15 analysis pipeline scripts (01 → 15)
├── tests/                  # Unit tests
├── src/
│   ├── records.py          # PaleoRecord ABC + 4 archive-type subclasses
│   ├── detectors.py        # CausalDetector ABC + StandardGranger + CARC
│   ├── factory.py          # RecordFactory + DetectorFactory (Factory pattern)
│   ├── pipeline.py         # CausalPipeline (Strategy pattern)
│   ├── config.py           # Central configuration
│   ├── io.py               # Data loading utilities
│   ├── arc.py              # Adaptive Residual Causality deconfounding
│   ├── granger.py          # Granger causality + age-perturbation utilities
│   ├── stacking.py         # Regional ensemble stacking
│   ├── synthetic.py        # Synthetic data generation (Scenario A)
│   ├── paleo_generator.py  # Class-based paleoclimate data generator
│   ├── validation.py       # Method comparison framework
│   ├── visualization.py    # Publication-quality plotting utilities
│   └── geolocation.py      # Record coordinates and regional metadata
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## Design Patterns

| Pattern | Where | Purpose |
|---------|-------|---------|
| **Factory** | `RecordFactory`, `DetectorFactory` | Create correct subclass by name/keyword |
| **Strategy** | `CausalPipeline` ↔ `CausalDetector` | Swap detection methods without changing pipeline |
| **Template Method** | `CausalDetector.detect()` | Define algorithm skeleton; subclasses fill steps |

## Requirements

- Python ≥ 3.9
- numpy, scipy, pandas, statsmodels, scikit-learn
- See `requirements.txt` for full list

## License

MIT — see [LICENSE](LICENSE).
