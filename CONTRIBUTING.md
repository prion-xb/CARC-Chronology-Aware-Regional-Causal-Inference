# Contributing to CARC

Thank you for your interest in contributing! CARC is a paleoclimate causal
inference method under active research. We welcome:

- Bug reports and feature requests via GitHub Issues
- Improvements to the synthetic validation framework
- New causal detection method wrappers (CCM, PCMCI, TE, …)
- Better figures and documentation

## Development Setup

```bash
git clone https://github.com/prion-xb/CARC-Chronology-Aware-Regional-Consensus-Framework-.git
cd CARC-Chronology-Aware-Regional-Consensus-Framework-
pip install -r requirements.txt
cp .env.example .env   # edit paths as needed
```

## Project Conventions

- **Library code** goes in `src/` with clear docstrings (NumPy style).
- **Analysis pipelines** go in `scripts/` with a numbered prefix indicating
  workflow order (01 → 12).
- **Configuration** lives in `src/config.py` — read paths from environment
  variables, not hardcoded absolute paths.
- **Data** in `proxy_records/` is tracked; generated outputs in `outputs/`
  and `figures/` are gitignored.

## Adding a New Method

1. Add a wrapper function in `src/validation.py` following the signature:
   ```python
   def mymethod_causality(data, dt, **kwargs) -> list[tuple]:
       ...
       return [(driver, target), ...]
   ```
2. Import your method in `src/__init__.py`.
3. Add a comparison in `scripts/10_method_validation.py`.

## Testing

```bash
python -m pytest tests/ -v
```

Run the synthetic validation (self-contained, no external data):

```bash
python scripts/03_synthetic_validation.py
```

## Commit Style

- Use present tense ("Add CCM wrapper", not "Added CCM wrapper")
- Reference issue numbers when relevant

## License

By contributing, you agree that your code will be licensed under the MIT
License (see [LICENSE](LICENSE)).
