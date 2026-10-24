#!/usr/bin/env python3
"""
02_generate_age_errors.py
-------------------------
Batch-generate age-error CSV files for all paleoclimate proxy records.

Uses ``src.config.get_sigma()`` as the **single source of truth** for all
age-model 1σ uncertainties, ensuring consistency between the generated error
files and the analysis pipeline (stacking, Granger causality, CARC).

Resolution order (implemented in ``src.config.get_sigma``):
1. ``SPECIAL_SIGMA`` — per-record published values (e.g. NGRIP=0.65 ka)
2. ``RECORD_AGE_MODEL`` → ``AGE_MODEL_SIGMA`` — by named age model
3. ``DEFAULT_SIGMA[archive_type]`` — by archive type
4. 1.5 ka — universal fallback

Output: proxy_errors/*_err.csv (two-column: age [ka], 1σ [ka])
"""

import os, sys, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np

from src.config import DATA_DIR, ERROR_DIR, get_sigma
from src.stacking import classify_record


def generate_error_file(input_path, output_dir=None):
    """Generate a per-point age-error CSV for a single record.

    Reads the record's age column, looks up the correct sigma via
    ``config.get_sigma()``, and writes a two-column CSV (age, sigma).

    Parameters
    ----------
    input_path : str
        Path to the raw proxy record CSV/XLSX file.
    output_dir : str or None
        Output directory (default: config.ERROR_DIR → proxy_errors/).
    """
    # --- load age column ---
    try:
        if input_path.endswith('.xlsx'):
            df_raw = pd.read_excel(input_path, engine='openpyxl')
        else:
            df_raw = pd.read_csv(input_path, header=None, dtype=str)
    except Exception as e:
        print(f"  SKIP {input_path}: read error ({e})")
        return

    if df_raw.shape[1] == 0:
        print(f"  SKIP {input_path}: empty file")
        return

    # First column is age
    age_series = pd.to_numeric(df_raw.iloc[:, 0], errors='coerce')
    valid_mask = ~age_series.isna()
    if not valid_mask.any():
        print(f"  SKIP {input_path}: no valid ages")
        return

    ages = age_series[valid_mask].values.astype(float)

    # --- determine sigma via the single source of truth ---
    fname = os.path.basename(input_path)
    stem = os.path.splitext(fname)[0]
    typ, region = classify_record(fname)  # typ = 'ice_core' | 'marine' | ...
    sigma = get_sigma(stem, archive_type=typ)

    # --- build constant-uncertainty column ---
    errors = np.full_like(ages, sigma)

    # --- write ---
    if output_dir is None:
        output_dir = ERROR_DIR
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, f"{stem}_err.csv")
    out_df = pd.DataFrame({0: ages, 1: errors})
    out_df.to_csv(out_path, index=False, header=False)
    print(f"  {stem}: σ={sigma:.2f} ka ({typ}, {region or 'unclassified'}) → {out_path}")


def batch_generate(input_dir=None, pattern="*.csv"):
    """Generate error files for all records in *input_dir*."""
    if input_dir is None:
        input_dir = DATA_DIR

    csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    xlsx_files = sorted(glob.glob(os.path.join(input_dir, "*.xlsx")))
    all_files = csv_files + xlsx_files

    # Exclude files that are already error files
    all_files = [f for f in all_files
                 if not f.endswith('_err.csv') and 'age_err' not in f.lower()]

    print(f"Found {len(all_files)} record files in {input_dir}")
    print(f"Output directory: {ERROR_DIR}")
    print()

    for f in all_files:
        generate_error_file(f)

    print(f"\nDone. {len(all_files)} error files generated.")


if __name__ == "__main__":
    batch_generate()
