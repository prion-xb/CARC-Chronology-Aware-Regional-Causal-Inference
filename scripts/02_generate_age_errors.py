#!/usr/bin/env python3
"""
02_generate_age_errors.py
-------------------------
Batch-generate age-error CSV files for all paleoclimate proxy records.

Maps each record to an age-model type and assigns per-point 1σ uncertainties
based on the age model (GICC05, AICC2012, LR04, Speleothem U-Th, etc.).

Output: data/proxy_errors/*_err.csv
"""

import os, sys, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np

from src.config import DATA_DIR, ERROR_DIR


# ---------------------------------------------------------------------------
# Age-model uncertainty functions (1σ, ka)
# ---------------------------------------------------------------------------
def lr04_uncertainty(age_ka):
    if age_ka <= 1000:   return 4.0
    elif age_ka <= 3000: return 10.0
    else:                return 25.0

def aicc2012_uncertainty(age_ka):
    if age_ka <= 120:    return 0.5
    elif age_ka <= 800:  return 0.5 + (3.0 - 0.5) / (800 - 120) * (age_ka - 120)
    else:                return 3.0

def loess_uncertainty(age_ka):       return 2.0
def generic_orbital_uncertainty(age_ka): return 3.0
def generic_ice_uncertainty(age_ka):     return 1.0
def speleothem_uncertainty(age_ka):      return 0.3


def infer_record_type(filename):
    name = os.path.basename(filename).lower()
    if any(k in name for k in ['odp', 'u130', 'md', 'ps75', 'kxy', 'ipwp',
                               'cariaco', 'u1476', 'u1537', 'opd']):
        return 'lr04'
    if any(k in name for k in ['weinan', 'xifeng', 'gulang', 'luochuan',
                               'jy10', 'jy12', 'xf06', 'loess', 'mgs']):
        return 'loess'
    if any(k in name for k in ['edc', 'edml', 'taldice', 'vostok', 'ngrip',
                               'dome_fuji', 'dome fuji']):
        return 'aicc2012'
    if any(k in name for k in ['cheng2016', 'botuvera', 'santana', 'lapa',
                               'marota', 'paixao', 'stalagmite', 'cave']):
        return 'speleothem'
    if 'ice' in name or 'dome' in name:
        return 'icecore_other'
    return 'generic_orbital'


UNC_FUNCS = {
    'lr04': lr04_uncertainty, 'loess': loess_uncertainty,
    'aicc2012': aicc2012_uncertainty, 'speleothem': speleothem_uncertainty,
    'icecore_other': generic_ice_uncertainty,
    'generic_orbital': generic_orbital_uncertainty,
}


def generate_error_file(input_path, output_dir=None):
    try:
        df_raw = pd.read_csv(input_path, header=None, dtype=str)
    except Exception as e:
        print(f"Read failed {input_path}: {e}")
        return

    if df_raw.shape[1] == 0:
        return

    age_series = pd.to_numeric(df_raw.iloc[:, 0], errors='coerce')
    valid_mask = ~age_series.isna()
    if not valid_mask.any():
        return

    ages = age_series[valid_mask].values
    rec_type = infer_record_type(input_path)
    unc_func = UNC_FUNCS.get(rec_type, generic_orbital_uncertainty)
    errors = np.array([unc_func(float(a)) for a in ages])

    out_df = pd.DataFrame({0: ages, 1: errors})

    if output_dir is None:
        output_dir = ERROR_DIR
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.basename(input_path)
    name, _ = os.path.splitext(base)
    out_path = os.path.join(output_dir, f"{name}_err.csv")
    out_df.to_csv(out_path, index=False, header=False)
    print(f"Generated: {out_path}  ({rec_type}, n={len(ages)})")


def batch_generate(input_dir=None, pattern="*.csv"):
    if input_dir is None:
        input_dir = DATA_DIR
    files = glob.glob(os.path.join(input_dir, "**", pattern), recursive=True)
    files = [f for f in files if not f.endswith('_err.csv') and 'age_err' not in f]
    print(f"Found {len(files)} files to process")
    for f in files:
        generate_error_file(f)


if __name__ == "__main__":
    batch_generate()
