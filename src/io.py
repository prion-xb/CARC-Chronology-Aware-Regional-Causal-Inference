"""
Data I/O utilities: loading, column detection, interpolation, standardization.
"""

import os
import pandas as pd
import numpy as np
from scipy import interpolate

from .config import DATA_DIR


def identify_columns(df):
    """
    Auto-detect age and value columns in a DataFrame.

    Returns (age_col, value_col) — column names or None.
    """
    age_col, value_col = None, None

    # Age column: look for keywords
    for col in df.columns:
        col_str = str(col).lower().strip()
        if any(kw in col_str for kw in ['age', 'ka', 'kyr', 'bp', 'year']):
            if age_col is None:
                age_col = col
        elif value_col is None and pd.api.types.is_numeric_dtype(df[col]) and 'unnamed' not in col_str:
            value_col = col

    # Fallback: first numeric column = age, second = value
    if age_col is None:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                age_col = col
                break
    if value_col is None:
        for col in df.columns:
            if col != age_col and pd.api.types.is_numeric_dtype(df[col]):
                value_col = col
                break

    return age_col, value_col


def safe_load_csv(filepath):
    """Load a CSV file with or without headers. Returns DataFrame with 'age' and 'value' columns."""
    try:
        df = pd.read_csv(filepath, header=None, skipinitialspace=True)
        if df.shape[1] < 2:
            return pd.DataFrame(columns=['age', 'value'])
        df = df.iloc[:, :2]
        df.columns = ['age', 'value']
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        return df.dropna()
    except Exception:
        return pd.DataFrame(columns=['age', 'value'])


def load_record(stem, data_dir=None):
    """
    Load a paleoclimate record by its stem name or full filename.

    Matching strategy (first match wins):
    1. Exact filename match (e.g. 'Erhardt_NGRIP_aerosol_d18O_step1.csv')
    2. Prefix/stem match (e.g. stem='NGRIP_d18O' matches files starting with that)

    Parameters
    ----------
    stem : str
        Record name stem or full filename.
    data_dir : str, optional
        Directory to search. Defaults to config.DATA_DIR.

    Returns
    -------
    age : np.ndarray or None
    value : np.ndarray or None
    """
    if data_dir is None:
        data_dir = DATA_DIR

    if not os.path.isdir(data_dir):
        return None, None

    files = os.listdir(data_dir)

    # Stage 1: exact filename match
    if stem in files:
        filepath = os.path.join(data_dir, stem)
        try:
            if stem.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath, engine='openpyxl')
        except Exception:
            return None, None

        age_col, val_col = identify_columns(df)
        if age_col is None or val_col is None:
            return None, None

        age = df[age_col].values.astype(float)
        val = df[val_col].values.astype(float)
        mask = np.isfinite(age) & np.isfinite(val)
        age, val = age[mask], val[mask]

        if len(age) >= 10:
            order = np.argsort(age)
            return age[order], val[order]

    # Stage 2: prefix/stem match
    for fname in files:
        if fname.startswith(stem) and (fname.endswith('.csv') or fname.endswith('.xlsx')):
            filepath = os.path.join(data_dir, fname)
            try:
                if fname.endswith('.csv'):
                    df = pd.read_csv(filepath)
                else:
                    df = pd.read_excel(filepath, engine='openpyxl')
            except Exception:
                return None, None

            age_col, val_col = identify_columns(df)
            if age_col is None or val_col is None:
                return None, None

            age = df[age_col].values.astype(float)
            val = df[val_col].values.astype(float)
            mask = np.isfinite(age) & np.isfinite(val)
            age, val = age[mask], val[mask]

            if len(age) < 10:
                return None, None

            order = np.argsort(age)
            return age[order], val[order]

    return None, None


def interpolate_to_grid(age, value, grid, kind='linear'):
    """
    Interpolate (age, value) to a regular grid. Fills NaN by forward-fill then
    backward-fill. Returns the interpolated array.
    """
    f = interpolate.interp1d(age, value, kind=kind, bounds_error=False, fill_value=np.nan)
    result = f(grid)

    # Forward-fill then backward-fill NaN
    valid_idx = np.where(~np.isnan(result))[0]
    if len(valid_idx) == 0:
        return result

    last_valid = result[valid_idx[0]]
    for i in range(len(result)):
        if np.isnan(result[i]):
            result[i] = last_valid
        else:
            last_valid = result[i]

    return result


def standardize(series):
    """Z-score standardize: (x - mean) / std."""
    mean = np.nanmean(series)
    std = np.nanstd(series)
    if std < 1e-10:
        return series - mean
    return (series - mean) / std


def load_all_records(data_dir=None, age_min=10.0, age_max=60.0, min_points=10):
    """
    Load all CSV/XLSX records from a directory, interpolate to a common grid,
    and return a list of (interp_array, sigma, region, record_name) tuples.

    Parameters
    ----------
    data_dir : str
        Directory containing record files.
    age_min, age_max : float
        Age range to extract.
    min_points : int
        Minimum data points required in range.

    Returns
    -------
    records : list of (np.ndarray, float, str, str)
    grid : np.ndarray
    """
    from .config import DT_HIGH_RES, get_sigma, RECORD_AGE_MODEL, AGE_MODEL_SIGMA
    from .stacking import classify_record

    if data_dir is None:
        data_dir = DATA_DIR

    grid = np.arange(age_min, age_max + DT_HIGH_RES, DT_HIGH_RES)
    records = []

    for fname in os.listdir(data_dir):
        if not (fname.endswith('.csv') or fname.endswith('.xlsx')):
            continue

        path = os.path.join(data_dir, fname)
        try:
            if fname.endswith('.csv'):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path, engine='openpyxl')
        except Exception:
            continue

        age_col, val_col = identify_columns(df)
        if age_col is None or val_col is None:
            continue

        age = df[age_col].astype(float).values
        value = df[val_col].astype(float).values
        mask = np.isfinite(age) & np.isfinite(value)
        age, value = age[mask], value[mask]

        if len(age) < min_points:
            continue
        order = np.argsort(age)
        age, value = age[order], value[order]

        in_range = (age >= age_min) & (age <= age_max)
        if np.sum(in_range) < min_points:
            continue
        age, value = age[in_range], value[in_range]

        typ, region = classify_record(fname)
        if region is None:
            continue

        rec_name = os.path.splitext(fname)[0]
        sigma = get_sigma(rec_name)

        f_interp = interpolate.interp1d(age, value, kind='linear',
                                        bounds_error=False, fill_value='extrapolate')
        interp = f_interp(grid)
        interp = np.nan_to_num(interp, nan=np.nanmean(interp))

        if np.std(interp) < 1e-6:
            continue

        interp = standardize(interp)
        records.append((interp, sigma, region, rec_name))

    return records, grid
