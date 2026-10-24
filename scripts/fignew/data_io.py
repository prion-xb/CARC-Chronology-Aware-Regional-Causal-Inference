# -*- coding: utf-8 -*-
"""Minimal, self-contained loaders for the raw proxy CSVs.

Reality of the raw data (verified 2026-09-07):
  * 79 of 83 proxy_records files are 2-column CSVs, but most are HEADERLESS -
    the first row is already data (pandas would call it the header).
  * 1 CSV has a stray third column; 3 files are .xlsx.
  * proxy_errors files mirror the record files (same names + '_err').

So: sniff the first row. If every cell parses as a number, there is no
header and we read with header=None. Everything else follows pandas defaults.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RECORDS_DIR = os.path.join(ROOT, "proxy_records")
ERRORS_DIR = os.path.join(ROOT, "proxy_errors")


def _is_header_row(first_row) -> bool:
    """True if the first CSV row is NOT all-numeric (i.e. it is a header)."""
    def num(x):
        try:
            float(str(x).strip())
            return True
        except ValueError:
            return False
    return not all(num(c) for c in first_row)


def read_csv_any(path: str) -> pd.DataFrame:
    """Read a 2-column (age, value) file.

    Three formats exist in the raw data:
      * headerless CSV / xlsx  -> first two columns, first row is data
      * CSV with a header row  -> first two columns
      * PANGAEA xlsx export    -> description block, a '*/' row, then a header
        row; age = the column whose name contains 'age', value = last column
    """
    pangaea = False
    if path.endswith(".xlsx"):
        raw = pd.read_excel(path, header=None)
        term = next((i for i in range(min(len(raw), 40))
                     if str(raw.iloc[i, 0]).strip() == "*/"), None)
        if term is not None:
            pangaea = True
            df = pd.read_excel(path, header=term + 1)
        else:
            has_header = not _is_header_row(raw.iloc[0])
            df = pd.read_excel(path, header=0 if has_header else None)
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            first = f.readline().strip()
        has_header = first and _is_header_row(first.split(","))
        df = pd.read_csv(path, header=0 if has_header else None,
                         on_bad_lines="skip")
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df = df.apply(pd.to_numeric, errors="coerce")
    if pangaea:
        age_col = next((c for c in df.columns
                        if "age" in str(c).lower()), df.columns[0])
        val_col = df.columns[-1]
        df = df[[age_col, val_col]]
    else:
        df = df.iloc[:, :2]
    df.columns = ["age_ka", "value"]
    return df.dropna()


def _find_file(directory: str, base: str) -> str | None:
    """Exact match first; fall back to tolerant prefix match (some files on
    disk carry trailing spaces or other small deviations from the table ID)."""
    for ext in (".csv", ".xlsx"):
        p = os.path.join(directory, base + ext)
        if os.path.exists(p):
            return p
    for p in glob.glob(os.path.join(directory, "*")):
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem.strip() == base.strip():
            return p
    return None


def read_record(file_id: str) -> pd.DataFrame:
    """Load one record by its file ID (with or without .csv/.xlsx suffix)."""
    base = file_id.split("_err")[0]
    p = _find_file(RECORDS_DIR, base)
    if p is None:
        raise FileNotFoundError(f"no record file for {file_id!r}")
    return read_csv_any(p)


def read_error(file_id: str) -> pd.DataFrame | None:
    """Load the paired age-error file; returns None when absent."""
    base = file_id.split("_err")[0]
    p = _find_file(ERRORS_DIR, base + "_err")
    if p is None:
        return None
    return read_csv_any(p)


def error_on_ages(err: pd.DataFrame, ages: np.ndarray) -> np.ndarray:
    """Sample the error grid onto record ages (nearest; constant between grid
    points - age errors are piecewise-constant by construction)."""
    e = err.sort_values("age_ka")
    return np.interp(ages, e["age_ka"].values, e["value"].values)


def list_record_ids() -> list[str]:
    """All record file IDs (stem of proxy_records/*)."""
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(RECORDS_DIR, "*")))
