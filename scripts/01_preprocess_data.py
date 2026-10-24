#!/usr/bin/env python3
"""
01_preprocess_data.py
---------------------
Load raw paleoclimate CSV records, compute quality metrics (SNR, trend,
resolution, coverage), assign composite scores, and classify records into
Tier 1/2/3.

Output: data_tiers_final.csv
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from scipy import stats

from src.config import DATA_DIR, OUTPUT_DIR, SPECIAL_SIGMA
from src.io import identify_columns, load_record, read_record_df
from src.stacking import classify_record  # canonical classification

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_START, TARGET_END = 10.0, 60.0   # ka
H1_START, H1_END = 14.0, 18.0           # Heinrich 1 window
H4_START, H4_END = 36.0, 40.0           # Heinrich 4 window

W_SNR, W_TREND, W_RES, W_COVERAGE = 0.35, 0.30, 0.20, 0.15
MIN_POINTS_IN_EVENT = 5

# High-quality records = records with published per-record age uncertainties.
# Single source of truth: SPECIAL_SIGMA in src/config.py (9 records).
HIGH_QUALITY_RECORDS = set(SPECIAL_SIGMA.keys())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_event_metrics(age, value, window_start, window_end):
    mask = (age >= window_start) & (age <= window_end)
    n_points = np.sum(mask)
    if n_points < MIN_POINTS_IN_EVENT:
        return {'SNR': np.nan, 'Trend_Score': 0.0, 'Trend_pValue': np.nan,
                'Trend_Slope': np.nan, 'N_Points': n_points, 'Coverage': 0.0}

    w_age, w_val = age[mask], value[mask]
    global_std = np.std(value)
    snr = np.std(w_val) / global_std if global_std > 0 else np.nan

    slope, intercept, r_value, p_value, std_err = stats.linregress(w_age, w_val)
    trend_score = abs(slope) / std_err if (p_value < 0.05 and std_err > 0) else 0.0

    if len(age) > 1:
        median_res = np.median(np.diff(age))
        expected = (window_end - window_start) / median_res
        coverage = min(1.0, n_points / expected) if expected > 0 else 0.0
    else:
        coverage = 0.0

    return {'SNR': snr, 'Trend_Score': trend_score, 'Trend_pValue': p_value,
            'Trend_Slope': slope, 'N_Points': n_points, 'Coverage': coverage}


def compute_high_freq_score(value):
    if len(value) < 10:
        return 0.0
    diffs = np.diff(value)
    ratio = np.std(diffs) / np.std(value) if np.std(value) > 0 else 0.0
    return min(10.0, ratio * 5)


def resolution_score(median_res_ka):
    if np.isnan(median_res_ka) or median_res_ka <= 0:
        return 0.0
    return max(0.0, 1 - np.log10(median_res_ka / 0.01) / np.log10(1.0 / 0.01)) * 10


def determine_record_type(filename):
    """Classify record type using the canonical stacking classifier.

    Wraps :func:`src.stacking.classify_record` and maps its output to
    human-readable type labels for the data-tier report.
    """
    typ, _ = classify_record(filename)
    _TYPE_LABELS = {
        'ice_core': 'Ice Core', 'marine': 'Marine Sediment',
        'speleothem': 'Speleothem', 'loess': 'Loess',
        'lake': 'Lake/Other', 'other': 'Other',
    }
    return _TYPE_LABELS.get(typ, 'Other')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    files = sorted([f for f in os.listdir(DATA_DIR)
                    if f.endswith('.csv') or f.endswith('.xlsx')])
    records = []

    for filename in files:
        filepath = os.path.join(DATA_DIR, filename)
        try:
            df = read_record_df(filepath)
        except Exception as e:
            print(f"Skip {filename}: read error ({e})")
            continue

        age_col, value_col = identify_columns(df)
        if age_col is None or value_col is None:
            print(f"Skip {filename}: cannot identify columns")
            continue

        df = df.dropna(subset=[age_col, value_col])
        age = df[age_col].values.astype(float)
        value = df[value_col].values.astype(float)
        sort_idx = np.argsort(age)
        age, value = age[sort_idx], value[sort_idx]

        if len(age) < 10:
            print(f"Skip {filename}: <10 points")
            continue

        record_name = os.path.splitext(filename)[0]
        rec_type = determine_record_type(filename)

        # Trim to the target interval (10–60 ka): quality metrics are scored
        # on the event-scale window, matching the paper's tiering semantics.
        in_target = (age >= TARGET_START) & (age <= TARGET_END)
        age, value = age[in_target], value[in_target]
        if len(age) < 10:
            print(f"Skip {filename}: <10 points in 10-60 ka")
            continue

        age_min, age_max = np.min(age), np.max(age)
        n_points = len(age)

        age_diffs = np.diff(age)
        res_median = np.median(age_diffs) if len(age_diffs) > 0 else np.nan
        coverage_global = 1.0  # all points lie within the target interval

        h1 = compute_event_metrics(age, value, H1_START, H1_END)
        h4 = compute_event_metrics(age, value, H4_START, H4_END)

        snr_best = np.nanmax([h1['SNR'], h4['SNR']])
        trend_best = max(h1['Trend_Score'], h4['Trend_Score'])
        event_coverage = max(h1['Coverage'], h4['Coverage'])
        has_event_data = (h1['N_Points'] >= MIN_POINTS_IN_EVENT or
                          h4['N_Points'] >= MIN_POINTS_IN_EVENT)

        hf_score = compute_high_freq_score(value)
        res_score = resolution_score(res_median)

        snr_norm = min(10.0, snr_best * 3) if not np.isnan(snr_best) else 0.0
        trend_norm = min(10.0, np.log1p(trend_best) * 2) if trend_best else 0.0
        cov_norm = min(10.0, coverage_global * 10)

        composite = (W_SNR * snr_norm + W_TREND * trend_norm +
                     W_RES * res_score + W_COVERAGE * cov_norm)
        composite += min(1.0, hf_score * 0.1)

        age_model_q = 'High (official errors available)' if record_name in HIGH_QUALITY_RECORDS else 'Unknown'

        if composite >= 5.0 and has_event_data and 'High' in age_model_q:
            tier = 'Tier 1'
        elif composite >= 3.0 and has_event_data:
            tier = 'Tier 2'
        else:
            tier = 'Tier 3'

        records.append({
            'Record': record_name, 'Type': rec_type,
            'Age_Min_ka': age_min, 'Age_Max_ka': age_max,
            'N_Points': n_points, 'Resolution_Median_ka': res_median,
            'Coverage_10_60ka': coverage_global,
            'Has_Event_Data': has_event_data,
            'SNR_H1': h1['SNR'], 'SNR_H4': h4['SNR'], 'SNR_Best': snr_best,
            'Trend_H1': h1['Trend_Score'], 'Trend_H4': h4['Trend_Score'],
            'Trend_Best': trend_best, 'Event_Coverage': event_coverage,
            'HighFreq_Score': hf_score, 'Resolution_Score': res_score,
            'Composite_Score': composite, 'Age_Model_Quality': age_model_q,
            'Tier': tier
        })

    df_result = pd.DataFrame(records)
    if df_result.empty:
        print("No valid records. Check DATA_DIR.")
        return

    df_result = df_result.sort_values('Composite_Score', ascending=False).reset_index(drop=True)
    out_path = os.path.join(OUTPUT_DIR, 'data_tiers_final.csv')
    df_result.to_csv(out_path, index=False, float_format='%.4f')
    print(f"Saved to {out_path}")

    print("\n=== Tier Summary ===")
    print(df_result['Tier'].value_counts())
    print("\n=== Tier 1 ===")
    print(df_result[df_result['Tier'] == 'Tier 1'][['Record', 'Type', 'Composite_Score']].to_string())


if __name__ == "__main__":
    main()
