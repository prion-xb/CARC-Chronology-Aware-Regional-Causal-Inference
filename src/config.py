"""
Central configuration for CARC analysis.

Override defaults by setting environment variables:
    CARC_DATA_DIR, CARC_ERROR_DIR, CARC_OUTPUT_DIR, CARC_FIGURE_DIR

Record-specific age-model mappings (SPECIAL_SIGMA, RECORD_AGE_MODEL,
TIER1_FILES, TIER1_SIGMA) are populated with the data used in the paper.
Override by setting a ``CARC_CONFIG_OVERRIDE`` env var pointing to a Python
file whose assignments are exec'd after the defaults.
"""

import os
import numpy as np

# ---------------------------------------------------------------------------
# Paths — set via environment variable or default to project-relative
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.environ.get(
    "CARC_DATA_DIR",
    os.path.join(PROJECT_ROOT, "proxy_records")
)
ERROR_DIR = os.environ.get(
    "CARC_ERROR_DIR",
    os.path.join(PROJECT_ROOT, "proxy_errors")
)
OUTPUT_DIR = os.environ.get(
    "CARC_OUTPUT_DIR",
    os.path.join(PROJECT_ROOT, "outputs")
)
FIGURE_DIR = os.environ.get(
    "CARC_FIGURE_DIR",
    os.path.join(PROJECT_ROOT, "figures")
)

# ---------------------------------------------------------------------------
# Analysis time windows
# ---------------------------------------------------------------------------
AGE_MIN_10_60 = 10.0   # ka
AGE_MAX_10_60 = 60.0   # ka
AGE_MIN_800 = 0.0      # ka
AGE_MAX_800 = 800.0    # ka

DT_HIGH_RES = 0.05     # ka — 50-year interpolation
DT_LOW_RES = 1.0       # ka — 1-kyr interpolation (800 ka analysis)

AGE_GRID_10_60 = np.arange(AGE_MIN_10_60, AGE_MAX_10_60 + DT_HIGH_RES, DT_HIGH_RES)
AGE_GRID_800 = np.arange(AGE_MIN_800, AGE_MAX_800 + DT_LOW_RES, DT_LOW_RES)

# ---------------------------------------------------------------------------
# CARC method parameters
# ---------------------------------------------------------------------------
N_PERTURB = 200               # Number of age perturbations (default)
ARC_ALPHA = 0.5                # Conservative ARC stripping strength
MAX_GRANGER_LAG = 5            # Maximum VAR lag order
ALPHA_LEVEL = 0.05             # Granger significance threshold
STABILITY_THRESHOLD = 0.95     # Honesty filter threshold
MIN_POINTS_IN_EVENT = 5        # Minimum data points in event window
NOISE_STD = 0.05               # Additive noise std in robustness tests

# ---------------------------------------------------------------------------
# Climate state windows (ka) — published, well-known periods
# ---------------------------------------------------------------------------
CLIMATE_WINDOWS = {
    'H1 event':               (14.0, 18.0),
    'H4 event':               (36.0, 40.0),
    'Bølling-Allerød':        (12.7, 14.7),
    'Last Glacial Maximum':   (19.0, 26.0),
    'GI-12 (interstadial)':   (44.0, 47.0),
    'GS-12 (stadial)':        (47.0, 49.0),
}

# ---------------------------------------------------------------------------
# Age model sigma by type (ka, 1σ) — from published literature
# ---------------------------------------------------------------------------
AGE_MODEL_SIGMA = {
    'AICC2012': 1.5, 'GICC05': 0.8, 'LR04': 2.0,
    'EDC3': 1.5, 'Speleothem': 0.5, 'Orbitally-tuned': 2.5,
    '14C': 1.0, 'Tephra': 1.0, 'Mixed': 1.5
}

DEFAULT_SIGMA = {
    'ice_core': 1.0, 'marine': 2.0, 'speleothem': 0.5,
    'loess': 2.5, 'lake': 1.5, 'other': 2.0
}

# ---------------------------------------------------------------------------
# Per-record age-model overrides
# ---------------------------------------------------------------------------
# Override map: record_name → age-model name (e.g. 'NGRIP_d18O': 'GICC05')
RECORD_AGE_MODEL = {
    # Ice cores — AICC2012 / GICC05
    'EDC_dD':                       'AICC2012',
    'EDC_CO2':                      'AICC2012',
    'EDC_CH4':                      'AICC2012',
    'EDC_Dust':                     'AICC2012',
    'DomeFuji_d18O':                'AICC2012',
    'DomeFuji_dD':                  'AICC2012',
    'NGRIP_d18O':                   'GICC05',
    # Marine — LR04 / orbitally-tuned
    'ODP983_IRD':                   'LR04',
    'ODP980_planktic_d18O':         'LR04',
    'ODP980_benthic_d18O':          'LR04',
    'U1308_RPI':                    'LR04',
    'U1385_CaTi':                   'LR04',
    'U1385_ZrSr':                   'LR04',
    'Cariaco_Mo':                   '14C',
    # Speleothems — U-Th
    'Cheng2016_d18O':               'Speleothem',
}

# Override map: record_name → sigma_ka (1σ age uncertainty)
SPECIAL_SIGMA = {
    # Tier 1 records (from published age-model literature, matched to Table S1)
    'NGRIP_d18O':                   0.65,
    'EDC_dD':                       0.83,
    'EDC_CO2':                      1.65,
    'EDC_CH4':                      1.66,
    'Cariaco_Mo':                   0.23,
    'Cheng2016_d18O':               0.14,
    'EDC_Dust':                     1.50,   # AICC2012, per err file
    # Other high-quality records
    'DomeFuji_d18O':                1.50,
    'DomeFuji_dD':                  1.50,
    'ODP983_IRD':                   2.00,
    'Lapa_d18O':                    0.39,   # paper line 161 "about 0.39 ka";
                                            # pair RMS 0.45 = mean(0.39, 0.50)
}

# Tier 1 core records (Table 1–4 in paper): label → filename in proxy_records/
# Labels are the canonical variable names used in scripts 05/06/15 and the paper.
TIER1_FILES = {
    'NGRIP':      'NGRIP_d18O.csv',
    'EDC_dD':     'EDC_dD.csv',
    'EDC_CO2':    'EDC_CO2.xlsx',
    'EDC_CH4':    'EDC_CH4.csv',
    'Cariaco':    'Cariaco_Mo.csv',
    'Cheng2016':  'Cheng2016_d18O.xlsx',
}

# Tier 1 sigma overrides: label → sigma_ka
TIER1_SIGMA = {
    'NGRIP':     0.65,
    'EDC_dD':    0.83,
    'EDC_CO2':   1.65,
    'EDC_CH4':   1.66,
    'Cariaco':   0.23,
    'Cheng2016': 0.14,
}

# Short aliases used by the synthetic-data generator in
# scripts/03_synthetic_validation.py ('CO2', 'CH4', 'Cheng').
TIER1_SIGMA_ALIASES = {
    'CO2':   'EDC_CO2',
    'CH4':   'EDC_CH4',
    'Cheng': 'Cheng2016',
}


def tier1_sigma(name):
    """Return the published 1σ age uncertainty for a Tier 1 variable.

    Resolves short aliases used by the synthetic generator
    (``TIER1_SIGMA_ALIASES``); falls back to 1.0 ka for unknown names.
    """
    return TIER1_SIGMA.get(TIER1_SIGMA_ALIASES.get(name, name), 1.0)

# ---------------------------------------------------------------------------
# Helper: get sigma for any record
# ---------------------------------------------------------------------------
def get_sigma(record_name, archive_type=None):
    """Return the age-model 1σ uncertainty (ka) for a given record.

    Resolution order:
    1. SPECIAL_SIGMA[record_name] — per-record published values
    2. RECORD_AGE_MODEL[record_name] → AGE_MODEL_SIGMA — by named age model
    3. DEFAULT_SIGMA[archive_type] — by archive type (ice_core, marine, etc.)
    4. 1.5 ka — universal fallback

    Parameters
    ----------
    record_name : str
        Record identifier (filename stem without extension).
    archive_type : str or None
        One of 'ice_core', 'marine', 'speleothem', 'loess', 'lake', 'other'.
        Used as final fallback when no per-record override exists.
    """
    if record_name in SPECIAL_SIGMA:
        return SPECIAL_SIGMA[record_name]
    age_model = RECORD_AGE_MODEL.get(record_name, None)
    if age_model is not None:
        return AGE_MODEL_SIGMA.get(age_model, 1.5)
    if archive_type is not None:
        return DEFAULT_SIGMA.get(archive_type, 1.5)
    return 1.5


# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
