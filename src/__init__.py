"""CARC: Chronology-Aware Robustness Causality — shared library."""

# =========================================================================
# OOP classes (NEW — Summer assignment architecture)
# =========================================================================
from .records import (
    PaleoRecord, IceCoreRecord, MarineRecord,
    SpeleothemRecord, TerrestrialRecord,
)
from .detectors import (
    CausalDetector, StandardGrangerDetector, CARCDetector,
)
from .factory import RecordFactory, DetectorFactory
from .pipeline import CausalPipeline

# =========================================================================
# Configuration
# =========================================================================
from .config import (
    DATA_DIR, ERROR_DIR, OUTPUT_DIR, FIGURE_DIR,
    AGE_GRID_10_60, AGE_GRID_800, DT_HIGH_RES, DT_LOW_RES,
    N_PERTURB, ARC_ALPHA, MAX_GRANGER_LAG, ALPHA_LEVEL,
    STABILITY_THRESHOLD, MIN_POINTS_IN_EVENT, NOISE_STD,
    CLIMATE_WINDOWS, AGE_MODEL_SIGMA, RECORD_AGE_MODEL,
    SPECIAL_SIGMA, DEFAULT_SIGMA, TIER1_FILES, TIER1_SIGMA,
    get_sigma,
)

# =========================================================================
# Data I/O
# =========================================================================
from .io import (
    identify_columns, safe_load_csv, load_record,
    interpolate_to_grid, standardize, load_all_records,
)

# =========================================================================
# ARC deconfounding
# =========================================================================
from .arc import (
    arc_deconfound, arc_deconfound_batch, extract_residuals_adaptive,
)

# =========================================================================
# Granger causality with age perturbation
# =========================================================================
from .granger import (
    select_var_lag, granger_test_pair, granger_min_pvalue,
    granger_detect, age_perturb, age_perturb_pair,
    robustness, is_honest,
)

# =========================================================================
# Regional stacking
# =========================================================================
from .stacking import (
    classify_record, build_region_dict, stack_region,
)

# =========================================================================
# Synthetic data
# =========================================================================
from .synthetic import (
    generate_scenario_A, generate_null_data, PaleoDataGenerator,
)

# =========================================================================
# Method validation
# =========================================================================
from .validation import (
    CausalMethodEvaluator,
    granger_causality, ccm_causality, pcmci_causality,
)
