"""Tests for CARC core library."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_imports():
    """Verify all src modules can be imported."""
    from src.config import DATA_DIR, N_PERTURB, ARC_ALPHA
    from src.arc import arc_deconfound, arc_deconfound_batch
    from src.granger import granger_test_pair, robustness, age_perturb
    from src.stacking import classify_record, build_region_dict
    from src.io import identify_columns, load_record, standardize
    from src.synthetic import generate_null_data, generate_scenario_A
    from src.validation import CausalMethodEvaluator
    assert DATA_DIR is not None
    assert 0 < ARC_ALPHA <= 1


def test_arc_deconfound_identity():
    """ARC with alpha=0 should return the original signal."""
    import numpy as np
    from src.arc import arc_deconfound

    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    proxy = rng.normal(0, 1, 100)
    residual = arc_deconfound(x, proxy, alpha=0.0)
    assert np.allclose(residual, x)


def test_arc_deconfound_full():
    """ARC with alpha=1 should remove all linear dependence on proxy."""
    import numpy as np
    from src.arc import arc_deconfound

    x = np.arange(100, dtype=float)
    proxy = x * 2  # perfect linear relationship
    residual = arc_deconfound(x, proxy, alpha=1.0)
    # After full removal, residual should be ~constant
    assert np.std(residual) < 1e-6 or np.allclose(residual, np.mean(residual))


def test_granger_test_pair_trivial():
    """Granger test on white noise should return a boolean."""
    import numpy as np
    from src.granger import granger_test_pair

    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, 500)
    Y = rng.normal(0, 1, 500)
    # White noise: either True (rare false positive) or False (expected)
    result = granger_test_pair(Y, X, maxlag=3)
    assert isinstance(result, (bool, np.bool_))


def test_classify_record_ice_core():
    """Classify_record should identify ice cores correctly."""
    from src.stacking import classify_record

    typ, region = classify_record('EDC_dD.csv')
    assert typ == 'ice_core'
    assert region == 'Antarctic'

    typ, region = classify_record('NGRIP_d18O.csv')
    assert typ == 'ice_core'
    assert region == 'North_Atlantic'


def test_classify_record_marine():
    """Classify_record should identify marine sediments."""
    from src.stacking import classify_record

    typ, region = classify_record('ODP983_IRD.csv')
    assert typ == 'marine'
    assert region == 'North_Atlantic'


def test_generate_null_data():
    """Null data generator should produce correct-length output."""
    import numpy as np
    from src.synthetic import generate_null_data

    X, Y, Z = generate_null_data(n=500, phi=0.85, beta=0.4, seed=42)
    assert len(X) == 500
    assert len(Y) == 500
    assert len(Z) == 500
    # X and Y should be correlated through Z when beta > 0
    corr = np.corrcoef(X, Y)[0, 1]
    assert abs(corr) > 0  # non-zero correlation expected


def test_generate_scenario_A():
    """Scenario A generator should produce 6 variables on a 10-60 ka grid."""
    from src.synthetic import generate_scenario_A

    data, edges, t_grid = generate_scenario_A(seed=42)
    assert len(data) == 6
    assert len(edges) == 3
    assert t_grid[0] >= 9.9
    assert t_grid[-1] <= 60.1
    for name in ['NGRIP', 'EDC_dD', 'CO2', 'CH4', 'Cariaco', 'Cheng']:
        assert name in data
        assert len(data[name]) == len(t_grid)


def test_standardize():
    """Standardize should produce zero-mean, unit-variance series."""
    import numpy as np
    from src.io import standardize

    x = np.arange(100, dtype=float) * 2 + 10
    z = standardize(x)
    assert abs(np.mean(z)) < 1e-10
    assert abs(np.std(z) - 1.0) < 1e-10


def test_paleo_generator_scenario_A():
    """PaleoDataGenerator scenario A should produce valid output."""
    import numpy as np
    from src.validation import PaleoDataGenerator

    gen = PaleoDataGenerator(scenario='A', random_seed=42)
    t, signals, forcing = gen.generate_dataset()

    assert len(signals) == 6
    assert 9.9 <= t[0] <= 10.1
    assert 59.9 <= t[-1] <= 60.1
    for v in gen.var_names:
        assert v in signals
        # Note: standardization is applied before the 10-60 ka trim,
        # so the trimmed subset mean may deviate slightly from 0.
        assert abs(np.mean(signals[v])) < 0.5


def test_paleo_generator_ensemble():
    """Ensemble generation should produce the requested number of realizations."""
    from src.validation import PaleoDataGenerator

    gen = PaleoDataGenerator(scenario='A', random_seed=42)
    ensemble = gen.generate_ensemble(n_realizations=10)
    assert len(ensemble) == 10
    for r in ensemble:
        assert 'signals' in r
        assert 'time' in r
        assert 'causal_edges' in r


def test_causal_method_evaluator():
    """CausalMethodEvaluator should return valid TPR/FPR."""
    from src.validation import PaleoDataGenerator, CausalMethodEvaluator, granger_causality

    gen = PaleoDataGenerator(scenario='A', random_seed=42)
    evaluator = CausalMethodEvaluator(gen, granger_causality, "Granger")
    audit = evaluator.evaluate(n_realizations=5, use_age_errors=False)

    assert 'TPR' in audit
    assert 'FPR' in audit
    assert 0 <= audit['TPR'] <= 1
    assert 0 <= audit['FPR'] <= 1
