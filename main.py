#!/usr/bin/env python3
"""
main.py — CARC (Chronology-Aware Robustness Causality) entry point.

This script demonstrates the object-oriented architecture of the CARC
paleoclimate causal inference framework. It showcases:

1. **Inheritance**: ``PaleoRecord`` ABC with 4 concrete archive-type subclasses
   (IceCoreRecord, MarineRecord, SpeleothemRecord, TerrestrialRecord)
   and ``CausalDetector`` ABC with 2 concrete subclasses
   (StandardGrangerDetector, CARCDetector).

2. **Factory Pattern**: ``RecordFactory`` creates the correct PaleoRecord
   subclass based on filename keywords; ``DetectorFactory`` creates the
   correct CausalDetector by method name.

3. **Strategy Pattern**: ``CausalPipeline`` accepts any CausalDetector,
   allowing method swapping without changing pipeline code.

Usage
-----
    python main.py                # Full pipeline: load records → CARC detect
    python main.py --demo         # Quick demo with synthetic data
    python main.py --method granger  # Use standard Granger (no CARC)
"""

import argparse
import sys
import os

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.records import PaleoRecord, IceCoreRecord, MarineRecord
from src.detectors import CausalDetector, StandardGrangerDetector, CARCDetector
from src.factory import RecordFactory, DetectorFactory
from src.pipeline import CausalPipeline


# =========================================================================
# Demo: synthetic data
# =========================================================================
def run_demo():
    """Demonstrate the OOP architecture with synthetic data."""
    print("=" * 60)
    print("  CARC — Object-Oriented Architecture Demo")
    print("=" * 60)

    # ---- Demonstrate inheritance ----
    print("\n[1] PaleoRecord inheritance hierarchy:")
    print("-" * 40)

    # Create records of different types using their constructors directly
    import numpy as np
    t = np.linspace(0, 100, 200)
    ice = IceCoreRecord(
        name="NGRIP_d18O_demo",
        age=t.copy(),
        value=np.sin(2 * np.pi * t / 41.0) + 0.3 * np.random.randn(200),
        sigma=0.65,
        region="North_Atlantic"
    )
    marine = MarineRecord(
        name="ODP983_IRD_demo",
        age=t.copy(),
        value=np.cos(2 * np.pi * t / 23.0) + 0.4 * np.random.randn(200),
        sigma=2.0,
        region="North_Atlantic"
    )

    for rec in [ice, marine]:
        print(f"  {rec}")
        print(f"    → archive_type = '{rec.archive_type}'")

    print(f"\n  isinstance(ice, PaleoRecord)    = {isinstance(ice, PaleoRecord)}")
    print(f"  isinstance(marine, PaleoRecord)  = {isinstance(marine, PaleoRecord)}")
    print(f"  isinstance(ice, IceCoreRecord)   = {isinstance(ice, IceCoreRecord)}")

    # ---- Demonstrate CausalDetector inheritance ----
    print("\n[2] CausalDetector inheritance hierarchy:")
    print("-" * 40)

    granger = StandardGrangerDetector(maxlag=5, alpha=0.05)
    carc = CARCDetector(n_perturb=50, arc_alpha=0.5)

    for det in [granger, carc]:
        print(f"  {det}")
        print(f"    → params = {det.params}")

    print(f"\n  isinstance(carc, CausalDetector)            = {isinstance(carc, CausalDetector)}")
    print(f"  isinstance(carc, StandardGrangerDetector)    = {isinstance(carc, StandardGrangerDetector)}")
    print(f"  isinstance(carc, CARCDetector)               = {isinstance(carc, CARCDetector)}")

    # ---- Demonstrate Factory pattern ----
    print("\n[3] Factory pattern:")
    print("-" * 40)

    # DetectorFactory
    d1 = DetectorFactory.create('granger', maxlag=5)
    d2 = DetectorFactory.create('carc', n_perturb=50, arc_alpha=0.5)
    print(f"  DetectorFactory.create('granger') → {d1}")
    print(f"  DetectorFactory.create('carc')    → {d2}")
    print(f"  Available methods: {DetectorFactory.list_available()}")

    # ---- Demonstrate Pipeline (Strategy pattern) ----
    print("\n[4] Pipeline with Strategy pattern:")
    print("-" * 40)

    # Build simple synthetic data dict
    n = 500
    np.random.seed(42)
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.8 * x[i - 1] + np.random.randn() * 0.3
        y[i] = 0.7 * y[i - 1] + 0.3 * x[i - 2] + np.random.randn() * 0.3  # x → y lag

    data = {'X': x, 'Y': y}
    sigma_dict = {'X': 0.5, 'Y': 0.5}

    # Strategy 1: Standard Granger
    pipeline = CausalPipeline(detector=d1)
    pipeline._data_dict = data
    pipeline._sigma_dict = sigma_dict
    edges_granger = pipeline.run(seed=123)['edges']
    print(f"  Standard Granger edges: {edges_granger}")

    # Strategy 2: CARC
    pipeline._detector = d2
    edges_carc = pipeline.run(seed=123)['edges']
    print(f"  CARC edges:             {edges_carc}")

    # ---- Summarize ----
    print("\n" + "=" * 60)
    print("  Architecture summary")
    print("=" * 60)
    print(f"""
    Classes defined:
      PaleoRecord (ABC)          — abstract base for proxy records
        ├── IceCoreRecord        — ice-core archives
        ├── MarineRecord         — marine sediment archives
        ├── SpeleothemRecord     — cave deposit archives
        └── TerrestrialRecord    — loess / lake / continental archives

      CausalDetector (ABC)       — abstract base for causal methods
        ├── StandardGrangerDetector — classical Granger causality
        └── CARCDetector            — chronology-aware robust causality

      RecordFactory              — creates PaleoRecord by file type
      DetectorFactory            — creates CausalDetector by name
      CausalPipeline             — orchestrates the full workflow

    Design patterns:
      • Factory Method  — RecordFactory, DetectorFactory
      • Template Method — CausalDetector.detect() skeleton
      • Strategy        — CausalPipeline accepts any detector
      • Inheritance     — 2 abstract base classes, 6 concrete subclasses

    Total classes: 11 (2 ABCs, 6 concrete, 2 factories, 1 pipeline)
    """)


# =========================================================================
# Full pipeline: real data
# =========================================================================
def run_full_pipeline(method='carc', age_min=10.0, age_max=60.0, **kwargs):
    """Run the complete CARC pipeline on real proxy records.

    Parameters
    ----------
    method : str
        'granger' or 'carc'.
    age_min, age_max : float
        Age range in ka.
    **kwargs
        Passed to detector constructor.
    """
    print(f"\n{'=' * 60}")
    print(f"  CARC Pipeline — {method.upper()} detection")
    print(f"  Age range: {age_min}–{age_max} ka")
    print(f"{'=' * 60}")

    # 1. Create pipeline with detector via factory
    print("\n[Step 1] Creating detector via DetectorFactory...")
    detector = DetectorFactory.create(method, **kwargs)
    pipeline = CausalPipeline(detector=detector)

    # 2. Load records via factory
    print("[Step 2] Loading records via RecordFactory...")
    pipeline.load_records(age_min=age_min, age_max=age_max)
    print(f"  Loaded {pipeline.n_records} records.")

    if pipeline.n_records == 0:
        print("  WARNING: No records found in the data directory.")
        print("  Please ensure proxy_records/ contains CSV/XLSX files.")
        return

    # Show record types
    type_counts = {}
    for rec in pipeline.records:
        t = rec.archive_type
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"  Archive types: {type_counts}")

    # 3. Set forcing proxy (use first ice-core record as common-forcing proxy)
    print("\n[Step 3] Configuring forcing proxy...")
    for rec in pipeline.records:
        if rec.archive_type == 'ice_core':
            proxy_vals = rec.interpolate(pipeline._age_grid)
            pipeline.set_forcing_proxy(proxy_vals)
            print(f"  Using '{rec.name}' as common-forcing proxy.")
            break
    else:
        print("  No ice-core record found; skipping ARC deconfounding.")

    # 4. Run detection
    print(f"\n[Step 4] Running {method.upper()} detection...")
    results = pipeline.run(seed=42)

    edges = results['edges']
    print(f"  Detected {len(edges)} significant causal edges.")

    # 5. Display results
    if edges:
        print("\n[Results] Significant causal edges (honesty-filtered):")
        print("-" * 60)
        df = pipeline.summarize()
        print(df.to_string(index=False))
    else:
        print("\n  No significant edges detected at current thresholds.")

    return pipeline


# =========================================================================
# CLI
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="CARC — Chronology-Aware Robustness Causality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                       # Full CARC pipeline (10–60 ka)
  python main.py --demo                # Quick architecture demo
  python main.py --method granger      # Standard Granger (no CARC)
  python main.py --age-min 0 --age-max 800  # Orbital-scale analysis
        """
    )
    parser.add_argument('--demo', action='store_true',
                        help='Run a quick demo of the OOP architecture.')
    parser.add_argument('--method', default='carc', choices=['granger', 'carc'],
                        help='Causal detection method (default: carc).')
    parser.add_argument('--age-min', type=float, default=10.0,
                        help='Minimum age in ka (default: 10.0).')
    parser.add_argument('--age-max', type=float, default=60.0,
                        help='Maximum age in ka (default: 60.0).')
    parser.add_argument('--n-perturb', type=int, default=200,
                        help='Number of age perturbations (default: 200).')
    parser.add_argument('--arc-alpha', type=float, default=0.5,
                        help='ARC stripping strength (default: 0.5).')
    parser.add_argument('--stability', type=float, default=0.95,
                        help='Honesty filter threshold (default: 0.95).')

    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        kwargs = {}
        if args.method == 'carc':
            kwargs = {
                'n_perturb': args.n_perturb,
                'arc_alpha': args.arc_alpha,
                'stability_threshold': args.stability,
            }
        run_full_pipeline(
            method=args.method,
            age_min=args.age_min,
            age_max=args.age_max,
            **kwargs
        )


if __name__ == '__main__':
    main()
