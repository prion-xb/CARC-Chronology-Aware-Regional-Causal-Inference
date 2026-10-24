"""
CausalPipeline — High-level orchestrator for CARC analysis.

CausalPipeline ties together record loading, detector configuration, ARC
deconfounding, and causal detection into a single workflow. It accepts any
CausalDetector via the Strategy pattern, allowing the user to swap methods
without changing pipeline code.
"""

import numpy as np
import pandas as pd

from .factory import RecordFactory, DetectorFactory
from .config import AGE_GRID_10_60, AGE_GRID_800
from .io import interpolate_to_grid


class CausalPipeline:
    """End-to-end causal inference pipeline for paleoclimate records.

    Workflow
    --------
    1. ``load_records()`` — load and filter records by age range
    2. ``set_detector(method, **kwargs)`` — choose causal method (or pass
       a pre-built detector at construction time)
    3. ``run()`` — execute detection and return results

    Parameters
    ----------
    detector : CausalDetector or None
        Pre-built detector instance. If None, call ``set_detector()``
        before ``run()``.
    age_grid : np.ndarray or None
        Common time grid for interpolation. Defaults to the 10–60 ka
        high-resolution grid.
    """

    def __init__(self, detector=None, age_grid=None):
        self._detector = detector
        self._age_grid = (age_grid if age_grid is not None
                          else AGE_GRID_10_60.copy())
        self._records = []
        self._data_dict = {}
        self._sigma_dict = {}
        self._forcing_proxy = None
        self._results = {}
        self._robustness_scores = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def detector(self):
        """The current CausalDetector instance."""
        return self._detector

    @property
    def records(self):
        """List of loaded PaleoRecord instances."""
        return self._records

    @property
    def results(self):
        """Detection results from the last ``run()`` call."""
        return dict(self._results)

    @property
    def n_records(self):
        """Number of loaded records."""
        return len(self._records)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def set_detector(self, method='carc', **kwargs):
        """Create and set a CausalDetector using the DetectorFactory.

        Parameters
        ----------
        method : str
            'granger' or 'carc'.
        **kwargs
            Forwarded to the detector constructor.
        """
        self._detector = DetectorFactory.create(method, **kwargs)

    def set_grid(self, grid):
        """Set the common interpolation time grid.

        Parameters
        ----------
        grid : np.ndarray (1D)
        """
        self._age_grid = np.asarray(grid, dtype=float)

    def set_forcing_proxy(self, proxy_values):
        """Set the common-forcing proxy signal for ARC deconfounding.

        Parameters
        ----------
        proxy_values : np.ndarray (1D)
            Must match the length of ``age_grid``.
        """
        self._forcing_proxy = np.asarray(proxy_values, dtype=float)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_records(self, records=None, data_dir=None,
                     age_min=10.0, age_max=60.0, min_points=10):
        """Load records into the pipeline.

        Parameters
        ----------
        records : list of PaleoRecord or None
            Pre-loaded records. If None, uses RecordFactory to load from disk.
        data_dir : str or None
            Directory to search (used only if records is None).
        age_min, age_max : float
            Age range filter (ka).
        min_points : int
            Minimum data points required.
        """
        if records is not None:
            self._records = list(records)
        else:
            factory = RecordFactory(data_dir=data_dir)
            self._records = factory.create_batch(
                age_min=age_min, age_max=age_max, min_points=min_points
            )

        # Build data dict and sigma dict
        self._data_dict = {}
        self._sigma_dict = {}
        for rec in self._records:
            interp = rec.interpolate(self._age_grid)
            self._data_dict[rec.name] = interp
            self._sigma_dict[rec.name] = rec.sigma

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run(self, seed=None):
        """Execute the causal detection pipeline.

        Returns
        -------
        dict
            {
                'edges': list of (driver, target),
                'robustness': dict {(driver,target): {...}},
                'n_records': int,
                'detector': str,
                'grid': np.ndarray,
            }
        """
        if self._detector is None:
            raise RuntimeError(
                "No detector set. Call set_detector() or pass a detector "
                "at construction time."
            )
        if not self._data_dict:
            raise RuntimeError(
                "No data loaded. Call load_records() first."
            )

        # Run detection
        if isinstance(self._detector, type(
            DetectorFactory.create('carc')
        )):
            # CARC detector — pass forcing proxy and sigma dict; make sure
            # the detector knows the grid spacing so age errors (ka) are
            # converted to grid index units in the perturbation.
            if hasattr(self._detector, '_dt'):
                self._detector._dt = self._age_grid[1] - self._age_grid[0]
            edges = self._detector.detect(
                self._data_dict,
                forcing_proxy=self._forcing_proxy,
                sigma_dict=self._sigma_dict,
                seed=seed,
            )
            robustness = self._detector.pairwise_scan(
                self._data_dict,
                forcing_proxy=self._forcing_proxy,
                sigma_dict=self._sigma_dict,
                seed=seed,
            )
        else:
            edges = self._detector.detect(self._data_dict)
            robustness = {}

        self._results = {
            'edges': edges,
            'robustness': robustness,
            'n_records': len(self._records),
            'detector': self._detector.name,
            'grid': self._age_grid,
        }
        return self._results

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def summarize(self):
        """Return a summary DataFrame of detected edges.

        Returns
        -------
        pd.DataFrame
            Columns: driver, target, robustness, honest
            Sorted by robustness (descending).
        """
        edges = self._results.get('edges', [])
        robustness = self._results.get('robustness', {})

        if not edges:
            return pd.DataFrame(
                columns=['driver', 'target', 'robustness', 'honest']
            )

        rows = []
        for driver, target in edges:
            info = robustness.get((driver, target), {})
            rows.append({
                'driver': driver,
                'target': target,
                'robustness': info.get('robustness', 0.0),
                'honest': info.get('honest', False),
            })

        df = pd.DataFrame(rows)
        df.sort_values('robustness', ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def to_csv(self, output_path):
        """Save the summary table to a CSV file.

        Parameters
        ----------
        output_path : str
            Path to output CSV.
        """
        df = self.summarize()
        df.to_csv(output_path, index=False, float_format='%.4f')

    def __repr__(self):
        return (f"CausalPipeline("
                f"detector={self._detector.name if self._detector else 'None'}, "
                f"n_records={len(self._records)}, "
                f"grid=[{self._age_grid[0]:.1f}..{self._age_grid[-1]:.1f}])")
