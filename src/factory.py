"""
Factory classes for creating PaleoRecord and CausalDetector instances.

Design Pattern
--------------
- **RecordFactory**: Simple Factory — creates the correct PaleoRecord
  subclass based on filename keyword classification.
- **DetectorFactory**: Simple Factory — creates the correct CausalDetector
  subclass (StandardGranger or CARC) based on a method name string.

Both factories encapsulate creation logic so that client code (pipelines,
scripts) does not need to know about concrete subclass constructors.
"""

import os
import re
import numpy as np

from .config import DATA_DIR, get_sigma
from .records import (
    PaleoRecord, IceCoreRecord, MarineRecord,
    SpeleothemRecord, TerrestrialRecord,
)
from .io import load_record, safe_load_csv, identify_columns
from .detectors import (
    CausalDetector, StandardGrangerDetector, CARCDetector,
)
from .stacking import classify_record  # canonical classification


# ---------------------------------------------------------------------------
# RecordFactory
# ---------------------------------------------------------------------------
class RecordFactory:
    """Factory that creates PaleoRecord instances from data files.

    Determines the correct subclass by delegating to
    :func:`src.stacking.classify_record`, then loads the data, assigns
    metadata (archive type, region, sigma), and returns the appropriate
    concrete record.

    Usage
    -----
    >>> factory = RecordFactory()
    >>> rec = factory.create_from_file('NGRIP_d18O')
    >>> records = factory.create_batch(age_min=0.0, age_max=800.0)
    """

    def __init__(self, data_dir=None):
        self._data_dir = data_dir if data_dir is not None else DATA_DIR

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    # Type mapping: stacking.classify_record → factory archive_type
    _TYPE_UNIFY = {
        'ice_core':   'ice_core',
        'marine':     'marine',
        'speleothem': 'speleothem',
        'loess':      'terrestrial',
        'lake':       'terrestrial',
        'other':      'marine',
    }

    @classmethod
    def classify(cls, filename):
        """Classify a filename into (archive_type, region).

        Delegates to :func:`src.stacking.classify_record` as the single
        canonical classification source, then maps type tags to the
        four PaleoRecord subclasses.
        """
        typ, region = classify_record(filename)
        archive_type = cls._TYPE_UNIFY.get(typ, 'marine')
        return archive_type, region

    # ------------------------------------------------------------------
    # Subclass mapping
    # ------------------------------------------------------------------
    _SUBCLASS_MAP = {
        'ice_core': IceCoreRecord,
        'marine': MarineRecord,
        'speleothem': SpeleothemRecord,
        'terrestrial': TerrestrialRecord,
    }

    @classmethod
    def _get_record_class(cls, archive_type):
        """Return the PaleoRecord subclass for a given archive_type string."""
        return cls._SUBCLASS_MAP.get(archive_type, MarineRecord)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------
    def create_from_file(self, stem):
        """Create a PaleoRecord from a record name stem or filename.

        Parameters
        ----------
        stem : str
            Record identifier (stem name or full filename).

        Returns
        -------
        PaleoRecord subclass instance, or None if loading fails.
        """
        age, value = load_record(stem, self._data_dir)
        if age is None or value is None or len(age) < 10:
            return None

        archive_type, region = self.classify(stem)
        sigma = get_sigma(stem)

        RecordClass = self._get_record_class(archive_type)
        rec = RecordClass(name=stem, age=age, value=value,
                          sigma=sigma, region=region)
        rec.standardize()
        return rec

    def create_batch(self, age_min=0.0, age_max=800.0, min_points=10):
        """Create PaleoRecord instances for all CSV/XLSX files in the data directory.

        Parameters
        ----------
        age_min, age_max : float
            Age range filter (ka).
        min_points : int
            Minimum data points required in the age range.

        Returns
        -------
        list of PaleoRecord subclass instances
        """
        records = []
        data_dir = self._data_dir

        if not os.path.isdir(data_dir):
            return records

        for fname in os.listdir(data_dir):
            if not (fname.endswith('.csv') or fname.endswith('.xlsx')):
                continue

            stem = os.path.splitext(fname)[0]
            rec = self.create_from_file(stem)
            if rec is None:
                continue

            # Age range filter
            in_range = (rec.age >= age_min) & (rec.age <= age_max)
            if np.sum(in_range) < min_points:
                continue

            # Trim to range
            rec._age = rec.age[in_range]
            rec._value = rec.value[in_range]
            if np.std(rec.value) < 1e-6:
                continue
            rec.standardize()

            records.append(rec)

        return records


# ---------------------------------------------------------------------------
# DetectorFactory
# ---------------------------------------------------------------------------
class DetectorFactory:
    """Factory that creates CausalDetector instances by method name.

    Supported methods
    -----------------
    - ``'granger'`` → :class:`StandardGrangerDetector`
    - ``'carc'``    → :class:`CARCDetector`

    Usage
    -----
    >>> factory = DetectorFactory()
    >>> det = factory.create('carc', n_perturb=200, arc_alpha=0.5)
    >>> print(factory.list_available())  # ['granger', 'carc']
    """

    _REGISTRY = {
        'granger': StandardGrangerDetector,
        'carc': CARCDetector,
    }

    @classmethod
    def create(cls, method='carc', **kwargs):
        """Create a CausalDetector instance.

        Parameters
        ----------
        method : str
            One of 'granger' or 'carc'.
        **kwargs
            Forwarded to the detector constructor.

        Returns
        -------
        CausalDetector subclass instance

        Raises
        ------
        ValueError
            If `method` is not recognised.
        """
        method = method.lower()
        if method not in cls._REGISTRY:
            raise ValueError(
                f"Unknown detector method: '{method}'. "
                f"Available: {cls.list_available()}"
            )
        DetectorClass = cls._REGISTRY[method]
        return DetectorClass(**kwargs)

    @classmethod
    def list_available(cls):
        """Return a list of available method name strings."""
        return list(cls._REGISTRY.keys())

    @classmethod
    def register(cls, name, detector_class):
        """Register a new detector class under a given name.

        Parameters
        ----------
        name : str
            Method name key.
        detector_class : type
            Must be a subclass of CausalDetector.
        """
        if not issubclass(detector_class, CausalDetector):
            raise TypeError(
                f"{detector_class} must be a subclass of CausalDetector"
            )
        cls._REGISTRY[name.lower()] = detector_class
