"""
PaleoRecord — Object-oriented representation of paleoclimate proxy records.

Provides an abstract base class PaleoRecord and four concrete subclasses
corresponding to major paleoclimate archive types. Each record encapsulates
its age-depth model, proxy values, age uncertainty (σ), geographic region,
and archive-specific metadata.

Class Hierarchy
---------------
PaleoRecord (ABC)
├── IceCoreRecord      — ice cores (NGRIP, EDC, Dome Fuji, WAIS, etc.)
├── MarineRecord       — marine sediment cores (ODP, U-series, etc.)
├── SpeleothemRecord   — cave deposits / speleothems (Cheng, Botuvera, etc.)
└── TerrestrialRecord  — loess, lake sediments, and other continental archives
"""

import os
import numpy as np
from abc import ABC, abstractmethod
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class PaleoRecord(ABC):
    """Abstract base class for a single paleoclimate proxy record.

    Encapsulates the time series (age, value), age-model uncertainty,
    geographic region, and archive-type metadata.

    Parameters
    ----------
    name : str
        Record identifier / stem name.
    age : np.ndarray (1D)
        Age axis in ka (thousands of years before present).
    value : np.ndarray (1D)
        Proxy values, same length as `age`.
    sigma : float
        Age-model 1σ uncertainty in ka.
    region : str
        Geographic region label.
    """

    def __init__(self, name, age, value, sigma, region):
        if len(age) != len(value):
            raise ValueError(
                f"age and value must have same length for record '{name}'"
            )
        self._name = name
        self._age = np.asarray(age, dtype=float)
        self._value = np.asarray(value, dtype=float)
        self._sigma = float(sigma)
        self._region = str(region)
        self._archive_type = self.archive_type  # resolved by subclass property

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def name(self):
        """Record identifier."""
        return self._name

    @property
    def age(self):
        """Age axis (ka)."""
        return self._age

    @property
    def value(self):
        """Proxy values (original units)."""
        return self._value

    @property
    def sigma(self):
        """Age-model 1σ uncertainty (ka)."""
        return self._sigma

    @sigma.setter
    def sigma(self, new_sigma):
        self._sigma = float(new_sigma)

    @property
    def region(self):
        """Geographic region."""
        return self._region

    @region.setter
    def region(self, new_region):
        self._region = str(new_region)

    @property
    @abstractmethod
    def archive_type(self):
        """Archive type string (set by subclass)."""
        ...

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------
    def interpolate(self, grid, kind='linear'):
        """Interpolate record values onto a regular time grid.

        Parameters
        ----------
        grid : np.ndarray (1D)
            Target age grid.
        kind : str
            Interpolation kind ('linear', 'cubic', etc.).

        Returns
        -------
        np.ndarray
            Interpolated values on `grid`.
        """
        f = interp1d(self._age, self._value, kind=kind,
                     bounds_error=False, fill_value=np.nan)
        result = f(grid)
        # forward-fill remaining NaN
        valid_idx = np.where(~np.isnan(result))[0]
        if len(valid_idx) > 0:
            last = result[valid_idx[0]]
            for i in range(len(result)):
                if np.isnan(result[i]):
                    result[i] = last
                else:
                    last = result[i]
        return result

    def standardize(self):
        """Z-score standardize the record values (in-place)."""
        mean = np.nanmean(self._value)
        std = np.nanstd(self._value)
        if std < 1e-10:
            self._value = self._value - mean
        else:
            self._value = (self._value - mean) / std
        return self

    def perturb_age(self, rng=None):
        """Perturb the age axis by N(0, sigma) and return re-sorted values.

        This mimics the effect of age-model uncertainty by adding Gaussian
        noise to each age estimate, re-sorting the series, and interpolating
        back to the original time positions.

        Parameters
        ----------
        rng : np.random.Generator or None

        Returns
        -------
        np.ndarray
            Value array after age perturbation (same length).
        """
        if rng is None:
            rng = np.random.default_rng()

        n = len(self._age)
        orig = np.arange(n)
        pert = orig + rng.normal(0, self._sigma, n)
        idx = np.argsort(pert)

        f = interp1d(pert[idx], self._value[idx], kind='linear',
                     fill_value='extrapolate', bounds_error=False)
        result = f(orig)

        valid = ~np.isnan(result)
        if valid.sum() < 10:
            return self._value.copy()
        return result

    def to_dict(self):
        """Return record metadata as a dictionary."""
        return {
            'name': self._name,
            'archive_type': self.archive_type,
            'region': self._region,
            'sigma': self._sigma,
            'n_points': len(self._age),
            'age_min': float(np.min(self._age)),
            'age_max': float(np.max(self._age)),
        }

    def __len__(self):
        return len(self._age)

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"name='{self._name}', "
                f"archive='{self.archive_type}', "
                f"region='{self._region}', "
                f"σ={self._sigma:.2f} ka, "
                f"n={len(self)})")


# ---------------------------------------------------------------------------
# Concrete subclasses
# ---------------------------------------------------------------------------
class IceCoreRecord(PaleoRecord):
    """Ice-core proxy record (δ¹⁸O, δD, gas concentrations, dust, etc.)."""

    @property
    def archive_type(self):
        return 'ice_core'


class MarineRecord(PaleoRecord):
    """Marine sediment core record (δ¹⁸O, SST, XRF elemental ratios, etc.)."""

    @property
    def archive_type(self):
        return 'marine'


class SpeleothemRecord(PaleoRecord):
    """Speleothem (cave deposit) record — typically U-Th dated δ¹⁸O time series."""

    @property
    def archive_type(self):
        return 'speleothem'


class TerrestrialRecord(PaleoRecord):
    """Terrestrial / continental archive — loess, lake sediment, or other.

    This subclass covers loess-palaeosol sequences, lake sediment cores, and
    other non-marine, non-ice, non-speleothem continental records.
    """

    @property
    def archive_type(self):
        return 'terrestrial'
