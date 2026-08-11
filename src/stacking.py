"""
Regional stacking: record classification, region grouping, and ensemble stacking
with age-perturbation.

Used in: 0-800.py, 堆叠A.py, 堆叠相位差.py, 网络相位差.py
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d


def classify_record(filename):
    """
    Classify a paleoclimate record by archive type and geographic region.

    Parameters
    ----------
    filename : str
        Record filename (e.g. '32_Jouzel-2007-dD.csv').

    Returns
    -------
    typ : str
        'ice_core', 'marine', 'speleothem', 'loess', 'lake', or 'other'
    region : str or None
        Geographic region, or None if unclassifiable.
    """
    name = filename.lower()

    # --- Type ---
    # Ice cores: EDC, Dome Fuji, NGRIP, WAIS, Vostok
    if any(k in name for k in ['edc', 'dome fuji', 'domefuji', 'ngrip', 'wais',
                                 'vostok', 'jouzel', 'bereiter', 'neem',
                                 'ch4 over past']):
        typ = 'ice_core'
    elif any(k in name for k in ['odp', 'u13', 'u14', 'u15', 'u1446', 'u1476', 'u1483',
                                 'u1542', 'm125', 'ps75', 'md0', 'md1', 'md2', 'md9',
                                 'md-', 'kx97', 'ipwp', 'cariaco', 'dsr', 'botuvera',
                                 'framstrait', 'barker', 'mcmanus', 'naafs',
                                 'clemens', 'g.ruber', 'g_ruber', 'd18og.rub',
                                 'ln k-al', 'ln ti-ca',
                                 'log fe-ca', 'log zr-fe',
                                 'sst=0-2.7', 'ohc-stack',
                                 'ba=128', 'br.csv']):
        typ = 'marine'
    elif any(k in name for k in ['speleo', 'cheng', 'dongge', 'sanbao', 'hulu']):
        typ = 'speleothem'
    elif any(k in name for k in ['loess', 'gulang', 'weinan', 'luochuan', 'xifeng',
                                 'yimaguan', 'zb13', 'chb14', 'gl-fe', 'gl_',
                                 'lc gt', 'ymg gt', 'pollen=0-800', 'gt32']):
        typ = 'loess'
    elif any(k in name for k in ['lake', 'titicaca', 'malawi', 'elgygytgyn',
                                 'baikal', 'ohrid', 'junin', 'bio silica']):
        typ = 'lake'
    else:
        typ = 'other'

    # --- Region ---
    # Antarctic & Southern Ocean
    if any(k in name for k in ['edc', 'dome fuji', 'domefuji', 'wais', 'vostok',
                                 'jouzel', 'bereiter', 'neem', 'ch4 over past']):
        region = 'Antarctic'
    elif any(k in name for k in ['odp1090', 'odp1094', 'u1537', 'ps75',
                                 'md97-2021', 'md00-2361', 'jaccard',
                                 'log fe-ca', 'log zr-fe']):
        region = 'Southern_Ocean'
    # North Atlantic & Nordic Seas
    elif any(k in name for k in ['ngrip', 'odp983', 'odp980', 'u1302', 'u1308',
                                 'u1313', 'u1391', 'framstrait', 'barker', 'mcmanus',
                                 'susceptibility=kappa', 'nps', 'rpi']):
        region = 'North_Atlantic'
    # East Asian Monsoon (speleothems + Chinese Loess Plateau)
    elif any(k in name for k in ['cheng', 'speleo', 'dongge', 'sanbao', 'hulu',
                                 'gulang', 'weinan', 'luochuan', 'xifeng',
                                 'yimaguan', 'zb13', 'chb14', 'gl-fe', 'gl_',
                                 'lc gt', 'ymg gt', 'pollen=0-800', 'jingyuan',
                                 'gt32']):
        region = 'East_Asian_Monsoon'
    # Tropical / Indo-Pacific
    elif any(k in name for k in ['cariaco', 'odp1089', 'odp1143', 'odp1146', 'odp967',
                                 'odp663', 'odp1002', 'odp184', 'ipwp',
                                 'kx97322', 'u1446', 'u1476', 'u1483', 'u1542',
                                 'u1385', 'md01-2443', 'md-2518', 'md-2519',
                                 'g.ruber', 'g_ruber', 'd18og.rub', 'md2622', 'clemens', 'naafs',
                                 'ohc-stack', 'malawi', 'odp 184',
                                 'ln k-al', 'ln ti-ca',
                                 'sst=0-2.7', 'ba=128', 'br.csv']):
        region = 'Tropical'
    # South America
    elif any(k in name for k in ['botuvera', 'titicaca', 'lapa', 'santana', 'paixao']):
        region = 'South_America'
    # High-latitude lakes
    elif any(k in name for k in ['elgygytgyn', 'baikal', 'ohrid', 'junin',
                                 'bio silica']):
        region = 'Other_Lakes'
    else:
        region = None

    return typ, region


def build_region_dict(records):
    """
    Group records by region. Filters out regions with < 2 records.

    Parameters
    ----------
    records : list of (interp_array, sigma, region, name)

    Returns
    -------
    region_dict : dict of {region: list of (interp, sigma, name)}
    """
    region_dict = {}
    for interp, sigma, region, name in records:
        region_dict.setdefault(region, []).append((interp, sigma, name))
    return {r: v for r, v in region_dict.items() if len(v) >= 2}


def stack_region(region_records, grid, n_pert, rng=None, smooth_sigma=5.0):
    """
    Create an age-perturbation ensemble stack for a region.

    For each perturbation:
    1. Perturb each record's time axis by its sigma
    2. Re-sort each record
    3. Take the median across records at each time point
    4. Gaussian-smooth the result

    Parameters
    ----------
    region_records : list of (interp_array, sigma, name)
    grid : np.ndarray (1D) — common time grid
    n_pert : int — number of perturbations
    rng : np.random.Generator or None
    smooth_sigma : float — Gaussian filter sigma in ka

    Returns
    -------
    stacked : np.ndarray of shape (n_pert, len(grid))
    """
    if rng is None:
        rng = np.random.default_rng()

    dt = grid[1] - grid[0]
    stacked = np.zeros((n_pert, len(grid)))

    for pert in range(n_pert):
        signals = []
        for interp, sigma, _ in region_records:
            age_pert = grid + rng.normal(0, sigma, len(grid))
            idx = np.argsort(age_pert)
            signals.append(interp[idx])
        stacked[pert] = np.median(signals, axis=0)

    # Gaussian smoothing
    for i in range(n_pert):
        stacked[i] = gaussian_filter1d(stacked[i], sigma=smooth_sigma / dt, mode='nearest')

    return stacked
