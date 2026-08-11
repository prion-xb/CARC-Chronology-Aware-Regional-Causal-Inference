"""
Geographic coordinates for CARC proxy records and regional centres.

Each record entry maps a record name stem to (latitude, longitude, archive_type).
Regional centres are approximate centroid locations used for network/map layouts.
"""

# ---------------------------------------------------------------------------
# Regional centroids (lat, lon) — approximate geographic centres
# ---------------------------------------------------------------------------
REGION_CENTROIDS = {
    "Antarctic":              (-75.0, 0.0),     # EPICA Dome C region
    "Southern_Ocean":         (-55.0, 0.0),     # Southern Ocean mid-latitudes
    "North_Atlantic":         (65.0, -30.0),    # Greenland / Nordic Seas
    "East_Asian_Monsoon":     (35.0, 108.0),    # Chinese Loess Plateau / cave region
    "Tropical":               (5.0, 90.0),      # Indo-Pacific warm pool
    "South_America":          (-15.0, -55.0),   # Amazon / eastern Brazil
    "Tropical_Marine":        (-5.0, 90.0),     # same as Tropical
    "South_American_Monsoon": (-15.0, -55.0),   # same as South_America
    "Other_Lakes":            (55.0, 120.0),    # Siberia / high-latitude lakes
    "Lake":                   (55.0, 120.0),    # same as Other_Lakes
}

# Regional short display names
REGION_LABELS = {
    "Antarctic":              "Antarctic",
    "Southern_Ocean":         "Southern Ocean",
    "North_Atlantic":         "North Atlantic",
    "East_Asian_Monsoon":     "East Asian Monsoon",
    "Tropical":               "Tropical",
    "South_America":          "South America",
    "Tropical_Marine":        "Tropical Marine",
    "South_American_Monsoon": "South American Monsoon",
    "Other_Lakes":            "High-Latitude Lakes",
    "Lake":                   "Lakes",
}

# Regional colour assignments (consistent across all figures)
REGION_COLORS = {
    "Antarctic":              "#2166AC",
    "Southern_Ocean":         "#4393C3",
    "North_Atlantic":         "#D73027",
    "East_Asian_Monsoon":     "#1B7837",
    "Tropical":               "#E66101",
    "South_America":          "#762A83",
    "Tropical_Marine":        "#E66101",
    "South_American_Monsoon": "#762A83",
    "Other_Lakes":            "#008080",
    "Lake":                   "#008080",
}

# ---------------------------------------------------------------------------
# Per-record coordinates (lat, lon)
# ---------------------------------------------------------------------------
RECORD_COORDS = {
    # --- Antarctic ice cores ---
    "EDC_dD":                 (-75.10, 123.35),
    "EDC_CO2":                (-75.10, 123.35),
    "EDC_CH4":                (-75.10, 123.35),
    "EDC_Dust":               (-75.10, 123.35),
    "DomeFuji_dD":            (-77.32, 39.70),
    "DomeFuji_d18O":          (-77.32, 39.70),
    "WAIS_CH4":               (-79.47, -112.09),
    "NGRIP_d18O":             (75.10, -42.32),

    # --- North Atlantic / Nordic Seas ---
    "ODP983_IRD":             (60.40, -23.64),
    "ODP983_NPS":             (60.40, -23.64),
    "ODP980_planktic_d18O":   (55.48, -14.70),
    "ODP980_benthic_d18O":    (55.48, -14.70),
    "ODP980_eNd":             (55.48, -14.70),
    "U1302-3_SiSr":           (50.15, -45.63),
    "U1302-3_d18O":           (50.15, -45.63),
    "U1302-3_MS":             (50.15, -45.63),
    "U1308_RPI":              (49.88, -24.23),
    "U1308_SiSr":             (49.88, -24.23),
    "U1313_nAlkanes":         (41.00, -32.96),
    "U1313_SST":              (41.00, -32.96),
    "FramStrait_IP25":        (78.90, 6.80),
    "FramStrait_PBIP25":      (78.90, 6.80),
    "FramStrait_DIP25":       (78.90, 6.80),

    # --- U1391 (Gulf of Cadiz / Iberian margin) ---
    "U1391_benthic_d18O":     (37.36, -9.41),
    "U1391_planktic_d18O":    (37.36, -9.41),

    # --- Southern Ocean ---
    "ODP1089_SST":            (-40.94, 9.89),
    "ODP1090_Dust":           (-42.91, 8.90),
    "ODP1090_FeMAR":          (-42.91, 8.90),
    "ODP1094_BaFe":           (-53.18, 5.13),
    "PS75-059-2_Fe":          (-54.20, -78.33),
    "PS75-076-2_Fe":          (-57.50, -78.33),
    "PS75-074-3_Fe":          (-56.36, -78.33),
    "U1537_MS_AR":            (-54.68, -57.58),
    "U1537_NGR":              (-54.68, -57.58),
    "MD97-2120_d18O":         (-45.53, 174.68),
    "MD97-2120_MgCa_SST":     (-45.53, 174.68),

    # --- Tropical ---
    "Cariaco_Mo":             (10.70, -64.93),
    "ODP1143_UK37_SST":       (9.36, 113.28),
    "ODP1146_Al":             (19.46, 116.27),
    "ODP1146_Terrigenous":    (19.46, 116.27),
    "ODP967_Dust":            (34.07, 32.73),
    "ODP967_d18O":            (34.07, 32.73),
    "ODP967_WetDryIndex":     (34.07, 32.73),
    "ODP663_bSiO2":           (-1.20, -11.90),
    "ODP663_CaCO3":           (-1.20, -11.90),
    "ODP663_Terrigenous":     (-1.20, -11.90),
    "U1385_ZrSr":             (37.57, -10.12),
    "U1446_RbCa":             (5.38, 90.38),
    "U1446_d18Osw":           (5.38, 90.38),
    "U1476_benthic_d18O":     (-15.13, 38.60),
    "U1483_benthic_d18O":     (-13.09, 121.60),
    "U1483_logKCa":           (-13.09, 121.60),
    "U1542_SST":              (-8.12, -83.31),
    "U1542_CHC":              (-8.12, -83.31),
    "KX97322-4_d18O":         (-3.50, 143.20),
    "KX97322-4_MgCa_SST":     (-3.50, 143.20),
    "IPWP_OHC":               (0.0, 120.0),
    "M125-73-3_lnKAl":        (-25.40, 38.30),
    "M125-73-3_lnTiCa":       (-25.40, 38.30),
    "MD01-2443_UK37_SST":     (37.80, -10.17),
    "MD01-2443_CaTi":         (37.80, -10.17),
    "MD05-2925_d18O":         (-10.34, 161.37),
    "MD05-2925_MgCa_SST":     (-10.34, 161.37),
    "MD00-2361_d18O":         (-41.38, 43.56),
    "MD00-2361_logFeCa":      (-41.38, 43.56),
    "MD00-2361_logZrFe":      (-41.38, 43.56),
    "MD01-2518_DSR":          (-25.52, 112.50),
    "MD01-2519_DSR":          (-27.50, 112.50),
    "Baikal_BioSilica":       (53.50, 108.00),

    # --- East Asian Monsoon ---
    "Cheng2016_d18O":         (32.0, 109.0),
    "Gulang_ms":              (37.50, 102.88),
    "Gulang_grain":           (37.50, 102.88),
    "Gulang_FeK":             (37.50, 102.88),
    "Weinan_ms":              (34.50, 109.50),
    "Weinan_grain":           (34.50, 109.50),
    "Luochuan_GT32":          (35.76, 109.42),
    "Luochuan_grain":         (35.76, 109.42),
    "Xifeng_ms":              (35.73, 107.63),
    "Xifeng_grain":           (35.73, 107.63),
    "Yimaguan_GT32":          (35.20, 108.00),
    "Jingyuan_ms":            (36.56, 104.68),
    "Jingyuan_grain":         (36.56, 104.68),
    "ZB13_Pollen":            (37.70, 107.40),
    "CHB14_KZr":              (35.00, 109.00),

    # --- South America ---
    "Botuvera_d13C":          (-27.22, -49.16),
    "Botuvera_d18O":          (-27.22, -49.16),
    "Santana_d18O":           (-24.50, -48.50),
    "Paixao_d18O":            (-12.50, -41.00),
    "Lapa_d18O":              (-13.88, -44.02),
    "Sudwala_d18O":           (-25.37, 30.70),
    "Marota_d18O":            (-13.00, -41.00),

    # --- Lakes ---
    "Elgygytgyn_MS":          (67.50, 172.00),
    "Elgygytgyn_SiTi":        (67.50, 172.00),
    "Elgygytgyn_TOC":         (67.50, 172.00),
    "Malawi_Ca":              (-12.00, 34.50),
    "Titicaca_MS":            (-16.00, -69.00),
    "Junin_GI":               (-11.00, -76.00),
    "Ohrid_CaK":              (41.04, 20.72),
}


def get_record_coords(record_name):
    """
    Return (lat, lon) for a record name.
    Falls back to fuzzy matching on record name prefix.
    """
    if record_name in RECORD_COORDS:
        return RECORD_COORDS[record_name]

    # Fuzzy match: try matching prefix
    for key in RECORD_COORDS:
        if record_name.startswith(key) or key.startswith(record_name):
            return RECORD_COORDS[key]

    return None, None


def get_region_centroid(region):
    """Return (lat, lon) for a region name."""
    return REGION_CENTROIDS.get(region, (0.0, 0.0))


def get_region_label(region):
    """Return a human-readable label for a region."""
    return REGION_LABELS.get(region, region.replace("_", " "))


def get_region_color(region):
    """Return the canonical colour for a region."""
    return REGION_COLORS.get(region, "#636363")


def get_all_record_coordinates():
    """Return list of (record_name, lat, lon) for all records with coordinates."""
    return [(name, lat, lon) for name, (lat, lon) in RECORD_COORDS.items()]
