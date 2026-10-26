"""
Standardized display names for the 80-record 10-60 ka dataset.

Record identifiers used inside the pipeline are repository file stems,
which carry the same scientific names as the clean record keys of
data/region_assignment_80_paper.csv (e.g. ``MD00-2361_logFeCa``); paper
figures and Table S6 use the standardized short names defined here
(e.g. ``MD00-2361 log Fe/Ca``), keyed through those keys (the same
naming as the paper's Section 4.3 and Table S2 inventory).
"""

import os

import pandas as pd

# project-relative path (src/ sits one level below the project root)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ASSIGNMENT_CSV = os.path.join(_ROOT, 'data', 'region_assignment_80_paper.csv')

# clean key -> standardized display name
DISPLAY_NAMES = {
    # North Atlantic
    'NGRIP_d18O': 'NGRIP δ¹⁸O',
    'ODP983_IRD': 'ODP 983 IRD',
    'ODP983_NPS': 'ODP 983 NPS',
    'ODP980_planktic_d18O': 'ODP 980 planktic δ¹⁸O',
    'ODP980_benthic_d18O': 'ODP 980 benthic δ¹⁸O',
    'U1308_RPI': 'U1308 RPI',
    'U1308_SiSr': 'U1308 Si/Sr',
    'U1302-3_MS': 'U1302-U1303 MS',
    'U1302-3_CaSr': 'U1302-U1303 Ca/Sr',
    'U1302-3_SiSr': 'U1302-U1303 Si/Sr',
    'U1302-3_d18O': 'U1302-U1303 δ¹⁸O',
    'U1313_SST_2.7Ma': 'U1313 SST (2.7 Ma)',
    'U1313_SST_0-2.7Ma': 'U1313 SST (0–2.7 Ma)',
    # Antarctic / Southern Ocean
    'EDC_dD': 'EDC δD',
    'EDC_CO2': 'EDC CO₂',
    'EDC_CH4': 'EDC CH₄',
    'EDC_Dust': 'EDC dust',
    'DomeFuji_dD': 'Dome Fuji δD',
    'DomeFuji_d18O': 'Dome Fuji δ¹⁸O',
    'ODP1090_Dust': 'ODP 1090 dust',
    'ODP1090_FeMAR': 'ODP 1090 Fe MAR',
    'ODP1094_BaFe': 'ODP 1094 Ba/Fe',
    'PS75-059-2_Fe': 'PS75/059-2 Fe',
    'PS75-076-2_Fe': 'PS75/076-2 Fe',
    'PS75-074-3_Fe': 'PS75/074-3 Fe',
    'U1537_NGR': 'U1537 NGR',
    'U1537_MS_AR': 'U1537 MS',
    'MD00-2361_d18O': 'MD00-2361 δ¹⁸O',
    'MD00-2361_logFeCa': 'MD00-2361 log Fe/Ca',
    'MD00-2361_logZrFe': 'MD00-2361 log Zr/Fe',
    'MD97-2120_d18O': 'MD97-2120 δ¹⁸O',
    # Tropical
    'Cariaco_Mo': 'Cariaco Mo',
    'U1542_SST': 'U1542 SST',
    'U1542_CHC': 'U1542 CHC',
    'U1385_ZrSr': 'U1385 Zr/Sr',
    'U1385_CaTi': 'U1385 Ca/Ti',
    'MD01-2443_CaTi': 'MD01-2443 Ca/Ti',
    'MD01-2443_UK37_SST': 'MD01-2443 alkenone SST',
    'U1446_RbCa': 'U1446 Rb/Ca',
    'U1446_d18Osw': 'U1446 δ¹⁸Osw',
    'MD05-2925_d18O': 'MD05-2925 δ¹⁸O',
    'MD05-2925_MgCa_SST': 'MD05-2925 Mg/Ca SST',
    'MD01-2518_DSR': 'MD01-2518 DSR',
    'MD01-2519_DSR': 'MD01-2519 DSR',
    'ODP967_Dust': 'ODP 967 Ba/Al',
    'ODP967_WetDryIndex': 'ODP 967 wet–dry',
    'ODP967_WetDryIndex_800ka': 'ODP 967 wet–dry (800 ka)',
    'ODP1089_SST': 'ODP 1089 SST',
    'U1483_benthic_d18O': 'U1483 benthic δ¹⁸O',
    'U1483_logKCa': 'U1483 log K/Ca',
    'IPWP_OHC': 'IPWP OHC-stack',
    'U1476_benthic_d18O': 'U1476 benthic δ¹⁸O',
    'KX97322-4_d18O': 'KX97322-4 δ¹⁸O',
    'KX97322-4_MgCa_SST': 'KX97322-4 Mg/Ca SST',
    'ODP663_CaCO3': 'ODP 663 CaCO₃',
    'ODP663_Terrigenous': 'ODP 663 terrigenous',
    'ODP663_bSiO2': 'ODP 663 bSiO₂',
    'ODP1143_UK37_SST': 'ODP 1143 UK′37 SST',
    'ODP1146_Al': 'ODP 1146 Al',
    'ODP1146_Terrigenous': 'ODP 1146 terrigenous',
    'U1391_planktic_d18O': 'U1391 planktic δ¹⁸O',
    'U1391_benthic_d18O': 'U1391 benthic δ¹⁸O',
    'Malawi_Ca': 'Lake Malawi Ca',
    'M125-73-3_lnKAl': 'M125-73-3 ln K/Al',
    'M125-73-3_lnTiCa': 'M125-73-3 ln Ti/Ca',
    # East Asian Monsoon
    'Cheng2016_d18O': 'Cheng 2016 δ¹⁸O',
    'Gulang_FeK': 'Gulang Fe/K',
    'Luochuan_GT32': 'Luochuan GT32',
    'Yimaguan_GT32': 'Yimaguan GT32',
    'CHB14_KZr': 'CHB14 K/Zr',
    'ZB13_Pollen': 'ZB13 pollen',
    # South America
    'Botuvera_d13C': 'Botuvera δ¹³C',
    'Botuvera_d18O': 'Botuvera δ¹⁸O',
    'Titicaca_MS': 'Titicaca MS',
    # High-Latitude Lakes
    'Elgygytgyn_MS': "El'gygytgyn MS",
    'Elgygytgyn_TOC': "El'gygytgyn TOC",
    'Elgygytgyn_SiTi': "El'gygytgyn Si/Ti",
    'Baikal_BioSilica': 'Baikal bioSi',
    'Junin_GI': 'Lake Junín GI',
    'Ohrid_CaK': 'Ohrid Ca/K',
}


def _raw_to_display():
    """Map repository file stem -> display name via the paper's clean
    record keys (data/region_assignment_80_paper.csv)."""
    m = pd.read_csv(_ASSIGNMENT_CSV, encoding='utf-8-sig')
    table = {}
    for clean, ids in zip(m['Record'], m['S2_File_IDs'].astype(str)):
        for raw in ids.split(';'):
            raw = raw.strip()
            if raw:
                table[raw] = DISPLAY_NAMES.get(clean, clean)
    return table


_RAW_TO_DISPLAY = _raw_to_display()


def display_name(record_id):
    """Standardized display name for a pipeline record identifier."""
    return _RAW_TO_DISPLAY.get(str(record_id).strip(), str(record_id))
