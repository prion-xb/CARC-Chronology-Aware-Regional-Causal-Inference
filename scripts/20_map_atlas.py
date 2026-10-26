#!/usr/bin/env python3
"""
20_map_atlas.py
---------------
Record-level scrutinization atlas: where each record sits on the map, how it
fares under the CARC honesty filters, which layer (observational noise /
age error / ARC) limits it, and how the three methods (Granger / PCMCI / CCM)
agree.

Inputs (outputs/pairwise_scan/, produced by scripts/19_full_pairwise_scan.py):
  record_profile.csv   per-record pass counts, recovery counts, metadata
  record_meta.csv      sigma / resolution / SNR
  shard_*.csv          per-pair robustness (all variants, dual thresholds)
  pcmci_pairs.csv      PCMCI per-pair results
  ccm_pairs.csv        CCM cross-map skills on Granger-surviving pairs

Outputs:
  figures/FigS_Record_Atlas.png/.pdf         map + layer-impact + thresholds +
                                             method agreement + top strip
  figures/FigS_Record_Strip.png/.pdf         all records, one row each
  figures/record_atlas_interactive.html      plotly map + hover + table view
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.config import OUTPUT_DIR, FIGURE_DIR, STABILITY_THRESHOLD
from src.geolocation import RECORD_COORDS, REGION_COLORS

SCAN_DIR = os.path.join(OUTPUT_DIR, 'pairwise_scan')
THRESH_80 = 0.80

# Project light-mode palette (validated reference)
INK, INK_SEC, INK_MUTED = '#0b0b0b', '#52514e', '#898781'
SURFACE, GRID = '#fcfcfb', '#c3c2b7'
BLUE, BLUE_HI = '#2a78d6', '#104281'
ORANGE, GREEN, AQUA = '#eb6834', '#1baf7a', '#008300'
ACCENT = '#eda100'

# Paper region names (Southern_Ocean merged into Antarctic for hue purposes,
# consistent with scripts 09/17/18)
PAPER_REGIONS = ['North_Atlantic', 'Antarctic', 'Tropical',
                 'East_Asian_Monsoon', 'South_America', 'Other_Lakes']
REGION_HUE = {
    'North_Atlantic': REGION_COLORS['North_Atlantic'],
    'Southern_Ocean': REGION_COLORS['Southern_Ocean'],
    'Antarctic': REGION_COLORS['Antarctic'],
    'Tropical': REGION_COLORS['Tropical'],
    'Tropical_Marine': REGION_COLORS['Tropical'],
    'East_Asian_Monsoon': REGION_COLORS['East_Asian_Monsoon'],
    'South_America': REGION_COLORS['South_America'],
    'South_American_Monsoon': REGION_COLORS['South_America'],
    'Other_Lakes': REGION_COLORS['Other_Lakes'],
    'Lake': REGION_COLORS['Other_Lakes'],
}
REGION_LABEL = {
    'North_Atlantic': 'North Atlantic', 'Southern_Ocean': 'Southern Ocean',
    'Antarctic': 'Antarctic', 'Tropical': 'Tropical', 'Tropical_Marine': 'Tropical',
    'East_Asian_Monsoon': 'E. Asian Monsoon', 'South_America': 'S. Am. Monsoon',
    'South_American_Monsoon': 'S. Am. Monsoon', 'Other_Lakes': 'High-Lat Lakes',
    'Lake': 'High-Lat Lakes',
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
# Explicit record-name -> coordinate mapping for the 80 scan records.
# File stems carry the paper's clean record keys (one file per key, see
# data/region_assignment_80_paper.csv), so each entry maps a stem to the
# RECORD_COORDS entry of the same key; variants of one site share
# coordinates. Kept as an explicit dict because 36_regional_submitted.py
# parses it at runtime for record -> file resolution.
RECORD_SITE = {
    'NGRIP_d18O': RECORD_COORDS['NGRIP_d18O'],
    'ODP983_IRD': RECORD_COORDS['ODP983_IRD'],
    'ODP983_NPS': RECORD_COORDS['ODP983_NPS'],
    'ODP980_planktic_d18O': RECORD_COORDS['ODP980_planktic_d18O'],
    'ODP980_benthic_d18O': RECORD_COORDS['ODP980_benthic_d18O'],
    'U1308_RPI': RECORD_COORDS['U1308_RPI'],
    'U1308_SiSr': RECORD_COORDS['U1308_SiSr'],
    'U1302-3_MS': RECORD_COORDS['U1302-3_MS'],
    'U1302-3_CaSr': RECORD_COORDS['U1302-3_CaSr'],
    'U1302-3_SiSr': RECORD_COORDS['U1302-3_SiSr'],
    'U1302-3_d18O': RECORD_COORDS['U1302-3_d18O'],
    'U1313_SST_2.7Ma': RECORD_COORDS['U1313_SST_2.7Ma'],
    'U1313_SST_0-2.7Ma': RECORD_COORDS['U1313_SST_0-2.7Ma'],
    'EDC_dD': RECORD_COORDS['EDC_dD'],
    'EDC_CO2': RECORD_COORDS['EDC_CO2'],
    'EDC_CH4': RECORD_COORDS['EDC_CH4'],
    'EDC_Dust': RECORD_COORDS['EDC_Dust'],
    'DomeFuji_dD': RECORD_COORDS['DomeFuji_dD'],
    'DomeFuji_d18O': RECORD_COORDS['DomeFuji_d18O'],
    'ODP1090_Dust': RECORD_COORDS['ODP1090_Dust'],
    'ODP1090_FeMAR': RECORD_COORDS['ODP1090_FeMAR'],
    'ODP1094_BaFe': RECORD_COORDS['ODP1094_BaFe'],
    'PS75-059-2_Fe': RECORD_COORDS['PS75-059-2_Fe'],
    'PS75-076-2_Fe': RECORD_COORDS['PS75-076-2_Fe'],
    'PS75-074-3_Fe': RECORD_COORDS['PS75-074-3_Fe'],
    'U1537_NGR': RECORD_COORDS['U1537_NGR'],
    'U1537_MS_AR': RECORD_COORDS['U1537_MS_AR'],
    'MD00-2361_d18O': RECORD_COORDS['MD00-2361_d18O'],
    'MD00-2361_logFeCa': RECORD_COORDS['MD00-2361_logFeCa'],
    'MD00-2361_logZrFe': RECORD_COORDS['MD00-2361_logZrFe'],
    'MD97-2120_d18O': RECORD_COORDS['MD97-2120_d18O'],
    'Cariaco_Mo': RECORD_COORDS['Cariaco_Mo'],
    'U1542_SST': RECORD_COORDS['U1542_SST'],
    'U1542_CHC': RECORD_COORDS['U1542_CHC'],
    'U1385_ZrSr': RECORD_COORDS['U1385_ZrSr'],
    'U1385_CaTi': RECORD_COORDS['U1385_CaTi'],
    'MD01-2443_CaTi': RECORD_COORDS['MD01-2443_CaTi'],
    'MD01-2443_UK37_SST': RECORD_COORDS['MD01-2443_UK37_SST'],
    'U1446_RbCa': RECORD_COORDS['U1446_RbCa'],
    'U1446_d18Osw': RECORD_COORDS['U1446_d18Osw'],
    'MD05-2925_d18O': RECORD_COORDS['MD05-2925_d18O'],
    'MD05-2925_MgCa_SST': RECORD_COORDS['MD05-2925_MgCa_SST'],
    'MD01-2518_DSR': RECORD_COORDS['MD01-2518_DSR'],
    'MD01-2519_DSR': RECORD_COORDS['MD01-2519_DSR'],
    'ODP967_Dust': RECORD_COORDS['ODP967_Dust'],
    'ODP967_WetDryIndex_800ka': RECORD_COORDS['ODP967_WetDryIndex_800ka'],
    'ODP967_WetDryIndex': RECORD_COORDS['ODP967_WetDryIndex'],
    'ODP1089_SST': RECORD_COORDS['ODP1089_SST'],
    'U1483_benthic_d18O': RECORD_COORDS['U1483_benthic_d18O'],
    'U1483_logKCa': RECORD_COORDS['U1483_logKCa'],
    'IPWP_OHC': RECORD_COORDS['IPWP_OHC'],
    'U1476_benthic_d18O': RECORD_COORDS['U1476_benthic_d18O'],
    'KX97322-4_d18O': RECORD_COORDS['KX97322-4_d18O'],
    'KX97322-4_MgCa_SST': RECORD_COORDS['KX97322-4_MgCa_SST'],
    'ODP663_CaCO3': RECORD_COORDS['ODP663_CaCO3'],
    'ODP663_Terrigenous': RECORD_COORDS['ODP663_Terrigenous'],
    'ODP663_bSiO2': RECORD_COORDS['ODP663_bSiO2'],
    'ODP1143_UK37_SST': RECORD_COORDS['ODP1143_UK37_SST'],
    'ODP1146_Al': RECORD_COORDS['ODP1146_Al'],
    'ODP1146_Terrigenous': RECORD_COORDS['ODP1146_Terrigenous'],
    'U1391_planktic_d18O': RECORD_COORDS['U1391_planktic_d18O'],
    'U1391_benthic_d18O': RECORD_COORDS['U1391_benthic_d18O'],
    'Malawi_Ca': RECORD_COORDS['Malawi_Ca'],
    'M125-73-3_lnKAl': RECORD_COORDS['M125-73-3_lnKAl'],
    'M125-73-3_lnTiCa': RECORD_COORDS['M125-73-3_lnTiCa'],
    'Cheng2016_d18O': RECORD_COORDS['Cheng2016_d18O'],
    'Gulang_FeK': RECORD_COORDS['Gulang_FeK'],
    'Luochuan_GT32': RECORD_COORDS['Luochuan_GT32'],
    'Yimaguan_GT32': RECORD_COORDS['Yimaguan_GT32'],
    'CHB14_KZr': RECORD_COORDS['CHB14_KZr'],
    'ZB13_Pollen': RECORD_COORDS['ZB13_Pollen'],
    'Botuvera_d13C': RECORD_COORDS['Botuvera_d13C'],
    'Botuvera_d18O': RECORD_COORDS['Botuvera_d18O'],
    'Titicaca_MS': RECORD_COORDS['Titicaca_MS'],
    'Elgygytgyn_MS': RECORD_COORDS['Elgygytgyn_MS'],
    'Elgygytgyn_TOC': RECORD_COORDS['Elgygytgyn_TOC'],
    'Elgygytgyn_SiTi': RECORD_COORDS['Elgygytgyn_SiTi'],
    'Baikal_BioSilica': RECORD_COORDS['Baikal_BioSilica'],
    'Junin_GI': RECORD_COORDS['Junin_GI'],
    'Ohrid_CaK': RECORD_COORDS['Ohrid_CaK'],
}


def _match_coords(name):
    """Coordinate lookup: exact key, substring fallback, then explicit map."""
    name = str(name).strip()
    if name in RECORD_COORDS:
        return RECORD_COORDS[name]
    best, blen = None, 0
    nl = name.lower()
    for key, coord in RECORD_COORDS.items():
        kl = key.lower()
        if kl in nl or nl in kl:
            if len(kl) > blen:
                best, blen = coord, len(kl)
    if best is not None:
        return best
    if name in RECORD_SITE:
        return RECORD_SITE[name]
    for k, v in RECORD_SITE.items():
        if k.strip() == name:
            return v
    return None


def load_atlas_data():
    prof = pd.read_csv(os.path.join(SCAN_DIR, 'record_profile.csv'))
    meta = pd.read_csv(os.path.join(SCAN_DIR, 'record_meta.csv'))
    extra = [c for c in ['resolution_ka', 'snr_proxy', 'diff_arc']
             if c not in prof.columns]
    if extra:
        prof = prof.merge(meta[['record'] + extra], on='record', how='left')

    # pair rows
    shards = sorted([f for f in os.listdir(SCAN_DIR) if f.startswith('shard_')])
    if shards:
        pairs = pd.concat([pd.read_csv(os.path.join(SCAN_DIR, f))
                           for f in shards], ignore_index=True)
    else:
        pairs = pd.DataFrame()

    # PCMCI per-record degree (lagged links only, tau >= 1)
    pcm_path = os.path.join(SCAN_DIR, 'pcmci_pairs.csv')
    pcm = pd.read_csv(pcm_path) if os.path.exists(pcm_path) else None
    if pcm is not None:
        pcm_dir = pcm[pcm['pcmci_lagged_tau_ge1']]
        deg_out = pcm_dir.groupby('source').size().to_dict()
        deg_in = pcm_dir.groupby('target').size().to_dict()
        prof['pcmci_out'] = prof['record'].map(deg_out).fillna(0).astype(int)
        prof['pcmci_in'] = prof['record'].map(deg_in).fillna(0).astype(int)
    else:
        prof['pcmci_out'] = prof['pcmci_in'] = 0

    # CCM confirmations per record
    ccm_path = os.path.join(SCAN_DIR, 'ccm_pairs.csv')
    ccm = pd.read_csv(ccm_path) if os.path.exists(ccm_path) else None
    if ccm is not None:
        ccm_ok = ccm.dropna(subset=['ccm_fwd', 'ccm_rev'])
        conf = ccm_ok[ccm_ok['ccm_fwd'] > ccm_ok['ccm_rev']]
        ccm_out = conf.groupby('source').size().to_dict()
        prof['ccm_out'] = prof['record'].map(ccm_out).fillna(0).astype(int)
    else:
        prof['ccm_out'] = 0

    prof['coords'] = prof['record'].apply(_match_coords)
    prof['lat'] = prof['coords'].apply(lambda c: c[0] if c else np.nan)
    prof['lon'] = prof['coords'].apply(lambda c: c[1] if c else np.nan)
    prof['region_hue'] = prof['region'].map(REGION_HUE).fillna(INK_MUTED)
    prof['region_label'] = prof['region'].map(REGION_LABEL).fillna('Other')
    prof['any_pass95'] = (prof['n_out_pass95'] + prof['n_in_pass95']) > 0
    prof['any_pass80'] = (prof['n_out_pass80'] + prof['n_in_pass80']) > 0
    prof['max_rate_out'] = prof['max_rate_out'].fillna(0)
    if 'n_trend_leak95' not in prof.columns:
        prof['n_trend_leak95'] = 0
    return prof, pairs, pcm, ccm


# ---------------------------------------------------------------------------
# Figure 1: atlas (map + side panels + top strip)
# ---------------------------------------------------------------------------
def draw_atlas(prof, pairs):
    plt.rcParams.update({'font.size': 9.5, 'font.family': 'Arial',
                         'axes.edgecolor': GRID, 'text.color': INK})
    fig = plt.figure(figsize=(7.0, 5.5), facecolor=SURFACE)  # 178 mm wide, same aspect
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.15],
                          height_ratios=[1.0, 0.72],
                          left=0.05, right=0.97, top=0.90, bottom=0.095,
                          wspace=0.20, hspace=0.42)

    # ---------------- Panel A: map ----------------
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    ax = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson())
    ax.set_facecolor('#eef0ee')
    ax.add_feature(cfeature.LAND, facecolor='#e6e4da', edgecolor='none', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#f4f6f7', edgecolor='none', zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor='#a9a89f', linewidth=0.4, zorder=1)
    gl = ax.gridlines(draw_labels=False, linewidth=0.3, color='#d8d7cf', zorder=1)

    have = prof.dropna(subset=['lat', 'lon'])
    noloc = prof[prof['lat'].isna()]
    # size channel: marker area ~ max outgoing rate (min visible dot at 0)
    s_min, s_max = 9.0, 95.0
    sizes = s_min + (s_max - s_min) * np.sqrt(have['max_rate_out'].clip(0, 1))
    colors = have['region_hue']
    ax.scatter(have['lon'], have['lat'], s=sizes, c=colors, alpha=0.85,
               transform=ccrs.PlateCarree(), edgecolors='none', zorder=3,
               linewidths=0)
    # status encoding on the marker edge (not hue):
    #   solid ink ring   = honest edge at 95%
    #   dashed ring      = honest edge at 80% only
    #   no ring          = no honest edge
    p95 = have[have['any_pass95']]
    if len(p95):
        ax.scatter(p95['lon'], p95['lat'],
                   s=s_min + (s_max - s_min) * np.sqrt(p95['max_rate_out'].clip(0, 1)),
                   facecolors='none', edgecolors=INK, linewidths=1.2,
                   transform=ccrs.PlateCarree(), zorder=4)
    p80 = have[have['any_pass80'] & ~have['any_pass95']]
    if len(p80):
        ax.scatter(p80['lon'], p80['lat'],
                   s=s_min + (s_max - s_min) * np.sqrt(p80['max_rate_out'].clip(0, 1)),
                   facecolors='none', edgecolors=INK, linewidths=0.9,
                   linestyle=(0, (3, 2)),
                   transform=ccrs.PlateCarree(), zorder=4)
    if len(noloc):
        ax.text(0.02, 0.04, f'{len(noloc)} records without coordinates '
                f'(see strip, *)', transform=ax.transAxes, fontsize=7,
                color=INK_MUTED)
    ax.set_title('Record atlas — marker area ∝ best outgoing robustness',
                 fontsize=8.5, color=INK, pad=4)

    # ---------------- Panels B1/B2: layer impact (split) ----------------
    # B1: blue = edges that FAIL at 95% but recover when the layer is removed.
    # B2: orange = edges that PASS at 95% but collapse under forced
    # differencing (shared low-frequency structure, i.e. trend leakage).
    gs_r = gs[0, 1].subgridspec(2, 1, hspace=0.55)
    ax_b1 = fig.add_subplot(gs_r[0])
    ax_b1.set_facecolor(SURFACE)
    layers = [('recov_nonoise', 'observational noise'),
              ('recov_noage', 'age error'),
              ('recov_noarc', 'ARC deconfounding')]
    totals = [int(prof[c].sum()) for c, _ in layers]
    y = np.arange(len(layers))[::-1]
    ax_b1.barh(y, totals, height=0.55,
               color=[BLUE, '#6da7ec', '#9ec5f4'], edgecolor='none')
    ax_b1.set_yticks(y)
    ax_b1.set_yticklabels([lab for _, lab in layers], fontsize=8.5, color=INK_SEC)
    ax_b1.set_xlabel('edges that fail at 95% but recover\nwhen the layer is removed',
                     fontsize=8, color=INK_MUTED)
    for yi, t in zip(y, totals):
        ax_b1.text(t, yi, f'  {t}', va='center', fontsize=8.5, color=INK)
    ax_b1.set_title('Which layer drives scrutiny?', fontsize=10,
                    fontweight='bold', color=INK, pad=4)
    ax_b1.tick_params(colors=INK_MUTED, labelsize=8)
    ax_b1.grid(axis='x', color='#e1e0d9', lw=0.5)
    for s in ['top', 'right']:
        ax_b1.spines[s].set_visible(False)

    ax_b2 = fig.add_subplot(gs_r[1])
    ax_b2.set_facecolor(SURFACE)
    n_leak = int(prof['n_trend_leak95'].sum())
    ax_b2.barh([0], [n_leak], height=0.45, color=ORANGE, edgecolor='none')
    ax_b2.text(n_leak, 0, f'  {n_leak}', va='center', fontsize=8.5, color=INK)
    ax_b2.set_yticks([0])
    ax_b2.set_yticklabels(['shared low-freq.\ntrend'], fontsize=8.5, color=INK_SEC)
    ax_b2.set_xlabel('edges that pass at 95% but collapse\nunder forced differencing',
                     fontsize=8, color=INK_MUTED)
    ax_b2.set_title('Trend-leak check', fontsize=10, fontweight='bold',
                    color=INK, pad=4)
    ax_b2.tick_params(colors=INK_MUTED, labelsize=8)
    ax_b2.grid(axis='x', color='#e1e0d9', lw=0.5)
    for s in ['top', 'right']:
        ax_b2.spines[s].set_visible(False)

    # ---------------- Panels B3/B4: thresholds + method agreement ----------------
    gs_b = gs[1, 1].subgridspec(1, 2, wspace=0.6)
    ax_b3 = fig.add_subplot(gs_b[0])
    ax_b3.set_facecolor(SURFACE)
    n95 = int(((pairs['pass95_full']).sum()) if len(pairs) else 0)
    n80 = int((pairs['pass80_full']).sum()) if len(pairs) else 0
    bars = ax_b3.bar([0, 1], [n95, n80], width=0.5,
                     color=[BLUE_HI, BLUE], edgecolor='none')
    ax_b3.set_xticks([0, 1])
    ax_b3.set_xticklabels(['95% honesty\nthreshold', '80% (relaxed)'],
                          fontsize=8.5, color=INK_SEC)
    ax_b3.set_ylabel('honest directed edges', fontsize=8, color=INK_MUTED)
    for b, v in zip(bars, [n95, n80]):
        # value inside the bar: above the bar it collided with the panel
        # title when the bar is tall (layout QA: text_overlap)
        ax_b3.text(b.get_x() + b.get_width() / 2, v / 2, f'{v}', ha='center',
                   va='center', fontsize=9, color='white', fontweight='bold')
    ax_b3.set_title('Threshold relaxation', fontsize=8.5, fontweight='bold',
                    color=INK, pad=4)
    ax_b3.tick_params(colors=INK_MUTED, labelsize=8)
    ax_b3.grid(axis='y', color='#e1e0d9', lw=0.5)
    for s in ['top', 'right']:
        ax_b3.spines[s].set_visible(False)

    # ---------------- Panel B4: method agreement ----------------
    ax_b4 = fig.add_subplot(gs_b[1])
    ax_b4.set_facecolor(SURFACE)
    # links reported per method (PCMCI: lagged tau>=1; CCM: fwd > rev among
    # the top-400 Granger-surviving pairs actually cross-mapped)
    n_gr = n95
    n_pcm = int(prof['pcmci_out'].sum())
    n_ccm = int(prof['ccm_out'].sum())
    bars4 = ax_b4.bar([0, 1, 2], [n_gr, n_pcm, n_ccm], width=0.5,
                      color=[BLUE_HI, AQUA, ORANGE], edgecolor='none')
    ax_b4.set_xticks([0, 1, 2])
    ax_b4.set_xticklabels(['Granger\n(95%)', 'PCMCI\n(lagged)', 'CCM\n(fwd>rev)*'],
                          fontsize=8, color=INK_SEC)
    ax_b4.set_ylabel('directed links', fontsize=8, color=INK_MUTED)
    for b, v in zip(bars4, [n_gr, n_pcm, n_ccm]):
        # value inside the bar: above the bar it collided with the panel
        # title when the bar is tall (layout QA: text_overlap)
        ax_b4.text(b.get_x() + b.get_width() / 2, v / 2, f'{v}', ha='center',
                   va='center', fontsize=9, color='white', fontweight='bold')
    ax_b4.set_title('Method agreement', fontsize=8.5, fontweight='bold',
                    color=INK, pad=4)
    ax_b4.tick_params(colors=INK_MUTED, labelsize=8)
    ax_b4.grid(axis='y', color='#e1e0d9', lw=0.5)
    for s in ['top', 'right']:
        ax_b4.spines[s].set_visible(False)
    # CCM footnote moved to the bottom caption band (layout QA)

    # ---------------- Bottom strip: top records (label column + bars) -------
    gs_strip = gs[1, 0].subgridspec(1, 2, width_ratios=[0.9, 1.0], wspace=0.02)
    ax_lab = fig.add_subplot(gs_strip[0])
    ax_strip = fig.add_subplot(gs_strip[1])
    ax_lab.set_facecolor(SURFACE)
    ax_strip.set_facecolor(SURFACE)
    top = prof.sort_values('max_rate_out', ascending=False).head(15)
    y = np.arange(len(top))[::-1]
    ax_strip.barh(y, top['max_rate_out'], height=0.62,
                  color=[c for c in top['region_hue']], edgecolor='none', alpha=0.85)
    # Defect letters, colour-coded: blue family = recovered by layer removal,
    # orange = trend leak. n = noise, a = age error, r = ARC, t = trend.
    DEFECTS = [('n', 'recov_nonoise', BLUE), ('a', 'recov_noage', '#6da7ec'),
               ('r', 'recov_noarc', '#9ec5f4'), ('t', 'n_trend_leak95', ORANGE)]
    for yi, (_, r) in zip(y, top.iterrows()):
        rec_mark = ''
        if r['any_pass95']:
            rec_mark = '●'
        elif r['any_pass80']:
            rec_mark = '◌'
        lab = f"{r['record'][:22]} {rec_mark}"
        # label lives in its own narrow column — never spills off-canvas
        ax_lab.text(1.0, yi, lab, ha='right', va='center', fontsize=7,
                    color=INK_SEC)
        x_def = max(r['max_rate_out'], 0) + 0.02
        for c, coldef, colr in DEFECTS:
            if r[coldef] > 0:
                ax_strip.text(x_def, yi, c, ha='left', va='center', fontsize=8,
                              color=colr, fontweight='bold')
                x_def += 0.06
    ax_lab.set_xlim(0, 1)
    ax_lab.set_ylim(-0.5, len(top) - 0.5)
    ax_lab.axis('off')
    ax_strip.set_xlim(0, 1.3)
    ax_strip.set_ylim(-0.5, len(top) - 0.5)
    ax_strip.set_yticks([])
    ax_strip.axvline(STABILITY_THRESHOLD, color=GRID, lw=0.8, ls=(0, (5, 3)))
    ax_strip.axvline(THRESH_80, color=GRID, lw=0.6, ls=(0, (1, 4)))
    ax_strip.set_xlabel('best outgoing robustness', fontsize=8.5,
                        color=INK_MUTED)
    ax_strip.set_title('Top 15 records', fontsize=10, fontweight='bold',
                       color=INK, pad=4)
    ax_strip.tick_params(colors=INK_MUTED, labelsize=8)
    ax_strip.grid(axis='x', color='#e1e0d9', lw=0.5)
    for s in ['top', 'right']:
        ax_strip.spines[s].set_visible(False)

    # legend for the map
    handles = [Patch(facecolor=c, alpha=0.85, label=REGION_LABEL[r])
               for r, c in REGION_HUE.items()
               if r in PAPER_REGIONS or r == 'Southern_Ocean']
    handles += [
        Line2D([], [], marker='o', linestyle='none', mfc='none', mec=INK,
               ms=7, label='honest edge at 95%'),
        Line2D([], [], marker='o', linestyle='none', mfc='none', mec=INK,
               ms=7, ls=(0, (3, 2)), label='honest edge at 80% only'),
    ]
    # legend inside the map panel (lower-left ocean): figure-level legends
    # at the bottom collided with the strip axes and the caption (layout QA)
    ax.legend(handles=handles, loc='lower left', ncol=2, frameon=False,
              fontsize=7, handlelength=1.2, labelspacing=0.5,
              columnspacing=0.8)

    # strip encoding explanation lives in its own caption band below the
    # panels (bottom=0.095), clear of every axes and the xlabels
    fig.text(0.03, 0.055,
             '● = honest at 95%, ◌ = 80% only · n/a/r = recovered by dropping '
             'noise/age-error/ARC · t = trend-leak',
             fontsize=7.5, color=INK_SEC, ha='left', va='bottom')
    fig.text(0.03, 0.033,
             '*CCM mapped only the top-400 Granger-surviving pairs (48%)',
             fontsize=7, color=INK_MUTED, ha='left', va='bottom')

    fig.text(0.05, 0.955, 'CARC record-level scrutinization atlas '
             '(10–60 ka, all-pairs)',
             fontsize=13, fontweight='bold', color=INK)

    # route through save_figure: the direct savefig path left the PNG in
    # RGBA, which journals reject on preflight
    from src.visualization import save_figure
    save_figure(fig, 'FigS_Record_Atlas', fmts=('pdf', 'png'), dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: full record strip (all records)
# ---------------------------------------------------------------------------
def draw_strip(prof):
    # Strip groups: region-contiguous blocks packed into three side-by-side
    # horizontal strips (NA + Southern Ocean + Antarctic | Tropical |
    # E. Asian + S. Am. + Lakes). Row pitch is fixed at 33 px @ 300 dpi by
    # sizing the strip height from the longest strip (34 rows), so 7-pt
    # record names always clear each other at the 178-mm print width
    # (layout QA: text_overlap).
    GROUP = {'North_Atlantic': 0, 'Southern_Ocean': 0, 'Antarctic': 0,
             'Tropical': 1,
             'East_Asian_Monsoon': 2, 'South_America': 2,
             'Other_Lakes': 2}
    GROUP_ORDER = [('North_Atlantic', 'Southern_Ocean', 'Antarctic'),
                   ('Tropical',),
                   ('East_Asian_Monsoon', 'South_America', 'Other_Lakes')]
    GROUP_TITLE = {0: 'North Atlantic + Antarctic/SO',
                   1: 'Tropical',
                   2: 'E. Asian + S. Am. + Lakes'}
    prof = prof.copy()
    strips = []
    for regions in GROUP_ORDER:
        sub = (prof[prof['region'].isin(regions)]
               .sort_values('max_rate_out', ascending=False))
        strips.append(sub)
    rows_max = max(len(s) for s in strips)

    plt.rcParams.update({'font.size': 8, 'font.family': 'Arial',
                         'axes.edgecolor': GRID, 'text.color': INK})
    # strip height = rows_max x 33 px @ 300 dpi; +0.94 in for titles,
    # xlabels and margins -> 178 mm wide figure, taller only as rows demand
    strip_h = 0.11 * rows_max
    fig_h = strip_h + 0.94
    fig = plt.figure(figsize=(7.0, fig_h), facecolor=SURFACE)
    h_frac = strip_h / fig_h
    x0, gap = 0.025, 0.02
    w = (0.985 - x0 - 2 * gap) / 3.0

    for si, sub in enumerate(strips):
        ax = fig.add_axes([x0 + si * (w + gap), 0.055, w, h_frac])
        ax.set_facecolor(SURFACE)
        n = len(sub)
        yy = np.arange(n)[::-1]
        ax.barh(yy, sub['max_rate_out'], height=0.72,
                color=sub['region_hue'], edgecolor='none', alpha=0.85)
        # 7-pt Consolas labels: status mark + no-coordinate star + name
        # (21 chars), right-aligned against the bar zone start (x = -0.05);
        # the negative-x label zone is sized to the widest label (layout QA)
        for yi, (_, r) in zip(yy, sub.iterrows()):
            mark = '●' if r['any_pass95'] else ('◌' if r['any_pass80'] else '○')
            noloc = '*' if pd.isna(r['lat']) else ' '
            lab = f"{mark}{noloc} {r['record'][:21]}"
            ax.text(-0.05, yi, lab, ha='right', va='center', fontsize=7,
                    color=INK_SEC, family='Consolas')
        ax.set_xlim(-2.42, 1.12)
        ax.set_ylim(-0.5, n - 0.3)
        ax.set_yticks([])
        ax.axvline(STABILITY_THRESHOLD, color=GRID, lw=0.8, ls=(0, (5, 3)))
        ax.axvline(THRESH_80, color=GRID, lw=0.6, ls=(0, (1, 4)))
        ax.set_xlabel('best outgoing robustness', fontsize=8,
                      color=INK_MUTED)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(colors=INK_MUTED, labelsize=7)
        ax.grid(axis='x', color='#e1e0d9', lw=0.5)
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)
        # group title at the top-left corner of each strip (above the axes,
        # below the figure subtitle — no collision by construction)
        ax.text(0.01, 1.0, GROUP_TITLE[si], transform=ax.transAxes,
                ha='left', va='bottom', fontsize=7.5, fontweight='bold',
                color=INK)

    fig.text(0.5, 0.975,
             'All records — best outgoing robustness and honest-edge status',
             fontsize=10, fontweight='bold', color=INK, ha='center')
    fig.text(0.5, 0.938,
             '(● = honest at 95%, ◌ = 80% only, ○ = none; * = no coordinates)',
             fontsize=8, color=INK_SEC, ha='center')

    # one-row region colour key between the subtitle and the strips
    key_order = ['North_Atlantic', 'Southern_Ocean', 'Antarctic', 'Tropical',
                 'East_Asian_Monsoon', 'South_America', 'Other_Lakes']
    handles = [Patch(facecolor=REGION_HUE[k], alpha=0.85,
                     label=REGION_LABEL[k]) for k in key_order]
    fig.legend(handles=handles, loc='upper center',
               bbox_to_anchor=(0.5, 0.905), ncol=7, frameon=False,
               fontsize=7.5, handlelength=1.0, columnspacing=0.9)

    # route through save_figure: the direct savefig path left the PNG in
    # RGBA, which journals reject on preflight
    from src.visualization import save_figure
    save_figure(fig, 'FigS_Record_Strip', fmts=('pdf', 'png'), dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Interactive HTML (plotly + table view)
# ---------------------------------------------------------------------------
def draw_interactive(prof):
    import plotly.graph_objects as go

    have = prof.dropna(subset=['lat', 'lon'])
    text = [
        f"<b>{r['record']}</b><br>region: {r['region_label']}<br>"
        f"σ = {r['sigma_ka']:.2f} ka · resolution ≈ {r['resolution_ka']:.1f} ka<br>"
        f"best out: {r['max_rate_out']:.2f} · mean out {r['mean_rate_out']:.2f} / "
        f"in {r['mean_rate_in']:.2f}<br>"
        f"honest 95%: {int(r['n_out_pass95'])} out / {int(r['n_in_pass95'])} in · "
        f"80%: {int(r['n_out_pass80'])}/{int(r['n_in_pass80'])}<br>"
        f"recovered by dropping noise {int(r['recov_nonoise'])} · "
        f"age-error {int(r['recov_noage'])} · ARC {int(r['recov_noarc'])}<br>"
        f"trend-leak (95% pass that collapses under differencing): "
        f"{int(r['n_trend_leak95'])}<br>"
        f"PCMCI deg: {int(r['pcmci_out'])} out / {int(r['pcmci_in'])} in · "
        f"CCM-confirmed out: {int(r['ccm_out'])}"
        for _, r in have.iterrows()]

    fig = go.Figure(go.Scattergeo(
        lon=have['lon'], lat=have['lat'],
        mode='markers',
        marker=dict(
            size=6 + 14 * np.sqrt(have['max_rate_out'].clip(0, 1)),
            color=have['region_hue'],
            opacity=0.85,
            line=dict(width=1.4 if True else 0,
                      color=['#0b0b0b' if p else 'rgba(0,0,0,0)'
                             for p in have['any_pass95']]),
        ),
        text=text, hoverinfo='text',
    ))
    fig.update_geos(projection_type='natural earth', showland=True,
                    landcolor='#e6e4da', oceancolor='#f4f6f7',
                    showcoastlines=True, coastlinecolor='#a9a89f',
                    showframe=False)
    fig.update_layout(
        paper_bgcolor='#fcfcfb', plot_bgcolor='#fcfcfb',
        font=dict(family='system-ui, Segoe UI, sans-serif', color='#0b0b0b',
                  size=12),
        title=dict(text='CARC record-level scrutinization atlas '
                        '(marker size ∝ best outgoing robustness; '
                        'dark ring = honest at 95%)', x=0.05),
        margin=dict(l=20, r=20, t=60, b=20),
    )

    # table view (accessibility / details)
    cols = ['record', 'region', 'sigma_ka', 'resolution_ka',
            'max_rate_out', 'n_out_pass95', 'n_in_pass95',
            'n_out_pass80', 'n_in_pass80', 'recov_nonoise', 'recov_noage',
            'recov_noarc', 'n_trend_leak95', 'pcmci_out', 'pcmci_in', 'ccm_out']
    tbl = prof[cols].sort_values('max_rate_out', ascending=False)
    tbl = tbl.rename(columns={c: c.replace('_', ' ') for c in cols})
    html_tbl = tbl.to_html(index=False, float_format='%.2f', border=0,
                           classes='tbl')

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>CARC record scrutinization atlas</title>
<style>
 body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
        background: #f9f9f7; color: #0b0b0b; margin: 16px; }}
 h1 {{ font-size: 18px; }}
 details {{ margin-top: 18px; }}
 .tbl {{ border-collapse: collapse; font-size: 12px; }}
 .tbl th, .tbl td {{ border-bottom: 1px solid #e1e0d9; padding: 3px 8px;
   text-align: right; }}
 .tbl th {{ position: sticky; top: 0; background: #fcfcfb; color: #52514e; }}
 .tbl td:first-child, .tbl th:first-child {{ text-align: left; }}
</style></head>
<body>
<h1>CARC record-level scrutinization atlas (10–60 ka)</h1>
{fig.to_html(full_html=False, include_plotlyjs='cdn')}
<details><summary>Table view — all {len(tbl)} records (click to expand)</summary>
{html_tbl}
</details>
</body></html>"""

    path = os.path.join(FIGURE_DIR, 'record_atlas_interactive.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
def main():
    prof, pairs, pcm, ccm = load_atlas_data()
    print(f"Records: {len(prof)} | with coordinates: {int(prof['lat'].notna().sum())}")
    if len(pairs):
        n95 = int(pairs['pass95_full'].sum())
        n80 = int(pairs['pass80_full'].sum())
        print(f"Honest directed edges: {n95} at 95%, {n80} at 80%")
        print(f"Recovered edges by layer: noise {int(prof['recov_nonoise'].sum())}, "
              f"age-error {int(prof['recov_noage'].sum())}, "
              f"ARC {int(prof['recov_noarc'].sum())}")
    draw_atlas(prof, pairs)
    draw_strip(prof)
    draw_interactive(prof)
    print("Done.")


if __name__ == "__main__":
    main()
