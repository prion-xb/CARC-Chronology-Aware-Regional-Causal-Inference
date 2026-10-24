#!/usr/bin/env python3
"""
31_figure5_escape.py
--------------------
Assembles the new main-paper Figure 5 ("escape diagnosis", Discussion /
Section 5.3) from the CSVs produced by scripts 24, 26, and 29 plus the
real-data robustness tables:

  (a) Age-error scaling (scripts/24): robustness of the three prescribed
      Scenario A edges and of the worst negative control as the published
      1-sigma age-model errors are scaled 0.1x-5x. The vertical line marks
      the published errors (scale 1). The 95% honesty filter is the
      horizontal dashed line; the arrow shows the escape direction
      (improve chronological precision).
  (b) Detectability phase diagram (scripts/26): filled robustness
      contours of NGRIP -> EDC_dD over {causal strength} x {lag/error
      ratio}. The bold 0.95 contour is the empirical escape boundary;
      the three canonical edge positions (strength 0.4, their Table 1
      lag/error ratios) sit deep in the sub-threshold regime.
  (c) Real-data projection: robustness vs the RMS 1-sigma age error of
      the pair. Synthetic reference curves (the three edges at strength
      0.4, from panel (a) data) behind; six-record pairs (Table 4 /
      conditional_causality_allpairs.csv, H1 window emphasized) and
      regional pairs (regional_robustness_windows.csv; pairs in Table 5
      highlighted) overlaid.

Requires the CSVs: age_error_scaling.csv, detectability_phase.csv,
stacking_noise_reduction.csv, conditional_causality_allpairs.csv,
regional_robustness_windows.csv; and the paper docx for Table 5 pairings
(highlight only -- skipped gracefully if unreadable).

Output: Fig8_Escape_Diagnosis.{pdf,png,tif} (+svg via the export wrapper).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import OUTPUT_DIR, DATA_DIR
from src.visualization import apply_style, save_figure, CARC_COLORS
from src.io import load_all_records
from src.stacking import build_region_dict

MAIN_DOCX = os.path.join(os.path.dirname(__file__), '..', 'paper',
                         'CARC_v3_final-ZZF_edited.docx')

# Published 1-sigma errors (Table S3 / scenarioA_final_params.csv)
AGE_ERR = {'NGRIP': 0.65, 'EDC_dD': 0.83, 'CO2': 1.65, 'CH4': 1.66,
           'Cariaco': 0.23, 'Cheng': 0.14}
# CSV record names in conditional_causality_allpairs.csv
CSV_NAME = {'NGRIP': 'NGRIP', 'EDC_dD': 'EDC_dD', 'CO2': 'EDC_CO2',
            'CH4': 'EDC_CH4', 'Cariaco': 'Cariaco', 'Cheng': 'Cheng2016'}

# Six-record classic pairs (Table 4 / S5)
SIX_PAIRS = [('NGRIP', 'EDC_dD'), ('EDC_dD', 'CO2'), ('CO2', 'CH4'),
             ('Cariaco', 'Cheng')]

EDGE_COLORS = {'NGRIP -> EDC_dD': CARC_COLORS['red'],
               'CO2 -> CH4': CARC_COLORS['blue'],
               'Cariaco -> Cheng': CARC_COLORS['green']}
EDGE_LABELS = {'NGRIP -> EDC_dD': 'NGRIP $\\to$ EDC $\\delta$D',
               'CO2 -> CH4': 'CO$_2$ $\\to$ CH$_4$',
               'Cariaco -> Cheng': 'Cariaco $\\to$ Cheng'}
EDGE_RMS_ERR = {'NGRIP -> EDC_dD':
                np.sqrt((AGE_ERR['NGRIP'] ** 2 + AGE_ERR['EDC_dD'] ** 2) / 2),
                'CO2 -> CH4':
                np.sqrt((AGE_ERR['CO2'] ** 2 + AGE_ERR['CH4'] ** 2) / 2),
                'Cariaco -> Cheng':
                np.sqrt((AGE_ERR['Cariaco'] ** 2 + AGE_ERR['Cheng'] ** 2) / 2)}
SIX_RMS_ERR = {(a, b): np.sqrt((AGE_ERR[a] ** 2 + AGE_ERR[b] ** 2) / 2)
               for a, b in SIX_PAIRS}

REGION_NAMES = {
    'North_Atlantic': 'North Atlantic',
    'Antarctic': 'Antarctic/Southern Ocean',
    'Southern_Ocean': 'Antarctic/Southern Ocean',
    'Tropical': 'Tropical',
    'East_Asian_Monsoon': 'East Asian Monsoon',
    'South_America': 'South American Monsoon',
    'Other_Lakes': 'High-latitude Lakes',
}

# Table 5 name variants -> REGION_NAMES spellings
T5_ALIASES = {'Antarctic/SO': 'Antarctic/Southern Ocean',
              'High-Latitude Lakes': 'High-latitude Lakes'}

WINDOW_KEYS = [('H1', 'H1 event'), ('H4', 'H4 event'),
               ('Bølling', 'Bølling-Allerød'),
               ('Last Glacial', 'Last Glacial Maximum'),
               ('GI-12', 'GI 12'), ('GI 12', 'GI 12'),
               ('GS-12', 'GS 12'), ('GS 12', 'GS 12')]


def match_window(w):
    for key, label in WINDOW_KEYS:
        if key.lower() in str(w).lower():
            return label
    return None


def load_table5_highlights():
    """Parse Table 5 pairings from the edited main docx (region pairs with
    >=95% phase consistency). Returns a set of (window, region_A, region_B)
    with canonical names, or an empty set if parsing fails."""
    try:
        import docx
        d = docx.Document(MAIN_DOCX)
        t = d.tables[4]
    except Exception:
        return set()
    out = set()
    for r in t.rows[1:]:
        cells = [c.text.strip() for c in r.cells]
        if len(cells) < 4 or not cells[0]:
            continue
        win = match_window(cells[0])
        pair = cells[1]
        if win is None or '→' not in pair:
            continue
        a, b = [p.strip() for p in pair.split('→')]
        out.add((win, a, b))
    return out


def panel_a(ax, df):
    for edge, c in EDGE_COLORS.items():
        sub = df[df['pair'] == edge].sort_values('error_scale')
        ax.plot(sub['error_scale'], sub['robustness'] * 100, 'o-',
                color=c, lw=2.2, ms=5.5, label=EDGE_LABELS[edge])
    cmax = df[df['pair'] == 'control_max'].sort_values('error_scale')
    cmean = df[df['pair'] == 'control_mean'].sort_values('error_scale')
    ax.fill_between(cmax['error_scale'], cmean['robustness'] * 100,
                    cmax['robustness'] * 100, color=CARC_COLORS['grey'],
                    alpha=0.22, lw=0, label='worst negative control '
                    '(mean-max band)')
    ax.axhline(95, color=CARC_COLORS['red'], lw=1.2, ls='--')
    ax.text(4.4, 97, '95% honesty filter', fontsize=8,
            color=CARC_COLORS['red'], ha='right')
    ax.axvline(1.0, color='0.3', lw=1.0, ls=':')
    ax.text(1.05, 3.5, 'published 1$\\sigma$', fontsize=8,
            color='0.3', rotation=90, va='bottom')
    ax.annotate('', xy=(0.14, 62), xytext=(0.85, 20),
                arrowprops=dict(arrowstyle='->', color=CARC_COLORS['navy'],
                                lw=1.6))
    ax.text(0.17, 66, 'escape: improve\nchronological precision',
            fontsize=8, color=CARC_COLORS['navy'], va='bottom')
    ax.set_xscale('log')
    ax.set_xlabel('Age-error scale factor (1 = published 1$\\sigma$)')
    ax.set_ylabel('Robustness (%)')
    ax.set_ylim(0, 100)
    ax.set_xlim(0.09, 5.5)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.legend(fontsize=7.5, frameon=True, loc='upper left')
    ax.set_title('(a) Escape paths', fontsize=9)


def panel_b(ax, df):
    piv = df.pivot_table(index='causal_strength', columns='lag_error_ratio',
                         values='robustness_focus')
    piv = piv.sort_index(ascending=True)
    piv = piv[sorted(piv.columns, reverse=True)]   # ratio decreases -> right
    X, Y = np.meshgrid(piv.columns.values, piv.index.values)
    Z = piv.values
    cs = ax.contourf(X, Y, Z, levels=[0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0],
                     vmin=0, vmax=1.0, cmap='YlGnBu', extend='both')
    ax.contour(X, Y, Z, levels=[0.95], colors=CARC_COLORS['red'],
               linewidths=2.2, linestyles='-')
    ax.contour(X, Y, Z, levels=[0.05], colors=CARC_COLORS['grey'],
               linewidths=0.8, linestyles=':')
    # canonical edge positions (strength 0.4, Table 1 lag/error ratios)
    ratios = {'NGRIP -> EDC_dD': 1.1, 'CO2 -> CH4': 0.9,
              'Cariaco -> Cheng': 12.1}
    for edge, c in EDGE_COLORS.items():
        ax.plot(ratios[edge], 0.4, 'o', ms=9, mfc=c, mec='white', mew=1.4,
                zorder=5)
    ax.text(0.28, 0.75, 'escape boundary\n(robustness = 95%)',
            fontsize=8, color=CARC_COLORS['red'], va='bottom')
    ax.text(6.2, 0.52, 'sub-threshold\nregime', fontsize=8,
            color='0.25', va='center', ha='center')
    ax.set_xscale('log')
    ax.set_xlabel('Lag / error ratio')
    ax.set_ylabel('Causal strength (injected $\\beta$)')
    ax.set_ylim(0.1, 1.6)
    ax.set_xlim(0.2, 13)
    ax.set_title('(b) Detectability diagram', fontsize=9)
    fig_ = ax.figure
    fig_.colorbar(cs, ax=ax, label='Robustness', fraction=0.046, pad=0.03)


def panel_c(ax, df_scale, six, reg):
    # synthetic reference curves at strength 0.4 (from panel (a) data)
    for edge, c in EDGE_COLORS.items():
        sub = df_scale[df_scale['pair'] == edge].sort_values('error_scale')
        x = EDGE_RMS_ERR[edge] * sub['error_scale']
        ax.plot(x, sub['robustness'] * 100, '-', color=c, lw=1.8,
                alpha=0.55, zorder=2)
        ax.plot(x.iloc[2], sub['robustness'].iloc[2] * 100, 'o', ms=7,
                color=c, zorder=4,
                label=f"{EDGE_LABELS[edge]} (synthetic, $\\beta$=0.4)")
    # six-record real pairs
    for i, ((a, b), xerr) in enumerate(SIX_RMS_ERR.items()):
        na, nb = CSV_NAME[a], CSV_NAME[b]
        sub = six[(six['record_A'] == na) & (six['record_B'] == nb)]
        if sub.empty:
            sub = six[(six['record_A'] == nb) & (six['record_B'] == na)]
            sub = sub.assign(v=np.maximum(sub['A_to_B'], sub['B_to_A']))
        else:
            sub = sub.assign(v=np.maximum(sub['A_to_B'], sub['B_to_A']))
        if sub.empty:
            continue
        ax.plot(np.full(len(sub), xerr), sub['v'] * 100, 'x', ms=6,
                color=CARC_COLORS['navy'], alpha=0.45, zorder=3)
        h1 = sub[sub['window'] == 'H1 event']
        if not h1.empty:
            ax.plot(xerr, h1['v'].iloc[0] * 100, 'D', ms=8,
                    color=CARC_COLORS['navy'], zorder=5)
    # regional pairs
    if reg is not None and not reg.empty:
        other = reg[~reg['t5']]
        if not other.empty:
            ax.plot(other['rms_err'], other['rob'] * 100, '+', ms=7, mew=1.2,
                    color=CARC_COLORS['grey'], alpha=0.5, zorder=3,
                    label='regional pairs (all windows)')
        t5 = reg[reg['t5']]
        if not t5.empty:
            ax.plot(t5['rms_err'], t5['rob'] * 100, 'o', ms=7, mfc='none',
                    mec=CARC_COLORS['orange'], mew=1.5, zorder=5,
                    label='regional pairs in Table 5 ($\\geq$95%)')
    ax.axhline(95, color=CARC_COLORS['red'], lw=1.2, ls='--')
    ax.set_xscale('log')
    ax.set_xlabel('RMS 1$\\sigma$ age error of the pair (ka)')
    ax.set_ylabel('Robustness (%)')
    ax.set_ylim(0, 100)
    ax.set_xlim(0.1, 3.5)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.legend(fontsize=7, frameon=True, loc='upper right')
    ax.set_title('(c) Pairings on the escape plane', fontsize=9)


def build_regional_rows():
    """Regional-pair rows: robustness (max direction, per window) vs RMS
    pair error from member records (10-60 ka), with a flag marking the
    pairs listed in Table 5 (>=95% phase consistency)."""
    rob = pd.read_csv(os.path.join(OUTPUT_DIR,
                                   'regional_robustness_windows.csv'))
    noise = pd.read_csv(os.path.join(OUTPUT_DIR, 'stacking_noise_reduction.csv'))
    if 'rms_age_error_ka' not in noise.columns:
        return None
    rms = {r['region']: r['rms_age_error_ka'] for _, r in noise.iterrows()}
    highlights = set()
    for (w, a, b) in load_table5_highlights():
        highlights.add((w, T5_ALIASES.get(a, a), T5_ALIASES.get(b, b)))
    rows = []
    for _, r in rob.iterrows():
        ra, rb = REGION_NAMES.get(r['region_A']), REGION_NAMES.get(r['region_B'])
        if ra not in rms or rb not in rms:
            continue
        win = match_window(r['window'])
        t5 = win is not None and ((win, ra, rb) in highlights
                                  or (win, rb, ra) in highlights)
        rows.append({'rms_err': float(np.sqrt((rms[ra] ** 2 + rms[rb] ** 2)
                                              / 2.0)),
                     'rob': float(max(r['A_to_B'], r['B_to_A'])),
                     'window': win, 't5': t5})
    return pd.DataFrame(rows) if rows else None


def main():
    apply_style('paper')

    scale = pd.read_csv(os.path.join(OUTPUT_DIR, 'age_error_scaling.csv'))
    phase = pd.read_csv(os.path.join(OUTPUT_DIR, 'detectability_phase.csv'))
    six = pd.read_csv(os.path.join(OUTPUT_DIR,
                                   'conditional_causality_allpairs.csv'))
    reg = build_regional_rows()

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.386))  # 178 mm wide, same aspect
    panel_a(axes[0], scale)
    panel_b(axes[1], phase)
    panel_c(axes[2], scale, six, reg)

    # explicit margins: tight_layout over-pads this row once the colorbar
    # is attached, squeezing each panel to ~1.3 in — too narrow for 9-pt
    # titles at the 178-mm column width
    fig.subplots_adjust(left=0.055, right=0.985, wspace=0.10,
                        bottom=0.155, top=0.885)
    save_figure(fig, 'Fig8_Escape_Diagnosis', fmts=('pdf', 'png', 'tif'))
    plt.close(fig)


if __name__ == '__main__':
    main()
