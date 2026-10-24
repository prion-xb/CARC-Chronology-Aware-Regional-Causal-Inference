#!/usr/bin/env python3
"""
41_scrutinization_figure.py
---------------------------
Main-text Fig. 5 — Global record-level attribution scrutiny.

Summarises the all-pairs CARC scan of scripts/19_full_pairwise_scan.py
(outputs/pairwise_scan/record_profile.csv) for the 80 records of the
10-60 ka dataset:

  (a) 80-record strip: raw 95%-honesty passes per record. The forced-
      difference honesty check (alldiff variant) leaves 0 edges in every
      record — every raw pass is trend leakage. Record labels use the
      standardized display names of src/record_names.py (Table S6).
  (b) Six regional composites (as in the main stacking): mean raw passes,
      mean trend-leakage edges, and mean edges recovered when age errors
      are removed or ARC deconfounding is disabled.
  (c) left: ablation ladder — mean recovered edges per record when one
      layer (observational noise / age error / ARC) is removed; right:
      CCM cross-map skill forward vs reverse for the 400 Granger-surviving
      pairs — symmetry implies common forcing rather than directionality.

Outputs:
  figures/Fig5_Global_Attribution_Scrutiny.{png,pdf} (+ layered .svg via
  scripts/export_layered_svg.py — semantic layers for Illustrator)
  outputs/table_s6_scrutiny_profile.csv  (Appendix Table S6, 80 rows)
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.config import OUTPUT_DIR, FIGURE_DIR
from src.record_names import display_name

SCAN_DIR = os.path.join(OUTPUT_DIR, 'pairwise_scan')
TIER_CSV = os.path.join(OUTPUT_DIR, 'data_tiers_final.csv')

# Project light-mode palette (validated reference, as in scripts/20_map_atlas.py)
INK, INK_SEC, INK_MUTED = '#0b0b0b', '#52514e', '#898781'
SURFACE, GRID = '#fcfcfb', '#c3c2b7'
BLUE, BLUE_HI = '#2a78d6', '#104281'
ORANGE, GREEN, AQUA, ACCENT = '#eb6834', '#1baf7a', '#008300', '#eda100'
TIER_FILL = {'Tier 1': '#D73027', 'Tier 2': '#E66101', 'Tier 3': '#E8B800'}

# Six paper composites (Southern_Ocean merged into Antarctic, as in §4.3)
REGION_ORDER = ['North Atlantic', 'Antarctic/Southern Ocean', 'Tropical',
                'East Asian Monsoon', 'South American Monsoon',
                'High-Latitude Lakes']
REGION_KEY = {
    'North_Atlantic': 'North Atlantic',
    'Antarctic': 'Antarctic/Southern Ocean',
    'Southern_Ocean': 'Antarctic/Southern Ocean',
    'Tropical': 'Tropical', 'Tropical_Marine': 'Tropical',
    'East_Asian_Monsoon': 'East Asian Monsoon',
    'South_America': 'South American Monsoon',
    'South_American_Monsoon': 'South American Monsoon',
    'Other_Lakes': 'High-Latitude Lakes', 'Lake': 'High-Latitude Lakes',
}
REGION_COLOR = {
    'North Atlantic': '#D73027', 'Antarctic/Southern Ocean': '#2166AC',
    'Tropical': '#E66101', 'East Asian Monsoon': '#1B7837',
    'South American Monsoon': '#762A83', 'High-Latitude Lakes': '#008080',
}

plt.rcParams.update({
    'font.size': 9,
    # Arial first: Illustrator-native (the layered SVG keeps text editable)
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'sans-serif'],
    'axes.facecolor': SURFACE, 'figure.facecolor': SURFACE,
    'axes.edgecolor': INK_SEC, 'axes.labelcolor': INK,
    'xtick.color': INK_SEC, 'ytick.color': INK_SEC,
    'text.color': INK, 'svg.fonttype': 'none',
})


def _short(name, n=34):
    s = str(name)
    return s if len(s) <= n else s[:n - 1] + '…'


def _region_short(r):
    return {'North Atlantic': 'North Atlantic',
            'Antarctic/Southern Ocean': 'Antarctic/SO',
            'Tropical': 'Tropical',
            'East Asian Monsoon': 'E. Asian',
            'South American Monsoon': 'S. Am.',
            'High-Latitude Lakes': 'High-Lat Lakes'}[r]


def load_data():
    prof = pd.read_csv(os.path.join(SCAN_DIR, 'record_profile.csv'))
    tier = pd.read_csv(TIER_CSV)
    prof = prof.merge(tier[['Record', 'Tier', 'Type']],
                      left_on='record', right_on='Record', how='left')
    assert prof['Tier'].notna().all(), 'tier merge incomplete'
    prof['region6'] = prof['region'].map(REGION_KEY)
    prof['total_pass95'] = prof['n_out_pass95'] + prof['n_in_pass95']
    prof['trend_leak'] = prof['n_trend_leak95'].clip(upper=prof['total_pass95'])
    prof['shortlag'] = prof['total_pass95'] - prof['trend_leak']
    ccm = pd.read_csv(os.path.join(SCAN_DIR, 'ccm_pairs.csv'))
    return prof, ccm


def _region_sort_key(r):
    return REGION_ORDER.index(r)


def panel_a(ax1, ax2, ax3, prof):
    """80-record strip: raw 95% passes (all of which are trend leakage —
    the forced-difference honesty check leaves 0 edges in every record).
    Three side-by-side horizontal-row strips, region-contiguous (NA +
    Antarctic | Tropical | E. Asian + S. Am. + Lakes): 34 rows is the
    longest strip, giving the 7-pt tier-coloured record names a ~13-px
    row pitch at 300 dpi (a 7-pt text box is ~10 px) — the original
    80-row single strip would need ~10 in of height at the 178-mm print
    width."""
    prof = prof.copy()
    prof['_rk'] = prof['region6'].map(_region_sort_key)
    prof['_disp'] = prof['record'].map(display_name)
    prof = prof.sort_values(['_rk', 'total_pass95', '_disp'],
                            ascending=[True, False, True])
    x = np.arange(len(prof))
    y = prof['trend_leak'].to_numpy()
    ytop = max(y.max() * 1.25, 40.0)
    # row boundaries: greedy over the region sequence, each strip <= 35
    # records (largest region, Tropical, has 34)
    uniq = list(dict.fromkeys(prof['region6'].tolist()))
    bounds, acc = [], 0
    for r in uniq:
        n = int((prof['region6'] == r).sum())
        if acc + n > 35 and acc:
            bounds.append(acc)
            acc = 0
        acc += n
    bounds.append(acc)
    # bounds are row lengths (acc resets), so strips are the cumulative spans
    cum = np.cumsum([0] + bounds)
    rows = list(zip(cum[:-1], cum[1:]))

    for ax, (xa, xb) in zip((ax1, ax2, ax3), rows):
        idx = np.arange(xa, xb)
        n_rows = len(idx)
        yy = np.arange(n_rows)[::-1]          # top row first
        ax.barh(yy, y[idx], height=0.72, color=ORANGE, edgecolor='none')
        for n, patch in enumerate(ax.patches):
            patch.set_gid(f'layer-bars-{xa + n:03d}')
        # horizontal record labels, tier colours (left of the bars)
        for yi, xi in zip(yy, idx):
            row = prof.iloc[xi]
            ax.text(-0.9, yi, _short(row['_disp'], n=12), ha='right',
                    va='center', fontsize=7,
                    color=TIER_FILL.get(row['Tier'], INK),
                    gid=f'layer-recordlabels-{xi:03d}')
        # region separators + names at the right edge of each block
        prev, n_reg = None, 0
        for yi, xi in zip(yy, idx):
            row = prof.iloc[xi]
            if row['region6'] != prev:
                ax.axhline(yi + 0.55, color=INK_MUTED, lw=0.7, zorder=1,
                           gid=f'layer-regions-{n_reg:03d}')
                ax.text(ytop, yi + 0.55, _region_short(row['region6']),
                        ha='right', va='bottom', fontsize=7.5,
                        fontweight='bold', color=REGION_COLOR[row['region6']],
                        zorder=2, gid=f'layer-regions-{100 + n_reg:03d}')
                prev = row['region6']
                n_reg += 1
        ax.set_ylim(-0.7, n_rows - 0.3)
        # 12-char 7-pt names span ~46 data units, so the left margin
        # must reach -53 or labels spill off the strip (and into the
        # neighbour strip's region names at 178 mm)
        ax.set_xlim(-53, ytop)
        ax.set_yticks([])
        ax.tick_params(axis='x', labelsize=8)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
    ax2.set_xlabel('Directed pair-edges passing the 95% honesty filter '
                   '(raw, ADF-gated differencing)', fontsize=8)
    # legend (left) + honesty headline (right) above the middle strip
    ax2.legend(loc='upper center', fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.30, 1.06), ncol=1, handlelength=1.2,
               handleheight=1.0, columnspacing=1.2)
    ax2.text(0.995, 1.06,
             'Forced-difference honesty check (alldiff):\n'
             '0 of 6,320 directed pairs survive in all 80 records',
             transform=ax2.transAxes, ha='right', va='bottom',
             fontsize=8.5, fontweight='bold', color=BLUE_HI,
             gid='layer-note-000')
    ax1.text(-0.02, 1.06, '(a)', transform=ax1.transAxes, fontsize=10,
             fontweight='bold', va='bottom', ha='right')


def panel_b(ax, prof):
    """Per-region means: raw passes, trend leakage, recovered edges."""
    g = (prof.groupby('region6')
         .agg(n=('record', 'size'),
              raw=('total_pass95', 'mean'),
              leak=('trend_leak', 'mean'),
              rec_age=('recov_noage', 'mean'),
              rec_arc=('recov_noarc', 'mean'))
         .reindex(REGION_ORDER))
    y = np.arange(len(REGION_ORDER))[::-1]
    h = 0.17
    series = [
        ('raw', BLUE, 'Raw passes (mean)', 'rawpasses'),
        ('leak', ORANGE, 'Trend leakage (mean)', 'trendleak'),
        ('rec_age', AQUA, 'Age errors removed', 'recovage'),
        ('rec_arc', ACCENT, 'ARC disabled', 'recovarc'),
    ]
    for i, (key, color, label, lyr) in enumerate(series):
        off = (i - 1.5) * h
        bars = ax.barh(y + off, g[key], height=h, color=color,
                       edgecolor='none')
        for n, patch in enumerate(bars):
            patch.set_gid(f'layer-{lyr}-{n:03d}')
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r}\n(n = {int(g.loc[r, 'n'])})" for r in REGION_ORDER],
                       fontsize=7.5)
    ax.set_xlabel('Mean directed edges per record')
    ax.set_xlim(0, g[['raw', 'leak', 'rec_age', 'rec_arc']].max().max() * 1.10)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', length=0)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    handles = [Patch(fc=c, ec='none', label=l) for k, c, l, lyr in series]
    ax.legend(handles=handles, loc='lower right', fontsize=7, frameon=False,
              ncol=1, handlelength=1.1, handleheight=0.9)
    ax.text(0.985, 1.02, 'alldiff: 0 honest edges',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            fontweight='bold', color=BLUE_HI)
    ax.text(-0.10, 1.0, '(b)', transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='bottom')


def panel_c(ax_left, ax_right, prof, ccm):
    """Ablation ladder + CCM symmetry."""
    # --- left: ablation ladder ---
    labs = ['Observational\nnoise', 'Age\nerror',
            'ARC\ndeconfounding']
    means = [prof['recov_nonoise'].mean(), prof['recov_noage'].mean(),
             prof['recov_noarc'].mean()]
    x = np.arange(3)
    ladder = ax_left.bar(x, means, width=0.55,
                         color=[INK_MUTED, BLUE, ACCENT], edgecolor='none')
    for n, patch in enumerate(ladder):
        patch.set_gid(f'layer-ladder-{n:03d}')
    for xi, m in zip(x, means):
        ax_left.text(xi, m + 0.4, f'{m:.1f}', ha='center', va='bottom',
                     fontsize=9, fontweight='bold', color=INK)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labs, fontsize=8)
    ax_left.set_ylabel('Mean recovered edges\nper record')
    ax_left.set_ylim(0, max(means) * 1.35)
    ax_left.tick_params(axis='x', length=0)
    for spine in ['top', 'right']:
        ax_left.spines[spine].set_visible(False)
    ax_left.text(0.5, 0.965, 'Which layer blocks inference',
                 transform=ax_left.transAxes, ha='center', va='top',
                 fontsize=8.5, fontweight='bold', color=INK)

    # --- right: CCM symmetry ---
    cmap = LinearSegmentedColormap.from_list('blue_dens', ['#ffffff', BLUE, BLUE_HI])
    hb = ax_right.hexbin(ccm['ccm_fwd'], ccm['ccm_rev'], gridsize=22,
                         cmap=cmap, mincnt=1, linewidths=0.2,
                         edgecolors=GRID)
    hb.set_gid('layer-hexbin-000')
    diag = ax_right.plot([0, 1], [0, 1], color=INK_SEC, lw=1.0, ls='--')
    diag[0].set_gid('layer-diag-000')
    mf, mr = ccm['ccm_fwd'].mean(), ccm['ccm_rev'].mean()
    ax_right.text(0.96, 0.08,
                  f'mean skill {mf:.2f} (forward) vs\n{mr:.2f} (reverse) — '
                  f'symmetric,\nno directional dominance',
                  transform=ax_right.transAxes, ha='right', va='bottom',
                  fontsize=8, color=INK)
    ax_right.set_xlabel('CCM cross-map skill, forward')
    ax_right.set_ylabel('CCM cross-map skill, reverse')
    ax_right.set_xlim(0, 1)
    ax_right.set_ylim(0, 1)
    ax_right.set_aspect('equal')
    ax_right.tick_params(labelsize=8)
    for spine in ['top', 'right']:
        ax_right.spines[spine].set_visible(False)
    ax_right.text(0.5, 0.965,
                  'CCM symmetry',
                  transform=ax_right.transAxes, ha='center', va='top',
                  fontsize=8.5, fontweight='bold', color=INK)
    cb = plt.colorbar(hb, ax=ax_right, fraction=0.046, pad=0.04)
    cb.set_label('pairs per cell', fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_edgecolor(INK_MUTED)
    ax_left.text(-0.12, 1.0, '(c)', transform=ax_left.transAxes, fontsize=10,
                 fontweight='bold', va='bottom')


def export_table_s6(prof, ccm):
    """Appendix Table S6: per-record scrutiny profile (80 rows), using the
    standardized display names of src/record_names.py with the repository
    file identifier kept for traceability (Table S2)."""
    out = prof.copy()
    out['_disp'] = out['record'].map(display_name)
    out = out.sort_values(['region6', 'total_pass95', '_disp'],
                          ascending=[True, False, True])[
        ['_disp', 'record', 'Type', 'region6', 'Tier', 'sigma_ka',
         'snr_proxy', 'n_out_pass95', 'n_in_pass95', 'n_trend_leak95',
         'n_out_pass95_alldiff', 'recov_nonoise', 'recov_noage',
         'recov_noarc', 'max_rate_out']].copy()
    out.insert(0, 'No.', range(1, len(out) + 1))
    out.columns = ['No.', 'Record', 'Source ID', 'Type', 'Region', 'Tier',
                   'sigma_ka', 'SNR', 'Pass95 out', 'Pass95 in',
                   'Trend-leak 95%', 'Pass95 alldiff', 'Recov -noise',
                   'Recov -age', 'Recov -ARC', 'Max rate out']
    out = out.round({'sigma_ka': 1, 'SNR': 1, 'Max rate out': 3})
    path = os.path.join(OUTPUT_DIR, 'table_s6_scrutiny_profile.csv')
    out.to_csv(path, index=False, encoding='utf-8-sig')
    print(f'Saved {path} ({len(out)} rows)')


def draw(prof, ccm):
    # (a) full-width on top as three side-by-side horizontal-row strips
    # (see panel_a docstring); (b)(c) below. Strip heights follow their
    # row counts (13-px pitch at 300 dpi), bottom-aligned so the three
    # x-axes share one baseline.
    # strip heights follow row counts at a 33-px (300-dpi) row pitch —
    # a 7-pt text box is ~39 px tall, so the pitch buys real clearance
    fig = plt.figure(figsize=(7.0, 7.2))  # 178 mm wide
    gs = fig.add_gridspec(2, 1, height_ratios=[1.75, 0.8],
                          left=0.045, right=0.985, top=0.955, bottom=0.06,
                          hspace=0.15)
    _cell = gs[0, 0].get_position(fig)
    w3 = _cell.width / 3.0
    ax_a1 = fig.add_axes([_cell.x0, _cell.y0, w3, 0.816 * _cell.height])
    ax_a2 = fig.add_axes([_cell.x0 + w3, _cell.y0, w3, 0.895 * _cell.height])
    ax_a3 = fig.add_axes([_cell.x0 + 2 * w3, _cell.y0, w3,
                          0.395 * _cell.height])
    gs_b = gs[1, 0].subgridspec(1, 3, width_ratios=[1.15, 0.9, 1.1],
                                wspace=0.35)
    ax_b = fig.add_subplot(gs_b[0, 0])
    ax_cl = fig.add_subplot(gs_b[0, 1])
    ax_cr = fig.add_subplot(gs_b[0, 2])

    panel_a(ax_a1, ax_a2, ax_a3, prof)
    panel_b(ax_b, prof)
    panel_c(ax_cl, ax_cr, prof, ccm)

    # semantic layer order for the layered SVG (scripts/export_layered_svg.py
    # reads this attribute automatically); unnamed artists fall back to the
    # generic frame/background/data/markers/text/annotations/legend classes
    fig._svg_layer_order = ('bars', 'regions', 'recordlabels', 'note',
                            'rawpasses', 'trendleak', 'recovage', 'recovarc',
                            'ladder', 'hexbin', 'diag')

    # route through save_figure so the PNG is flattened to RGB and the
    # TIFF gets LZW compression (journal-ready raster post-processing)
    from src.visualization import save_figure
    save_figure(fig, 'Fig5_Global_Attribution_Scrutiny',
                fmts=('pdf', 'png', 'tif'), dpi=300)
    print('Saved Fig5_Global_Attribution_Scrutiny (+ .pdf/.tif)')
    plt.close(fig)


def main():
    prof, ccm = load_data()
    print(f"Records: {len(prof)} | raw 95% passes mean "
          f"{prof['total_pass95'].mean():.1f} | alldiff honest passes total "
          f"{int(prof['n_out_pass95_alldiff'].sum() + prof['n_in_pass95_alldiff'].sum())}")
    print(f"Ablation means (noise/age/ARC): "
          f"{prof['recov_nonoise'].mean():.1f}/{prof['recov_noage'].mean():.1f}/"
          f"{prof['recov_noarc'].mean():.1f}")
    print(f"CCM pairs: {len(ccm)} | fwd {ccm['ccm_fwd'].mean():.2f} / "
          f"rev {ccm['ccm_rev'].mean():.2f}")
    export_table_s6(prof, ccm)
    draw(prof, ccm)
    print('Done.')


if __name__ == '__main__':
    main()
