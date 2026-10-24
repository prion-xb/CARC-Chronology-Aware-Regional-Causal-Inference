#!/usr/bin/env python3
"""
32_framework_figure.py
----------------------
The CARC framework schematic (manuscript Fig. 2): the four methodological
layers grounded in Theorems 1-4, with the final decision node separating the
statistically reproducible regime from the sub-threshold regime.

Layout is a vertical flow of four layer bands (colour-coded in the fixed
CARC categorical order) plus a split decision node. All numbers shown
(200 realizations, alpha = 0.5, 95% filter) are the canonical values used
in the pipeline and cited in the manuscript.

Output: figures/Fig2_CARC_Framework.{pdf,png} (+svg via the export wrapper).

Layered SVG (Adobe Illustrator editing source): artists carry semantic gids
and ``fig._svg_layer_order`` names the layers bottom to top — input, l1..l4
(one band + its texts per layer), flow (vertical arrows), decision (node +
branches + outcome labels), footer (figure-level texts). See AI_EDITING.md.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from src.visualization import apply_style, save_figure, CARC_COLORS, \
    CARC_COLORS_LIGHT

INK = '#1a1a18'
INK_SEC = '#52514e'
SURFACE = '#ffffff'

# layer -> (accent colour, light fill)
LAYER_COLORS = {
    1: (CARC_COLORS['blue'],   CARC_COLORS_LIGHT['blue']),
    2: (CARC_COLORS['orange'], CARC_COLORS_LIGHT['orange']),
    3: (CARC_COLORS['teal'],   CARC_COLORS_LIGHT['teal']),
    4: (CARC_COLORS['purple'], CARC_COLORS_LIGHT['purple']),
}

LAYERS = [
    # (tag, title, body lines)
    (1, 'Layer 1 · Regional stacking + chronology perturbation',
     ['Six regional composites (median stack) of 80 records',
      'Monte Carlo age perturbations: 200 realizations',
      'Suppresses local noise (Theorem 1)']),
    (2, 'Layer 2 · Adaptive conservative deconfounding (ARC)',
     ['Removes part of the common-forcing signal (α = 0.5)',
      'NGRIP millennial trend as the common-forcing proxy',
      'Preserves the AR structure (Theorem 2)']),
    (3, 'Layer 3 · Pairwise causality testing',
     ['Deconfounded residuals → directional test per realization',
      'Granger test (default) · CCM / PCMCI (modular)',
      'FPR floor under common forcing (Theorem 3)']),
    (4, 'Layer 4 · Reproducibility filter',
     ['Edge retained only if it passes in ≥ 95% of realizations',
      'Robustness R(A→B) = share of passing realizations',
      'Filter honesty bound (Theorem 4)']),
]


def _layer_band(ax, y, height, idx, title, lines):
    """One rounded band: title left, body lines right, theorem tag.

    Every artist gets a semantic gid (``layer-l<idx>-...``) so the layered
    SVG export puts one CARC layer band per Illustrator layer.
    """
    accent, light = LAYER_COLORS[idx]
    x0, x1, y0 = 0.02, 0.98, y
    band = FancyBboxPatch((x0, y0), x1 - x0, height,
                          boxstyle='round,pad=0.004,rounding_size=0.012',
                          fc=light, ec=accent, lw=1.0, zorder=1)
    band.set_gid(f'layer-l{idx}-band')
    ax.add_patch(band)
    # left column: layer tag
    tag = ax.text(0.045, y0 + height - 0.028, f'LAYER {idx}',
                  fontsize=11, fontweight='bold', color=accent, ha='left',
                  va='top', zorder=3)
    tag.set_gid(f'layer-l{idx}-tag')
    ttl = ax.text(0.045, y0 + height - 0.062, title,
                  fontsize=10.5, fontweight='bold', color=INK, ha='left',
                  va='top', zorder=3)
    ttl.set_gid(f'layer-l{idx}-title')
    for k, line in enumerate(lines):
        t = ax.text(0.045, y0 + height - 0.096 - k * 0.030, line,
                    fontsize=8.5, color=INK_SEC, ha='left', va='top', zorder=3)
        t.set_gid(f'layer-l{idx}-text-{k:02d}')


def main():
    apply_style('paper')
    fig = plt.figure(figsize=(7.0, 5.542), facecolor=SURFACE)  # 178 mm wide, same aspect
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')

    # ---------------------------------------------------------------- input
    in_tag = ax.text(0.02, 0.985, 'Input', fontsize=10, fontweight='bold',
                     color=CARC_COLORS['grey'], ha='left', va='top')
    in_tag.set_gid('layer-input-tag')
    in_desc = ax.text(0.02, 0.952,
                      '80 proxy records · 6 regions · published age-model '
                      'errors (1σ)',
                      fontsize=11, color=INK, ha='left', va='top',
                      fontweight='bold')
    in_desc.set_gid('layer-input-desc')
    in_arrow = FancyArrowPatch((0.5, 0.942), (0.5, 0.936),
                               arrowstyle='-|>', mutation_scale=16,
                               lw=1.0, color=INK_SEC, zorder=2)
    in_arrow.set_gid('layer-input-arrow')
    ax.add_patch(in_arrow)

    # ---------------------------------------------------------------- layers
    # y_top is the bottom of the first band; the top clears the input line
    band_h, gap = 0.160, 0.0175
    y_top = 0.9325 - band_h
    for k, (idx, title, lines) in enumerate(LAYERS):
        y = y_top - k * (band_h + gap)
        _layer_band(ax, y, band_h, idx, title, lines)
        if k < 3:
            arrow = FancyArrowPatch((0.5, y), (0.5, y - gap + 0.002),
                                    arrowstyle='-|>', mutation_scale=16,
                                    lw=1.0, color=INK_SEC, zorder=2)
            arrow.set_gid(f'layer-flow-{k}')
            ax.add_patch(arrow)

    # --------------------------------------------------------- decision node
    y_split = 0.205
    dec_arrow = FancyArrowPatch((0.5, y_split + 0.030), (0.5, y_split + 0.004),
                                arrowstyle='-|>', mutation_scale=16,
                                lw=1.0, color=INK_SEC, zorder=2)
    dec_arrow.set_gid('layer-flow-3')
    ax.add_patch(dec_arrow)
    dec_title = ax.text(0.5, y_split - 0.004, 'Decision', fontsize=11,
                        fontweight='bold', color=INK, ha='center', va='top')
    dec_title.set_gid('layer-decision-title')
    dec_q = ax.text(0.5, y_split - 0.034, 'R(A→B) ≥ 95% ?', fontsize=11.5,
                    fontweight='bold', color=INK, ha='center', va='top')
    dec_q.set_gid('layer-decision-question')

    # left branch: reproducible regime
    branch_l = FancyArrowPatch((0.5, y_split - 0.050),
                               (0.26, y_split - 0.090),
                               arrowstyle='-|>', mutation_scale=15,
                               lw=0.9, color=INK_SEC, zorder=2,
                               connectionstyle='arc3,rad=-0.15')
    branch_l.set_gid('layer-decision-arrow-pass')
    ax.add_patch(branch_l)
    pass_title = ax.text(0.19, y_split - 0.100, 'reproducible regime',
                         fontsize=9.5, fontweight='bold',
                         color=CARC_COLORS['green'], ha='center', va='top')
    pass_title.set_gid('layer-decision-pass-title')
    pass_note = ax.text(0.19, y_split - 0.132, 'directional claim',
                        fontsize=8.5, color=INK_SEC, ha='center', va='top')
    pass_note.set_gid('layer-decision-pass-note')
    # right branch: sub-threshold regime
    branch_r = FancyArrowPatch((0.5, y_split - 0.050),
                               (0.76, y_split - 0.090),
                               arrowstyle='-|>', mutation_scale=15,
                               lw=0.9, color=INK_SEC, zorder=2,
                               connectionstyle='arc3,rad=0.15')
    branch_r.set_gid('layer-decision-arrow-fail')
    ax.add_patch(branch_r)
    fail_title = ax.text(0.81, y_split - 0.100, 'sub-threshold regime',
                         fontsize=9.5, fontweight='bold',
                         color=CARC_COLORS['grey'], ha='center', va='top')
    fail_title.set_gid('layer-decision-fail-title')
    fail_note = ax.text(0.81, y_split - 0.132, 'no directional claim',
                        fontsize=8.5, color=INK_SEC, ha='center', va='top')
    fail_note.set_gid('layer-decision-fail-note')

    foot_title = fig.text(0.02, 0.028,
                          'CARC — Chronology-Aware Robustness Criterion',
                          fontsize=13, fontweight='bold', color=INK)
    foot_title.set_gid('layer-footer-title')
    foot_note = fig.text(0.02, 0.004,
                         'Four layers grounded in Theorems 1–4; all thresholds '
                         'are the canonical pipeline values',
                         fontsize=8.5, color=INK_SEC)
    foot_note.set_gid('layer-footer-note')

    # semantic SVG layers (bottom to top) for the Adobe Illustrator workflow:
    # one Illustrator layer per CARC layer band, plus input / flow arrows /
    # decision node / footer — see AI_EDITING.md
    fig._svg_layer_order = ('input', 'l1', 'l2', 'l3', 'l4',
                            'flow', 'decision', 'footer')

    save_figure(fig, 'Fig2_CARC_Framework', fmts=('pdf', 'png', 'tif'))
    plt.close(fig)


if __name__ == '__main__':
    main()
