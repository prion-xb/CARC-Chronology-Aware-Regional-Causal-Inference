#!/usr/bin/env python3
"""
sync_figure_folders.py
----------------------
Sort the CARC figure deliverables into category folders for the
Illustrator / submission workflow.

The canonical pipeline output stays in figures/ (every figure script saves
there). This script COPIES the deliverables into:

  figures/01_main_text/        main-text figures (Fig1..Fig7 + Fig5 Escape)
  figures/02_supplementary/    supplementary figures (FigS*)
  figures/03_validation/       method-validation / reviewer-ammunition figures
  figures/04_demo/             demo figures (NOT for the paper)

Each folder contains a README.md with a one-line explanation and editing
notes per figure. Re-run this script after any figure is regenerated to
re-sync the curated copies:

  python scripts/figure_tools/sync_figure_folders.py

Files whose stem starts with _tmp_ (layout scratch) are never copied.
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config import FIGURE_DIR

EXTS = ('.png', '.pdf', '.svg', '.tif')

# stem -> category folder (files are matched by stem, any extension)
# 2026-08-20 figure reorder: the framework flowchart is now the paper's
# Fig. 1 and the map (FigS9) / quality summary (FigS10) moved to the
# appendix; new stems are the canonical deliverables (old stems kept on
# disk for traceability but no longer synced).
MAIN = ('Fig1_CARC_Framework', 'Fig2_Layer1_Stacking',
        'Fig3_MultiParameter_Sensitivity', 'Fig4_SlidingWindow_CARC',
        'Fig5_Global_Attribution_Scrutiny', 'Fig6_Regional_Pairwise_Matrix',
        'Fig7_Escape_Diagnosis')
# NOTE: the old bare-stem 300-dpi TIFs (Fig1.tif ... Fig4.tif, generated
# 2026-07-30/08-03 by the pre-revision pipeline) no longer match the current
# figures and were archived to figures/00_archive/ on 2026-08-18. Do NOT add
# bare 'Fig1'..'Fig4' stems back here — if 300-dpi TIFs are needed again,
# export them from the current layered SVGs instead.
SUPP = ('FigS9_Global_Proxy_Map', 'FigS10_Record_Quality_Summary',
        'FigS11_Stacking_Noise_Reduction', 'FigS12_Theoretical_FPR_Heatmap',
        'FigS13_Method_Comparison', 'FigS_Conditional_Causality',
        'FigS_Directional_Robustness', 'FigS_Record_Atlas',
        'FigS_Record_Strip', 'FigS_Regional_Pairwise_Matrix',
        'FigS_SixRecord_Pairwise_Matrix', 'FigS_Stacking_10-60',
        'FigS_Stacking_0-800', 'scenarioA_pairwise_matrix')
VALIDATION = ('carc_fpr_vs_beta',
              'carc_synthetic_validation', 'arc_validation', 'ccm_audit_phi',
              'violin_TPR_multipanel', 'violin_FPR_multipanel',
              # demoted from main text on 2026-08-18 (superseded by Fig6,
              # which matches the paper's Fig. 3 caption)
              'Fig5_Validation_MultiPanel')
DEMO = ('CARC_Validation_MultiPanel', 'simulated_series_examples',
        'timeseries_stack')

CATEGORY_OF = {}
for stem in MAIN:
    CATEGORY_OF[stem] = '01_main_text'
for stem in SUPP:
    CATEGORY_OF[stem] = '02_supplementary'
for stem in VALIDATION:
    CATEGORY_OF[stem] = '03_validation'
for stem in DEMO:
    CATEGORY_OF[stem] = '04_demo'


def main():
    copied, skipped = [], []
    for name in sorted(os.listdir(FIGURE_DIR)):
        path = os.path.join(FIGURE_DIR, name)
        if not os.path.isfile(path):
            continue
        stem, ext = os.path.splitext(name)
        if stem.startswith('_tmp_'):
            skipped.append(name)
            continue
        if stem == 'record_atlas_interactive':      # html companion of the atlas
            cat = '02_supplementary'
        elif stem in CATEGORY_OF:
            cat = CATEGORY_OF[stem]
        else:
            skipped.append(name)
            continue
        dst_dir = os.path.join(FIGURE_DIR, cat)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(path, os.path.join(dst_dir, name))
        copied.append(f'{cat}/{name}')

    print(f'Copied {len(copied)} file(s) into category folders.')
    for c in copied:
        print(f'  {c}')
    if skipped:
        print(f'Skipped (not copied): {", ".join(skipped)}')
    missing = [f'{s}.*' for s in ('FigS13_Method_Comparison',
                                  'violin_TPR_multipanel',
                                  'violin_FPR_multipanel')
               if not any(n.startswith(s) for n in os.listdir(FIGURE_DIR))]
    if missing:
        print(f'Note: expected but not yet generated: {", ".join(missing)}')


if __name__ == '__main__':
    main()
