#!/usr/bin/env python3
"""
export_layered_svg.py
---------------------
Run a CARC figure script with the layout-QA hook active and emit a layered,
text-editable SVG sibling for every .png it saves, for the Adobe Illustrator
editing workflow (see AI_EDITING.md).

Usage:
  python scripts/export_layered_svg.py scripts/13_publication_figures.py [args...]

This is the *final regeneration* entry point: every figure the script saves
is (1) checked for text overlap / clipping by src.layout_qa (report written
to ``$QA_REPORT`` or outputs/qa_layout_report.jsonl), (2) written as PNG/PDF
as usual, and (3) exported as a layered SVG with one
``<g id="layer-...">`` per element class inside every axes group.

Set QA_NO_SAVE=1 to run the QA checks without writing any files.
"""

import os
import runpy
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
import matplotlib.figure


def _make_dual_savefig(orig_savefig):
    """Wrap savefig: after the real (QA-checked) save, add a layered SVG
    sibling for every .png that hits disk."""
    def _dual_savefig(self, *args, **kwargs):
        out = orig_savefig(self, *args, **kwargs)
        if os.environ.get('QA_NO_SAVE') == '1':
            return out                 # QA runs: nothing should hit disk
        fname = args[0] if args else kwargs.get('fname', '')
        if not str(fname).lower().endswith('.png'):
            return out                 # svg emitted for png only (pdf pairs)
        svg_path = os.path.splitext(str(fname))[0] + '.svg'
        try:
            from src.svg_export import save_layered_svg
            save_layered_svg(self, svg_path)
        except Exception as exc:
            print(f'[svg_export] FAILED {svg_path}: {exc!r}')
        return out
    return _dual_savefig


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)

    target = os.path.abspath(sys.argv[1])
    if not os.path.exists(target):
        print("target not found:", target)
        sys.exit(1)

    # one report file per target script, so sequential runs never clobber;
    # must be set before importing layout_qa (it reads the env at import)
    script_key = os.path.splitext(os.path.basename(target))[0]
    report = os.environ.get("QA_REPORT",
                            os.path.join("outputs", f"qa_layout_{script_key}.jsonl"))
    if os.path.exists(report):
        os.remove(report)
    os.environ["QA_REPORT"] = report

    from src.layout_qa import install_qa_hook, summarize
    install_qa_hook()
    # chain: our svg wrapper delegates to the QA hook, which delegates to
    # the real savefig — one savefig call therefore QA-checks *and* writes
    matplotlib.figure.Figure.savefig = _make_dual_savefig(
        matplotlib.figure.Figure.savefig)
    # Illustrator-friendly defaults for scripts that do not call apply_style:
    # Arial first (Windows/AI native), and mathtext glyphs (σ β α δ ...) also
    # in Arial — DejaVu math glyphs trigger missing-glyph warnings in AI.
    matplotlib.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Arial',
        'mathtext.it': 'Arial:italic',
        'mathtext.bf': 'Arial:bold',
        'mathtext.cal': 'Arial:italic',
    })
    sys.argv = [target] + sys.argv[2:]      # let the script see its own args
    runpy.run_path(target, run_name="__main__")

    print("\n----- layout QA summary -----")
    summarize(report)


if __name__ == '__main__':
    main()
