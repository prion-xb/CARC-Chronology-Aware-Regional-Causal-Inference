#!/usr/bin/env python3
"""
qa_figures.py — run any figure script under the layout QA hook.

Usage:
    python scripts/figure_tools/qa_figures.py scripts/17_regional_pairwise_matrix.py
    QA_NO_SAVE=1 python scripts/figure_tools/qa_figures.py scripts/17_regional_pairwise_matrix.py
    python scripts/figure_tools/qa_figures.py --summary

The target script is executed unchanged; every ``fig.savefig`` is intercepted
and inspected for overlapping text, out-of-canvas text, colliding axes and
legends that cover data. Issues are appended to ``outputs/qa_layout_report.jsonl``
and printed as a compact table at the end.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    if sys.argv[1] == "--summary":
        from src.layout_qa import summarize
        summarize(os.environ.get("QA_REPORT", "outputs/qa_layout_report.jsonl"))
        sys.exit(0)

    target = os.path.abspath(sys.argv[1])
    if not os.path.exists(target):
        print("target not found:", target)
        sys.exit(1)

    # one report file per target script, so parallel runs never clobber;
    # must be set before importing layout_qa (it reads the env at import)
    script_key = os.path.splitext(os.path.basename(target))[0]
    report = os.environ.get("QA_REPORT",
                            os.path.join("outputs", f"qa_layout_{script_key}.jsonl"))
    if os.path.exists(report):
        os.remove(report)
    os.environ["QA_REPORT"] = report

    from src.layout_qa import install_qa_hook, summarize

    install_qa_hook()
    sys.argv = [target] + sys.argv[2:]      # let the script see its own args
    import runpy
    runpy.run_path(target, run_name="__main__")

    print("\n----- layout QA summary -----")
    summarize(report)
