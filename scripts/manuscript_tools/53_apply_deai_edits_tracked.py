"""Second de-AI pass: the edits that 52_apply_deai_edits.py refused.

Those refusals were correct. The main text carries live tracked-change markup
(328 w:ins / 280 w:del from two authors), which fragments a sentence into runs
separated by w:ins/w:del elements. Rewriting across such a boundary would move
text into or out of another author's tracked revision.

Every edit here is therefore scoped to a span that lies inside a single run, so
no revision boundary is crossed and the markup is left exactly as the authors
left it. The em-dashes that Word stored as standalone runs are handled by
position instead, in `dash_runs`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from docx import Document  # noqa: E402
from docx_edit import iter_paragraphs, replace_in_paragraph  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EMDASH = "—"

# (anchor that identifies the paragraph, old, new) -- old must sit inside one run
MAIN = [
    ("may be replaced by CCM, PCMCI",
     "—so that CARC functions as a general robustness check rather than as a single fixed procedure.",
     "; CARC then functions as a general robustness check, not as a single fixed procedure."),
    ("Equation (3) expresses the F-statistic inflation",
     "underlying parameters—autoregressive coefficients, forcing loadings, and the persistence of the common driver.",
     "underlying parameters: autoregressive coefficients, forcing loadings, and the persistence of the common driver."),
    ("Equation (3) expresses the F-statistic inflation",
     "How often would a researcher using standard Granger causality on untreated proxy data erroneously conclude that X causes Y? Theorem 3 translates the F-statistic inflation into a lower bound on this rate, and the accompanying numerical evaluation shows that the bound is far from academic.",
     "Theorem 3 translates the F-statistic inflation into a lower bound on this rate, and the numerical evaluation below shows the bound to be substantial in practice."),
    ("Design objective. Rigorously propagate",
     "rather than relying on a single ", "instead of relying on a single "),
    ("Panel (c), left: ablation ladder",
     "when one layer—observational noise, ", "when one layer, observational noise, "),
    ("Panel (c), left: ablation ladder",
     "—is removed; right:", ", is removed; right:"),
    ("The present study differs from previous work",
     " rather than to a generic ", " and not to a generic "),
    ("At present, therefore, no record-to-record attribution claim",
     "most likely to unlock inference", "most likely to enable inference"),
]

SUPP = [
    ("Table S6 reports the per-record profile",
     "when one layer—observational noise, age perturbation, or ARC ",
     "when one layer, observational noise, age perturbation, or ARC "),
    ("Table S6 reports the per-record profile",
     "—is removed. Record names", ", is removed. Record names"),
    ("(ii) Six-record Granger analysis",
     "The repository reconstruction—unconditional first-differencing",
     "The repository reconstruction (unconditional first-differencing"),
    ("(ii) Six-record Granger analysis",
     "200 draws—lies systematically", "200 draws) lies systematically"),
]

# Paragraphs where Word stored each em-dash as its own run: anchor -> replacements
# in document order, one per standalone em-dash run.
DASH_RUNS = {
    "Left (Diagnosis): the four theorems": [": "],
    "every directed record pair (6,320 in total)": [": "],
    "At present, therefore, no record-to-record attribution claim": [", ", ", ", ", ", ", "],
}


def paragraphs_matching(doc, anchor):
    return [p for p in iter_paragraphs(doc) if anchor in p.text]


def apply_spans(doc, edits, label):
    ok = fail = 0
    for anchor, old, new in edits:
        pars = paragraphs_matching(doc, anchor)
        if len(pars) != 1:
            print(f"  FAIL anchor matched {len(pars)} paragraphs: {anchor[:50]!r}")
            fail += 1
            continue
        try:
            n = replace_in_paragraph(pars[0], old, new)
        except ValueError as exc:
            print(f"  FAIL {old[:60]!r}: {exc}")
            fail += 1
            continue
        if n != 1:
            print(f"  FAIL {old[:60]!r}: {n} replacements")
            fail += 1
        else:
            ok += 1
    print(f"{label} spans: {ok} applied, {fail} failed")
    return fail


def apply_dash_runs(doc, label):
    fail = 0
    for anchor, repls in DASH_RUNS.items():
        pars = paragraphs_matching(doc, anchor)
        if len(pars) != 1:
            continue  # this paragraph lives in the other document
        runs = [r for r in pars[0].runs if r.text == EMDASH]
        if len(runs) != len(repls):
            print(f"  FAIL {anchor[:45]!r}: {len(runs)} dash runs, expected {len(repls)}")
            fail += 1
            continue
        for run, txt in zip(runs, repls):
            run.text = txt
        print(f"  ok   {anchor[:45]!r}: {len(runs)} dash runs rewritten")
    return fail


if __name__ == "__main__":
    bad = 0
    for path, edits, label in [
        (ROOT / "paper" / "Research Pape5.docx", MAIN, "main text"),
        (ROOT / "paper" / "CARC_supplementary_edited.docx", SUPP, "supplementary"),
    ]:
        doc = Document(path)
        bad += apply_spans(doc, edits, label)
        bad += apply_dash_runs(doc, label)
        doc.save(path)
    sys.exit(1 if bad else 0)
