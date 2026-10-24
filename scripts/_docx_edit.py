"""Small helpers for surgical edits to the CARC paper .docx files.

Why this is not a one-liner: python-docx splits a paragraph into runs at
arbitrary points, so a phrase the reader sees as contiguous is usually cut
across several runs. Worse, these paragraphs also carry OMML equation objects
(`m:oMath`) that are siblings of the runs, not runs themselves. Flattening a
paragraph into its first run therefore silently drags text across the
equations and scrambles the sentence.

`replace_in_paragraph` instead maps the match onto the exact runs it covers
and rewrites only those, refusing when the span would jump over a non-run
element such as an equation.
"""
from __future__ import annotations

from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn

_RUN = qn("w:r")
# spelling-check markers carry no content; treat them as transparent
_PROOF = qn("w:proofErr")


def iter_paragraphs(doc):
    """Yield every paragraph in the document body, including those in tables."""
    def walk(parent):
        if hasattr(parent, "element") and hasattr(parent.element, "body"):
            elm = parent.element.body
        else:
            elm = parent._tc
        for child in elm.iterchildren():
            if child.tag == qn("w:p"):
                yield Paragraph(child, parent)
            elif child.tag == qn("w:tbl"):
                tbl = Table(child, parent)
                for row in tbl.rows:
                    for cell in row.cells:
                        yield from walk(cell)

    yield from walk(doc)


def _run_positions(par):
    """Return (runs, starts, concat) where starts[i] is run i's offset in concat."""
    runs, starts, pos = [], [], 0
    for r in par.runs:
        runs.append(r)
        starts.append(pos)
        pos += len(r.text)
    return runs, starts, "".join(r.text for r in runs)


def _spans_foreign_element(par, first_run, last_run):
    """True if a non-run element (e.g. an equation) sits between the two runs."""
    if first_run is last_run:
        return False
    children = list(par._p.iterchildren())
    try:
        i0 = children.index(first_run._r)
        i1 = children.index(last_run._r)
    except ValueError:
        return True
    return any(c.tag not in (_RUN, _PROOF) for c in children[i0 + 1:i1])


def replace_in_paragraph(par, old, new, count=0):
    """Replace `old` with `new` in one paragraph, preserving runs and equations.

    Returns the number of replacements made. `count=0` means replace all.
    Raises ValueError if a match would span an equation object.
    """
    made, search_from = 0, 0
    while count == 0 or made < count:
        runs, starts, text = _run_positions(par)
        # Search forward from the end of the last replacement: `new` legitimately
        # contains `old` when an edit appends a sentence, and restarting at 0
        # would then match its own output forever.
        idx = text.find(old, search_from)
        if idx < 0:
            break
        end = idx + len(old)
        search_from = idx + len(new)

        i0 = max(i for i, s in enumerate(starts) if s <= idx)
        i1 = max(i for i, s in enumerate(starts) if s < end)
        if _spans_foreign_element(par, runs[i0], runs[i1]):
            raise ValueError(
                f"match spans an equation object; edit manually:\n  {old[:90]!r}"
            )

        off0 = idx - starts[i0]
        off1 = end - starts[i1]
        head = runs[i0].text[:off0]
        tail = runs[i1].text[off1:]
        if i0 == i1:
            runs[i0].text = head + new + tail
        else:
            runs[i0].text = head + new
            for k in range(i0 + 1, i1):
                runs[k].text = ""
            runs[i1].text = tail
        made += 1
    return made


def replace_everywhere(doc, old, new, expect=None):
    """Apply `replace_in_paragraph` across the whole document.

    `expect` asserts the total number of replacements, so a silently-missed or
    accidentally-duplicated edit fails loudly instead of shipping.
    """
    n = sum(replace_in_paragraph(par, old, new) for par in iter_paragraphs(doc))
    if expect is not None and n != expect:
        raise AssertionError(f"{old[:70]!r}: made {n} replacements, expected {expect}")
    return n


def find_paragraph(doc, startswith=None, contains=None):
    """Return the single paragraph matching the predicate, or raise."""
    hits = []
    for par in iter_paragraphs(doc):
        t = par.text.strip()
        if (startswith and t.startswith(startswith)) or (contains and contains in t):
            hits.append(par)
    if len(hits) != 1:
        raise LookupError(f"expected 1 match, got {len(hits)}")
    return hits[0]


def has_equation(par):
    """True if the paragraph carries an OMML equation object."""
    return any(c.tag != _RUN for c in par._p.iterchildren() if "oMath" in c.tag)
