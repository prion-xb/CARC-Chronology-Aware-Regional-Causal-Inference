r"""List every tracked change (w:ins / w:del) in the manuscript for author review.

Read-only: the .docx is never modified. Walk word/document.xml body paragraphs in
document order, attribute each revision to the current section heading, pair
adjacent same-author del+ins into REPLACE rows, classify PURELY MECHANICAL edits
(whitespace / dash / quote swaps) so they can be summarised, and list every
SUBSTANTIVE edit individually.

Outputs:
  outputs/revision_list.md  — human-readable review list
  outputs/revision_list.csv — full rows, untruncated

Section headings are paragraphs whose visible text matches
  ^(Abstract|References|Acknowledgements|\d+(?:\.\d+)+\s+\S|\d\s+\S)
The task suggested \d+(\.\d+)*\s+\S, but that also matched the run-in lead-in
"12 negative controls" (a sentence label, not a heading); every real numbered
heading in this document carries a dot (2.1, 3.5, 4.3, ...), so dotted numbers
or a single-digit section number are required instead.

Reconciliation: pairs*2 + lone inserts + lone deletes must equal the raw
w:ins / w:del element counts (328 / 280). Nested revisions (a w:del inside a
w:ins, e.g. Word wrapping a deletion in an empty insertion) are enumerated in
document order exactly like sibling ones, so every element is counted once.
"""
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parent.parent
DOCX = ROOT / "paper" / "Research Pape5.docx"
MD_OUT = ROOT / "outputs" / "revision_list.md"
CSV_OUT = ROOT / "outputs" / "revision_list.csv"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def q(name):
    return "{%s}%s" % (W, name)


def mq(name):
    return "{%s}%s" % (M, name)


W_P, W_INS, W_DEL = q("p"), q("ins"), q("del")
W_T, W_DELTEXT, W_TAB, W_BR = q("t"), q("delText"), q("tab"), q("br")
M_T = mq("t")

HEADING_RE = re.compile(
    r"^(Abstract|References|Acknowledgements|\d+(?:\.\d+)+\s+\S|\d\s+\S)"
)
WS_RE = re.compile(r"\s+")
DASHES = "\u2013\u2014-"
QUOTES = "\u2018\u2019\u201a\u201b\u201c\u201d"
REM_DASH = str.maketrans("", "", DASHES)
REM_QUOTE = str.maketrans("", "", QUOTES)

CTX = 50  # chars of plain context captured before / after a change


def visible_text(p):
    """Visible paragraph text: w:t and m:t everywhere except inside w:del."""
    parts = []

    def walk(elem):
        for ch in elem:
            if ch.tag == W_DEL:
                continue
            if ch.tag in (W_T, M_T):
                parts.append(ch.text or "")
            elif ch.tag == W_TAB:
                parts.append("\t")
            elif ch.tag == W_BR:
                parts.append("\n")
            else:
                walk(ch)

    walk(p)
    return "".join(parts)


def event_text(el):
    """Text carried by one revision element.

    Equation runs store their text in m:t instead of w:t, so both are read
    (a w:ins carries w:t / m:t, a w:del carries w:delText / m:t). Formatting
    revisions wrapped inside w:rPr carry no text and come out empty.
    """
    tags = {W_T, M_T} if el.tag == W_INS else {W_DELTEXT, M_T}
    parts = []
    for ch in el.iter():
        if ch.tag in tags:
            parts.append(ch.text or "")
        elif ch.tag == W_TAB:
            parts.append("\t")
        elif ch.tag == W_BR:
            parts.append("\n")
    return "".join(parts)


def walk_items(elem, items, inside_rev):
    """Ordered items of a paragraph: ('plain', str) chunks and ('rev', element).

    Generic recursion over every child, so revisions nested inside runs,
    equation zones (w:oMath) or paragraph/run properties are all enumerated in
    document order exactly once. Plain text (w:t / m:t outside revisions) is
    emitted as plain chunks; w:tab / w:br become "\t" / "\n".
    """
    for ch in elem:
        tag = ch.tag
        if tag in (W_INS, W_DEL):
            items.append(("rev", ch))
            walk_items(ch, items, True)
        elif tag in (W_T, M_T):
            if not inside_rev:
                items.append(("plain", ch.text or ""))
        elif tag == W_TAB:
            if not inside_rev:
                items.append(("plain", "\t"))
        elif tag == W_BR:
            if not inside_rev:
                items.append(("plain", "\n"))
        else:
            walk_items(ch, items, inside_rev)


def mechanical_class(deleted, inserted):
    """(is_mechanical, subtype) — purely mechanical if only whitespace, dashes
    (– — -) or curly/straight quotes differ."""
    d = WS_RE.sub("", deleted)
    i = WS_RE.sub("", inserted)
    if d == i:
        return True, "whitespace"
    if d.translate(REM_DASH) == i.translate(REM_DASH):
        return True, "dash"
    if d.translate(REM_QUOTE) == i.translate(REM_QUOTE):
        return True, "quote"
    if d.translate(REM_DASH).translate(REM_QUOTE) == i.translate(REM_DASH).translate(REM_QUOTE):
        return True, "dash+quote"
    return False, ""


def process_paragraph(p, section, counts):
    """Emit rows for one paragraph; update counts (element / author tallies).

    Tallies are taken in a separate pass over every revision element, before
    pairing, so each w:ins / w:del element is counted exactly once regardless
    of whether it later joins a REPLACE pair.
    """
    items = []
    walk_items(p, items, False)

    for kind, el in items:
        if kind != "rev":
            continue
        author = el.get(q("author")) or "?"
        counts["elements"][el.tag] += 1
        counts["authors"][author][el.tag] += 1
        date = el.get(q("date")) or ""
        if date:
            counts["dates"].append(date[:10])

    # cumulative plain-text offsets, so each rev knows its position in the
    # surrounding plain context
    plain_parts, offsets, cur = [], [], 0
    for kind, payload in items:
        offsets.append(cur)
        if kind == "plain":
            plain_parts.append(payload)
            cur += len(payload)
    plain_all = "".join(plain_parts)

    rows = []
    i = 0
    while i < len(items):
        kind, el = items[i]
        if kind != "rev":
            i += 1
            continue

        author = el.get(q("author")) or "?"

        # pair this event with the next if they are adjacent revs, one ins one
        # del, by the same author, with no plain run text between them
        if i + 1 < len(items):
            kind2, el2 = items[i + 1]
            if (
                kind2 == "rev"
                and el2.tag != el.tag
                and (el2.get(q("author")) or "?") == author
            ):
                del_el, ins_el = (el, el2) if el.tag == W_DEL else (el2, el)
                deleted = event_text(del_el)
                inserted = event_text(ins_el)
                off = offsets[i]
                rows.append(
                    dict(
                        section=section,
                        author=author,
                        date=(ins_el.get(q("date")) or del_el.get(q("date")) or "")[:10],
                        type="REPLACE",
                        deleted=deleted,
                        inserted=inserted,
                        before=plain_all[max(0, off - CTX): off],
                        after=plain_all[off: off + CTX],
                        mech=mechanical_class(deleted, inserted),
                    )
                )
                i += 2
                continue

        text = event_text(el)
        off = offsets[i]
        if el.tag == W_INS:
            rows.append(
                dict(
                    section=section, author=author, date=date[:10], type="INSERT",
                    deleted="", inserted=text,
                    before=plain_all[max(0, off - CTX): off],
                    after=plain_all[off: off + CTX],
                    mech=mechanical_class("", text),
                )
            )
        else:
            rows.append(
                dict(
                    section=section, author=author, date=date[:10], type="DELETE",
                    deleted=text, inserted="",
                    before=plain_all[max(0, off - CTX): off],
                    after=plain_all[off: off + CTX],
                    mech=mechanical_class(text, ""),
                )
            )
        i += 1
    return rows


def md_cell(text, limit):
    text = (text or "").replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    text = WS_RE.sub(" ", text).strip()
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "\u2026"
    return text


def main():
    doc = Document(DOCX)
    body = doc.element.body

    counts = {
        "elements": Counter(),                # w:ins / w:del totals
        "authors": defaultdict(Counter),      # author -> Counter({'w:ins': n, 'w:del': n})
        "dates": [],
    }
    section = "(front matter)"
    rows = []
    for p in body.iter(W_P):
        vt = visible_text(p).strip()
        if vt and HEADING_RE.match(vt):
            section = vt
        rows.extend(process_paragraph(p, section, counts))

    # ---- reconciliation -------------------------------------------------
    n_ins = counts["elements"][W_INS]
    n_del = counts["elements"][W_DEL]
    n_pairs = sum(1 for r in rows if r["type"] == "REPLACE")
    n_lone_ins = sum(1 for r in rows if r["type"] == "INSERT")
    n_lone_del = sum(1 for r in rows if r["type"] == "DELETE")
    accounted_ins = n_pairs + n_lone_ins
    accounted_del = n_pairs + n_lone_del
    ok_ins = accounted_ins == n_ins
    ok_del = accounted_del == n_del

    print("=" * 70)
    print("TRACKED-CHANGE REVIEW: paper/Research Pape5.docx")
    print(f"raw elements : {n_ins} w:ins, {n_del} w:del (total {n_ins + n_del})")
    print(f"rows         : {n_pairs} REPLACE, {n_lone_ins} INSERT, {n_lone_del} DELETE")
    print(f"recon ins    : pairs*2 half + lone = {n_pairs} + {n_lone_ins} "
          f"= {accounted_ins} vs {n_ins} raw  {'OK' if ok_ins else 'MISMATCH'}")
    print(f"recon del    : pairs*2 half + lone = {n_pairs} + {n_lone_del} "
          f"= {accounted_del} vs {n_del} raw  {'OK' if ok_del else 'MISMATCH'}")
    for author, c in counts["authors"].items():
        print(f"  author {author!r}: {c[W_INS]} ins, {c[W_DEL]} del")
    if counts["dates"]:
        print(f"date range   : {min(counts['dates'])} .. {max(counts['dates'])}")

    mech = [r for r in rows if r["mech"][0]]
    subst = [r for r in rows if not r["mech"][0]]
    print(f"rows         : {len(mech)} mechanical (summarised), "
          f"{len(subst)} substantive (listed individually)")
    print("=" * 70)
    if not (ok_ins and ok_del):
        print("RECONCILIATION FAILED - inspect nested/weird markup")
        sys.exit(1)

    # ---- outputs --------------------------------------------------------
    MD_OUT.parent.mkdir(parents=True, exist_ok=True)

    with open(MD_OUT, "w", encoding="utf-8") as f:
        f.write("# Revision list - Research Pape5.docx\n\n")
        f.write("Review of the co-authors' tracked changes before accepting. "
                "Read-only scan of the .docx; source: `scripts/55_revision_list.py`.\n\n")
        f.write("## Totals\n\n")
        f.write(f"- Insertions (`w:ins`): **{n_ins}**\n")
        f.write(f"- Deletions (`w:del`): **{n_del}**\n")
        f.write(f"- Revision rows: {n_pairs} REPLACE, {n_lone_ins} INSERT, "
                f"{n_lone_del} DELETE\n")
        f.write(f"- Date range: {min(counts['dates'])} .. {max(counts['dates'])}\n\n")
        f.write("| Author | Insertions | Deletions |\n|---|---|---|\n")
        for author in sorted(counts["authors"]):
            c = counts["authors"][author]
            f.write(f"| {author} | {c[W_INS]} | {c[W_DEL]} |\n")

        f.write("\n## Summary of mechanical changes\n\n")
        f.write("Purely mechanical edits (whitespace, dash, or quote swaps only) "
                "are summarised here instead of being listed one by one.\n\n")
        f.write("| Author | Change type | Count |\n|---|---|---|\n")
        mech_counter = Counter(
            (r["author"], f'{r["type"]} [{r["mech"][1]}]') for r in mech
        )
        for (author, ctype), n in sorted(mech_counter.items()):
            f.write(f"| {author} | {ctype} | {n} |\n")

        f.write("\n## Substantive changes\n\n")
        f.write("| # | Section | Author | Type | Deleted | Inserted | Context |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for idx, r in enumerate(subst, 1):
            f.write(
                f"| {idx} | {md_cell(r['section'], 40)} | {r['author']} "
                f"| {r['type']} | {md_cell(r['deleted'], 100)} "
                f"| {md_cell(r['inserted'], 100)} "
                f"| \u2026{md_cell(r['before'], 60)}\u2026 \u2192 \u2026{md_cell(r['after'], 60)}\u2026 |\n"
            )
        f.write(f"\n_{len(subst)} substantive rows; full untruncated text in "
                f"`outputs/revision_list.csv`._\n")

    with open(CSV_OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "section", "author", "date", "type",
                    "deleted", "inserted", "context_before", "context_after",
                    "mechanical"])
        for idx, r in enumerate(rows, 1):
            w.writerow([
                idx, r["section"], r["author"], r["date"], r["type"],
                r["deleted"], r["inserted"], r["before"], r["after"],
                "True" if r["mech"][0] else "False",
            ])

    print(f"wrote {MD_OUT}")
    print(f"wrote {CSV_OUT}")


if __name__ == "__main__":
    main()
