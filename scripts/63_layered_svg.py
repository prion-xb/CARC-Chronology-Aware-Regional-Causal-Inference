#!/usr/bin/env python3
"""Post-process MATLAB element SVGs into semantically named layers.

MATLAB's SVG writer (Batik) emits one flat stream of <g> elements with no
layer structure, so an element arrives in Illustrator as a single tangle.
This script reads each element SVG, classifies every top-level group by what
it draws, and rewrites the file with named layers:

    background  canvas surface (behind everything)
    grid        grid lines
    axes        axis spines and tick marks
    data        bars, lines, markers, patches, embedded images
    labels      axis labels, tick numerals, titles, annotations
    legend      legend markers and text

Group order inside each layer is preserved, so the drawing is visually
unchanged; only the layer structure is added. Note: MATLAB's SVG writer
(Batik) outlines text to <path> glyphs, so SVG text is NOT editable in
Illustrator. The sibling PDF (carc.save_element writes both) keeps text as
real <text>, so edit wording in the PDF and paths/styling in the SVG.

Usage:
    python scripts/63_layered_svg.py               # every element SVG
    python scripts/63_layered_svg.py <path.svg>    # one file
    python scripts/63_layered_svg.py <dir>         # all SVGs under a dir
"""
import os
import re
import sys

from xml.etree import ElementTree as ET

SVG_NS = "http://www.w3.org/2000/svg"
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
ET.register_namespace("", SVG_NS)
ET.register_namespace("inkscape", INKSCAPE_NS)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")

# z-order of the layers, bottom first; matches how MATLAB draws the figure
LAYERS = ["background", "grid", "axes", "data", "labels", "legend"]

# palette from matlab/+carc/style.m (the SVG writer emits rgb() values)
COLORS = {
    "ink": (26, 26, 26),        # INK        #1a1a1a
    "ink_sec": (74, 74, 74),    # INK_SEC
    "ink_muted": (122, 122, 122),  # INK_MUTED
    "grid": (217, 217, 212),    # GRID
    "member": (185, 185, 178),  # MEMBER
    "blue": (40, 116, 166),     # BLUE
    "orange": (214, 137, 16),   # ORANGE
    "pale": (249, 232, 213),    # PALE
    "red": (192, 57, 43),       # RED
}


def parse_color(s):
    """rgb(1,2,3) or #1a1a1a -> (r, g, b) or None."""
    m = re.search(r"rgb\((\d+),(\d+),(\d+)\)", s or "")
    if m:
        return tuple(int(x) for x in m.groups())
    m = re.search(r"#([0-9a-fA-F]{6})", s or "")
    if m:
        h = m.group(1)
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    return None


def near(c, ref, tol=12):
    return c is not None and all(abs(a - b) <= tol for a, b in zip(c, ref))


def _style_of(src):
    style = {}
    for part in (src.get("style", "") or "").split(";"):
        if ":" in part:
            k, _, v = part.partition(":")
            style[k.strip()] = v.strip()
    for att in ("fill", "stroke", "stroke-dasharray"):
        if src.get(att) and att not in style:
            style[att] = src.get(att)
    return style


def group_info(el):
    """Style + child facts for one group. Batik puts the drawing style on
    the child elements, not the <g> itself, so read both. Text appears in
    two forms:
      - outlined text: a <g> with font-size + text colour as fill (and
        stroke, same colour) whose glyph paths set stroke:none
      - native text: a real <text> element (MATLAB annotations, some
        tick labels)"""
    kids = [k for k in el]
    child_styles = [_style_of(k) for k in kids]
    kids_tags = {k.tag.split("}")[-1] for k in kids}
    info = {
        "group_style": _style_of(el),
        "child_styles": child_styles,
        "kids": kids,
        "n_children": len(kids),
        "has_text": "text" in kids_tags,
        "has_line": "line" in kids_tags,
        "has_rect": "rect" in kids_tags,
        "has_path": "path" in kids_tags,
        "has_image": "image" in kids_tags,
    }
    # effective style: group style, with each child's own values taking
    # precedence (SVG cascade)
    style = dict(info["group_style"])
    for cs in child_styles:
        style.update(cs)
    info["style"] = style
    gs = info["group_style"]
    info["has_font_size"] = bool(gs.get("font-size"))
    info["fill"] = parse_color(gs.get("fill", ""))
    info["stroke"] = parse_color(gs.get("stroke", ""))
    # outlined text: font-size + text colour + glyph children that do not
    # stroke (a graphics group with font-size but no stroked children is
    # also caught; such groups carry fill:none children and are excluded
    # by the fill check)
    info["is_outlined_text"] = (
        info["has_font_size"]
        and kids_tags <= {"path", "use"}
        and gs.get("fill") not in (None, "none")
        and all(cs.get("stroke") in (None, "none") for cs in child_styles)
    )
    # tick marks are short line segments; measure the longest line child
    info["max_line_len"] = 0.0
    for k in kids:
        if k.tag.split("}")[-1] == "line":
            x1 = float(k.get("x1", 0)); x2 = float(k.get("x2", 0))
            y1 = float(k.get("y1", 0)); y2 = float(k.get("y2", 0))
            info["max_line_len"] = max(
                info["max_line_len"], ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    return info


def classify(info):
    """Return the layer name for one collected group."""
    st = info["style"]
    fill, stroke = info["fill"], info["stroke"]

    if info["has_text"] or info["is_outlined_text"]:
        return "labels"
    if info["has_image"]:
        return "data"
    if info["has_rect"]:
        # canvas surface is the only full-surface fill
        if near(fill, (255, 255, 255)) and near(stroke, (255, 255, 255)):
            return "background"
        return "data"
    if info["has_line"] or info["has_path"]:
        # dashed grid lines (GridColor GRID) -> grid
        if st.get("stroke-dasharray"):
            if near(stroke, COLORS["grid"], 25) or near(stroke, COLORS["member"], 25):
                return "grid"
        # axis spines / tick marks are muted ink, ticks are short
        if near(stroke, COLORS["ink_muted"]) and info["max_line_len"] < 400:
            return "axes"
        return "data"
    return "data"


def collect_draw_groups(root):
    """Every element that actually draws something, at any nesting depth:
    <g> elements with primitive children, plus bare primitives (MATLAB
    emits some annotations as raw <text>/<line> elements, unwrapped).
    A <g> that contains other drawing groups is a wrapper and is not
    collected itself; its children are. Batik nests per-panel, so wrappers
    can sit several levels deep and a text group can be wrapped inside a
    graphics group."""
    primitives = {"text", "line", "rect", "path", "image", "use"}
    found = []

    def walk(el):
        for k in list(el):
            tag = k.tag.split("}")[-1]
            if tag == "g":
                kids_tags = {c.tag.split("}")[-1] for c in k}
                if kids_tags & primitives:
                    found.append(k)
                if kids_tags & {"g", "defs"}:
                    walk(k)
            elif tag in primitives:
                found.append(k)

    walk(root)
    # Keep every collected group unless its *parent* is also collected (a
    # wrapper that merely contains other drawing groups carries nothing of
    # its own). A group with primitive children stays even if it also
    # contains collected sub-groups — its own primitives would otherwise be
    # orphaned. Bare primitives always stay.
    outermost = []
    for grp in found:
        if grp.tag.split("}")[-1] != "g":
            outermost.append(grp)
            continue
        parent = None
        for cand in root.iter():
            if any(grp is c for c in cand):
                parent = cand
                break
        if parent is None or parent not in found:
            outermost.append(grp)
    return outermost


def layered_svg(path):
    tree = ET.parse(path)
    root = tree.getroot()

    # Collect every drawing group (through the nested Batik wrappers) and
    # re-parent them into semantic layer groups at the root level. All
    # <defs> (clip paths, embedded font glyphs) stay where they are.
    # Bare primitives get wrapped in a <g> so each layer holds only groups.
    groups = collect_draw_groups(root)
    parents = {}
    wrapped = {}
    for el in groups:
        for cand in root.iter():
            if any(el is c for c in cand):
                parents[el] = cand
                break
        if el.tag.split("}")[-1] != "g":
            # keep the transform on the element: it is expressed in its
            # original parent's coordinate space, moving it to the wrapper
            # would shift the drawing
            wrapped[id(el)] = ET.Element("g")

    # classify first, then detach every element from its wrapper in one pass
    buckets = {name: [] for name in LAYERS}
    for el in groups:
        if el in parents:
            node = wrapped.get(id(el), el)
            if node is el:
                buckets.setdefault(classify(group_info(el)), []).append(el)
                parents[el].remove(el)
            else:
                node.append(el)
                buckets.setdefault(classify(group_info(node)), []).append(node)
                parents[el].remove(el)

    # drop now-empty wrapper <g>s
    for cand in root.iter():
        for g in list(cand):
            if (g.tag.split("}")[-1] == "g" and len(g) == 0
                    and g.get("id") not in LAYERS):
                cand.remove(g)

    for name in LAYERS:
        if not buckets.get(name):
            continue
        g = ET.SubElement(root, "g")
        g.set("id", name)
        g.set("{%s}groupmode" % INKSCAPE_NS, "layer")
        g.set("{%s}label" % INKSCAPE_NS, name)
        g.set("display", "inline")
        for el in buckets[name]:
            g.append(el)

    tree.write(path, encoding="utf-8", xml_declaration=True)
    return {name: len(buckets.get(name, [])) for name in LAYERS
            if buckets.get(name)}


def main(argv):
    targets = argv[1:]
    if not targets:
        targets = [
            os.path.join("Figures", "elements", "matlab"),
            os.path.join("Figures", "elements"),
        ]
    files = []
    for t in targets:
        if os.path.isfile(t):
            files.append(t)
        elif os.path.isdir(t):
            for dirpath, _, names in os.walk(t):
                files.extend(os.path.join(dirpath, n) for n in names
                             if n.endswith(".svg"))
        else:
            print(f"not found: {t}", file=sys.stderr)
            return 1
    files = sorted(set(os.path.normpath(f) for f in files
                       if "layered" not in f))
    ok = 0
    skipped = 0
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                if "inkscape:groupmode" in fh.read():
                    skipped += 1
                    continue
            counts = layered_svg(f)
            print(f"layered: {f}  " + ", ".join(
                f"{k}:{v}" for k, v in counts.items() if v))
            ok += 1
        except Exception as e:  # noqa: BLE001 - one bad file must not stop the rest
            print(f"FAILED: {f}: {e}", file=sys.stderr)
    print(f"\n{ok} layered, {skipped} already layered, "
          f"{len(files) - ok - skipped} failed / {len(files)} SVGs")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
