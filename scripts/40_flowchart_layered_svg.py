#!/usr/bin/env python3
"""Build the layered AI-editable SVG for the TikZ framework flowchart.

Canonical source: ../flow_diagram_tikz.md (TikZ standalone, edited by the user).
Mirror:           Figures/_tikz_gt/flow.tex         (byte-identical to flow_diagram_tikz.md)
Layer wrapper:    Figures/_tikz_gt/flow_layers.tex  (conditional rendering per
                  layer; must be manually kept in sync with the picture geometry)
Output:           figures/Fig2_CARC_Framework_tikz.svg

Why per-layer compilation (fixes the old hand-written SVG's layout drift/tofu):
  * Each semantic layer is compiled separately with latex + dvisvgm. All
    non-target nodes stay present as invisible phantoms (\\phantom text +
    draw=none), so every pass has the exact same geometry and viewBox.
    Merging layers can therefore never shift/overlap elements (no layout drift).
  * dvisvgm converts every glyph to vector paths, so the merged SVG has zero
    font dependency and can never show tofu/replacement glyphs (no tofu).
    Trade-off: text is outlines, not editable text objects; for text edits
    use figures/Fig2_CARC_Framework_tikz.pdf (AI opens it with live text).

Layers (bottom -> top): background -> theory -> input -> treatment ->
    l1 -> l2 -> l3 -> l4 -> flow -> links -> decision.
"""

import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TIKZ_DIR = ROOT / "Figures" / "_tikz_gt"
MD_SRC = ROOT / "flow_diagram_tikz.md"
FLOW_TEX = TIKZ_DIR / "flow.tex"
LAYERS_TEX = TIKZ_DIR / "flow_layers.tex"
WORK = TIKZ_DIR / "layers_tmp"
OUT = ROOT / "figures" / "Fig2_CARC_Framework_tikz.svg"
REF_FULL = TIKZ_DIR / "flow.svg"  # single-pass full render, for element-count QA

LAYER_ORDER = ["theory", "input", "treatment", "l1", "l2", "l3", "l4",
               "flow", "links", "decision"]

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

DRAWING_TAGS = {"path", "rect", "ellipse", "circle", "polygon", "polyline",
                "text", "image", "line"}


def run(cmd, cwd, what):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if p.returncode != 0:
        print(f"[FAIL] {what}\n--- stdout ---\n{p.stdout[-3000:]}\n"
              f"--- stderr ---\n{p.stderr[-3000:]}", file=sys.stderr)
        sys.exit(1)


def sync_mirror():
    """Keep Figures/_tikz_gt/flow.tex byte-identical to flow_diagram_tikz.md."""
    if not MD_SRC.exists():
        sys.exit(f"[FAIL] canonical source missing: {MD_SRC}")
    md = MD_SRC.read_text(encoding="utf-8")
    if FLOW_TEX.exists() and FLOW_TEX.read_text(encoding="utf-8") == md:
        print("[sync] flow_diagram_tikz.md == flow.tex (no change)")
        return
    shutil.copyfile(MD_SRC, FLOW_TEX)
    print(f"[sync] copied flow_diagram_tikz.md -> {FLOW_TEX.relative_to(ROOT)}")
    print("       NOTE: flow_layers.tex (layer wrapper) is NOT auto-synced;\n"
          "       update it by hand if the picture geometry changed.")


def compile_layer(key):
    latex = shutil.which("latex")
    dvisvgm = shutil.which("dvisvgm")
    if not latex or not dvisvgm:
        sys.exit("[FAIL] latex/dvisvgm not found on PATH (MiKTeX required)")
    WORK.mkdir(exist_ok=True)
    driver = WORK / f"lay_{key}.tex"
    driver.write_text(
        f"\\expandafter\\def\\csname isLAY{key}\\endcsname{{1}}"
        f"\\input{{../flow_layers}}",
        encoding="utf-8")
    job = f"lay_{key}"
    for i in (1, 2):
        run([latex, "-interaction=nonstopmode", "-halt-on-error",
             f"-jobname={job}", driver.name], WORK, f"latex {key} pass {i}")
    # --no-fonts: every glyph becomes a vector path; without it dvisvgm
    # references system font names (cmbx8, cmmi6, ...) with no embedded
    # data, which Illustrator falls back on -> wrong metrics + tofu.
    run([dvisvgm, "--no-fonts", f"{job}.dvi", "-o", f"{job}.svg"], WORK,
        f"dvisvgm {key}")


def layer_content(key):
    path = WORK / f"lay_{key}.svg"
    raw = path.read_text(encoding="utf-8")
    if "@font-face" in raw or "font-family" in raw:
        sys.exit(f"[FAIL] {key}: SVG still references fonts; "
                 f"expected pure vector paths (would garble in Illustrator)")
    tree = ET.parse(path)
    root = tree.getroot()
    parts = []
    for child in root:
        tag = child.tag.rsplit("}", 1)[-1]
        if tag in ("defs", "style", "metadata", "title", "desc"):
            continue
        if tag == "g" and child.attrib.get("id") == "page1":
            parts.extend(child)
        else:
            parts.append(child)
    return root.attrib, "".join(ET.tostring(c, encoding="unicode")
                                for c in parts)


def qa_element_count(merged_text):
    """Every drawn element belongs to exactly one layer, so the merged file
    must contain exactly as many drawing elements as the single-pass full
    render (flow.svg) plus the white background rect."""
    if not REF_FULL.exists():
        print("[qa] reference flow.svg missing; skipping element-count check")
        return
    ref = ET.parse(REF_FULL).getroot()
    ref_n = sum(1 for e in ref.iter()
                if e.tag.rsplit("}", 1)[-1] in DRAWING_TAGS)
    m = ET.fromstring(merged_text)
    m_n = sum(1 for e in m.iter()
              if e.tag.rsplit("}", 1)[-1] in DRAWING_TAGS)
    if m_n == ref_n + 1:
        print(f"[qa] element count OK: merged {m_n} = flow.svg {ref_n} + 1 bg rect")
    else:
        print(f"[WARN] element count mismatch: merged {m_n} vs "
              f"flow.svg {ref_n} + 1 (layers partition may be incomplete)")


def merge():
    layers, viewboxes = {}, set()
    for key in LAYER_ORDER:
        compile_layer(key)
        attrs, content = layer_content(key)
        layers[key] = content
        viewboxes.add(attrs.get("viewBox"))
        print(f"[layer] {key:10s} viewBox={attrs.get('viewBox')} "
              f"elements={content.count('<path') + content.count('<g')}")
    if len(viewboxes) != 1:
        sys.exit(f"[FAIL] per-layer viewBoxes differ (would misalign layers): {viewboxes}")
    vb = viewboxes.pop()

    root_attrs, _ = layer_content(LAYER_ORDER[0])
    width, height = root_attrs.get("width"), root_attrs.get("height")
    svg = ET.Element(f"{{{SVG_NS}}}svg", {
        "version": "1.1",
        "xmlns": SVG_NS,
        "xmlns:xlink": "http://www.w3.org/1999/xlink",
        "width": width, "height": height, "viewBox": vb,
    })
    bg = ET.SubElement(svg, f"{{{SVG_NS}}}g", {"id": "layer-background"})
    x, y, w, h = vb.split()
    ET.SubElement(bg, f"{{{SVG_NS}}}rect",
                  {"x": x, "y": y, "width": w, "height": h, "fill": "#ffffff"})
    for key in LAYER_ORDER:
        g = ET.SubElement(svg, f"{{{SVG_NS}}}g", {"id": f"layer-{key}"})
        for el in ET.fromstring(f"<g xmlns='{SVG_NS}'>{layers[key]}</g>"):
            g.append(el)

    header = f"""<?xml version="1.0" encoding="UTF-8"?>
<!--
  Fig2_CARC_Framework_tikz.svg — layered SVG (Adobe Illustrator editing source)
  ==================================================================
  Generated by scripts/40_flowchart_layered_svg.py, compiling the TikZ source
  in the root-level flow_diagram_tikz.md layer by layer (latex + dvisvgm):
    * each layer is compiled separately; non-target layers stay present as
      \\phantom placeholders with draw=none, so the geometry is identical to
      the full figure → all layers share one viewBox, and merging cannot
      misalign elements;
    * dvisvgm converts every glyph to vector paths → zero font dependency;
      Illustrator/browsers open without tofu (trade-off: text is outlines,
      not editable text; for text edits use
      figures/Fig2_CARC_Framework_tikz.pdf, or edit flow_diagram_tikz.md and
      regenerate).
  Layers (bottom → top): background → theory → input → treatment →
    l1 → l2 → l3 → l4 → flow → links → decision
  Regenerate: edit flow_diagram_tikz.md → sync Figures/_tikz_gt/flow.tex
    (byte-identical) and flow_layers.tex (layer wrapper, synced manually) → run
      python scripts/40_flowchart_layered_svg.py
  The old hand-written text SVG is archived at
    figures/00_archive/Fig2_CARC_Framework_tikz_text.svg.
-->
"""
    body = ET.tostring(svg, encoding="unicode")
    merged = header + body
    OUT.write_text(merged, encoding="utf-8")
    qa_element_count(merged)
    print(f"[done] {OUT.relative_to(ROOT)} "
          f"({len(merged.encode('utf-8')) // 1024} KB, viewBox={vb})")
    print(f"[done] layers: background + {', '.join(LAYER_ORDER)}")


if __name__ == "__main__":
    if not LAYERS_TEX.exists():
        sys.exit(f"[FAIL] layer wrapper missing: {LAYERS_TEX}")
    sync_mirror()
    merge()
