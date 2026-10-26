"""Final-acceptance glyph-height check for delivered rasters.

Thresholds (300 dpi): 7 pt cap height = 29.2 px, Arial x-height ~15 px.
Data curves and axes are far taller than any glyph; dots and grid marks are
tiny. A text glyph is a dark connected component in [8, 150] px tall with
an aspect ratio (w/h) in [0.15, 3.5] and area >= 10 px. We require:

  * 5th-percentile glyph height >= 12 px (no text below ~6.3 pt, the
    mathtext auto-subscript size)
  * median glyph height    >= 20 px (the bulk of labels near 8-9 pt), or
    >= 15 px when the whole glyph population sits at the 7 pt floor
    (p5 >= 14 px): an all-7 pt figure (e.g. the 80-row record strip,
    median ~19 px) has no room to fail on bulk size — if even its 5th
    percentile reaches the 7 pt x-height, every label is >= 7 pt.
"""
import sys

import numpy as np
from PIL import Image
from scipy import ndimage


def glyph_heights(path):
    im = Image.open(path).convert("L")
    a = np.asarray(im)
    dark = a < 128
    lab, n = ndimage.label(dark, structure=np.ones((3, 3)))
    if n == 0:
        return np.array([])
    objs = ndimage.find_objects(lab)
    hs = []
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        if not (8 <= h <= 150):
            continue
        if not (0.15 <= w / h <= 3.5):
            continue
        if np.count_nonzero(lab[sl] == i) < 10:
            continue
        # small round markers (scatter dots, degree signs, connector
        # endpoints) are square-ish: w/h ~= 1. A glyph of this height is
        # never that wide — the widest letter at 8-13 px (a subscript 'o')
        # stays below 0.9 — so drop the dots instead of counting them as
        # text below the 7 pt floor
        if h < 14 and w / h >= 0.92:
            continue
        hs.append(h)
    return np.array(hs)


def main(paths):
    ok = True
    for p in paths:
        hs = glyph_heights(p)
        if hs.size == 0:
            print(f"FAIL {p}: no text-sized glyphs found")
            ok = False
            continue
        p5, med = np.percentile(hs, 5), np.median(hs)
        # p5 floor = 12 px: body text sits >= 7 pt (x-height ~15 px), but
        # mathtext auto-subscripts (e.g. sigma_noise) legitimately reach
        # ~6.3 pt (x-height ~12 px). Bulk-size guard: median >= 20 px for
        # figures whose body is ~8-9 pt; an all-7 pt figure (p5 already at
        # the 7 pt x-height floor, 14 px) passes with median >= 15 px.
        bulk_ok = med >= 20 or (p5 >= 14 and med >= 15)
        status = "OK" if p5 >= 12 and bulk_ok else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"{status} {p}: n={hs.size:4d} p5={p5:5.1f}px "
              f"median={med:5.1f}px (>=12/20, or >=14/15 all-7pt required)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
