"""
Layout QA for CARC figures.

Monkeypatches ``Figure.savefig`` so that every figure a script saves is
inspected first: after a full render, text artists, axes boxes and legends
are compared pairwise for collisions, and every text is checked against the
figure boundary (catches clipping / tight-bbox surprises).

Usage (no changes needed in the target script)::

    python scripts/figure_tools/qa_figures.py scripts/17_regional_pairwise_matrix.py

Report lines are JSON, one per issue, appended to the report file, e.g.::

    {"fig": "FigS_Regional_Pairwise_Matrix",
     "kind": "text_overlap",
     "artists": ["...", "..."],
     "overlap_frac": 0.42,
     "where": "axes[3]"}

Kinds:
  text_overlap      two text blocks intersect (>=2% of the smaller box)
  text_outside      text extends beyond the figure canvas (would clip/crop)
  axes_overlap      two axes frames intersect
  legend_over_axes  a legend bbox intersects the axes it belongs to
  suptitle_overlap  suptitle intersects the top-most axes
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure
from matplotlib.text import Text
from matplotlib.legend import Legend


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _intersect_area(b1, b2):
    """Intersection area of two Bbox objects (0 when disjoint)."""
    x0 = max(b1.x0, b2.x0)
    y0 = max(b1.y0, b2.y0)
    x1 = min(b1.x1, b2.x1)
    y1 = min(b1.y1, b2.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _overlap_frac(b1, b2):
    """Overlap area as a fraction of the smaller of the two boxes."""
    area = _intersect_area(b1, b2)
    if area <= 0:
        return 0.0
    smaller = min(b1.width * b1.height, b2.width * b2.height)
    return area / smaller if smaller > 0 else 0.0


def _describe_artist(a):
    txt = getattr(a, "get_text", lambda: "")()
    if isinstance(txt, str) and txt:
        return f"text{chr(34)}{txt[:48]}{chr(34)}"
    return type(a).__name__


def _all_texts(fig):
    """Text artists to check, excluding tick labels (dedup by artist id)."""
    out, seen = [], set()
    def _add(t, ax):
        if id(t) not in seen:
            seen.add(id(t))
            out.append((t, ax))
    for ax in fig.axes:
        for t in ax.texts:
            _add(t, ax)
        if ax.get_title():
            _add(ax.title, ax)
    if fig._suptitle is not None:
        _add(fig._suptitle, None)
    for t in fig.texts:               # suptitle may also live in fig.texts
        _add(t, None)
    return out


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _text_extent(t, renderer):
    """Text-only extent of an artist: for annotations this EXCLUDES the
    connector arrow (Text.get_window_extent), so a thin connector crossing
    a caption does not read as a text collision; empty-string annotations
    (pure arrows) return None and are skipped by the text checks."""
    from matplotlib.text import Annotation, Text
    if isinstance(t, Annotation):
        if not t.get_text():
            return None
        return Text.get_window_extent(t, renderer)
    return t.get_window_extent(renderer)


def _check_text_pairs(fig):
    issues = []
    renderer = fig.canvas.get_renderer()
    items = _all_texts(fig)
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            t1, ax1 = items[i]
            t2, ax2 = items[j]
            b1 = _text_extent(t1, renderer)
            b2 = _text_extent(t2, renderer)
            if b1 is None or b2 is None:
                continue
            frac = _overlap_frac(b1, b2)
            if frac < 0.02:
                continue
            issues.append({
                "kind": "text_overlap",
                "artists": [_describe_artist(t1), _describe_artist(t2)],
                "overlap_frac": round(frac, 3),
                "boxes": [[round(b1.x0, 1), round(b1.y0, 1),
                           round(b1.x1, 1), round(b1.y1, 1)],
                          [round(b2.x0, 1), round(b2.y0, 1),
                           round(b2.x1, 1), round(b2.y1, 1)]],
                "where": (ax1.get_label() if ax1 else "figure"),
            })
    return issues


def _check_text_outside(fig):
    issues = []
    fb = fig.bbox
    pad = fb.width * 0.004   # tiny tolerance for antialiasing
    renderer = fig.canvas.get_renderer()
    for t, ax in _all_texts(fig):
        b = _text_extent(t, renderer)
        if b is None:
            continue
        if (b.x0 < fb.x0 - pad or b.y0 < fb.y0 - pad
                or b.x1 > fb.x1 + pad or b.y1 > fb.y1 + pad):
            issues.append({
                "kind": "text_outside",
                "artists": [_describe_artist(t)],
                "box": [round(b.x0, 1), round(b.y0, 1),
                        round(b.x1, 1), round(b.y1, 1)],
                "where": (ax.get_label() if ax else "figure"),
            })
    return issues


def _check_axes_overlap(fig):
    issues = []
    axs = fig.axes
    for i in range(len(axs)):
        for j in range(i + 1, len(axs)):
            b1 = axs[i].get_window_extent()
            b2 = axs[j].get_window_extent()
            # twinx/twiny axes share an identical box by design — skip them
            if (abs(b1.x0 - b2.x0) < 3 and abs(b1.y0 - b2.y0) < 3
                    and abs(b1.x1 - b2.x1) < 3 and abs(b1.y1 - b2.y1) < 3):
                continue
            if _overlap_frac(b1, b2) > 0.01:
                issues.append({
                    "kind": "axes_overlap",
                    "artists": [axs[i].get_label(), axs[j].get_label()],
                    "overlap_frac": round(_overlap_frac(b1, b2), 3),
                    "where": "figure",
                })
    return issues


def _check_legend_over_axes(fig):
    """Figure-level legends (fig.legend) must not cover the axes; in-axes
    legends (loc='best' etc.) live inside axes by design and are skipped."""
    issues = []
    if fig.legends:
        axes = sorted(fig.axes, key=lambda a: a.get_window_extent().y0)
        for leg in fig.legends:
            lb = leg.get_window_extent(fig.canvas.get_renderer())
            for ax in axes[:2]:
                ab = ax.get_window_extent()
                frac = _overlap_frac(lb, ab)
                if frac > 0.02:
                    issues.append({
                        "kind": "legend_over_axes",
                        "artists": ["fig.legend"],
                        "overlap_frac": round(frac, 3),
                        "where": ax.get_label(),
                    })
                    break
    return issues


def _check_suptitle(fig):
    issues = []
    if not fig._suptitle:
        return issues
    st = fig._suptitle
    sb = st.get_window_extent(fig.canvas.get_renderer())
    axes = sorted(fig.axes, key=lambda a: a.get_window_extent().y1)
    for ax in axes[-2:]:
        ab = ax.get_window_extent()
        frac = _overlap_frac(sb, ab)
        if frac > 0.01:
            issues.append({
                "kind": "suptitle_overlap",
                "artists": ["suptitle", f"axes({ax.get_label()})"],
                "overlap_frac": round(frac, 3),
                "where": "figure",
            })
            break
    return issues


# ---------------------------------------------------------------------------
# Map-label defusing
# ---------------------------------------------------------------------------

def defuse_annotations(fig, annotations, obstacles=(), max_k=3.0, step=0.15,
                       rotations=(0.0,)):
    """Resolve label collisions by lengthening / rotating connector offsets.

    For each ``(annot, base_dx, base_dy)`` in ``annotations`` (an
    ``ax.annotate`` created with ``textcoords="offset points"``), try
    candidate offsets ``k * R(theta) . (dx, dy)`` — theta from ``rotations``
    (degrees, 0 first), k from 1.0 upward — and keep the first one whose
    window extent clears every other label, every obstacle, and the figure
    canvas. Used by map figures where hand-tuned offsets pile up.

    Returns the final ``(dx, dy)`` offsets per annotation.
    """
    import math
    renderer = fig.canvas.get_renderer()
    fig.canvas.draw()
    obs = []
    for o in obstacles:
        if hasattr(o, "get_window_extent"):
            obs.append(o.get_window_extent(renderer))
        else:
            obs.append(o)
    fb = fig.bbox
    ann = [a[0] for a in annotations]
    base = [(a[1], a[2]) for a in annotations]
    # text-only extents: the connector arrow is not part of the label box,
    # otherwise a label whose anchor sits inside an obstacle can never clear
    box = [_text_extent(a, renderer) for a in ann]
    orig = [a.xyann for a in ann]

    def set_off(i, dx, dy):
        ann[i].xyann = (dx, dy)

    def candidates(i):
        dx0, dy0 = base[i]
        for th in rotations:
            c, s = math.cos(math.radians(th)), math.sin(math.radians(th))
            rx, ry = dx0 * c - dy0 * s, dx0 * s + dy0 * c
            k = 1.0
            while k <= max_k + 1e-9:
                yield (rx * k, ry * k)
                k += step

    def clear(nb, i):
        if (nb.x0 < fb.x0 or nb.y0 < fb.y0
                or nb.x1 > fb.x1 or nb.y1 > fb.y1):
            return False
        if any(j != i and _overlap_frac(nb, box[j]) >= 0.02
               for j in range(len(ann))):
            return False
        return not any(_overlap_frac(nb, ob) >= 0.02 for ob in obs)

    for _ in range(60):
        # most-collided labels first, so crowded corners get sorted out early
        order = sorted(range(len(ann)),
                       key=lambda i: -sum(
                           1 for j in range(len(ann)) if j != i
                           and _overlap_frac(box[i], box[j]) >= 0.02))
        changed = False
        for i in order:
            bi = box[i]
            hit = any(j != i and _overlap_frac(bi, box[j]) >= 0.02
                      for j in range(len(ann)))
            if not hit:
                hit = any(_overlap_frac(bi, ob) >= 0.02 for ob in obs)
            if not hit:
                continue
            for dx, dy in candidates(i):
                set_off(i, dx, dy)
                nb = _text_extent(ann[i], renderer)
                if clear(nb, i):
                    box[i] = nb
                    changed = True
                    break
            else:
                set_off(i, *orig[i])     # restore the original offset
        if not changed:
            break
    return [ann[i].xyann for i in range(len(ann))]


def defuse_edge_labels(fig, items, obstacles=()):
    """Resolve edge-label collisions by sliding labels along their edges.

    ``items`` is a list of ``(Text, mid_xy, unit_xy, length_deg, perp_xy)``:
    the label can slide ``t * length_deg`` along the edge direction and sit
    ``d`` degrees on either side of the line. Greedily relocates each
    colliding label to the first candidate position that clears every other
    label, every obstacle (e.g. node captions, arrows), and the canvas.
    """
    renderer = fig.canvas.get_renderer()
    fig.canvas.draw()
    obs = []
    for o in obstacles:
        if hasattr(o, "get_window_extent"):
            obs.append(o.get_window_extent(renderer))
        else:
            obs.append(o)
    fb = fig.bbox
    texts = [it[0] for it in items]
    box = [t.get_window_extent(renderer) for t in texts]
    orig = [t.get_position() for t in texts]
    slides = (0.0, 0.12, -0.12, 0.25, -0.25, 0.40, -0.40, 0.60, -0.60)
    perps = (5.0, -5.0, 9.0, -9.0, 14.0, -14.0, 20.0, -20.0, 28.0, -28.0)

    def set_pos(i, t_deg, d_deg):
        it = items[i]
        texts[i].set_position((
            it[1][0] + it[2][0] * t_deg + it[4][0] * d_deg,
            it[1][1] + it[2][1] * t_deg + it[4][1] * d_deg))

    def clear(nb, i):
        if (nb.x0 < fb.x0 or nb.y0 < fb.y0
                or nb.x1 > fb.x1 or nb.y1 > fb.y1):
            return False
        if any(j != i and _overlap_frac(nb, box[j]) >= 0.02
               for j in range(len(texts))):
            return False
        return not any(_overlap_frac(nb, ob) >= 0.02 for ob in obs)

    for _ in range(40):
        order = sorted(range(len(texts)),
                       key=lambda i: -sum(
                           1 for j in range(len(texts)) if j != i
                           and _overlap_frac(box[i], box[j]) >= 0.02))
        changed = False
        for i in order:
            bi = box[i]
            hit = any(j != i and _overlap_frac(bi, box[j]) >= 0.02
                      for j in range(len(texts)))
            if not hit:
                hit = any(_overlap_frac(bi, ob) >= 0.02 for ob in obs)
            if not hit:
                continue
            found = False
            for t_deg in slides:
                tt = t_deg * items[i][3]
                for d_deg in perps:
                    set_pos(i, tt, d_deg)
                    nb = texts[i].get_window_extent(renderer)
                    if clear(nb, i):
                        box[i] = nb
                        changed = True
                        found = True
                        break
                if found:
                    break
            if not found:
                texts[i].set_position(orig[i])
        if not changed:
            break


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def check_figure(fig, name="figure", min_report_frac=0.02):
    """Run all checks on a drawn figure; return list of issue dicts."""
    fig.canvas.draw()
    issues = []
    issues += _check_text_pairs(fig)
    issues += _check_text_outside(fig)
    issues += _check_axes_overlap(fig)
    issues += _check_legend_over_axes(fig)
    issues += _check_suptitle(fig)
    for it in issues:
        it["fig"] = name
    return issues


_REPORT_PATH = os.environ.get("QA_REPORT", "outputs/qa_layout_report.jsonl")
_NO_SAVE = os.environ.get("QA_NO_SAVE", "") == "1"
_orig_savefig = matplotlib.figure.Figure.savefig
_checked = set()          # id(fig): one check per figure (png+pdf save twice)


def _report_path():
    return os.environ.get("QA_REPORT", _REPORT_PATH)


def _patched_savefig(self, *args, **kwargs):
    fname = args[0] if args else kwargs.get("fname", "figure")
    name = os.path.splitext(os.path.basename(str(fname)))[0]
    if id(self) in _checked:
        return _orig_savefig(self, *args, **kwargs)
    _checked.add(id(self))
    try:
        issues = check_figure(self, name=name)
    except Exception as exc:                       # QA must never break a run
        print(f"[layout_qa] check failed for {name}: {exc!r}")
        issues = []
    if issues:
        with open(_report_path(), "a", encoding="utf-8") as fh:
            for it in issues:
                fh.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"[layout_qa] {name}: {len(issues)} issue(s) "
              f"-> {_report_path()}")
    else:
        print(f"[layout_qa] {name}: OK")
    if _NO_SAVE:
        return None
    return _orig_savefig(self, *args, **kwargs)


def install_qa_hook():
    matplotlib.figure.Figure.savefig = _patched_savefig


def summarize(report_path=_REPORT_PATH):
    """Print a compact per-figure summary of the report file."""
    if not os.path.exists(report_path):
        print("no report file:", report_path)
        return
    from collections import Counter, defaultdict
    rows = [json.loads(ln) for ln in open(report_path, encoding="utf-8")]
    by_fig = defaultdict(Counter)
    for r in rows:
        by_fig[r["fig"]][r["kind"]] += 1
    for fig, counts in by_fig.items():
        detail = ", ".join(f"{k}x{n}" for k, n in counts.items())
        print(f"{fig:<45} {detail}")
    print(f"total: {len(rows)} issues across {len(by_fig)} figure(s)")


if __name__ == "__main__":
    summarize()
