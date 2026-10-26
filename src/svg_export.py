"""
Layered SVG export for Adobe Illustrator editing.

Saves a matplotlib figure as SVG with semantic layer groups per axes, so
that Adobe Illustrator's Layers panel shows one sublayer per element class
(frame / background / data / markers / text / annotations / legend). Text
is exported as editable text objects (``svg.fonttype='none'``), and each
group carries a human-readable id, e.g. ``axes_1 > layer-text``.

Schematic figures may opt into figure-specific semantic layers: tag artists
with ``set_gid('layer-<name>-...')`` and set ``fig._svg_layer_order`` — see
``save_layered_svg``.

The SVG is the *editing source*; the PNG/PDF siblings remain the final
render. After editing in Illustrator, export the print file from AI — see
docs/AI_EDITING.md.

Usage (direct)::

    from src.svg_export import save_layered_svg
    save_layered_svg(fig, 'figures/Fig1_Global_Proxy_Map.svg')

or run any figure script through scripts/figure_tools/export_layered_svg.py, which
emits an .svg sibling for every .png the script saves.
"""

import os
import re
import xml.etree.ElementTree as ET

import matplotlib
from matplotlib.axis import Tick, XAxis, YAxis
from matplotlib.collections import Collection, PathCollection
from matplotlib.image import AxesImage
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.spines import Spine
from matplotlib.text import Annotation, Text

SVG_NS = 'http://www.w3.org/2000/svg'

# Layer paint order (bottom to top). 'frame' covers spines, ticks, tick
# labels, axis titles and gridlines; everything else is self-explanatory.
LAYER_ORDER = ('frame', 'background', 'data', 'markers',
               'text', 'annotations', 'legend')


# ---------------------------------------------------------------------------
# Artist classification
# ---------------------------------------------------------------------------

def _dedupe_gids(fig):
    """Suffix duplicate gids with -<nnn> so every SVG id is unique."""
    seen = {}
    for ax in fig.axes:
        artists = ax.get_children()
        for artist in artists:
            gid = artist.get_gid()
            if not gid or not gid.startswith('layer-'):
                continue
            n = seen.get(gid, 0)
            if n:
                artist.set_gid(f'{gid}-{n:03d}')
            seen[gid] = n + 1
    for artist in fig.get_children():
        gid = artist.get_gid()
        if not gid or not gid.startswith('layer-'):
            continue
        n = seen.get(gid, 0)
        if n:
            artist.set_gid(f'{gid}-{n:03d}')
        seen[gid] = n + 1


def _classify_fig_artists(fig):
    """Return {layer_name: [artists]} for every drawable artist of *fig*.

    Every artist is assigned a gid ``<layer>-<nnn>``; the SVG post-pass
    merges these into one ``<g id="layer-<layer>">`` per axes.
    """
    layers = {name: [] for name in LAYER_ORDER}

    for ax in fig.axes:
        frame_ids = set()
        for lb in ax.get_xticklabels() + ax.get_yticklabels():
            frame_ids.add(id(lb))
        for obj in (ax.title, ax.xaxis.label, ax.yaxis.label):
            if obj is not None:
                frame_ids.add(id(obj))
        for obj in (ax._left_title, ax._right_title):
            if obj is not None:
                frame_ids.add(id(obj))
        grid_ids = set(id(ln) for ln in ax.xaxis.get_gridlines()
                       + ax.yaxis.get_gridlines())

        for artist in ax.get_children():
            if (id(artist) in frame_ids or id(artist) in grid_ids
                    or isinstance(artist, (XAxis, YAxis, Tick, Spine))):
                layer = 'frame'
            elif isinstance(artist, Text):
                layer = 'annotations' if isinstance(artist, Annotation) \
                    else 'text'
            elif isinstance(artist, Legend):
                layer = 'legend'
            elif isinstance(artist, PathCollection):
                layer = 'markers'
            elif isinstance(artist, (Collection, AxesImage)):
                layer = 'data'
            elif isinstance(artist, Line2D):
                layer = 'data'
            elif isinstance(artist, Patch):
                layer = 'background'
            else:
                layer = 'data'
            layers[layer].append(artist)

    # figure-level artists (legends, suptitle, free text)
    for artist in fig.get_children():
        if isinstance(artist, Text):
            layers['text'].append(artist)
        elif isinstance(artist, Legend):
            layers['legend'].append(artist)
        elif isinstance(artist, Patch):
            layers['background'].append(artist)
        elif isinstance(artist, (Collection, Line2D, AxesImage)):
            layers['data'].append(artist)

    # tag every artist with its layer so the SVG writer emits an id.
    # Artists the figure script already tagged with a semantic
    # ``layer-<name>-...`` gid (custom layers, e.g. schematic figures) keep
    # their own layer; everything else gets the generic class layer.
    for name, artists in layers.items():
        for n, artist in enumerate(artists):
            gid = artist.get_gid()
            if gid and gid.startswith('layer-'):
                continue
            artist.set_gid(f'layer-{name}-{n:03d}')

    # SVG ids must be document-unique: the same semantic tag can be reused
    # in several axes (e.g. one reference line per panel), so duplicate gids
    # get a -<nnn> suffix. Layer grouping matches the ``layer-<name>-``
    # prefix, so the bucket is unaffected.
    _dedupe_gids(fig)
    return layers


# ---------------------------------------------------------------------------
# SVG post-processing: merge per-artist gids into per-axes layer groups
# ---------------------------------------------------------------------------

def _group_layers_in_svg(path, order=None):
    """Rewrite the SVG in place: inside every group that contains layer
    artists, wrap all ``<g id="<layer>-*">`` children into one
    ``<g id="layer-<layer>">`` per layer.

    Paint order (bottom to top) is given by *order* for the names it lists
    (generic class names may be interleaved, e.g. shading between
    background and data); generic class layers NOT listed in *order* keep
    their default positions before the custom ones — so e.g. the axes
    background rect never covers figure content. Unlisted semantic
    ``layer-<name>-<nnn>`` gids follow in first-appearance order."""
    # generic classes not repositioned by *order* stay at their default
    # bottom slots
    ordered = [n for n in LAYER_ORDER if n not in (order or [])] \
        + list(order or [])
    ET.register_namespace('', SVG_NS)
    # keep matplotlib's namespace prefixes verbatim: Illustrator resolves
    # <use xlink:href=...> marker references only under the xlink prefix —
    # an auto-generated ns0/ns4 prefix leaves scatter markers invisible
    for _prefix, _uri in (
            ('xlink', 'http://www.w3.org/1999/xlink'),
            ('dc', 'http://purl.org/dc/elements/1.1/'),
            ('cc', 'http://creativecommons.org/ns#'),
            ('rdf', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#')):
        ET.register_namespace(_prefix, _uri)
    tree = ET.parse(path)
    root = tree.getroot()

    def group_children(parent):
        children = list(parent)
        buckets = {}          # layer name -> [children], insertion-ordered
        rest = []
        for child in children:
            cid = child.get('id', '')
            placed = False
            if cid.startswith('layer-'):
                tail = cid[len('layer-'):]
                for name in ordered:
                    if tail.startswith(name + '-'):
                        buckets.setdefault(name, []).append(child)
                        placed = True
                        break
                if not placed:
                    m = re.match(r'^(.+?)-\d+$', tail)
                    if m:
                        buckets.setdefault(m.group(1), []).append(child)
                        placed = True
            if not placed:
                rest.append(child)
        if not buckets:
            return
        # rebuild: layer wrappers in paint order, then anything untagged
        for child in children:
            parent.remove(child)
        for name in ordered:
            if name in buckets:
                wrapper = ET.SubElement(parent, f'{{{SVG_NS}}}g')
                wrapper.set('id', f'layer-{name}')
                for child in buckets.pop(name):
                    wrapper.append(child)
        for name, layer_children in buckets.items():   # leftover semantic names
            wrapper = ET.SubElement(parent, f'{{{SVG_NS}}}g')
            wrapper.set('id', f'layer-{name}')
            for child in layer_children:
                wrapper.append(child)
        for child in rest:
            parent.append(child)

    # recurse over every group in the document (figure root, axes groups)
    for gel in list(root.iter(f'{{{SVG_NS}}}g')):
        group_children(gel)
    tree.write(path, encoding='utf-8', xml_declaration=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_layered_svg(fig, path, layer_order=None, **save_kwargs):
    """Save *fig* as a layered, text-editable SVG at *path*.

    Assigns layer gids to all artists, saves with ``svg.fonttype='none'``
    (text stays editable), then post-processes the file so every axes group
    contains one ``<g id="layer-...">`` per element class.

    Schematic figures can request semantic layers instead of the generic
    class layers: tag artists with ``artist.set_gid('layer-<name>-...')``
    and pass the layer names (bottom to top) in *layer_order*, or set
    ``fig._svg_layer_order`` — the export wrapper
    (``scripts/figure_tools/export_layered_svg.py``) reads the figure attribute
    automatically.

    ``save_kwargs`` are passed through to ``fig.savefig`` (e.g.
    ``facecolor=...``).
    """
    path = os.fspath(path)
    order = layer_order or getattr(fig, '_svg_layer_order', None)
    _classify_fig_artists(fig)
    with matplotlib.rc_context({'svg.fonttype': 'none'}):
        fig.savefig(path, format='svg', metadata=None, **save_kwargs)
    _group_layers_in_svg(path, order)
    print(f'[svg_export] {path}')
    return path
