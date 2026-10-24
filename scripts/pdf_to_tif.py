#!/usr/bin/env python3
"""
pdf_to_tif.py — render a PDF page to a submission-grade TIFF.

Uses PyMuPDF (fitz) to rasterize and Pillow to write LZW-compressed TIFF with
the correct DPI metadata. No Ghostscript or ImageMagick needed.

  python scripts/pdf_to_tif.py in.pdf out.tif            # 600 dpi, RGB, LZW
  python scripts/pdf_to_tif.py in.pdf out.tif --dpi 300
  python scripts/pdf_to_tif.py in.pdf out.tif --page 2 --gray

Notes for journal submission:
  * line art / flowcharts: 600-1200 dpi; halftone photos: 300 dpi.
    Elsevier and most AGU/Nature-family journals accept 300 dpi for colour
    combination art and want >=1000 dpi for pure line art.
  * LZW is lossless and universally accepted; do NOT use JPEG-in-TIFF.
  * --alpha is off: the page is flattened onto white, since TIFF readers
    disagree about how to composite alpha.
"""
import argparse
import os
import sys

import fitz  # PyMuPDF
from PIL import Image


def pdf_to_tif(src, dst, dpi=600, page=1, gray=False, compression='tiff_lzw'):
    doc = fitz.open(src)
    if not 1 <= page <= doc.page_count:
        raise SystemExit(f'{src}: page {page} out of range (1-{doc.page_count})')
    p = doc.load_page(page - 1)

    # render at the requested dpi; colorspace RGB (or GRAY), no alpha channel
    cs = fitz.csGRAY if gray else fitz.csRGB
    pix = p.get_pixmap(dpi=dpi, colorspace=cs, alpha=False)

    im = Image.frombytes('L' if gray else 'RGB', (pix.width, pix.height),
                         pix.samples)
    im.save(dst, format='TIFF', compression=compression, dpi=(dpi, dpi))
    doc.close()

    w_in, h_in = pix.width / dpi, pix.height / dpi
    print(f'{os.path.basename(src)} p{page} -> {os.path.basename(dst)}')
    print(f'   {pix.width} x {pix.height} px @ {dpi} dpi '
          f'= {w_in:.2f} x {h_in:.2f} in ({w_in * 25.4:.0f} x '
          f'{h_in * 25.4:.0f} mm)')
    print(f'   mode={"L" if gray else "RGB"} compression={compression} '
          f'size={os.path.getsize(dst) / 1e6:.1f} MB')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--dpi', type=int, default=600)
    ap.add_argument('--page', type=int, default=1)
    ap.add_argument('--gray', action='store_true', help='8-bit greyscale')
    ap.add_argument('--compression', default='tiff_lzw',
                    choices=['tiff_lzw', 'tiff_deflate', 'packbits', 'none'])
    a = ap.parse_args()
    if not os.path.exists(a.src):
        raise SystemExit(f'not found: {a.src}')
    pdf_to_tif(a.src, a.dst, a.dpi, a.page, a.gray, a.compression)


if __name__ == '__main__':
    sys.exit(main())
