#!/usr/bin/env python3
"""One-off fix for Fig2_CARC_Framework_tikz.svg: make the math faithful to the
compiled LaTeX ground truth (figures/_tikz_gt/, MiKTeX + dvisvgm).

Fixes:
  1. Theory boxes T1-T4 resized + titles/formulas wrapped exactly as TeX
     breaks them (T1: 3 lines, T2: 2, T3: 4 with radical line, T4: 4).
  2. Accents R-tilde / Theta-hat become explicit floating glyphs (U+02DC /
     U+02C6 in dy/dx-shifted tspans) instead of combining marks.
  3. Blackboard E / script N / script G get Cambria Math font-family.
  4. Matrix subscripts get commas (N,1 / 1,T / K,1 ...) per LaTeX.
  5. L3 F-statistic becomes a stacked fraction with a real bar.
  6. L4 becomes a small stacked 1/M fraction + Sigma with side limits.
  7. Link anchor y's follow the new theory box midpoints.
"""
import io, sys

P = 'figures/Fig2_CARC_Framework_tikz.svg'
s = open(P, encoding='utf-8').read()
orig = s
errs = []

def rep(old, new, n=1):
    global s
    c = s.count(old)
    if c != n:
        errs.append(f'count {c} != {n}: {old[:80]!r}')
    s = s.replace(old, new)

CM = "Cambria Math, 'Segoe UI Symbol', serif"

# ---------------------------------------------------------------- header
rep("""  Fonts: Arial (repo convention, see AI_EDITING.md). Math is Unicode text
  with baseline-shifted sub/superscripts so it stays editable in Illustrator.
  Rare glyphs: double-struck E (U+1D53C) and combining tilde/circumflex may
  need a font with good math coverage on AI import (Segoe UI Symbol / Cambria
  Math) - see AI_EDITING.md section 6.""",
"""  Fonts: Arial (repo convention, see AI_EDITING.md). Math is Unicode text
  with baseline-shifted sub/superscripts so it stays editable in Illustrator.
  Accents (tilde/hat) are explicit floating glyphs (U+02DC / U+02C6 in dy/dx
  shifted tspans) - NOT combining marks, which render misplaced in the VS
  Code preview and on AI import. Board/script glyphs (U+1D53C / U+1D4A9 /
  U+1D4A2) carry font-family=\"""" + CM + """\".
  Layout follows the compiled flow_diagram_tikz.md (MiKTeX + dvisvgm, kept in
  figures/_tikz_gt/): theory titles wrap exactly as TeX breaks them, L3
  shows the F-statistic as a stacked fraction with a bar, L4 a small
  stacked 1/M fraction with Sigma side limits.""")

# ---------------------------------------------------------------- theory
rep('<rect x="0.94" y="3.7" width="4.62" height="6.4" rx="0.141"',
    '<rect x="0.94" y="2.5" width="4.62" height="8.3" rx="0.141"')
rep('<text x="3.25" y="3.45"', '<text x="3.25" y="2.25"')
rep('<rect x="1.54" y="2.9" width="3.42" height="1.0" rx="0.141"',
    '<rect x="1.54" y="2.9" width="3.42" height="1.4" rx="0.141"')
rep('<text x="3.25" y="3.24" font-family="Arial, Helvetica, sans-serif" font-size="0.3175" text-anchor="middle">Theorem 1: Residual Correlation<tspan x="3.25" dy="0.406"><tspan font-style="italic">Cov</tspan><tspan>(</tspan><tspan font-style="italic">X</tspan><tspan baseline-shift="sub" font-size="0.229" font-style="italic">t−1</tspan><tspan>, </tspan><tspan font-style="italic">u</tspan><tspan baseline-shift="sub" font-size="0.229" font-style="italic">t</tspan><tspan>) ≠ 0</tspan></tspan></text>',
    '<text x="3.25" y="3.22" font-family="Arial, Helvetica, sans-serif" font-size="0.3175" text-anchor="middle">Theorem 1: Residual<tspan x="3.25" dy="0.406">Correlation</tspan><tspan x="3.25" dy="0.406"><tspan font-style="italic">Cov</tspan><tspan>(</tspan><tspan font-style="italic">X</tspan><tspan baseline-shift="sub" font-size="0.229" font-style="italic">t−1</tspan><tspan>, </tspan><tspan font-style="italic">u</tspan><tspan baseline-shift="sub" font-size="0.229" font-style="italic">t</tspan><tspan>) ≠ 0</tspan></tspan></text>')
rep('<rect x="1.54" y="4.7" width="3.42" height="1.0" rx="0.141"',
    '<rect x="1.54" y="4.75" width="3.42" height="1.05" rx="0.141"')
rep('<text x="3.25" y="5.04" font-family="Arial, Helvetica, sans-serif" font-size="0.3175" text-anchor="middle">Theorem 2: Inflated <tspan font-style="italic">F</tspan><tspan x="3.25" dy="0.406"><tspan>𝔼[</tspan><tspan font-style="italic">F</tspan><tspan>] = 1 + </tspan><tspan font-style="italic">λ</tspan><tspan baseline-shift="sub" font-size="0.229">eff</tspan><tspan font-style="italic">/q</tspan></tspan></text>',
    '<text x="3.25" y="5.07" font-family="Arial, Helvetica, sans-serif" font-size="0.3175" text-anchor="middle">Theorem 2: Inflated <tspan font-style="italic">F</tspan><tspan x="3.25" dy="0.406"><tspan font-family="' + CM + '">𝔼</tspan><tspan>[</tspan><tspan font-style="italic">F</tspan><tspan>] = 1 + </tspan><tspan font-style="italic">λ</tspan><tspan baseline-shift="sub" font-size="0.229">eff</tspan><tspan font-style="italic">/q</tspan></tspan></text>')
rep('<rect x="1.54" y="6.5" width="3.42" height="1.0" rx="0.141"',
    '<rect x="1.54" y="6.25" width="3.42" height="1.9" rx="0.141"')
rep('<text x="3.25" y="6.84" font-family="Arial, Helvetica, sans-serif" font-size="0.3175" text-anchor="middle">Theorem 3: FPR Lower Bound<tspan x="3.25" dy="0.406"><tspan>FPR ≥ 1 − Φ(</tspan><tspan font-style="italic">z</tspan><tspan baseline-shift="sub" font-size="0.229">1−α</tspan><tspan> − √(</tspan><tspan font-style="italic">λ</tspan><tspan baseline-shift="sub" font-size="0.229">eff</tspan><tspan>/2))</tspan></tspan></text>',
    '<text x="3.25" y="6.57" font-family="Arial, Helvetica, sans-serif" font-size="0.3175" text-anchor="middle">Theorem 3: FPR<tspan x="3.25" dy="0.406">Lower Bound</tspan><tspan x="3.25" dy="0.406">FPR ≥</tspan><tspan x="1.93" y="7.788">1 − Φ(z<tspan baseline-shift="sub" font-size="0.229">1−α</tspan> − √</tspan><tspan x="3.88" y="7.788"><tspan font-style="italic">λ</tspan><tspan baseline-shift="sub" font-size="0.229">eff</tspan><tspan font-style="italic">/2</tspan></tspan></text>\n      <line x1="3.80" y1="7.55" x2="4.58" y2="7.55" stroke="#000000" stroke-width="0.014"/>')
rep('<rect x="1.54" y="8.3" width="3.42" height="1.0" rx="0.141"',
    '<rect x="1.54" y="8.6" width="3.42" height="1.8" rx="0.141"')
rep('<text x="3.25" y="8.64" font-family="Arial, Helvetica, sans-serif" font-size="0.3175" text-anchor="middle">Theorem 4: Nonlinear Extension<tspan x="3.25" dy="0.406">linear estimate is conservative</tspan></text>',
    '<text x="3.25" y="8.92" font-family="Arial, Helvetica, sans-serif" font-size="0.3175" text-anchor="middle">Theorem 4: Nonlinear<tspan x="3.25" dy="0.406">Extension</tspan><tspan x="3.25" dy="0.406">linear estimate</tspan><tspan x="3.25" dy="0.406">is conservative</tspan></text>')
rep('<text transform="rotate(-90 0.64 6.9)" x="0.64" y="6.9"',
    '<text transform="rotate(-90 0.64 6.65)" x="0.64" y="6.65"')

# ---------------------------------------------------------------- links
rep('<path d="M 4.96 3.4 H 10.54 V 4.8"', '<path d="M 4.96 3.6 H 10.54 V 4.8"')
rep('<path d="M 4.96 5.2 H 10.54 V 8.375"', '<path d="M 4.96 5.275 H 10.54 V 8.375"')
rep('<path d="M 4.96 7.0 H 10.54 V 12.505"', '<path d="M 4.96 7.2 H 10.54 V 12.505"')
rep('<path d="M 4.96 8.8 H 10.54 V 16.635"', '<path d="M 4.96 9.5 H 10.54 V 16.635"')

# ------------------------------------------------ matrix subscript commas
for a, b in [('11','1,1'),('12','1,2'),('1T','1,T'),('21','2,1'),('22','2,2'),
             ('2T','2,T'),('N1','N,1'),('N2','N,2'),('NT','N,T'),
             ('K1','K,1'),('KT','K,T')]:
    rep(f'>{a}<', f'>{b}<', n={'11':4,'1T':4,'K1':2,'KT':2}.get(a, 2))

# ---------------------------------------------------------------- layer 1
rep('<rect x="10.54" y="4.3" width="5.0" height="1.0" rx="0.106"',
    '<rect x="10.54" y="4.3" width="5.0" height="1.35" rx="0.106"')
rep('<text x="13.04" y="4.64" font-family="Arial, Helvetica, sans-serif" font-size="0.3175" font-weight="bold" text-anchor="middle">Layer 1: Chronology-Aware Stacking<tspan x="13.04" dy="0.406" font-weight="normal">',
    '<text x="13.04" y="4.64" font-family="Arial, Helvetica, sans-serif" font-size="0.3175" font-weight="bold" text-anchor="middle">Layer 1: Chronology-<tspan x="13.04" dy="0.406">Aware Stacking</tspan><tspan x="13.04" dy="0.406" font-weight="normal">')
rep('<line x1="13.04" y1="5.3" x2="13.04" y2="7.7"',
    '<line x1="13.04" y1="5.65" x2="13.04" y2="7.7"')
rep('<line x1="15.54" y1="4.8" x2="16.54" y2="4.8"',
    '<line x1="15.54" y1="4.975" x2="16.54" y2="4.975"')
rep('<polygon points="16.54,4.8 16.24,4.7 16.24,4.9"',
    '<polygon points="16.54,4.975 16.24,4.875 16.24,5.075"')
rep('<text x="16.04" y="4.58"', '<text x="16.04" y="4.755"')
rep('<text x="6.7" y="5.75"', '<text x="6.7" y="5.85"')
rep('<tspan>~ </tspan><tspan font-style="italic">N</tspan>',
    '<tspan>∼ </tspan><tspan font-family="' + CM + '">𝒩</tspan>')

# ---------------------------------------------------------------- layer 2
rep('<tspan font-weight="bold" font-style="italic">R̃</tspan><tspan> = </tspan><tspan font-weight="bold" font-style="italic">R</tspan><tspan> − </tspan><tspan font-style="italic">α</tspan><tspan font-weight="bold" font-style="italic">Θ̂</tspan><tspan font-weight="bold" font-style="italic">P</tspan>',
    '<tspan font-weight="bold" font-style="italic">R</tspan><tspan dy="-0.08" dx="-0.15" font-size="0.229">˜</tspan><tspan dx="0.05"> = </tspan><tspan font-weight="bold" font-style="italic">R</tspan><tspan> − </tspan><tspan font-style="italic">α</tspan><tspan font-weight="bold" font-style="italic">Θ</tspan><tspan dy="-0.08" dx="-0.16" font-size="0.229">ˆ</tspan><tspan font-weight="bold" font-style="italic" dx="0.07">P</tspan>')
for sub in ['1,1', '1,T', 'K,1', 'K,T']:
    rep(f'<tspan font-style="italic">R̃</tspan><tspan baseline-shift="sub" font-size="0.141" font-style="italic">{sub}</tspan>',
        f'<tspan font-style="italic">R</tspan><tspan dy="-0.05" dx="-0.09" font-size="0.141">˜</tspan><tspan baseline-shift="sub" font-size="0.141" font-style="italic" dx="0.01">{sub}</tspan>')
rep('<tspan font-weight="bold" font-style="italic">R̃</tspan><tspan baseline-shift="sub" font-size="0.229" font-weight="bold" font-style="italic">K×T</tspan>',
    '<tspan font-weight="bold" font-style="italic">R</tspan><tspan dy="-0.08" dx="-0.15" font-size="0.229">˜</tspan><tspan baseline-shift="sub" font-size="0.229" font-weight="bold" font-style="italic" dx="0.05">K×T</tspan>')
rep('<tspan font-weight="bold" font-style="italic">Θ̂</tspan><tspan>: OLS</tspan>',
    '<tspan font-weight="bold" font-style="italic">Θ</tspan><tspan dy="-0.07" dx="-0.145" font-size="0.229">ˆ</tspan><tspan dx="0.055">: OLS</tspan>')

# ---------------------------------------------------------------- layer 3
rep('<text x="13.04" y="11.79" font-family="Arial, Helvetica, sans-serif" font-size="0.3175" font-weight="bold" text-anchor="middle">Layer 3: Chronology-Perturbed<tspan x="13.04" dy="0.406">Causal Test</tspan>',
    '<text x="13.04" y="11.79" font-family="Arial, Helvetica, sans-serif" font-size="0.3175" font-weight="bold" text-anchor="middle">Layer 3: Chronology-<tspan x="13.04" dy="0.406">Perturbed Causal Test</tspan>')
for sub in ['t−1', 't−p']:
    rep(f'<tspan font-weight="bold" font-style="italic">R̃</tspan><tspan baseline-shift="sub" font-size="0.229" font-style="italic">{sub}</tspan>',
        f'<tspan font-weight="bold" font-style="italic">R</tspan><tspan dy="-0.08" dx="-0.15" font-size="0.229">˜</tspan><tspan baseline-shift="sub" font-size="0.229" font-style="italic" dx="0.05">{sub}</tspan>')
rep('<tspan>]</tspan></tspan><tspan x="13.04" dy="0.406" font-weight="normal">',
    '<tspan>], </tspan></tspan><tspan x="13.04" dy="0.406" font-weight="normal">')
rep('<tspan x="13.04" dy="0.406" font-weight="normal"><tspan font-style="italic">F</tspan><tspan baseline-shift="sub" font-size="0.229" font-style="italic">ij</tspan><tspan> = (</tspan><tspan font-style="italic">SSR</tspan><tspan baseline-shift="sub" font-size="0.229" font-style="italic">R</tspan><tspan> − </tspan><tspan font-style="italic">SSR</tspan><tspan baseline-shift="sub" font-size="0.229" font-style="italic">U</tspan><tspan>)/</tspan><tspan font-style="italic">q</tspan></tspan><tspan x="13.04" dy="0.406" font-weight="normal"><tspan font-style="italic">SSR</tspan><tspan baseline-shift="sub" font-size="0.229" font-style="italic">U</tspan><tspan>/(</tspan><tspan font-style="italic">n</tspan><tspan> − </tspan><tspan font-style="italic">p</tspan><tspan> − </tspan><tspan font-style="italic">q</tspan><tspan> − 1)</tspan></tspan></text>',
    '<tspan x="11.66" y="13.08" font-weight="normal"><tspan font-style="italic">F</tspan><tspan baseline-shift="sub" font-size="0.229" font-style="italic">ij</tspan><tspan> = </tspan></tspan><tspan x="12.41" y="12.93" font-weight="normal" font-size="0.229"><tspan>(</tspan><tspan font-style="italic">SSR</tspan><tspan baseline-shift="sub" font-size="0.161" font-style="italic">R</tspan><tspan> − </tspan><tspan font-style="italic">SSR</tspan><tspan baseline-shift="sub" font-size="0.161" font-style="italic">U</tspan><tspan>)/</tspan><tspan font-style="italic">q</tspan></tspan><tspan x="12.41" y="13.19" font-weight="normal" font-size="0.229"><tspan font-style="italic">SSR</tspan><tspan baseline-shift="sub" font-size="0.161" font-style="italic">U</tspan><tspan>/(</tspan><tspan font-style="italic">n</tspan><tspan> − </tspan><tspan font-style="italic">p</tspan><tspan> − </tspan><tspan font-style="italic">q</tspan><tspan> − 1)</tspan></tspan></text>\n      <line x1="12.36" y1="13.06" x2="14.42" y2="13.06" stroke="#000000" stroke-width="0.028"/>')
for sub in ['1,t−1', 'K,t−p', '1,T−1', 'K,T−p']:
    rep(f'<tspan font-style="italic">R̃</tspan><tspan baseline-shift="sub" font-size="0.141" font-style="italic">{sub}</tspan>',
        f'<tspan font-style="italic">R</tspan><tspan dy="-0.05" dx="-0.09" font-size="0.141">˜</tspan><tspan baseline-shift="sub" font-size="0.141" font-style="italic" dx="0.01">{sub}</tspan>')

# ---------------------------------------------------------------- layer 4
rep('<tspan x="13.04" dy="0.406" font-weight="normal"><tspan font-style="italic">S</tspan><tspan baseline-shift="sub" font-size="0.229" font-style="italic">i→j</tspan><tspan> = (1/</tspan><tspan font-style="italic">M</tspan><tspan>) Σ</tspan><tspan baseline-shift="sub" font-size="0.229" font-style="italic">m=1</tspan><tspan baseline-shift="super" font-size="0.229" font-style="italic">M</tspan><tspan> 1(</tspan><tspan font-style="italic">F</tspan><tspan baseline-shift="sub" font-size="0.229" font-style="italic">ij</tspan><tspan baseline-shift="super" font-size="0.229">(m)</tspan><tspan> &gt; </tspan><tspan font-style="italic">F</tspan><tspan baseline-shift="sub" font-size="0.229">crit</tspan><tspan>),</tspan></tspan>',
    '<tspan x="11.25" y="16.30" font-weight="normal"><tspan font-style="italic">S</tspan><tspan baseline-shift="sub" font-size="0.229" font-style="italic">i→j</tspan><tspan> = </tspan></tspan><tspan x="12.12" y="16.09" font-weight="normal" font-size="0.229">1</tspan><tspan x="12.12" y="16.40" font-weight="normal" font-size="0.229" font-style="italic">M</tspan><tspan x="12.50" y="16.30" font-weight="normal">Σ</tspan><tspan x="12.55" y="16.02" font-weight="normal" font-size="0.229" font-style="italic">M</tspan><tspan x="12.55" y="16.53" font-weight="normal" font-size="0.229" font-style="italic">m=1</tspan><tspan x="13.02" y="16.30" font-weight="normal"><tspan font-weight="bold">1</tspan><tspan>(</tspan><tspan font-style="italic">F</tspan><tspan baseline-shift="sub" font-size="0.229" font-style="italic">ij</tspan><tspan baseline-shift="super" font-size="0.229">(m)</tspan><tspan> &gt; </tspan><tspan font-style="italic">F</tspan><tspan baseline-shift="sub" font-size="0.229">crit</tspan><tspan>),</tspan></tspan>')
rep('<line x1="15.54" y1="16.635" x2="16.54" y2="16.635"',
    '<line x1="12.08" y1="16.22" x2="12.38" y2="16.22" stroke="#000000" stroke-width="0.028"/>\n      <line x1="15.54" y1="16.635" x2="16.54" y2="16.635"')

# ---------------------------------------------------------------- decision
rep('<tspan font-style="italic">G</tspan><tspan> = {(</tspan>',
    '<tspan font-family="' + CM + '">𝒢</tspan><tspan> = {(</tspan>')

if errs:
    print('ERRORS:'); [print(' ', e) for e in errs]; sys.exit(1)

assert 'R̃' not in s and 'Θ̂' not in s and '𝔼[' not in s, 'combining/board glyphs remain'
open(P, 'w', encoding='utf-8', newline='\n').write(s)
print(f'OK: {len(s) - len(orig):+d} chars written to {P}')
