"""Deliver the QA-passed figures into the CARC_final2 submission package.

Copies each rendered figure to Figures/正文 or Figures/supplementary under
the journal file names (Fig1.pdf … Fig7.tif, FigS1.png … FigS10.tif), then
runs the acceptance checks:

  * raster width = 2100 px (178 mm at 300 dpi)
  * PNG/TIFF mode RGB (no alpha) and TIFF LZW-compressed
  * Fig1.pdf has embedded fonts (vector, replaces the old JPG)

Nothing is overwritten silently: an existing destination is moved to
Figures/_backup_orig first.

Conflicts to report to the user (docx claims both): FigS3 == Fig2 content
(33_layer1_stacking) and FigS7 == Fig6 content (17_regional_pairwise).
"""
import os
import shutil
import sys

REPO = r"D:\PythonWorkSpace\pythonProject5\CARC_submit\figures"
PKG = r"D:\PythonWorkSpace\pythonProject6\二\CARC_final2\Figures"
MAIN = os.path.join(PKG, "正文")
SUPP = os.path.join(PKG, "supplementary")
BACKUP = os.path.join(PKG, "_backup_orig")

# source file (repo figures/) -> destination (package)
MAIN_MANIFEST = [
    ("Fig2_CARC_Framework.pdf", "Fig1.pdf"),            # 32 framework
    ("Fig2_CARC_Framework.png", "Fig1.png"),
    ("FigS_Layer1_Stacking_10-60.tif", "Fig2.tif"),     # 33 (docx Fig. 2)
    ("Fig6_MultiParameter_Sensitivity.tif", "Fig3.tif"),# 14
    ("Fig7_SlidingWindow_CARC.tif", "Fig4.tif"),        # 15
    ("Fig5_Global_Attribution_Scrutiny.tif", "Fig5.tif"),  # 41
    ("FigS_Regional_Pairwise_Matrix.tif", "Fig6.tif"),  # 17
    ("Fig8_Escape_Diagnosis.tif", "Fig7.tif"),          # 31
]
SUPP_MANIFEST = [
    ("FigS_Record_Atlas.png", "FigS1.png"),             # 20
    ("FigS_Record_Strip.png", "FigS2.png"),             # 20
    ("FigS_Layer1_Stacking_10-60.png", "FigS3.png"),    # 33 (same as Fig2 — docx conflict, see header)
    ("FigS_Conditional_Causality.png", "FigS4.png"),    # 06
    ("FigS_SixRecord_Pairwise_Matrix.png", "FigS5.png"),# 06
    ("FigS_Directional_Robustness.png", "FigS6.png"),   # 09
    ("FigS_Regional_Pairwise_Matrix.tif", "FigS7.tif"), # 17 (same as Fig6 — docx conflict, see header)
    ("scenarioA_pairwise_matrix.tif", "FigS8.tif"),     # 16
    ("Fig1_Global_Proxy_Map.png", "FigS9.png"),         # 13 fig1
    ("Fig3_Record_Quality_Summary.png", "FigS10.png"),  # 13 fig3
]

TIF_NAMES = {d for _, d in MAIN_MANIFEST + SUPP_MANIFEST if d.endswith(".tif")}
PDF_NAMES = {d for _, d in MAIN_MANIFEST + SUPP_MANIFEST if d.endswith(".pdf")}


def install(dst_dir, manifest):
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(BACKUP, exist_ok=True)
    for src_name, dst_name in manifest:
        src = os.path.join(REPO, src_name)
        dst = os.path.join(dst_dir, dst_name)
        if not os.path.exists(src):
            print(f"  MISSING SOURCE: {src}")
            continue
        if os.path.exists(dst):
            bak = os.path.join(BACKUP, dst_name)
            shutil.move(dst, bak)
            print(f"  backed up existing {dst_name} -> _backup_orig")
        shutil.copy2(src, dst)
        print(f"  {dst_name}  <-  {src_name}")


def check(dst_dir, manifest):
    from PIL import Image
    import re
    ok = True
    for _, dst_name in manifest:
        path = os.path.join(dst_dir, dst_name)
        if not os.path.exists(path):
            print(f"  FAIL {dst_name}: missing")
            ok = False
            continue
        if dst_name in TIF_NAMES or dst_name.endswith(".png"):
            im = Image.open(path)
            w, h = im.size
            if dst_name in TIF_NAMES:
                comp = im.info.get("compression")
                if w != 2100:
                    print(f"  FAIL {dst_name}: width {w} != 2100 px")
                    ok = False
                if im.mode != "RGB":
                    print(f"  FAIL {dst_name}: mode {im.mode} != RGB")
                    ok = False
                if comp != "tiff_lzw":
                    print(f"  FAIL {dst_name}: compression {comp} != LZW")
                    ok = False
            if dst_name.endswith(".png"):
                if w != 2100:
                    print(f"  FAIL {dst_name}: width {w} != 2100 px")
                    ok = False
                if im.mode != "RGB":
                    print(f"  FAIL {dst_name}: mode {im.mode} != RGB")
                    ok = False
        if dst_name in PDF_NAMES:
            raw = open(path, "rb").read()
            if b"/Type /Font" not in raw and b"/Type/ Font" not in raw \
                    and not re.search(rb"/FontFile\d?", raw):
                print(f"  FAIL {dst_name}: no embedded font found")
                ok = False
            else:
                print(f"  OK   {dst_name}: embedded fonts")
    return ok


def main():
    print("== 正文 ==")
    install(MAIN, MAIN_MANIFEST)
    print("== supplementary ==")
    install(SUPP, SUPP_MANIFEST)
    print("== acceptance ==")
    ok1 = check(MAIN, MAIN_MANIFEST)
    ok2 = check(SUPP, SUPP_MANIFEST)
    if ok1 and ok2:
        print("ALL DELIVERED FILES PASS")
    else:
        print("SOME FILES FAILED — see above")
        sys.exit(1)


if __name__ == "__main__":
    main()
