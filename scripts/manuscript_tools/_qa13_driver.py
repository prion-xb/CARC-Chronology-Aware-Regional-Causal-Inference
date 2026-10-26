"""QA driver for 13_publication_figures.py — runs only the three figures
delivered to the package (map, regional stacks, record quality), skipping
the DDR network and validation figures."""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))

spec = importlib.util.spec_from_file_location(
    'pubfigs', os.path.join(HERE, '..', '13_publication_figures.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

quality_df = m.load_record_quality()
if quality_df is not None:
    print(f"Loaded quality data: {len(quality_df)} records")
m.figure1_map(quality_df, m.FIGURE_DIR)
m.figure2_regional_stacks(m.FIGURE_DIR)
m.figure3_record_quality(quality_df, m.FIGURE_DIR)
print("Done.")
