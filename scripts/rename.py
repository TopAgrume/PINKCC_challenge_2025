import os
from pathlib import Path

p = Path("output_ct_scans_corrected")
for f in os.listdir(p):
  os.rename(p / f, p / f"{f.split('_')[0]}.nii.gz")
