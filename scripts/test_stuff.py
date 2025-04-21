# %%
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ocd.dataset.dataset import SampleUtils

# %%
nifti = SampleUtils.load_from_path("../DatasetChallenge/CT/MSKCC/330680.nii.gz")

# %%
SampleUtils.display_slice(nifti[1], figsize=(10, 6))

# %%
SampleUtils.display_slice(SampleUtils.normalize_ct(nifti[1]), figsize=(10, 6))

# %%
nifti = SampleUtils.load_from_path(
  "../DatasetChallenge/Segmentation/MSKCC/347580.nii.gz"
)
# %%
type(nifti)
np.unique(nifti[1])

# %%
path = Path("..") / "2D_CT_SCANS" / "scan"
files = sorted(os.listdir(path))
files_prefix = set(map(lambda x: x.split("_")[0], files))
d = defaultdict(int)
for file in files:
  d[file.split("_")[0]] += 1
print(sorted(d.items(), key=lambda x: x[1]))

# %%
path = Path("..") / "2D_CT_SCANS" / "scan"
files = os.listdir(path)
files = set(map(lambda x: x.split("_")[0], files))
for file in files:
  plt.imshow(np.load(path / f"{file}_0.npy"))
  plt.title(file)
  plt.show()
