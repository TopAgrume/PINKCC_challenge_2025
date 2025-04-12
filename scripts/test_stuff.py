# %%
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
