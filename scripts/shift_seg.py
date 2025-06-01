from pathlib import Path

import nibabel as nib
import numpy as np

PATH = Path("DatasetChallengeV2") / "Segmentation" / "TCGA-13-0793.nii.gz"

file = nib.loadsave.load(PATH)
data = file.get_fdata()

data = np.concatenate([np.zeros((512, 512, 1)), data[:, :, :-1]], axis=2)
pred_file = nib.nifti1.Nifti1Image(
  data.astype(np.float64),
  file.affine,  # pyright: ignore
)
nib.loadsave.save(pred_file, "TCGA-13-0793.nii.gz")
