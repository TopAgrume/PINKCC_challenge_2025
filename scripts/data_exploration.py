import os
from pathlib import Path

import nibabel as nib
from tqdm import tqdm

from ocd.dataset.dataset import SampleUtils

p = Path("DatasetChallenge")

for f in tqdm(os.listdir(p / "CT")):
  file_scan, data_scan = SampleUtils.load_from_path(p / "CT" / f)
  _, data_seg = SampleUtils.load_from_path(p / "Segmentation" / f)

  new_scan = nib.nifti1.Nifti1Image(data_seg, file_scan.affine)
  nib.loadsave.save(new_scan, p / "Segmentation" / f)

# new_seg = nib.nifti1.Nifti1Image(data_seg[:, :, 42:], file_seg.affine)
# nib.loadsave.save(new_seg, "seg_" + name)
