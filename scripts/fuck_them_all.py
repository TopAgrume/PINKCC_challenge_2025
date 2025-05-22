from pathlib import Path

import nibabel as nib
import numpy as np

from ocd.dataset.dataset import SampleUtils

name = "TCGA-13-0768.nii.gz"
p = Path("DatasetChallenge")

file_scan, data_scan = SampleUtils.load_from_path(p / "CT" / "TCGA" / name)
file_seg, data_seg = SampleUtils.load_from_path(p / "Segmentation" / "TCGA" / name)

# new_scan = nib.nifti1.Nifti1Image(data_scan[:, :, 42:], file_scan.affine)
# nib.loadsave.save(new_scan, "scan_" + name)
new_seg = nib.nifti1.Nifti1Image(
  np.concatenate([data_seg[:, :, 1:], np.zeros((512, 512, 1))], axis=2), file_seg.affine
)
nib.loadsave.save(new_seg, "seg_" + name)
