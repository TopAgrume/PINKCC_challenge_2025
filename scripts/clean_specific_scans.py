from pathlib import Path

import matplotlib.pyplot as plt

from ocd.dataset.dataset import SampleUtils

name = "333076.nii.gz"
p = Path("DatasetChallenge")

file_scan, data_scan = SampleUtils.load_from_path(p / "CT" / "MSKCC" / name)
file_seg, data_seg = SampleUtils.load_from_path(p / "Segmentation" / "MSKCC" / name)

plt.imshow(data_scan[:, :, 60])
plt.show()

plt.imshow(data_scan[:, :, 61])
plt.show()
# new_scan = nib.nifti1.Nifti1Image(data_scan[:, :, 42:], file_scan.affine)
# nib.loadsave.save(new_scan, "scan_" + name)
# new_seg = nib.nifti1.Nifti1Image(data_seg[:, :, 42:], file_seg.affine)
# nib.loadsave.save(new_seg, "seg_" + name)
