from pathlib import Path

import nibabel
from nibabel import nifti1
from nibabel.orientations import (
  aff2axcodes,
)

from scripts.convert_dataset import SampleUtils

if False:
  file, data = SampleUtils.load_from_path(
    Path("nnUNet_raw_data")
    / "Dataset001_OvarianCancerDestroyer"
    / "imagesTr"
    / "OvarianCancerDestroyer_245_0000.nii.gz"
  )
  scan, scan_data = SampleUtils.load_from_path(
    Path("nnUNet_raw_data")
    / "Dataset001_OvarianCancerDestroyer"
    / "labelsTr"
    / "OvarianCancerDestroyer_245.nii.gz"
  )

  axcodes = aff2axcodes(file.affine)
  print(axcodes)

  new_scan = nifti1.Nifti1Image(data[:, :, ::-1], file.affine)

  nibabel.loadsave.save(
    new_scan,
    Path("nnUNet_raw_data")
    / "Dataset001_OvarianCancerDestroyer"
    / "imagesTr"
    / "OvarianCancerDestroyer_245_0000.nii.gz",
  )

  new_seg = nifti1.Nifti1Image(scan_data[:, :, ::-1], scan.affine)
  nibabel.loadsave.save(
    new_seg,
    Path("nnUNet_raw_data")
    / "Dataset001_OvarianCancerDestroyer"
    / "labelsTr"
    / "OvarianCancerDestroyer_245.nii.gz",
  )

if True:
  file, data = SampleUtils.load_from_path(
    Path("nnUNet_raw_data")
    / "Dataset001_OvarianCancerDestroyer"
    / "imagesTr"
    / "OvarianCancerDestroyer_278_0000.nii.gz"
  )

  new_seg = nifti1.Nifti1Image(
    data.clip(min=data.min() / 10, max=data.max() / 10), file.affine
  )
  nibabel.loadsave.save(
    new_seg,
    Path("nnUNet_raw_data")
    / "Dataset001_OvarianCancerDestroyer"
    / "imagesTr"
    / "OvarianCancerDestroyer_278_0000.nii.gz",
  )
