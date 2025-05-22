from pathlib import Path

import dicom2nifti

dicom2nifti.dicom_series_to_nifti(
  Path("2.000000-AXIAL-59179"), Path("output") / "test.nii.gz"
)
