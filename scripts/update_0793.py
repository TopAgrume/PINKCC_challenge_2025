from pathlib import Path

import nibabel as nib
from nibabel.orientations import (
  aff2axcodes,
  apply_orientation,
  axcodes2ornt,
  inv_ornt_aff,
  ornt_transform,
)


def crop_first_physical_slice(seg_path: Path, output_path: Path):
  print(f"Loading: {seg_path}")
  img = nib.load(str(seg_path))
  data = img.get_fdata()
  affine = img.affine
  header = img.header.copy()

  # Reorient to RAS (Right-Anterior-Superior)
  current_axcodes = aff2axcodes(affine)
  target_axcodes = ("R", "A", "S")
  print(f"Current orientation: {current_axcodes} -> Target: {target_axcodes}")

  current_ornt = axcodes2ornt(current_axcodes)
  target_ornt = axcodes2ornt(target_axcodes)
  transform = ornt_transform(current_ornt, target_ornt)

  data_ras = apply_orientation(data, transform)
  affine_ras = affine @ inv_ornt_aff(transform, data.shape)

  print(f"Shape after RAS orientation: {data_ras.shape}")

  # Crop the FIRST slice along the superior (Z) axis
  data_cropped = data_ras[:, :, 1:]

  # Update affine translation to reflect the removed slice
  affine_ras[:3, 3] += affine_ras[:3, 2]  # move origin forward by 1 voxel in Z

  # Save new image
  cropped_img = nib.Nifti1Image(data_cropped, affine_ras, header)
  nib.save(cropped_img, str(output_path))
  print(f"Saved cropped segmentation (first slice removed): {output_path}")


if __name__ == "__main__":
  seg_path = Path(
    "/home/alex/PINKCC_challenge_2025/DatasetChallenge/Segmentation/TCGA/TCGA-13-0793.nii.gz"
  )
  output_path = Path(
    "/home/alex/PINKCC_challenge_2025/DatasetChallenge/Segmentation/TCGA/TCGA-13-0793.nii.gz"
  )
  crop_first_physical_slice(seg_path, output_path)
