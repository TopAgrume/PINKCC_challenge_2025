import os
from pathlib import Path

import nibabel as nib
from nibabel.orientations import (
  aff2axcodes,
  apply_orientation,
  axcodes2ornt,
  inv_ornt_aff,
  ornt_transform,
)

test_set_path = Path("../TEST_SET")
filenames = os.listdir(test_set_path)

preds_name_to_id = {
  f"OvarianCancerDestroyer_{k}_0000.nii.gz": v
  for k, v in zip(range(246, 296), filenames, strict=False)
}

output_dir = Path("preds")
output_dir.mkdir(exist_ok=True)

preds_dir = Path(
  "../old_results/unetr_pp_results/unetr_pp/3d_fullres/Task001_PINKCC/unetr_pp_trainer_synapse__unetr_pp_Plansv2.1/fold_0/validation_raw"
)

for k, v in preds_name_to_id:
  base_image_header = nib.loadsave.load(test_set_path / v).header

  img = nib.loadsave.load(preds_dir / k)
  affine = img.affine  # type: ignore
  data = img.get_fdata()  # type: ignore

  # check current orientation
  axcodes = aff2axcodes(affine)

  current_ornt = axcodes2ornt(axcodes)
  target_ornt = axcodes2ornt(("R", "A", "I"))

  transform = ornt_transform(current_ornt, target_ornt)
  reoriented_data = apply_orientation(data, transform)
  new_affine = affine @ inv_ornt_aff(transform, data.shape)

  data = reoriented_data
  affine = new_affine
  new_img = nib.nifti1.Nifti1Image(data, affine, base_image_header)
  nib.loadsave.save(new_img, output_dir / v)
