from pathlib import Path

import nibabel as nib
from nibabel.orientations import (
  aff2axcodes,
  apply_orientation,
  axcodes2ornt,
  inv_ornt_aff,
  ornt_transform,
)
from tqdm import tqdm

filenames = [
  "4.nii.gz",
  "41.nii.gz",
  "3.nii.gz",
  "6.nii.gz",
  "15.nii.gz",
  "32.nii.gz",
  "30.nii.gz",
  "37.nii.gz",
  "5.nii.gz",
  "35.nii.gz",
  "20.nii.gz",
  "27.nii.gz",
  "7.nii.gz",
  "45.nii.gz",
  "1.nii.gz",
  "36.nii.gz",
  "50.nii.gz",
  "39.nii.gz",
  "46.nii.gz",
  "24.nii.gz",
  "10.nii.gz",
  "13.nii.gz",
  "31.nii.gz",
  "18.nii.gz",
  "34.nii.gz",
  "14.nii.gz",
  "47.nii.gz",
  "22.nii.gz",
  "40.nii.gz",
  "19.nii.gz",
  "49.nii.gz",
  "38.nii.gz",
  "8.nii.gz",
  "28.nii.gz",
  "21.nii.gz",
  "33.nii.gz",
  "43.nii.gz",
  "9.nii.gz",
  "2.nii.gz",
  "26.nii.gz",
  "16.nii.gz",
  "12.nii.gz",
  "29.nii.gz",
  "44.nii.gz",
  "11.nii.gz",
  "48.nii.gz",
  "17.nii.gz",
  "42.nii.gz",
  "25.nii.gz",
  "23.nii.gz",
]

preds_name_to_id = {
  f"OvarianCancerDestroyer_{k}_0000.nii.gz": v
  for k, v in zip(range(50), filenames, strict=False)
}

output_dir = Path("preds")
output_dir.mkdir(exist_ok=True)

preds_dir = Path(
  "../old_results/unetr_pp_results/unetr_pp/3d_fullres/Task001_PINKCC/unetr_pp_trainer_synapse__unetr_pp_Plansv2.1/fold_0/validation_raw"
)

for k, v in tqdm(preds_name_to_id):
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
