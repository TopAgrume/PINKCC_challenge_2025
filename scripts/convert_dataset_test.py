import json
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


def standardize_orientation(
  nifti_file_path: str, ct_image: bool = False, is_segmentation: bool = False
):
  """Ensure RAS+ orientation (standard for nnU-Net)"""
  img = nib.load(nifti_file_path)  # type: ignore
  affine = img.affine  # type: ignore
  data = img.get_fdata()  # type: ignore
  modified = False

  # check current orientation
  axcodes = aff2axcodes(affine)

  # nnUNet expects data in LPS orientation (Left, Posterior, Superior))
  # See doc: http://www.grahamwideman.com/gw/brain/orientation/orientterms.htm
  if axcodes != ("L", "P", "S"):
    print(
      f"Non-compliant orientation {axcodes}, converting to LPS: {nifti_file_path}..."
    )

    current_ornt = axcodes2ornt(axcodes)
    target_ornt = axcodes2ornt(("L", "P", "S"))

    transform = ornt_transform(current_ornt, target_ornt)
    reoriented_data = apply_orientation(data, transform)
    new_affine = affine @ inv_ornt_aff(transform, data.shape)

    # New nifti image with the reoriented data
    data = reoriented_data
    affine = new_affine
    modified = True

  # Create the final image
  if modified:
    new_img = nib.Nifti1Image(data, affine, img.header)  # type: ignore
    return new_img, modified
  else:
    return img, False


def convert_dataset_to_nnunet_format(
  dataset_dir: Path, output_dir: Path, val_split=0.2, test_split=0.15
) -> Path:
  task_name = "Task001_MSKCC"
  task_dir = output_dir / task_name
  images_dir = task_dir / "imagesTr"
  labels_dir = task_dir / "labelsTr"
  images_test_dir = task_dir / "imagesTs"

  os.makedirs(images_dir, exist_ok=True)
  os.makedirs(labels_dir, exist_ok=True)
  os.makedirs(images_test_dir, exist_ok=True)
  print(f"Created nnUNet directory structure at: {task_dir}")

  all_paths = os.listdir(dataset_dir)

  dataset_json = {
    "name": task_name,
    "description": "Ovarian cancer segmentation from CT scans",
    "reference": "",
    "licence": "",
    "release": "1.0",
    "modality": {
      "0": "CT",
    },
    "labels": {
      "0": "background",
      "1": "primary_tumor",
      "2": "metastasis",
    },
    "numTraining": 0,
    "numTest": 0,
    "file_ending": ".nii.gz",
    "training": [],
    "test": len(all_paths),
  }

  for i, path in enumerate(all_paths):
    case_id = f"{i:03d}"

    print(f"\nProcessing Case {case_id}...")
    print(f"  Input CT: {path}")
    ct_dest = images_dir / f"{task_name}_{case_id}_0000.nii.gz"
    std_ct_img, _ = standardize_orientation(path, ct_image=False, is_segmentation=False)
    print(f"  Saving processed CT to: {ct_dest}")
    nib.save(std_ct_img, ct_dest)  # type: ignore

    dataset_json["training"].append(
      {
        "image": f"./imagesTr/{task_name}_{case_id}_0000.nii.gz",
      }
    )

  print("\nWriting dataset.json")
  with open(task_dir / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=4)

  return task_dir


if __name__ == "__main__":
  dataset_dir = Path("../TEST_SET")
  output_dir = Path("nnFormer_raw_data")

  os.makedirs(output_dir, exist_ok=True)
  convert_dataset_to_nnunet_format(dataset_dir, output_dir)
