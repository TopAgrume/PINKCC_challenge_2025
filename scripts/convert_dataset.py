import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.orientations import (
  aff2axcodes,
  apply_orientation,
  axcodes2ornt,
  inv_ornt_aff,
  ornt_transform,
)

from ocd.dataset.dataset import Dataset, SampleUtils

INVERTED_SCAN_IDS = [
  "330684",
  "330697",
  "330699",
  "330718",
  "333015",
  "333017",
  "333020",
  "333021",
  "333022",
  "333036",
  "374142",
  "374143",
  "374149",
  "374177",
  "374196",
  "TCGA-13-0799",
  "TCGA-13-0904",
  "TCGA-13-1405",
  "TCGA-13-1488",
  "TCGA-13-1499",
  "TCGA-13-1507",
  "TCGA-13-1511",
  "TCGA-61-1906",
  "TCGA-61-2016",
  "TCGA-13-0724",
]

SKIP_SCAN_IDS = [
  "TCGA-09-2054",
  "TCGA-24-1614",
  "TCGA-09-0364",
  "333076",
  # "TCGA-13-0768",
]

CROP_SCAN_ID = "333020"
CROP_START_Z = 43

REMOVE_LAST_SLICE = "TCGA-13-0724"

TEST = ["333020", "TCGA-13-0793"]


def is_test_scan(file_path):
  """Check if the scan is in the list of inverted scans"""
  return any(scan_id in str(file_path) for scan_id in TEST)


def is_inverted_scan(file_path):
  """Check if the scan is in the list of inverted scans"""
  return any(scan_id in str(file_path) for scan_id in INVERTED_SCAN_IDS)


def needs_cropping(file_path):
  """Check if the scan needs z-axis cropping"""
  return CROP_SCAN_ID in str(file_path)


def needs_remove_last_slice(file_path):
  return REMOVE_LAST_SLICE in str(file_path)


def should_skip_scan(file_path):
  """Check if the scan should be skipped"""
  return any(scan_id in str(file_path) for scan_id in SKIP_SCAN_IDS)


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

  if ct_image:
    print(f"Applying custom normalization to: {nifti_file_path}")
    data = SampleUtils.normalize_ct(ct_data=data)
    modified = True
    print("Normalization complete.")

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

  if is_inverted_scan(nifti_file_path):
    print(f"Found inverted scan, flipping superior-inferior axis: {nifti_file_path}")

    # Flip the data along the superior-inferior axis (typically axis 2 in LPS)
    data = np.flip(data, axis=2)

    # Update the affine to reflect the flip
    # We need to negate the third column of the affine matrix and adjust the translation
    flip_mat = np.eye(4)
    flip_mat[2, 2] = -1
    flip_mat[2, 3] = data.shape[2] - 1

    affine = affine @ flip_mat
    modified = True

  # Check if this scan needs z-axis cropping
  if needs_cropping(nifti_file_path):
    print(
      f"Cropping scan along z-axis from index {CROP_START_Z} to end: {nifti_file_path}"
    )

    # Crop the data along the z-axis (keep slices from CROP_START_Z to end)
    original_z_size = data.shape[2]
    data = data[:, :, :-CROP_START_Z]
    modified = True

    print(
      f"  Original z-dimension: {original_z_size}, New z-dimension: {data.shape[2]}"
    )

  if needs_remove_last_slice(nifti_file_path):
    data = data[:, :, 1:]
    modified = True

  if is_segmentation:
    print(f" Ensuring integer labels for segmentation: {nifti_file_path}")
    # Round to nearest integer and convert to int type
    unique_before = np.unique(data)
    data = np.round(data).astype(np.int32)
    unique_after = np.unique(data)
    print(f" Labels before: {unique_before}")
    print(f" Labels after:  {unique_after}")
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
  """
  Convert the dataset to nnUNet format

  Args:
      dataset_dir: Input dataset directory (DatasetChallenge)
      output_dir: Output directory for nnUNet format
      val_split: Percentage of data to use for validation
      test_split: Percentage of data to use for testing
  """
  print("\n=== Special Case Handling ===")
  print(f"Inverted scans that will be flipped: {', '.join(INVERTED_SCAN_IDS)}")
  print(f"Scans that will be skipped: {', '.join(SKIP_SCAN_IDS)}")
  print(f"Scan that will be cropped from z={CROP_START_Z} to end: {CROP_SCAN_ID}")
  print("===========================\n")

  # --- nnUNet Directory Structure ---
  task_name = "OvarianCancerDestroyer"
  nb_tasks = (
    len([f for f in os.listdir(output_dir) if os.path.isdir(output_dir / f)])
    if output_dir.exists()
    else 0
  )
  task_id = nb_tasks + 1
  task_dir = output_dir / f"Dataset{task_id:03d}_{task_name}"
  images_dir = task_dir / "imagesTr"
  labels_dir = task_dir / "labelsTr"
  images_test_dir = task_dir / "imagesTs"

  os.makedirs(images_dir, exist_ok=True)
  os.makedirs(labels_dir, exist_ok=True)
  os.makedirs(images_test_dir, exist_ok=True)
  print(f"Created nnUNet directory structure at: {task_dir}")

  # --- Load and Split Data ---
  ds = Dataset(base_dir=dataset_dir, random_state=42)
  train_pairs, val_pairs, test_pairs = ds.get_dataset_splits()
  all_training_pairs = train_pairs + val_pairs + test_pairs

  training_count = sum(
    1
    for pair in all_training_pairs
    if not (should_skip_scan(pair[0]) or should_skip_scan(pair[1]))
  )
  test_count = sum(
    1
    for pair in test_pairs
    if not (should_skip_scan(pair[0]) or should_skip_scan(pair[1]))
  )

  dataset_json = {
    "name": task_name,
    "description": "Ovarian cancer segmentation from CT scans",
    "reference": "",
    "licence": "",
    "release": "1.0",
    "channel_names": {
      "0": "CT",
    },
    "labels": {
      "background": 0,
      "primary_tumor": 1,
      "metastasis": 2,
    },
    "numTraining": training_count,
    "numTest": 0,
    "file_ending": ".nii.gz",
    "training": [],
    "test": [],
  }

  if False:
    # --- Process Training/Validation Data ---
    print(f"\nProcessing {len(all_training_pairs)} training/validation cases...")
    valid_case_count = 0
    for i, (ct_path, seg_path) in enumerate(train_pairs + val_pairs + test_pairs):
      if should_skip_scan(ct_path) or should_skip_scan(seg_path):
        print(f"\nSkipping Case {i}: {ct_path}")
        continue

      case_id = f"{valid_case_count:03d}"
      valid_case_count += 1

      print(f"\nProcessing Case {case_id}...")
      print(f"  Input CT: {ct_path}")
      ct_dest = images_dir / f"{task_name}_{case_id}_0000.nii.gz"
      std_ct_img, modif = standardize_orientation(
        ct_path, ct_image=False, is_segmentation=False
      )
      print(f"  Saving processed CT to: {ct_dest}")
      nib.save(std_ct_img, ct_dest)  # type: ignore

      print(f"  Input Seg: {seg_path}")
      seg_dest = labels_dir / f"{task_name}_{case_id}.nii.gz"
      std_seg_img, modif = standardize_orientation(
        seg_path, ct_image=False, is_segmentation=True
      )
      print(f"  Saving processed Seg to: {seg_dest}")
      nib.save(std_seg_img, seg_dest)  # type: ignore

      seg_data = std_seg_img.get_fdata()  # type: ignore
      unique_values = np.unique(seg_data)
      print(
        f"  Case {case_id}: Unique segmentation values after processing: {unique_values}"
      )

      dataset_json["training"].append(
        {
          "image": f"./imagesTr/{task_name}_{case_id}_0000.nii.gz",
          "label": f"./labelsTr/{task_name}_{case_id}.nii.gz",
        }
      )

  # --------------- ADDING FOR TESTING --------------
  print("ON PASSE AU TEST SET C'EST PARTIIII C'EST PARTIIIIIIIIII")
  valid_case_count = 246
  TEST_DIR = Path("../TEST_SET")
  all_paths = os.listdir(TEST_DIR)
  for path in all_paths:
    case_id = f"{valid_case_count:03d}"
    valid_case_count += 1

    print(f"\nProcessing Case {case_id}...")
    print(f"  Input CT: {path}")
    ct_dest = images_dir / f"{task_name}_{case_id}_0000.nii.gz"
    std_ct_img, _ = standardize_orientation(
      str(TEST_DIR / path), ct_image=False, is_segmentation=False
    )
    print(f"  Saving processed CT to: {ct_dest}")
    nib.save(std_ct_img, ct_dest)  # type: ignore

    false_label = nib.nifti1.Nifti1Image(
      np.zeros(std_ct_img.get_fdata().shape),  # pyright: ignore
      std_ct_img.affine,  # pyright: ignore
      std_ct_img.header,
    )
    seg_dest = labels_dir / f"{task_name}_{case_id}.nii.gz"
    nib.save(false_label, seg_dest)  # type: ignore

    dataset_json["training"].append(
      {
        "image": f"./imagesTr/{task_name}_{case_id}_0000.nii.gz",
        "label": f"./labelsTr/{task_name}_{case_id}.nii.gz",
      }
    )

  print("\nWriting dataset.json")
  with open(task_dir / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=4)

  print("\nDataset conversion complete.")
  print(f"nnUNet Task Name: {task_name}")
  print(f"Task ID: {task_id}")
  print(f"Data location: {task_dir}")
  return task_dir


if __name__ == "__main__":
  dataset_dir = Path("DatasetChallenge")
  output_dir = Path("nnUNet_raw_data")

  os.makedirs(output_dir, exist_ok=True)
  convert_dataset_to_nnunet_format(dataset_dir, output_dir)
