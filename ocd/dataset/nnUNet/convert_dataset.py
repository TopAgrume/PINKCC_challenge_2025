import os
import json
import shutil
import nibabel as nib
import numpy as np

from pathlib import Path
from ocd.dataset.dataset import Dataset, SampleUtils
from nibabel.orientations import (
    aff2axcodes, apply_orientation, axcodes2ornt,
    inv_ornt_aff, ornt_transform
)

def standardize_orientation(nifti_file_path: str, ct_image: bool = False):
    """Ensure RAS+ orientation (standard for nnU-Net)"""
    img = nib.load(nifti_file_path) # type: ignore
    affine = img.affine # type: ignore
    data = img.get_fdata() # type: ignore
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
        print(f"Non-compliant orientation {axcodes}, converting to LPS: {nifti_file_path}...")

        current_ornt = axcodes2ornt(axcodes)
        target_ornt = axcodes2ornt(("L", "P", "S"))

        transform = ornt_transform(current_ornt, target_ornt)
        reoriented_data = apply_orientation(data, transform)
        new_affine = affine @ inv_ornt_aff(transform, data.shape)

        # new nifti image with the reoriented data
        new_img = nib.Nifti1Image(reoriented_data, new_affine) # type: ignore

        modified = True
        return new_img, modified

    # if only normalization happened (no reorientation needed)
    elif modified:
        new_img = nib.Nifti1Image(data, affine, img.header) # type: ignore
        return new_img, modified

    # if neither normalization nor reorientation happened
    else:
        return img, False


def convert_dataset_to_nnunet_format(dataset_dir: Path, output_dir: Path, val_split=0.2, test_split=0.15) -> Path:
    """
    Convert the dataset to nnUNet format

    Args:
        dataset_dir: Input dataset directory (DatasetChallenge)
        output_dir: Output directory for nnUNet format
        val_split: Percentage of data to use for validation
        test_split: Percentage of data to use for testing
    """
    # --- nnUNet Directory Structure ---
    task_name = "OvarianCancerDestroyer"
    nb_tasks = len([f for f in os.listdir(output_dir) if os.path.isdir(output_dir / f)]) if output_dir.exists() else 0
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
    all_training_pairs = train_pairs + val_pairs

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
        "numTraining": len(all_training_pairs),
        "numTest": len(test_pairs),
        "file_ending": ".nii.gz",
        "training": [],
        "test": [],
    }

    # --- Process Training/Validation Data ---
    print(f"\nProcessing {len(all_training_pairs)} training/validation cases...")
    for i, (ct_path, seg_path) in enumerate(train_pairs + val_pairs):
        case_id = f"{i:03d}"

        print(f"\nProcessing Case {case_id}...")
        print(f"  Input CT: {ct_path}")
        ct_dest = images_dir / f"{task_name}_{case_id}_0000.nii.gz"
        std_ct_img, modif = standardize_orientation(ct_path, ct_image=True)
        print(f"  Saving processed CT to: {ct_dest}")
        nib.save(std_ct_img, ct_dest) # type: ignore

        print(f"  Input Seg: {seg_path}")
        seg_dest = labels_dir / f"{task_name}_{case_id}.nii.gz"
        std_seg_img, modif = standardize_orientation(seg_path, ct_image=False)
        print(f"  Saving processed Seg to: {seg_dest}")
        nib.save(std_seg_img, seg_dest) # type: ignore

        seg_data = std_seg_img.get_fdata() # type: ignore
        unique_values = np.unique(seg_data)
        print(f"  Case {case_id}: Unique segmentation values after processing: {unique_values}")

        dataset_json["training"].append({
            "image": f"./imagesTr/{task_name}_{case_id}_0000.nii.gz",
            "label": f"./labelsTr/{task_name}_{case_id}.nii.gz",
        })

    # --- Process Test Data ---
    print(f"\nProcessing {len(test_pairs)} test cases...")
    for i, (ct_path, seg_path) in enumerate(test_pairs):
        case_id = f"OCD_test_{i:04d}"

        print(f"\nProcessing Test Case {case_id}...")
        print(f"  Input CT: {ct_path}")
        ct_dest = images_test_dir / f"{task_name}_{case_id}_0000.nii.gz"
        std_ct_img, modif = standardize_orientation(ct_path, ct_image=True)
        print(f"  Saving processed test CT to: {ct_dest}")
        nib.save(std_ct_img, ct_dest) # type: ignore

        dataset_json["test"].append(f"./imagesTs/{task_name}_{case_id}_0000.nii.gz")

    print(f"\nWriting dataset.json")
    with open(task_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"\nDataset conversion complete.")
    print(f"nnUNet Task Name: {task_name}")
    print(f"Task ID: {task_id}")
    print(f"Data location: {task_dir}")
    return task_dir

if __name__ == "__main__":
    dataset_dir = Path("DatasetChallenge")
    output_dir = Path("nnUNet_raw_data")

    os.makedirs(output_dir, exist_ok=True)
    convert_dataset_to_nnunet_format(dataset_dir, output_dir)