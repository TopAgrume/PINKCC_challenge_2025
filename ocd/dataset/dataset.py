import os
import nibabel as nib
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from nibabel.orientations import (
  aff2axcodes,
  apply_orientation,
  axcodes2ornt,
  inv_ornt_aff,
  ornt_transform,
)
from sklearn.model_selection import train_test_split
from ocd.dataset.sample_utils import SampleUtils

from ocd.utils import INVERSED, create_folds_dataframe

class Dataset:
  """
  Dataset for CT and segmentation data
  """

  def __init__(self, base_dir: Path = Path("DatasetChallenge"), random_state: int = 42):
    """
    Initialize the dataset path

    Args:
        base_dir: Base directory containing the dataset
        random_state: Dataset split random seed
    """
    self.base_dir = base_dir
    self.datasets = ["MSKCC", "TCGA"]
    self.data = self._loading_dataset_paths()
    print(f"Random seed: {random_state}")

    # === MSKCC ===
    MSKCC_pairs = self._create_paired_dataset(datasets=self.data, dataset_name="MSKCC")
    MSKCC_train, MSKCC_val, MSKCC_test = self._split_dataset(
      MSKCC_pairs, dataset_name="MSKCC", random_state=random_state
    )

    # === MSKCC ===
    TCGA_pairs = self._create_paired_dataset(datasets=self.data, dataset_name="TCGA")

    TCGA_train, TCGA_val, TCGA_test = self._split_dataset(
      TCGA_pairs, dataset_name="TCGA", random_state=random_state
    )

    self.num_samples = len(MSKCC_pairs) + len(TCGA_pairs)
    self.train_pairs = MSKCC_train + TCGA_train
    self.val_pairs = MSKCC_val + TCGA_val
    self.test_pairs = MSKCC_test + TCGA_test

  def get_last_slice(self, data: np.ndarray, delete_last_n: int = 15) -> int:
    # TODO: implement

    return 0

  def convert_CT_scans_to_images(
    self, output_dir: Path, n_folds: int = 5, seed: int = 42
  ):
    """
    Convert the 3D CT scans from datasets to 2D images.
    The output structure is the following:
      - output_dir/scan/{CT_scan_name}_{slice_index}.jpg
      - output_dir/segmentation/{CT_scan_name}_{slice_index}.jpg

    Args:
        `output_dir`: path of the output directory

    Returns:
      None
    """
    TO_EXCLUDE = ["TCGA-13-0762"]  # only need to exclude this one for 2D

    ct_to_fold_df = create_folds_dataframe(TO_EXCLUDE, n_folds, seed)

    scan_paths = []
    seg_paths = []
    for dataset in self.datasets:
      scan_paths.extend(self.data[f"CT_{dataset}"])
      seg_paths.extend(self.data[f"Segmentation_{dataset}"])

    all_paths = zip(sorted(scan_paths), sorted(seg_paths), strict=False)

    os.mkdir(output_dir)
    os.mkdir(output_dir / "scan")
    os.mkdir(output_dir / "seg")

    rows = []
    for scan_path, seg_path in tqdm(all_paths):
      if any(exclude in str(scan_path) for exclude in TO_EXCLUDE):
        continue

      for path, dir_name in [(scan_path, "scan"), (seg_path, "seg")]:
        file, data = SampleUtils.load_from_path_nib(path)
        affine = file.affine  # pyright: ignore
        axcodes = aff2axcodes(affine)
        if axcodes != ("L", "P", "S"):
          current_ornt = axcodes2ornt(axcodes)
          target_ornt = axcodes2ornt(("L", "P", "S"))

          transform = ornt_transform(current_ornt, target_ornt)
          reoriented_data = apply_orientation(data, transform)
          new_affine = affine @ inv_ornt_aff(transform, data.shape)

          data = nib.nifti1.Nifti1Image(reoriented_data, new_affine).get_fdata()

        file_name = path.name.split(".")[0]
        assert data.shape[:2] == (512, 512)
        if file_name in INVERSED:
          data = data[:, :, ::-1]
        limit = self.get_last_slice(data)
        for i in range(limit):
          image = data[:, :, i]
          np.save(
            output_dir / dir_name / f"{file_name}_{i}",
            image.astype(np.float16),
          )
          if dir_name == "seg":
            fold_id = ct_to_fold_df[ct_to_fold_df["path"].str.contains(file_name)][
              "fold"
            ].item()
            rows.append(
              (
                f"{file_name}_{i}",
                fold_id,
                len(np.unique(image)) != 1,
              )
            )

    pd.DataFrame(rows, columns=["path", "fold_id", "has_labels"]).to_csv(
      output_dir / "metadata.csv"
    )

  def _loading_dataset_paths(self, verify: bool = True) -> dict:
    """
    Explore NIFTI files in the DatasetChallenge directory structure

    Args:
        base_dir: Base directory containing the dataset
        verify: If True, verify that CT and segmentation counts match

    Returns:
        Dictionary mapping dataset types to file paths
    """
    datasets = {}
    CT_length = []
    segmentation_length = []

    # CT data
    for dataset in self.datasets:
      path = os.path.join(self.base_dir, "CT", dataset)
      if os.path.exists(path):
        datasets[f"CT_{dataset}"] = [
          Path(os.path.join(path, f)) for f in os.listdir(path)
        ]
      CT_length.append(len(datasets[f"CT_{dataset}"]))
      print(f"CT_{dataset}: {CT_length[-1]} files")
      print(f"  Sample file: {os.path.basename(datasets[f'CT_{dataset}'][0])}")

    # Segmentation data
    seg_datasets = ["MSKCC", "TCGA"]
    for dataset in seg_datasets:
      path = os.path.join(self.base_dir, "Segmentation", dataset)
      if os.path.exists(path):
        datasets[f"Segmentation_{dataset}"] = [
          Path(os.path.join(path, f)) for f in os.listdir(path)
        ]
      segmentation_length.append(len(datasets[f"Segmentation_{dataset}"]))
      print(f"Segmentation_{dataset}: {segmentation_length[-1]} files")
      print(
        f"  Sample file: {os.path.basename(datasets[f'Segmentation_{dataset}'][0])}"
      )

    if verify:
      assert CT_length == segmentation_length, "Not the same amount sample/GT"

    return datasets

  def _create_paired_dataset(
    self, datasets: dict[str, list[str]], dataset_name: str = "MSKCC"
  ) -> list[tuple[str, str]]:
    """
    Create paired dataset of CT and segmentation files

    Args:
        datasets: Dictionary with dataset paths from loading_dataset_paths
        dataset_name: Name of the dataset to use (e.g., 'MSKCC', 'TCGA')

    Returns:
        List of (ct_path, seg_path) tuples
    """
    ct_paths = datasets.get(f"CT_{dataset_name}", [])
    seg_paths = datasets.get(f"Segmentation_{dataset_name}", [])

    if not ct_paths or not seg_paths:
      raise ValueError(f"Dataset {dataset_name} not found in datasets dictionary")

    # ensure correct pairing by sorting
    ct_paths.sort()
    seg_paths.sort()

    pairs = []
    for ct_path, seg_path in zip(ct_paths, seg_paths, strict=False):
      ct_file = os.path.basename(ct_path)
      seg_file = os.path.basename(seg_path)

      ct_id = ct_file.split(".")[0]
      seg_id = seg_file.split(".")[0]

      if ct_id == seg_id:
        pairs.append((ct_path, seg_path))
      else:
        print(f"Warning: Mismatched pair - CT: {ct_file}, Segmentation: {seg_file}")

    return pairs

  def _split_dataset(
    self,
    paired_data: list[tuple[str, str]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    dataset_name: str = "",
  ) -> tuple[list, list, list]:
    """
    Split dataset into train, validation, and test sets

    Args:
        paired_data: List of (ct_path, seg_path) tuples
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_pairs, val_pairs, test_pairs)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, (
      "Ratios must sum to 1"
    )

    train_pairs, temp_pairs = train_test_split(
      paired_data, train_size=train_ratio, random_state=random_state
    )

    val_size = val_ratio / (val_ratio + test_ratio)
    val_pairs, test_pairs = train_test_split(
      temp_pairs, train_size=val_size, random_state=random_state
    )

    print(
      f"Dataset split {dataset_name}: Train={len(train_pairs)}, "
      f"Val={len(val_pairs)}, Test={len(test_pairs)}"
    )
    return train_pairs, val_pairs, test_pairs

  def get_dataset_splits(self):
    return self.train_pairs, self.val_pairs, self.test_pairs

  def __len__(self):
    """Return the total number of samples"""
    return self.num_samples
