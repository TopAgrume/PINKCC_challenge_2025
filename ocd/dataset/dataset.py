import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.filebasedimages import FileBasedImage
from nibabel.orientations import (
  aff2axcodes,
  apply_orientation,
  axcodes2ornt,
  inv_ornt_aff,
  ornt_transform,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ocd.utils import INVERSED, create_folds_dataframe


class SampleUtils:
  """
  Usefull operation on CT and segmentation data
  """

  @classmethod
  def load_from_path(cls, file_path: str | Path) -> tuple[FileBasedImage, np.ndarray]:
    """
    Load NIFTI file content

    Args:
        file_path: Path to the NIFTI file

    Returns:
        Tuple of (nib.Nifti1Image, numpy.ndarray)
    """
    img = nib.loadsave.load(file_path)
    data = img.get_fdata()  # pyright: ignore
    return img, data

  @classmethod
  def display_slice(
    cls,
    data: np.ndarray,
    slice_idx=None,
    axis=2,
    figsize=(20, 16),
    cmap="gray",
    vmin=None,
    vmax=None,
    title=None,
  ) -> None:
    """
    Display a single slice from a 3D volume

    Args:
        data: 3D numpy array
        slice_idx: Index of the slice to display, defaults to middle slice
        axis: Axis along which to take the slice (0, 1, or 2)
        figsize: Figure size as (width, height)
        cmap: Colormap for the display
        vmin, vmax: Min and max values for color scaling
        title: Title for the plot
    """
    if slice_idx is None:
      slice_idx = data.shape[axis] // 2

    if axis == 0:
      slice_data = data[slice_idx, :, :]
    elif axis == 1:
      slice_data = data[:, slice_idx, :]
    else:
      slice_data = data[:, :, slice_idx]

    plt.figure(figsize=figsize)
    plt.imshow(slice_data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()

    if title:
      plt.title(title)
    else:
      plt.title(f"Slice {slice_idx} along axis {axis}")

    plt.axis("on")
    plt.tight_layout()
    plt.show()

  @classmethod
  def normalize_ct(
    cls,
    ct_data: np.ndarray,
    window_center: int = 40,
    window_width: int = 400,
    output_range: tuple[float, float] = (-1, 1),
  ) -> np.ndarray:
    """
    Apply windowing and normalization to CT data

    Args:
        ct_data: Raw CT data in Hounsfield units
        window_center: Center of the windowing operation (typically 40 for soft tissue)
        window_width: Width of the window (typically 400 for soft tissue)
        output_range: Desired output range for normalization

    Returns:
        Normalized CT data
    """
    # windowing
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2

    windowed_data = np.clip(ct_data, min_val, max_val)

    # normalize
    out_min, out_max = output_range
    normalized = out_min + (windowed_data - min_val) * (out_max - out_min) / (
      max_val - min_val
    )

    return normalized


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

  def get_last_slice(self, data: np.ndarray, margin: int = 8) -> tuple[int, int]:
    has_labels: np.ndarray = np.any(data != 0, axis=(0, 1))
    last_index = min(
      data.shape[2] - 1 - has_labels[::-1].argmax() + margin, data.shape[2] - 1
    )
    first_index = max(has_labels.argmax() - margin, 0)
    return first_index, last_index  # pyright: ignore

  def convert_CT_scans_to_images(
    self, output_dir: Path, n_folds: int = 5, seed: int = 69
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
    TO_EXCLUDE = [
      "TCGA-13-0762",
      "TCGA-13-0793",
      "TCGA-09-2054",
      "TCGA-24-1614",
      "TCGA-09-0364",
    ]

    ct_to_fold_df = create_folds_dataframe(TO_EXCLUDE, n_folds, seed)

    scan_paths = []
    seg_paths = []
    for dataset in self.datasets:
      scan_paths.extend(self.data[f"CT_{dataset}"])
      seg_paths.extend(self.data[f"Segmentation_{dataset}"])

    all_paths = zip(sorted(scan_paths), sorted(seg_paths), strict=False)

    if output_dir.exists():
      shutil.rmtree(output_dir)

    os.mkdir(output_dir)
    os.mkdir(output_dir / "scan")
    os.mkdir(output_dir / "seg")

    rows = []
    for scan_path, seg_path in tqdm(all_paths):
      if any(exclude in str(scan_path) for exclude in TO_EXCLUDE):
        print(f"\nSkipped {scan_path}...")
        continue

      for path, dir_name in [(seg_path, "seg"), (scan_path, "scan")]:
        file, data = SampleUtils.load_from_path(path)

        if dir_name == "seg":
          limit_inf, limit_sup = self.get_last_slice(data)
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
        for i in range(limit_inf, limit_sup + 1):  # pyright: ignore
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

    pd.DataFrame(rows, columns=["path", "fold_id", "has_labels"]).to_csv(  # pyright:ignore
      output_dir / "metadata.csv"
    )

  def _iterative_scan_through_component(
    self, start_i: int, start_j: int, start_k: int, scan: np.ndarray, seg: np.ndarray
  ) -> list[float]:
    original_value = seg[start_i, start_j, start_k]
    if original_value == 0 or original_value == -1:
      return []

    res_values = []
    stack = [(start_i, start_j, start_k)]

    while stack:
      i, j, k = stack.pop()

      if not (
        0 <= i < scan.shape[0]
        and 0 <= j < scan.shape[1]
        and 0 <= k < scan.shape[2]
        and seg[i, j, k] == original_value
      ):
        continue

      res_values.append(scan[i, j, k])
      seg[i, j, k] = -1

      if i < scan.shape[0] - 1 and seg[i + 1, j, k] == original_value:
        stack.append((i + 1, j, k))
      # (i-1, j, k)
      if i > 0 and seg[i - 1, j, k] == original_value:
        stack.append((i - 1, j, k))
      # (i, j+1, k)
      if j < scan.shape[1] - 1 and seg[i, j + 1, k] == original_value:
        stack.append((i, j + 1, k))
      # (i, j-1, k)
      if j > 0 and seg[i, j - 1, k] == original_value:
        stack.append((i, j - 1, k))
      # (i, j, k+1)
      if k < scan.shape[2] - 1 and seg[i, j, k + 1] == original_value:
        stack.append((i, j, k + 1))
      # (i, j, k-1)
      if k > 0 and seg[i, j, k - 1] == original_value:
        stack.append((i, j, k - 1))

    return res_values

  def _find_labels_and_get_stats(
    self, scan: np.ndarray, seg: np.ndarray
  ) -> tuple[list[tuple[float, float, int]], list[tuple[float, float, int]]]:
    primaries = []
    metastases = []
    x, y, z = scan.shape

    for i in range(x):
      for j in range(y):
        for k in range(z):
          original_label_at_pixel = seg[i, j, k]

          if original_label_at_pixel != 0 and original_label_at_pixel != -1:
            values = self._iterative_scan_through_component(i, j, k, scan, seg)

            if not values:
              continue

            values_np = np.array(values)
            mean_val = values_np.mean()
            std_val = values_np.std()
            n_pixels = len(values_np)
            stats_tuple = (mean_val, std_val, n_pixels)

            if original_label_at_pixel == 1:
              primaries.append(stats_tuple)
            else:
              metastases.append(stats_tuple)

    return (primaries, metastases)

  def get_labels_stats(self):
    TO_EXCLUDE = [
      "TCGA-13-0762",
      "TCGA-09-2054",
      "TCGA-24-1614",
      "TCGA-09-0364",
      "333076",
    ]

    scan_paths = []
    seg_paths = []
    for dataset in self.datasets:
      scan_paths.extend(self.data[f"CT_{dataset}"])
      seg_paths.extend(self.data[f"Segmentation_{dataset}"])

    all_paths = zip(sorted(scan_paths), sorted(seg_paths), strict=False)

    all_mean_primaries, all_mean_metastases = [], []
    all_std_primaries, all_std_metastases = [], []
    with open("labels_stats.txt", "w") as f:
      for scan_path, seg_path in tqdm(all_paths):
        if any(exclude in str(scan_path) for exclude in TO_EXCLUDE):
          print(f"\nSkipped {scan_path}...")
          continue

        f.write(f"{scan_path.name}\n")
        _, scan = SampleUtils.load_from_path(scan_path)
        _, seg = SampleUtils.load_from_path(seg_path)
        primaries, metastases = self._find_labels_and_get_stats(scan, seg)

        # -- PRIMARY --

        if len(primaries) != 0:
          mean_primaries, std_primaries, pixel_primaries = zip(*primaries, strict=False)
          mean_primaries, std_primaries, pixel_primaries = (
            np.array(mean_primaries),
            np.array(std_primaries),
            np.array(pixel_primaries),
          )
          f.write(
            f"  {len(primaries)} PRIMARIES mean {mean_primaries.mean():.2f}"
            f" std {std_primaries.mean():.2f} min {mean_primaries.min():.2f}"
            f" max {mean_primaries.max():.2f}\n"
          )
          [
            f.write(
              f"   -> {i} - mean {mean_primaries[i]:.2f} std {std_primaries[i]:.2f}"
              f" n_pixels {pixel_primaries[i]}\n"
            )
            for i in range(len(primaries))
          ]
          all_mean_primaries.append(mean_primaries.mean())
          all_std_primaries.append(std_primaries.mean())

        # -- METASTASE --

        if len(metastases) != 0:
          mean_metastases, std_metastases, pixel_metastases = zip(
            *metastases, strict=False
          )
          mean_metastases, std_metastases, pixel_metastases = (
            np.array(mean_metastases),
            np.array(std_metastases),
            np.array(pixel_metastases),
          )
          f.write(
            f"  {len(metastases)} METASTASES mean {mean_metastases.mean():.2f}"
            f" std {std_metastases.mean():.2f} min {mean_metastases.min():.2f}"
            f" max {mean_metastases.max():.2f}\n"
          )
          [
            f.write(
              f"   -> {i} - mean {mean_metastases[i]:.2f} std {std_metastases[i]:.2f}"
              f" n_pixels {pixel_metastases[i]}\n"
            )
            for i in range(len(metastases))
          ]
          all_mean_metastases.append(mean_metastases.mean())
          all_std_metastases.append(std_metastases.mean())

      # -- Final stats --

      all_mean_primaries, all_mean_metastases = (
        np.array(all_mean_primaries),
        np.array(all_mean_metastases),
      )
      all_std_primaries, all_std_metastases = (
        np.array(all_std_primaries),
        np.array(all_std_metastases),
      )
      f.write(
        f"\nPRIMARIES mean {all_mean_primaries.mean():.2f}"
        f" std {all_std_primaries.mean():.2f}\n"
      )
      f.write(
        f"METASTASES mean {all_mean_metastases.mean():.2f}"
        f" std {all_std_metastases.mean():.2f}\n"
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
