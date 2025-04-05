import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.filebasedimages import FileBasedImage
from sklearn.model_selection import train_test_split


class SampleUtils:
  """
  Usefull operation on CT and segmentation data
  """

  @classmethod
  def load_from_path(cls, file_path: str) -> tuple[FileBasedImage, np.ndarray]:
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
    data,
    slice_idx=None,
    axis=2,
    figsize=(20, 16),
    cmap="gray",
    vmin=None,
    vmax=None,
    title=None,
  ):
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

  def __init__(self, base_dir: str = "DatasetChallenge"):
    """
    Initialize the dataset path

    Args:
        base_dir: Base directory containing the dataset
    """
    self.base_dir = base_dir
    self.data = self._loading_dataset_paths()

    # === MSKCC ===
    MSKCC_pairs = self._create_paired_dataset(datasets=self.data, dataset_name="MSKCC")
    MSKCC_train, MSKCC_val, MSKCC_test = self._split_dataset(
      MSKCC_pairs, dataset_name="MSKCC"
    )

    # === MSKCC ===
    TCGA_pairs = self._create_paired_dataset(datasets=self.data, dataset_name="TCGA")

    TCGA_train, TCGA_val, TCGA_test = self._split_dataset(
      TCGA_pairs, dataset_name="TCGA"
    )

    self.num_samples = len(MSKCC_pairs) + len(TCGA_pairs)
    self.train_pairs = MSKCC_train + TCGA_train
    self.val_pairs = MSKCC_val + TCGA_val
    self.test_pairs = MSKCC_test + TCGA_test

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
    ct_datasets = ["MSKCC", "TCGA"]
    for dataset in ct_datasets:
      path = os.path.join(self.base_dir, "CT", dataset)
      if os.path.exists(path):
        datasets[f"CT_{dataset}"] = [os.path.join(path, f) for f in os.listdir(path)]
      CT_length.append(len(datasets[f"CT_{dataset}"]))
      print(f"CT_{dataset}: {CT_length[-1]} files")
      print(f"  Sample file: {os.path.basename(datasets[f'CT_{dataset}'][0])}")

    # Segmentation data
    seg_datasets = ["MSKCC", "TCGA"]
    for dataset in seg_datasets:
      path = os.path.join(self.base_dir, "Segmentation", dataset)
      if os.path.exists(path):
        datasets[f"Segmentation_{dataset}"] = [
          os.path.join(path, f) for f in os.listdir(path)
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
      f"Dataset split {dataset_name}: Train={len(train_pairs)},"
      f"Val={len(val_pairs)}, Test={len(test_pairs)}"
    )
    return train_pairs, val_pairs, test_pairs

  def get_dataset_splits(self):
    return self.train_pairs, self.val_pairs, self.test_pairs

  def __len__(self):
    """Return the total number of samples"""
    return self.num_samples
