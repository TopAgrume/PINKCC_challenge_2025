from collections.abc import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from ocd.dataset.dataset import SampleUtils


class CancerVolumeDataset(TorchDataset):
  """PyTorch Dataset for working with full 3D volumes"""

  def __init__(
    self,
    paired_data: list[tuple[str, str]],
    transforms: Callable | None = None,
    preload: bool = False,
    window_center: int = 40,
    window_width: int = 400,
    patch_size: tuple[int, int, int] | None = None,
  ):
    """
    Initialize the PyTorch dataset

    Args:
        paired_data: List of (ct_path, seg_path) tuples
        transforms: Optional transforms to apply to the data
        preload: If True, preload all data into memory
        window_center: Center of window for CT normalization
        window_width: Width of window for CT normalization
        patch_size: Optional size of patches to extract (D, H, W)
    """
    self.paired_data = paired_data
    self.transforms = transforms
    self.preload = preload
    self.window_center = window_center
    self.window_width = window_width
    self.patch_size = patch_size

    self.data_cache = {}

    if preload:
      self._preload_data()

  def _preload_data(self):
    """Preload all volumes into memory"""
    print("Preloading data into memory...")
    for idx, (ct_path, seg_path) in enumerate(self.paired_data):
      _, ct_data = SampleUtils.load_from_path(ct_path)
      _, seg_data = SampleUtils.load_from_path(seg_path)

      ct_data = SampleUtils.normalize_ct(
        ct_data, window_center=self.window_center, window_width=self.window_width
      )

      self.data_cache[idx] = (ct_data, seg_data)

  def __len__(self):
    """Return the number of volumes"""
    return len(self.paired_data)

  def _extract_random_patch(self, volume, mask):
    """Extract a random patch of specified size from the volume"""
    d, h, w = volume.shape
    assert self.patch_size
    pd, ph, pw = self.patch_size

    # ensure patch size doesn't exceed volume dimensions
    pd, ph, pw = min(pd, d), min(ph, h), min(pw, w)

    # random starting point
    d_start = np.random.randint(0, d - pd + 1)
    h_start = np.random.randint(0, h - ph + 1)
    w_start = np.random.randint(0, w - pw + 1)

    # extract patches
    vol_patch = volume[
      d_start : d_start + pd, h_start : h_start + ph, w_start : w_start + pw
    ]
    mask_patch = mask[
      d_start : d_start + pd, h_start : h_start + ph, w_start : w_start + pw
    ]

    return vol_patch, mask_patch

  def __getitem__(self, idx):
    """Get a volume pair (image volume, mask volume)"""
    # volume data
    if self.preload and idx in self.data_cache:
      ct_volume, seg_volume = self.data_cache[idx]
    else:  # delayed loading
      ct_path, seg_path = self.paired_data[idx]
      _, ct_volume = SampleUtils.load_from_path(ct_path)
      _, seg_volume = SampleUtils.load_from_path(seg_path)

      ct_volume = SampleUtils.normalize_ct(
        ct_volume, window_center=self.window_center, window_width=self.window_width
      )

    # extract patch if specified
    if self.patch_size:
      ct_volume, seg_volume = self._extract_random_patch(ct_volume, seg_volume)

    if self.transforms:
      ct_volume, seg_volume = self.transforms(ct_volume, seg_volume)

    ct_volume = ct_volume[np.newaxis, ...]  # shape: (1, D, H, W)
    seg_volume = seg_volume[np.newaxis, ...]  # shape: (1, D, H, W)

    ct_tensor = torch.from_numpy(ct_volume.astype(np.float32))
    seg_tensor = torch.from_numpy(seg_volume.astype(np.float32))

    return ct_tensor, seg_tensor

  @staticmethod
  def get_dataloader(dataset, batch_size=2, num_workers=4, shuffle=True):
    """
    Create a PyTorch DataLoader from the dataset

    Args:
        dataset: Instance of MedicalVolumeDataset
        batch_size: Batch size for training (typically small for 3D volumes)
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the data

    Returns:
        torch.utils.data.DataLoader
    """
    return DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=shuffle,
      num_workers=num_workers,
      pin_memory=torch.cuda.is_available(),
    )
