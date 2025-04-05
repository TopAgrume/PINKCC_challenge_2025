import os
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional, Callable, Dict

from ocd.dataset.dataset import SampleUtils


class CancerSegmentationDataset(TorchDataset):
    """PyTorch Dataset for cancer image segmentation task"""

    def __init__(self,
                 paired_data: List[Tuple[str, str]],
                 transforms: Optional[Callable] = None,
                 preload: bool = False,
                 slice_axis: int = 2,
                 window_center: int = 40,
                 window_width: int = 400):
        """
        Initialize the PyTorch dataset

        Args:
            paired_data: List of (ct_path, seg_path) tuples
            transforms: Optional transforms to apply to the data
            preload: If True, preload all data into memory
            slice_axis: Axis along which to extract 2D slices (0, 1, or 2)
            window_center: Center of window for CT normalization
            window_width: Width of window for CT normalization
        """
        self.paired_data = paired_data
        self.transforms = transforms
        self.preload = preload
        self.slice_axis = slice_axis
        self.window_center = window_center
        self.window_width = window_width

        self.data_cache = {}
        self.slice_map = []

        if preload:
            self._preload_data()
        else:
            self._build_slice_map()


    def _get_metadata(self, idx: int) -> Dict:
        """Get metadata for a sample"""
        ct_path, seg_path = self.paired_data[idx]
        ct_img = nib.load(ct_path)
        seg_img = nib.load(seg_path)

        return {
            'ct_path': ct_path,
            'seg_path': seg_path,
            'patient_id': os.path.basename(ct_path).split('.')[0],
            'ct_affine': ct_img.affine,
            'seg_affine': seg_img.affine,
            'ct_shape': ct_img.shape,
            'seg_shape': seg_img.shape
        }


    def _get_slice(self, volume: np.ndarray, slice_idx: Optional[int] = None) -> np.ndarray:
        if self.slice_axis == 0:
            return volume[slice_idx, :, :]
        elif self.slice_axis == 1:
            return volume[:, slice_idx, :]
        else:
            return volume[:, :, slice_idx]


    def _visualize_sample(self, idx: int, slice_idx: Optional[int] = None,
                       with_mask: bool = True, alpha: float = 0.5,
                       figsize: Tuple[int, int] = (12, 5)):
        """Visualize a sample with optional overlay of segmentation mask"""
        ct_path, seg_path = self.paired_data[idx]
        ct_data = SampleUtils.load_from_path(ct_path)
        seg_data = SampleUtils.load_from_path(seg_path)

        if slice_idx is None:
            slice_idx = ct_data.shape[self.slice_axis] // 2

        ct_slice = self._get_slice(ct_data, slice_idx)
        seg_slice = self._get_slice(seg_data, slice_idx)

        fig, axes = plt.subplots(1, 2 if with_mask else 1, figsize=figsize)

        if not with_mask:
            axes = [axes]

        # display CT slice
        axes[0].imshow(ct_slice, cmap='gray')
        axes[0].set_title(f'CT Slice {slice_idx}')
        axes[0].axis('on')

        if with_mask:
            # overlay
            axes[1].imshow(ct_slice, cmap='gray')
            mask = seg_slice > 0
            masked_data = np.ma.masked_where(~mask, seg_slice)
            axes[1].imshow(masked_data, cmap='jet', alpha=alpha)
            axes[1].set_title(f'Segmentation overlay (Alpha={alpha})')
            axes[1].axis('on')

        plt.tight_layout()
        plt.show()
        return fig


    def _preload_data(self):
        """Preload all volumes into memory"""
        print("Preloading data into memory...")
        for idx, (ct_path, seg_path) in enumerate(self.paired_data):
            _, ct_data = SampleUtils.load_from_path(ct_path)
            _, seg_data = SampleUtils.load_from_path(seg_path)

            ct_data = SampleUtils.normalize_ct(
                ct_data,
                window_center=self.window_center,
                window_width=self.window_width
            )

            self.data_cache[idx] = (ct_data, seg_data)

            # slice mapping for this volume
            self._add_volume_to_slice_map(idx, ct_data.shape[self.slice_axis])


    def _build_slice_map(self):
        """Build mapping from flat index to (volume_idx, slice_idx)"""
        for vol_idx, (ct_path, _) in enumerate(self.paired_data):
            img = nib.load(ct_path)
            n_slices = img.shape[self.slice_axis]

            # Add slices from this volume to the slice map
            self._add_volume_to_slice_map(vol_idx, n_slices)


    def _add_volume_to_slice_map(self, vol_idx, n_slices):
        """Add volume slices to the slice map"""
        for slice_idx in range(n_slices):
            self.slice_map.append((vol_idx, slice_idx))


    def __len__(self):
        """Return the total number of slices across all volumes"""
        return len(self.slice_map)


    def __getitem__(self, idx):
        """Get a single slice pair (image, mask)"""
        vol_idx, slice_idx = self.slice_map[idx]

        # volume data
        if self.preload and vol_idx in self.data_cache:
            ct_volume, seg_volume = self.data_cache[vol_idx]
        else: # delayed loading
            ct_path, seg_path = self.paired_data[vol_idx]
            _, ct_volume = SampleUtils.load_from_path(ct_path)
            _, seg_volume = SampleUtils.load_from_path(seg_path)

            ct_volume = SampleUtils.normalize_ct(
                ct_volume,
                window_center=self.window_center,
                window_width=self.window_width
            )

        if self.slice_axis == 0:
            ct_slice = ct_volume[slice_idx, :, :]
            seg_slice = seg_volume[slice_idx, :, :]
        elif self.slice_axis == 1:
            ct_slice = ct_volume[:, slice_idx, :]
            seg_slice = seg_volume[:, slice_idx, :]
        else:
            ct_slice = ct_volume[:, :, slice_idx]
            seg_slice = seg_volume[:, :, slice_idx]

        ct_slice = ct_slice[np.newaxis, ...]  # shape: (1, H, W)
        seg_slice = seg_slice[np.newaxis, ...]  # shape: (1, H, W)

        if self.transforms:
            ct_slice, seg_slice = self.transforms(ct_slice, seg_slice)

        ct_tensor = torch.from_numpy(ct_slice.astype(np.float32)) #TODO: np.float64
        seg_tensor = torch.from_numpy(seg_slice.astype(np.float32)) #TODO: np.float64

        return ct_tensor, seg_tensor


    @staticmethod
    def get_dataloader(dataset, batch_size=16, num_workers=4, shuffle=True):
        """
        Create a PyTorch DataLoader from the dataset

        Args:
            dataset: Instance of MedicalSegmentationDataset
            batch_size: Batch size for training
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
            pin_memory=torch.cuda.is_available()
        )