import os
import torch
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union

from ct_sam3d.ct_sam.builder import build_sam
from ct_sam3d.ct_sam.predictor import SamPredictor
from ct_sam3d.ct_sam.utils.frame import voxel_to_world
from ct_sam3d.ct_sam.utils.io_utils import load_module_from_file
from ct_sam3d.ct_sam.utils.resample import (
    resample_itkimage_torai,
    crop_roi_with_center,
    flip_itkimage_torai
)
from ocd.dataset.dataset import SampleUtils, Dataset


class CTSAM3DSegmenter:
    """
    A class to integrate CT-SAM3D model with existing dataset structure for
    interactive segmentation of CT scans using point prompts.
    """

    def __init__(
        self,
        checkpoint_path: str
    ):
        """
        Initialize the CT-SAM3D segmenter with the model checkpoint.

        Args:
            checkpoint_path: Path to the model checkpoint file (.pth)
        """
        self.checkpoint_path = checkpoint_path
        self.config_file = os.path.join(os.path.dirname(checkpoint_path), "config.py")

        if not os.path.isfile(self.config_file):
            raise FileNotFoundError(f"Config file not found at {self.config_file}")

        # ----- load model configuration -----
        cfg_module = load_module_from_file(self.config_file)
        self.cfg = cfg_module.cfg
        self.cfg.update({"checkpoint": checkpoint_path})

        # ----- initialize model -----
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam = build_sam(self.cfg)
        self.sam.to(self.device)

        self.predictor = SamPredictor(self.sam, self.cfg.dataset)

        self.current_image = None
        self.current_mask_gt = None
        self.mask_input = None
        self.image_patch = None
        self.image_spacing = None

        print(f"CT-SAM3D initialized on device: {self.device}")

    def load_sample_from_dataset(
        self,
        dataset_pair: Tuple[str, str],
        resample_spacing: List[float] = [1.5, 1.5, 1.5]
    ) -> Tuple[sitk.Image, Optional[sitk.Image]]:
        """
        Load a CT scan and its segmentation mask from a dataset pair.

        Args:
            dataset_pair: Tuple of (ct_path, seg_path) from Dataset class
            resample_spacing: Target spacing for resampling

        Returns:
            Tuple of (SimpleITK CT image, SimpleITK segmentation mask)
        """
        ct_path, seg_path = dataset_pair

        # load image with sitk
        ct_sitk, _ = SampleUtils.load_from_path_sitk(ct_path)

        # ensure RAI orientation
        ct_sitk = SampleUtils.ensure_rai_orientation(ct_sitk)

        # resample CT to target spacing
        ct_sitk = resample_itkimage_torai(
            sitk_im=ct_sitk, # type: ignore
            spacing=resample_spacing,
            interpolator="linear",
            pad_value=-1024
        )

        # load segmentation mask if available
        mask_sitk = None
        if seg_path:
            try:
                mask_sitk, _ = SampleUtils.load_from_path_nib(seg_path)
                mask_sitk = SampleUtils.ensure_rai_orientation(mask_sitk)

                mask_sitk = resample_itkimage_torai(
                    sitk_im=ct_sitk, # type: ignore
                    spacing=resample_spacing,
                    interpolator="nearest",
                    pad_value=0
                )
            except Exception as e:
                print(f"Failed to load segmentation mask: {e}")

        self.current_image = ct_sitk
        self.current_mask_gt = mask_sitk
        self.image_spacing = resample_spacing

        return ct_sitk, mask_sitk # type: ignore

    def crop_patch(
        self,
        center_voxel: List[int],
        patch_size: List[int] = [64, 64, 64]
    ) -> Tuple[sitk.Image, Optional[sitk.Image]]:
        """
        Crop a patch from the current image centered at the specified voxel.

        Args:
            center_voxel: Center voxel coordinates [x, y, z]
            patch_size: Size of the patch to crop [x, y, z]

        Returns:
            Tuple of (cropped CT patch, cropped mask patch if available)
        """
        if self.current_image is None:
            raise ValueError("No image loaded. Call load_sample_from_dataset first.")

        # Convert voxel coordinates to world coordinates
        center_world = voxel_to_world(self.current_image, center_voxel)

        # Get image axes
        x_axis, y_axis, z_axis = np.array(self.current_image.GetDirection()).reshape(3, 3).transpose()

        # Crop CT patch
        image_patch = crop_roi_with_center(
            self.current_image,
            center_world,
            self.current_image.GetSpacing(),
            x_axis, y_axis, z_axis,
            patch_size,
            "linear",
            -1024
        )

        # Crop mask patch if available
        mask_patch = None
        if self.current_mask_gt is not None:
            mask_patch = crop_roi_with_center(
                self.current_mask_gt,
                center_world,
                self.current_mask_gt.GetSpacing(),
                x_axis, y_axis, z_axis,
                patch_size,
                "nearest",
                0
            )

        self.image_patch = image_patch
        self.predictor.set_image(image_patch)

        return image_patch, mask_patch

    def predict_from_points(
        self,
        points: List[List[int]],
        labels: List[int],
        mode: str = "patch"
    ) -> Tuple[sitk.Image, float]:
        """
        Predict segmentation from point prompts.

        Args:
            points: List of point coordinates [[x, y, z], ...]
            labels: List of point labels (1 for positive, 0 for negative)
            mode: Prediction mode ('patch' or 'full_image')

        Returns:
            Tuple of (segmentation mask, DSC score if ground truth available)
        """
        if mode == "patch" and self.image_patch is None:
            raise ValueError("No patch loaded. Call crop_patch first.")

        if mode == "full_image" and self.current_image is None:
            raise ValueError("No image loaded. Call load_sample_from_dataset first.")

        points_array = np.array(points)
        labels_array = np.array(labels)

        # Predict mask
        mask, scores, logits = self.predictor.predict(
            point_coords=points_array,
            point_labels=labels_array,
            multimask_output=False,
            mask_input=self.mask_input
        )

        # Save logits for chained predictions
        self.mask_input = logits

        # Calculate DSC if ground truth is available
        dsc = None
        if self.current_mask_gt is not None and mode == "patch":
            # Get the mask array
            mask_array = sitk.GetArrayFromImage(mask)

            # Get the ground truth mask for the target label
            target_point = points[0]  # Use the first point to determine target label
            mask_gt_array = sitk.GetArrayFromImage(self.current_mask_gt)
            mask_gt_patch = mask_gt_array[target_point[2], target_point[1], target_point[0]]
            mask_gt_bin = (mask_gt_array == mask_gt_patch).astype(np.int32)

            # Calculate DSC
            dsc = self.compute_dsc(mask_array, mask_gt_bin)

        return mask, dsc

    def clear_mask(self):
        """
        Clear the current mask input to start a new prediction.
        """
        self.mask_input = None

    @staticmethod
    def compute_dsc(predict: np.ndarray, targets: np.ndarray, threshold: float = 0.0, smooth: float = 1.0) -> float:
        """
        Compute Dice Similarity Coefficient.

        Args:
            predict: Predicted binary mask
            targets: Target binary mask
            threshold: Threshold for binarization
            smooth: Smoothing factor to avoid division by zero

        Returns:
            DSC score
        """
        # convert to binary if not already
        if not np.all(np.isin(predict, [0, 1])):
            predict = (predict > threshold).astype(np.int32)

        if not np.all(np.isin(targets, [0, 1])):
            targets = (targets > threshold).astype(np.int32)

        # calculate DSC
        intersection = (predict * targets).sum()
        dice = (2.0 * intersection + smooth) / (predict.sum() + targets.sum() + smooth)

        return dice

    def visualize_slice(
        self,
        image: Union[sitk.Image, np.ndarray],
        mask: Optional[Union[sitk.Image, np.ndarray]] = None,
        slice_idx: Optional[int] = None,
        axis: int = 2,
        alpha: float = 0.5,
        figsize: Tuple[int, int] = (10, 8),
        window_center: int = 40,
        window_width: int = 400,
        points: Optional[List[List[int]]] = None,
        labels: Optional[List[int]] = None
    ) -> None:
        """
        Visualize a slice from the 3D volume with optional mask overlay.

        Args:
            image: 3D image (SimpleITK or numpy)
            mask: Optional segmentation mask (SimpleITK or numpy)
            slice_idx: Slice index (default: middle slice)
            axis: Axis along which to take the slice (0, 1, or 2)
            alpha: Transparency of the mask overlay
            figsize: Figure size
            window_center: Window center for CT visualization
            window_width: Window width for CT visualization
            points: Optional list of points to visualize
            labels: Optional list of point labels (1: positive, 0: negative)
        """
        if isinstance(image, sitk.Image):
            image_array = sitk.GetArrayFromImage(image)
        else:
            image_array = image

        if mask is not None and isinstance(mask, sitk.Image):
            mask_array = sitk.GetArrayFromImage(mask)
        elif mask is not None:
            mask_array = mask
        else:
            mask_array = None

        if slice_idx is None:
            slice_idx = image_array.shape[axis] // 2

        if axis == 0:
            image_slice = image_array[slice_idx, :, :]
            mask_slice = mask_array[slice_idx, :, :] if mask_array is not None else None
        elif axis == 1:
            image_slice = image_array[:, slice_idx, :]
            mask_slice = mask_array[:, slice_idx, :] if mask_array is not None else None
        else:
            image_slice = image_array[:, :, slice_idx]
            mask_slice = mask_array[:, :, slice_idx] if mask_array is not None else None

        # windowing to CT
        min_val = window_center - window_width / 2
        max_val = window_center + window_width / 2
        image_slice = np.clip(image_slice, min_val, max_val)
        image_slice = (image_slice - min_val) / (max_val - min_val)

        plt.figure(figsize=figsize)
        plt.imshow(image_slice, cmap='gray')

        # Overlay mask if available
        if mask_slice is not None:
            mask_rgb = np.zeros((*mask_slice.shape, 4))
            mask_rgb[mask_slice > 0, 0] = 1.0
            mask_rgb[mask_slice > 0, 3] = alpha
            plt.imshow(mask_rgb)

        # Plot points if available
        if points is not None and labels is not None:
            for point, label in zip(points, labels):
                # Only show points that are on this slice
                if point[axis] == slice_idx:
                    marker_style = 'o' if label == 1 else 'x'
                    marker_color = 'green' if label == 1 else 'red'

                    if axis == 0:
                        x, y = point[2], point[1]
                    elif axis == 1:
                        x, y = point[2], point[0]
                    else:
                        x, y = point[1], point[0]

                    plt.plot(x, y, marker=marker_style, markersize=10,
                             markeredgewidth=2, color=marker_color)

        plt.title(f"Slice {slice_idx} along axis {axis}")
        plt.colorbar(label="Intensity")
        plt.axis('on')
        plt.tight_layout()
        plt.show()

    def visualize_comparison(
        self,
        predicted_mask: sitk.Image,
        slice_idx: Optional[int] = None,
        axis: int = 2,
        figsize: Tuple[int, int] = (18, 6),
        window_center: int = 40,
        window_width: int = 400
    ) -> None:
        """
        Visualize a comparison between the image, predicted mask, and ground truth mask.

        Args:
            predicted_mask: Predicted segmentation mask (SimpleITK)
            slice_idx: Slice index (default: middle slice)
            axis: Axis along which to take the slice (0, 1, or 2)
            figsize: Figure size
            window_center: Window center for CT visualization
            window_width: Window width for CT visualization
        """
        if self.current_image is None:
            raise ValueError("No image loaded. Call load_sample_from_dataset first.")

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        if slice_idx is None:
            slice_idx = sitk.GetArrayFromImage(self.current_image).shape[axis] // 2

        image_array = sitk.GetArrayFromImage(self.current_image)
        pred_array = sitk.GetArrayFromImage(predicted_mask)

        if axis == 0:
            image_slice = image_array[slice_idx, :, :]
            pred_slice = pred_array[slice_idx, :, :]
        elif axis == 1:
            image_slice = image_array[:, slice_idx, :]
            pred_slice = pred_array[:, slice_idx, :]
        else:
            image_slice = image_array[:, :, slice_idx]
            pred_slice = pred_array[:, :, slice_idx]

        min_val = window_center - window_width / 2
        max_val = window_center + window_width / 2
        image_slice = np.clip(image_slice, min_val, max_val)
        image_slice = (image_slice - min_val) / (max_val - min_val)

        axes[0].imshow(image_slice, cmap='gray')
        axes[0].set_title('Original CT')
        axes[0].axis('on')

        axes[1].imshow(image_slice, cmap='gray')
        pred_mask = np.zeros((*pred_slice.shape, 4))
        pred_mask[pred_slice > 0, 0] = 1.0  # Red channel
        pred_mask[pred_slice > 0, 3] = 0.5  # Alpha
        axes[1].imshow(pred_mask)
        axes[1].set_title('Predicted Mask')
        axes[1].axis('on')

        axes[2].imshow(image_slice, cmap='gray')

        if self.current_mask_gt is not None:
            gt_array = sitk.GetArrayFromImage(self.current_mask_gt)

            if axis == 0:
                gt_slice = gt_array[slice_idx, :, :]
            elif axis == 1:
                gt_slice = gt_array[:, slice_idx, :]
            else:
                gt_slice = gt_array[:, :, slice_idx]

            # Determine the target label
            # Use the most common non-zero label in the predicted mask slice
            non_zero_labels = gt_slice[gt_slice > 0]
            if len(non_zero_labels) > 0:
                target_label = np.bincount(non_zero_labels.astype(int)).argmax()

                gt_mask = np.zeros((*gt_slice.shape, 4))
                gt_mask[gt_slice == target_label, 1] = 1.0
                gt_mask[gt_slice == target_label, 3] = 0.5
                axes[2].imshow(gt_mask)

                # calculate DSC for this slice
                gt_binary = (gt_slice == target_label).astype(np.int32)
                pred_binary = (pred_slice > 0).astype(np.int32)
                dsc = self.compute_dsc(pred_binary, gt_binary)
                axes[2].set_title(f'Ground Truth (DSC: {dsc:.4f})')
            else:
                axes[2].set_title('Ground Truth (No label in this slice)')
        else:
            axes[2].set_title('Ground Truth (Not available)')

        axes[2].axis('on')

        plt.tight_layout()
        plt.show()