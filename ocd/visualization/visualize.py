# ocd/visualization/visualize.py
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple

from ocd.dataset.sample_utils import SampleUtils

def visualize_prediction_vs_gt(
    ct_path: Path,
    gt_mask_path: Path,
    pred_mask_path: Path,
    slice_idx: Optional[int] = None,
    axis: int = 2, # 0=Sagittal, 1=Coronal, 2=Axial
    figsize: Tuple[int, int] = (18, 6),
    window_center: int = 40,
    window_width: int = 400,
    title: str = "nnUNet Prediction vs Ground Truth"
) -> None:
    """
    Visualizes a comparison between the original CT, ground truth mask, and predicted mask.
    """
    print(f"\n--- Visualizing Comparison ---")
    print(f"CT: {ct_path}")
    print(f"GT: {gt_mask_path}")
    print(f"Prediction: {pred_mask_path}")

    try:
        ct_sitk, _ = SampleUtils.load_from_path_sitk(ct_path)
        gt_sitk, _ = SampleUtils.load_from_path_sitk(gt_mask_path)
        pred_sitk, _ = SampleUtils.load_from_path_sitk(pred_mask_path)

        ct_sitk = SampleUtils.ensure_rai_orientation(ct_sitk)
        gt_sitk = SampleUtils.ensure_rai_orientation(gt_sitk)
        pred_sitk = SampleUtils.ensure_rai_orientation(pred_sitk)


        # usefull summary
        if ct_sitk.GetSize() != gt_sitk.GetSize() or ct_sitk.GetSize() != pred_sitk.GetSize():
            print("Warning: Image, GT, and Prediction dimensions do not match after loading/orientation.")
            print(f"CT Size:   {ct_sitk.GetSize()}")
            print(f"GT Size:   {gt_sitk.GetSize()}")
            print(f"Pred Size: {pred_sitk.GetSize()}")

        # --- Calculate DSC ---
        # Binarize masks for DSC calculation (label 1 vs background)
        gt_binary = sitk.BinaryThreshold(gt_sitk, lowerThreshold=1, upperThreshold=255, insideValue=1, outsideValue=0)
        pred_binary = sitk.BinaryThreshold(pred_sitk, lowerThreshold=1, upperThreshold=255, insideValue=1, outsideValue=0)

        overlap_measures = sitk.LabelOverlapMeasuresImageFilter()
        try:
            overlap_measures.Execute(gt_binary, pred_binary)
            dsc = overlap_measures.GetDiceCoefficient()
            dsc_text = f"{dsc:.4f}"
        except Exception as dsc_e:
            print(f"Could not calculate DSC: {dsc_e}")
            dsc_text = "Error"


        # --- Visualization ---
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(f"{title} (Overall DSC: {dsc_text})", fontsize=16)


        img_np = sitk.GetArrayFromImage(ct_sitk)     # Z, Y, X
        gt_np = sitk.GetArrayFromImage(gt_sitk)      # Z, Y, X
        pred_np = sitk.GetArrayFromImage(pred_sitk)  # Z, Y, X

        np_axis = 2 - axis

        # find a slice with some label in GT or Pred for better visualization
        if slice_idx is None:
            interesting_slice = np.argmax(np.sum(gt_np + pred_np > 0, axis=(1,2))) # Sum over Y,X; find max along Z
            if interesting_slice > 0 :
                 slice_idx = interesting_slice
            else:
                 slice_idx = img_np.shape[np_axis] // 2 # Default to middle if no label found
        else:
             slice_idx = min(slice_idx, img_np.shape[np_axis] - 1) # Ensure valid index


        if np_axis == 0: # Axial
            img_slice = img_np[slice_idx, :, :]
            gt_slice = gt_np[slice_idx, :, :]
            pred_slice = pred_np[slice_idx, :, :]
            axis_labels = ('X', 'Y')
        elif np_axis == 1: # Coronal
            img_slice = img_np[:, slice_idx, :]
            gt_slice = gt_np[:, slice_idx, :]
            pred_slice = pred_np[:, slice_idx, :]
            axis_labels = ('X', 'Z')
        else: # Sagittal
            img_slice = img_np[:, :, slice_idx]
            gt_slice = gt_np[:, :, slice_idx]
            pred_slice = pred_np[:, :, slice_idx]
            axis_labels = ('Y', 'Z')

        # ct normalization
        min_val = window_center - window_width / 2
        max_val = window_center + window_width / 2
        img_slice_display = np.clip(img_slice, min_val, max_val)
        img_slice_display = (img_slice_display - min_val) / (max_val - min_val)

        axis_names = ["Sagittal", "Coronal", "Axial"]
        slice_title = f"{axis_names[axis]} Slice {slice_idx}"

        # plot original CT
        axes[0].imshow(img_slice_display.T, cmap='gray', origin='lower')
        axes[0].set_title(f'Original CT\n{slice_title}')
        axes[0].set_xlabel(axis_labels[0])
        axes[0].set_ylabel(axis_labels[1])
        axes[0].axis('on')

        # plot GT overlay
        axes[1].imshow(img_slice_display.T, cmap='gray', origin='lower')
        gt_mask_rgb = np.zeros((*gt_slice.shape, 4))
        gt_mask_rgb[gt_slice == 1] = [0.0, 0.0, 1.0, 0.5]  # Blue overlay (Primary Tumor)
        gt_mask_rgb[gt_slice == 2] = [1.0, 1.0, 0.0, 0.5]  # Yellow overlay (Metastasis)
        axes[1].imshow(gt_mask_rgb.transpose(1, 0, 2), origin='lower')
        axes[1].set_title(f'CT + Ground Truth\n{slice_title}')
        axes[1].set_xlabel(axis_labels[0])
        axes[1].set_ylabel(axis_labels[1])
        axes[1].axis('on')

        # plot prediction overlay
        axes[2].imshow(img_slice_display.T, cmap='gray', origin='lower')
        pred_mask_rgb = np.zeros((*pred_slice.shape, 4))
        pred_mask_rgb[pred_slice == 1] = [1.0, 0.0, 0.0, 0.5]  # Red overlay (Primary Tumor)
        pred_mask_rgb[pred_slice == 2] = [0.0, 1.0, 0.0, 0.5]  # Green overlay (Metastasis)
        axes[2].imshow(pred_mask_rgb.transpose(1, 0, 2), origin='lower')
        axes[2].set_title(f'CT + Prediction\n{slice_title}')
        axes[2].set_xlabel(axis_labels[0])
        axes[2].set_ylabel(axis_labels[1])
        axes[2].axis('on')

        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        # plt.show()
        path = pred_mask_path.resolve().parent
        plt.savefig(str(path) + 'plot.png')

    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()