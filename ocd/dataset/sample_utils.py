import SimpleITK as sitk
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from nibabel.orientations import (
  aff2axcodes,
  apply_orientation,
  axcodes2ornt,
  inv_ornt_aff,
  ornt_transform,
)
from pathlib import Path
from typing import Tuple, Union
from nibabel.filebasedimages import FileBasedImage


class SampleUtils:
  """
  Usefull operation on CT and segmentation data
  """

  @classmethod
  def load_from_path_nib(cls, file_path: str | Path) -> tuple[FileBasedImage, np.ndarray]:
    """
    Load NIFTI file content using nibabel

    Args:
        file_path: Path to the NIFTI file

    Returns:
        Tuple of (nib.Nifti1Image, numpy.ndarray)
    """
    img = nib.loadsave.load(file_path)
    data = img.get_fdata()  # pyright: ignore
    return img, data

  @classmethod
  def load_from_path_nib_sitk(cls, file_path: Union[str, Path]) -> Tuple[sitk.Image, np.ndarray]:
    """
    Load NIFTI file content using SimpleITK

    Args:
        file_path: Path to the NIFTI file

    Returns:
        Tuple of (sitk.Image, numpy.ndarray)
    """
    img = sitk.ReadImage(str(file_path))
    data = sitk.GetArrayFromImage(img)
    return img, data

  @classmethod
  def ensure_rai_orientation(cls, image: Union[FileBasedImage, sitk.Image]) -> Union[sitk.Image, FileBasedImage]:
    """
    Ensure the image is in the RAI orientation (Right, Anterior, Inferior)
    which is commonly used in medical imaging.

    Args:
        image: 3D numpy array

    Returns:
        Reoriented SimpleITK image in RAI orientation
    """
    if isinstance(image, sitk.Image):
        reorient_filter = sitk.DICOMOrientImageFilter()
        reorient_filter.SetDesiredCoordinateOrientation("RAI")
        return reorient_filter.Execute(image)
    else:
        affine = image.affine  # pyright: ignore

        axcodes = aff2axcodes(affine)
        if axcodes != ("R", "A", "I"):
            current_ornt = axcodes2ornt(axcodes)
            target_ornt = axcodes2ornt(("R", "A", "I"))

            transform = ornt_transform(current_ornt, target_ornt)
            data = image.get_fdata()  # pyright: ignore
            reoriented_data = apply_orientation(data, transform)
            new_affine = affine @ inv_ornt_aff(transform, data.shape)

            return nib.nifti1.Nifti1Image(reoriented_data, new_affine)
        else:
            return image

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
    if isinstance(data, sitk.Image):
      data = sitk.GetArrayFromImage(data)
      axis = 2 - axis

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
    ct_data: Union[sitk.Image, np.ndarray],
    window_center: int = 40,
    window_width: int = 400,
    output_range: Tuple[float, float] = (-1, 1),
  ) -> Union[sitk.Image, np.ndarray]:
    """
    Apply windowing and normalization to CT data

    Args:
        ct_data: Raw CT data in Hounsfield units (SimpleITK image or numpy array)
        window_center: Center of the windowing operation (typically 40 for soft tissue)
        window_width: Width of the window (typically 400 for soft tissue)
        output_range: Desired output range for normalization

    Returns:
        Normalized CT data in the same format as input
    """
    is_sitk = isinstance(ct_data, sitk.Image)

    if is_sitk:
      data_np = sitk.GetArrayFromImage(ct_data)
    else:
      data_np = ct_data

    # windowing
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    windowed_data = np.clip(data_np, min_val, max_val)

    # normalize
    out_min, out_max = output_range
    normalized = out_min + (windowed_data - min_val) * (out_max - out_min) / (max_val - min_val)

    if is_sitk:
      result = sitk.GetImageFromArray(normalized)
      result.SetOrigin(ct_data.GetOrigin())
      result.SetSpacing(ct_data.GetSpacing())
      result.SetDirection(ct_data.GetDirection())
      return result
    else:
      return normalized