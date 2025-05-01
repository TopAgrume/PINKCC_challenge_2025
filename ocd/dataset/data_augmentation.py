import numpy as np
from monai.transforms.compose import Compose
from monai.transforms.intensity.dictionary import (
  RandAdjustContrastd,
  RandGaussianNoised,
)
from monai.transforms.spatial.dictionary import (
  Rand2DElasticd,
  RandFlipd,
  RandRotated,
  RandZoomd,
)
from monai.transforms.utility.dictionary import (
  ToTensorD,
)

TRANSFORMS = Compose(
  [
    RandGaussianNoised(keys=["image"], prob=0.4, mean=0.0, std=0.05),
    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.8, 1.5)),
    # Randomly scale intensity range (useful for CT windowing variations)
    # Adjust ranges based on typical HU values for your specific CT scans
    # Example: scaling a common lung window (-1000 to 400 HU)
    # Note: You might want to apply intensity normalization/windowing BEFORE augmentation,
    # but random variations around that window can be augmentation.
    # This example shifts a hypothetical [-500, 500] range slightly.
    # RandScaleIntensityRanged(
    #   keys=["image"],
    #   міжfrom_range=[-500, 500],  # Source intensity range (adjust based on your data)
    #   to_range=[-500, 500],  # Destination intensity range (can be same)
    #   prob=0.2,
    #   międzyfactor=(0.8, 1.2),  # Randomly scale the range
    # ),
    RandRotated(
      keys=["image", "mask"],
      range_x=(-np.pi / 12, np.pi / 12),  # Rotate up to +/- 15 degrees
      prob=0.5,
      mode=["bilinear", "nearest"],  # Bilinear for image, Nearest for mask
      padding_mode="border",  # or 'border', 'reflection'
    ),
    RandFlipd(
      keys=["image", "mask"],
      spatial_axis=[0],  # Flip along height (axis 0)
      prob=0.5,
    ),
    RandFlipd(
      keys=["image", "mask"],
      spatial_axis=[1],  # Flip along width (axis 1)
      prob=0.5,
    ),
    RandZoomd(
      keys=["image", "mask"],
      min_zoom=0.9,
      max_zoom=1.1,  # Zoom between 90% and 110%
      prob=0.25,
      mode=["bilinear", "nearest"],  # Bilinear for image, Nearest for mask
      padding_mode="edge",
    ),
    # Random affine transform (rotation, scale, shear, translate combined)
    # You might use this instead of or in addition to separate transforms.
    # Be careful not to apply too many spatial transforms consecutively if performance is critical.
    # RandAffined(
    #     keys=['image', 'mask'],
    #     rotate_range=(-np.pi/12, np.pi/12),
    #     scale_range=(-0.1, 0.1), # e.g., scale by 90% to 110%
    #     translate_range=(20, 20), # e.g., translate by +/- 20 pixels
    #     shear_range=None, # Add shear if desired, e.g., [(-0.05, 0.05)] * spatial_dims
    #     prob=0.5,
    #     mode=['bilinear', 'nearest'],
    #     padding_mode='zeros'
    # ),
    Rand2DElasticd(
      keys=["image", "mask"],
      spacing=(20, 20),  # Controls the grid spacing of deformation
      magnitude_range=(1, 2),  # Controls the magnitude of deformation
      prob=0.3,
      mode=["bilinear", "nearest"],
      padding_mode="zeros",
    ),
    ToTensorD(keys=["image", "mask"]),
  ]
)
