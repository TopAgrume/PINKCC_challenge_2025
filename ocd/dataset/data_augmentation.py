import numpy as np
from monai.transforms.compose import Compose
from monai.transforms.intensity.dictionary import (
  RandAdjustContrastd,
  RandGaussianNoised,
  RandGaussianSmoothd,
  RandScaleIntensityd,
  RandShiftIntensityd,
)
from monai.transforms.spatial.dictionary import (
  Rand2DElasticd,
  RandFlipd,
  RandRotated,
  RandZoomd,
)

TRANSFORMS = Compose(
  [
    RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.1),
    RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(1, 1.8)),
    RandRotated(
      keys=["image", "mask"],
      range_x=(-np.pi / 12, np.pi / 12),
      prob=0.5,
      mode=["bilinear", "nearest"],
      padding_mode="border",
    ),
    RandFlipd(
      keys=["image", "mask"],
      spatial_axis=[0],
      prob=0.5,
    ),
    RandFlipd(
      keys=["image", "mask"],
      spatial_axis=[1],
      prob=0.5,
    ),
    RandZoomd(
      keys=["image", "mask"],
      min_zoom=0.85,
      max_zoom=1.15,
      prob=0.5,
      mode=["bilinear", "nearest"],
      padding_mode="edge",
    ),
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    RandGaussianSmoothd(
      keys=["image"], sigma_x=(0.5, 1.15), sigma_y=(0.5, 1.15), prob=0.5
    ),
    Rand2DElasticd(
      keys=["image", "mask"],
      spacing=(25, 25),
      magnitude_range=(1, 2),
      prob=0.5,
      mode=["bilinear", "nearest"],
      padding_mode="zeros",
    ),
  ]
)
