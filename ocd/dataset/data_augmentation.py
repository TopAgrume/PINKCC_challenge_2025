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

TRANSFORMS = Compose(
  [
    RandGaussianNoised(keys=["image"], prob=0.4, mean=0.0, std=0.05),  #
    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.8, 1.5)),  #
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
      min_zoom=0.9,
      max_zoom=1.1,
      prob=0.25,
      mode=["bilinear", "nearest"],
      padding_mode="edge",
    ),
    Rand2DElasticd(
      keys=["image", "mask"],
      spacing=(20, 20),
      magnitude_range=(1, 2),
      prob=0.3,
      mode=["bilinear", "nearest"],
      padding_mode="zeros",
    ),
  ]
)
