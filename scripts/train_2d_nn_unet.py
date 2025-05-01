import os
from pathlib import Path

import torch
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from ocd import OUTPUT_DIR
from ocd.config import TrainConfig2D
from ocd.dataset.data_augmentation import TRANSFORMS
from ocd.loops.loop_2d import training_loop_2d
from ocd.loss import WeightedSegmentationLoss
from ocd.models.nn_unet_2d import NNUnet2D

if __name__ == "__main__":
  if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
  logger.add(OUTPUT_DIR / "training_loop_2d.log")

  device = "cuda" if torch.cuda.is_available() else "cpu"
  ce_label_smoothing = 0.05

  config = TrainConfig2D(
    dataset_path=Path("DatasetChallenge"),
    image_dataset_path=Path("2D_CT_SCANS"),
    model=(NNUnet2D, dict(in_channels=1, num_pool=5, base_num_features=32)),
    optimizer=(AdamW, dict(lr=3e-4)),
    scheduler=(StepLR, dict(step_size=100, gamma=0.99)),
    criterion=(
      WeightedSegmentationLoss,
      dict(ce_label_smoothing=ce_label_smoothing),
    ),
    augmentations=TRANSFORMS,
    device=device,
    batch_size=10,
    epochs=12,
    ce_label_smoothing=ce_label_smoothing,
  )
  config.save_config()

  training_loop_2d(config=config, create_2d_dataset=False)
