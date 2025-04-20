from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from ocd.config import TrainConfig2D
from ocd.loops.loop_2d import training_loop_2d
from ocd.loss import WeightedSegmentationLoss
from ocd.models.nn_unet_2d import NNUnet2D

if __name__ == "__main__":
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = NNUnet2D(in_channels=1).to(device)
  optimizer = AdamW(params=model.parameters(), lr=3e-4)
  scheduler = StepLR(optimizer=optimizer, step_size=2, gamma=0.98)

  config = TrainConfig2D(
    dataset_path=Path("DatasetChallenge"),
    image_dataset_path=Path("2D_CT_SCANS"),
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=WeightedSegmentationLoss(),
    device=device,
    batch_size=10,
  )

  training_loop_2d(config=config, create_2d_dataset=False)
