from pathlib import Path

import torch.nn as nn
import torch.optim


class TrainConfig2D:
  def __init__(
    self,
    dataset_path: Path,
    image_dataset_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    criterion: nn.Module,
    device: str,
    seed: int = 42,
    batch_size: int = 32,
    epochs: int = 50,
  ) -> None:
    self.scan_dataset_path = dataset_path
    self.image_dataset_path = image_dataset_path
    self.seed = seed

    self.batch_size = batch_size
    self.epochs = epochs
    self.model = model
    self.criterion = criterion
    self.optimizer = optimizer
    self.scheduler = scheduler

    self.device = device

    # TODO: add augmentations

  def save_model_checkpoint(self):
    pass

  def save_config(self):
    pass
