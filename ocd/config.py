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
    loss: nn.Module,
    seed: int = 42,
    batch_size: int = 64,
    epochs: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
  ) -> None:
    self.scan_dataset_path = dataset_path
    self.image_dataset_path = image_dataset_path
    self.seed = 42

    self.batch_size = 64
    self.epochs = epochs
    self.model = model
    self.loss = loss
    self.optimizer = optimizer

    self.device = device

    # TODO: add augmentations, scheduler

  def save(self):
    pass
