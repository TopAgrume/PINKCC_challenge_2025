from pathlib import Path
from pprint import pprint  # Import standard pprint

import torch.nn as nn
import torch.optim

from ocd import OUTPUT_DIR


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

  def save_config(self):
    with open(OUTPUT_DIR / "config", "w") as f:
      config_dict = vars(self).copy()
      config_dict["model"] = repr(self.model)
      config_dict["optimizer"] = (
        f"{type(self.optimizer).__name__}"
        f"(lr={self.optimizer.defaults.get('lr', 'N/A')})"
      )
      config_dict["scheduler"] = (
        f"{type(self.scheduler).__name__}"
        f"(step_size={getattr(self.scheduler, 'step_size', 'N/A')},"
        f" gamma={getattr(self.scheduler, 'gamma', 'N/A')})"
      )
      config_dict["criterion"] = repr(self.criterion)
      config_dict["scan_dataset_path"] = str(self.scan_dataset_path)
      config_dict["image_dataset_path"] = str(self.image_dataset_path)

      pprint(config_dict, stream=f, indent=2, width=120)
