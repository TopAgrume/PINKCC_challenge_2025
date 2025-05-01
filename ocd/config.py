from pathlib import Path
from pprint import pprint
from typing import Any  # Import standard pprint

import torch.nn as nn
import torch.optim
from monai.transforms.compose import Compose

from ocd import OUTPUT_DIR


class TrainConfig2D:
  def __init__(
    self,
    dataset_path: Path,
    image_dataset_path: Path,
    model: tuple[type[nn.Module], dict[str, Any]],
    optimizer: tuple[type[torch.optim.Optimizer], dict[str, Any]],
    scheduler: tuple[type[torch.optim.lr_scheduler.LRScheduler], dict[str, Any]],
    criterion: tuple[type[nn.Module], dict[str, Any]],
    augmentations: Compose,
    device: str,
    seed: int = 42,
    batch_size: int = 10,
    epochs: int = 5,
    num_classes: int = 3,
    ce_label_smoothing: float = 0.05,
  ) -> None:
    self.scan_dataset_path = dataset_path
    self.image_dataset_path = image_dataset_path
    self.seed = seed
    self.num_classes = num_classes
    self.ce_label_smoothing = ce_label_smoothing

    self.batch_size = batch_size
    self.epochs = epochs
    self.model = model
    self.criterion = criterion
    self.optimizer = optimizer
    self.scheduler = scheduler

    self.device = device
    self.augmentations = augmentations

  def get_model_optimizer_scheduler(
    self,
  ) -> tuple[
    nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, nn.Module
  ]:
    model = self.model[0](**self.model[1]).to(self.device)
    optimizer = self.optimizer[0](params=model.parameters(), **self.optimizer[1])
    scheduler = self.scheduler[0](optimizer=optimizer, **self.scheduler[1])
    criterion = self.criterion[0](**self.criterion[1])
    return model, optimizer, scheduler, criterion

  def save_config(self):
    with open(OUTPUT_DIR / "config", "w") as f:
      config_dict = vars(self).copy()
      config_dict["model"] = repr(self.model[0](**self.model[1]))
      config_dict["optimizer"] = f"{self.optimizer[0].__name__}"
      config_dict["scheduler"] = (
        f"{self.scheduler[0].__name__} with {self.scheduler[1]}"
      )
      config_dict["criterion"] = repr(self.criterion[0](**self.criterion[1]))
      config_dict["scan_dataset_path"] = str(self.scan_dataset_path)
      config_dict["image_dataset_path"] = str(self.image_dataset_path)
      transform_names = [type(t).__name__ for t in self.augmentations.transforms]
      config_dict["augmentations"] = f"Compose(transforms={transform_names})"
      config_dict["ce_label_smoothing"] = str(self.ce_label_smoothing)

      pprint(config_dict, stream=f, indent=2, width=120)
