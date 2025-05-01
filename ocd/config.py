from pathlib import Path
from pprint import pprint  # Import standard pprint

import torch.nn as nn
import torch.optim
from monai.transforms.compose import Compose

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
      transform_names = [type(t).__name__ for t in self.augmentations.transforms]
      config_dict["augmentations"] = f"Compose(transforms={transform_names})"
      config_dict["ce_label_smoothing"] = str(self.ce_label_smoothing)

      pprint(config_dict, stream=f, indent=2, width=120)
