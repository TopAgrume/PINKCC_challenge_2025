from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_score(
  preds: torch.Tensor,
  targets: torch.Tensor,
  num_classes: int,
  mode: Literal["loss", "score"],
  ignore_index: int = 0,
  eps: float = 1e-6,
) -> torch.Tensor:
  predictions = (
    F.one_hot(torch.argmax(preds, dim=1), num_classes=num_classes)
    .permute(0, 3, 1, 2)
    .float()
  )
  targets_one_hot = (
    F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
  )
  dice_losses = []
  for class_idx in range(num_classes):
    if class_idx == ignore_index:
      continue

    probs_class = predictions[:, class_idx, :, :]
    targets_class = targets_one_hot[:, class_idx, :, :]

    intersection = torch.sum(probs_class * targets_class, dim=(1, 2))

    pred_sum = torch.sum(probs_class, dim=(1, 2))

    target_sum = torch.sum(targets_class, dim=(1, 2))

    score = (2 * intersection + eps) / (pred_sum + target_sum + eps)
    if mode == "score":
      dice_losses.append(score)
    else:
      dice_losses.append(1 - score)

  return torch.mean(torch.stack(dice_losses))


class WeightedSegmentationLoss(nn.Module):
  def __init__(
    self,
    ce_label_smoothing: float,
    weight_dice: float = 0.7,
    weight_ce: float = 0.3,
    num_classes: int = 3,
    ignore_index: int = 0,
  ):
    super().__init__()
    self.ce_label_smoothing = ce_label_smoothing
    self.weight_dice = weight_dice
    self.weight_ce = weight_ce
    self.num_classes = num_classes
    self.ignore_index = ignore_index

  def forward(self, preds: torch.Tensor, targets: torch.Tensor):
    if self.num_classes > 1:
      if preds.shape[1] != self.num_classes:
        raise ValueError(
          "channel's number in prediction is different than number of classes"
        )

      return self.weight_dice * dice_score(
        preds, targets, self.num_classes, mode="loss", ignore_index=self.ignore_index
      ) + self.weight_ce * nn.CrossEntropyLoss(label_smoothing=self.ce_label_smoothing)(
        preds, targets.long()
      )
    else:
      raise NotImplementedError("Not implemented yet")
