from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def dilate2d(
  volume: torch.Tensor, kernel_size: int = 3, stride: int = 1
) -> torch.Tensor:
  return F.max_pool2d(
    volume, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2
  )


def dilated_dice_score(
  preds: torch.Tensor,
  targets: torch.Tensor,
  num_classes: int,
  mode: Literal["loss", "score"],
  kernel_size_dilation: int = 3,
  epsilon: float = 1e-8,
):
  preds_one_hot = (
    F.one_hot(torch.argmax(preds, dim=1), num_classes=num_classes)
    .permute(0, 3, 1, 2)
    .float()
  )
  targets_one_hot = (
    F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
  )

  dilated_dice_scores_per_class = []
  nb_positive_target_per_class = []
  for c in range(1, num_classes):
    target_c = targets_one_hot[:, c : c + 1, :, :]
    pred_c = preds_one_hot[:, c : c + 1, :, :]

    nb_positive_target = torch.sum(target_c, dim=[2, 3])
    nb_positive_pred = torch.sum(pred_c, dim=[2, 3])

    nb_positive_target_per_class.append(nb_positive_target)

    dilated_pred_c = dilate2d(pred_c, kernel_size=kernel_size_dilation)
    dilated_target_c = dilate2d(target_c, kernel_size=kernel_size_dilation)

    intersect_count_target_over_dilated_pred = torch.sum(
      target_c * dilated_pred_c, dim=[2, 3]
    )
    intersect_count_pred_over_dilated_target = torch.sum(
      pred_c * dilated_target_c, dim=[2, 3]
    )

    numerator = (
      intersect_count_target_over_dilated_pred
      + intersect_count_pred_over_dilated_target
    )
    denominator = nb_positive_target + nb_positive_pred + epsilon

    per_item_per_class_dilated_dice = numerator / denominator
    if mode == "loss":
      dilated_dice_scores_per_class.append(1 - per_item_per_class_dilated_dice)
    else:
      dilated_dice_scores_per_class.append(per_item_per_class_dilated_dice)

  all_classes_dilated_dice = torch.cat(dilated_dice_scores_per_class, dim=1)

  all_nb_positive_target = torch.cat(nb_positive_target_per_class, dim=1)

  presence_mask = all_nb_positive_target > 0

  num_present_classes_per_item = torch.sum(presence_mask.float(), dim=1)

  masked_scores = all_classes_dilated_dice * presence_mask.float()
  sum_scores_per_item = torch.sum(masked_scores, dim=1)
  mean_dilated_dice_per_item = sum_scores_per_item / (
    num_present_classes_per_item + epsilon
  )

  mean_dilated_dice = torch.mean(mean_dilated_dice_per_item)

  return mean_dilated_dice


class DBCECriterion(nn.Module):
  def __init__(
    self,
    ce_label_smoothing: float,
    num_classes: int = 3,
    dilation_kernel_size: int = 3,
    eps: float = 1e-8,
  ):
    super().__init__()

    self.num_classes = num_classes
    self.eps = eps
    self.ce_label_smoothing = ce_label_smoothing
    self.dilation_kernel_size = dilation_kernel_size

  def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    ce_loss_map = F.cross_entropy(
      preds, targets, reduction="none", label_smoothing=self.ce_label_smoothing
    )

    with torch.no_grad():
      Y_one_hot = (
        F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
      )

      W_all_classes = torch.zeros_like(Y_one_hot, device=preds.device)

      for c in range(self.num_classes):
        Yc = Y_one_hot[:, c : c + 1, :, :]

        Dc = dilate2d(
          Yc,
          kernel_size=self.dilation_kernel_size,
          stride=1,
        )
        Dc = (Dc > 0).float()

        area_dc = torch.sum(Dc, dim=(1, 2, 3), keepdim=True)

        weight_factor = 1.0 / (1.0 + area_dc + self.eps)
        Wc = weight_factor * Dc

        W_all_classes[:, c : c + 1, :, :] = Wc

      M, _ = torch.max(W_all_classes, dim=1, keepdim=False)

    weighted_ce_loss_map = M * ce_loss_map
    loss_per_sample = torch.sum(weighted_ce_loss_map, dim=(1, 2))

    return loss_per_sample.mean()


def dice_score(
  preds: torch.Tensor,
  targets: torch.Tensor,
  num_classes: int,
  mode: Literal["loss", "score"],
  eps: float = 1e-8,
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
  for class_idx in range(1, num_classes):
    preds_class = predictions[:, class_idx, :, :]
    targets_class = targets_one_hot[:, class_idx, :, :]

    intersection = torch.sum(preds_class * targets_class, dim=(1, 2))

    pred_sum = torch.sum(preds_class, dim=(1, 2))

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
    mode: Literal["normal", "dilated"],
    weight_dice: float = 1.7,
    weight_ce: float = 1.0,
    num_classes: int = 3,
    ignore_index: int = 0,
  ):
    super().__init__()
    self.mode = mode
    self.ce_label_smoothing = ce_label_smoothing
    self.weight_dice = weight_dice
    self.weight_ce = weight_ce
    self.num_classes = num_classes
    self.ignore_index = ignore_index

  def forward(self, preds: torch.Tensor, targets: torch.Tensor):
    if self.mode == "normal":
      return self.weight_dice * dice_score(
        preds, targets, self.num_classes, mode="loss"
      ) + self.weight_ce * nn.CrossEntropyLoss(label_smoothing=self.ce_label_smoothing)(
        preds, targets.long()
      )
    else:
      return self.weight_dice * dilated_dice_score(
        preds, targets, self.num_classes, mode="loss"
      ) + self.weight_ce * DBCECriterion(ce_label_smoothing=self.ce_label_smoothing)(
        preds, targets.long()
      )
