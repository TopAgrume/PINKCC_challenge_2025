import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedSegmentationLoss(nn.Module):
  """
  A weighted sum of soft Dice, Precision, and Recall losses for segmentation.

  Calculates the loss based on predicted probabilities and ground truth targets.
  Supports both binary (num_classes=1 or 2) and multi-class segmentation.
  Metrics are calculated per class (excluding ignore_index) and then averaged.
  """

  def __init__(
    self,
    weight_dice=1.0,
    weight_precision=0.5,
    weight_recall=0.5,
    num_classes=3,
    ignore_index=0,
    smooth=1e-6,
  ):
    """
    Args:
        weight_dice (float): Weight for the Dice loss component.
        weight_precision (float): Weight for the Precision loss component.
        weight_recall (float): Weight for the Recall loss component.
        num_classes (int): The total number of classes in the segmentation task.
                            Set to 1 if the model outputs a single channel for the foreground probability.
                            Set to 2 or more for multi-class output logits.
        ignore_index (int): Class index to ignore when calculating metrics (e.g., background class 0).
                            Only applicable for num_classes > 1.
        smooth (float): A small value added to the denominator to prevent division by zero.
    """
    super().__init__()
    self.weight_dice = weight_dice
    self.weight_precision = weight_precision
    self.weight_recall = weight_recall
    self.num_classes = num_classes
    self.ignore_index = ignore_index
    self.smooth = smooth

    # Validate weights
    if not all(
      isinstance(w, int | float) and w >= 0
      for w in [weight_dice, weight_precision, weight_recall]
    ):
      raise ValueError("Loss weights must be non-negative numbers.")

    # Validate num_classes
    if not isinstance(num_classes, int) or num_classes < 1:
      raise ValueError("num_classes must be a positive integer.")

    # Validate ignore_index
    if not isinstance(ignore_index, int) or (
      self.num_classes > 1 and (ignore_index < 0 or ignore_index >= num_classes)
    ):
      if self.num_classes > 1:
        raise ValueError(
          f"ignore_index must be between 0 and num_classes-1 for multi-class (num_classes={num_classes})."
        )
      # For num_classes=1, ignore_index is not strictly used in per-class loop, but keep validation simple
      if ignore_index < 0:
        raise ValueError("ignore_index must be non-negative.")

  def forward(self, predictions, targets):
    """
    Calculates the weighted segmentation loss.

    Args:
        predictions (torch.Tensor): Model predictions (logits). Shape (N, C, H, W) for multi-class
                                    or (N, 1, H, W) for binary foreground logits.
        targets (torch.Tensor): Ground truth labels. Shape (N, H, W) with class indices for multi-class
                                or (N, H, W) with 0/1 for binary.

    Returns:
        torch.Tensor: The calculated total weighted loss.
    """
    # Ensure targets are on the same device as predictions
    targets = targets.to(predictions.device)

    if self.num_classes == 1:
      # Binary case: Model outputs a single channel (logits for foreground)
      if predictions.shape[1] != 1:
        raise ValueError(
          f"For num_classes=1, predictions must have 1 channel, but got {predictions.shape[1]}."
        )
      probs = torch.sigmoid(predictions)  # (N, 1, H, W)
      # Ensure targets are float and same shape as probs for element-wise ops
      # Assuming targets are (N, H, W) with 0/1
      targets = targets.float().unsqueeze(1)  # (N, 1, H, W)

      # Calculate losses for the single foreground class
      dice_loss = self._soft_dice_loss(probs, targets)
      precision_loss = self._soft_precision_loss(probs, targets)
      recall_loss = self._soft_recall_loss(probs, targets)

    else:
      # Multi-class case: Model outputs C channels (logits for C classes)
      if predictions.shape[1] != self.num_classes:
        raise ValueError(
          f"For num_classes={self.num_classes}, predictions must have {self.num_classes} channels, but got {predictions.shape[1]}."
        )

      probs = torch.softmax(predictions, dim=1)  # (N, C, H, W)
      # Convert targets to one-hot encoding (N, H, W) -> (N, C, H, W)
      targets_one_hot = (
        F.one_hot(targets.long(), num_classes=self.num_classes)
        .permute(0, 3, 1, 2)
        .float()
      )  # (N, C, H, W)

      dice_losses = []
      precision_losses = []
      recall_losses = []

      # Calculate metrics for each class (excluding ignore_index)
      for class_idx in range(self.num_classes):
        if class_idx == self.ignore_index:
          continue

        probs_class = probs[:, class_idx, :, :]  # (N, H, W)
        targets_class = targets_one_hot[:, class_idx, :, :]  # (N, H, W)

        dice_losses.append(self._soft_dice_loss(probs_class, targets_class))
        precision_losses.append(self._soft_precision_loss(probs_class, targets_class))
        recall_losses.append(self._soft_recall_loss(probs_class, targets_class))

      # Average losses across classes that were included
      # Use torch.stack and torch.mean for robustness with empty lists if all classes ignored (unlikely)
      dice_loss = (
        torch.mean(torch.stack(dice_losses))
        if dice_losses
        else torch.tensor(0.0, device=predictions.device)
      )
      precision_loss = (
        torch.mean(torch.stack(precision_losses))
        if precision_losses
        else torch.tensor(0.0, device=predictions.device)
      )
      recall_loss = (
        torch.mean(torch.stack(recall_losses))
        if recall_losses
        else torch.tensor(0.0, device=predictions.device)
      )

    # Calculate the total weighted loss
    total_loss = (
      self.weight_dice * dice_loss
      + self.weight_precision * precision_loss
      + self.weight_recall * recall_loss
    )

    return total_loss

  def _soft_dice_loss(self, predictions, targets):
    """Calculates the soft Dice loss (1 - Dice coefficient)."""
    # predictions, targets are (N, H, W) or (N, 1, H, W) probability/target tensors

    # Flatten spatial dimensions for metric calculation
    predictions = predictions.view(predictions.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    # Calculate intersection and sum of areas
    intersection = (predictions * targets).sum(
      dim=1
    )  # Sum over flattened spatial dimensions for each item in batch
    sum_of_areas = predictions.sum(dim=1) + targets.sum(dim=1)

    # Calculate Dice coefficient per item in batch
    dice_coefficient = (2.0 * intersection + self.smooth) / (sum_of_areas + self.smooth)

    # Return 1 - mean Dice coefficient over the batch
    return 1.0 - dice_coefficient.mean()

  def _soft_precision_loss(self, predictions, targets):
    """Calculates the soft Precision loss (1 - Precision)."""
    # predictions, targets are (N, H, W) or (N, 1, H, W) probability/target tensors

    # Flatten spatial dimensions
    predictions = predictions.view(predictions.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    # Calculate True Positives and False Positives per item in batch
    true_positives = (predictions * targets).sum(dim=1)
    false_positives = (predictions * (1.0 - targets)).sum(dim=1)

    # Calculate Precision per item in batch
    precision = (true_positives + self.smooth) / (
      true_positives + false_positives + self.smooth
    )

    # Return 1 - mean Precision over the batch
    return 1.0 - precision.mean()

  def _soft_recall_loss(self, predictions, targets):
    """Calculates the soft Recall loss (1 - Recall)."""
    # predictions, targets are (N, H, W) or (N, 1, H, W) probability/target tensors

    # Flatten spatial dimensions
    predictions = predictions.view(predictions.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    # Calculate True Positives and False Negatives per item in batch
    true_positives = (predictions * targets).sum(dim=1)
    false_negatives = ((1.0 - predictions) * targets).sum(dim=1)

    # Calculate Recall per item in batch
    recall = (true_positives + self.smooth) / (
      true_positives + false_negatives + self.smooth
    )

    # Return 1 - mean Recall over the batch
    return 1.0 - recall.mean()
