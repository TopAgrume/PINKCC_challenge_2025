import os
import sys

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from ocd import OUTPUT_DIR
from ocd.config import TrainConfig2D
from ocd.dataset.dataset import Dataset
from ocd.dataset.dataset2d import BalancedBatchSampler, Dataset2D
from ocd.loss import dice_score


def eval_model(
  model: nn.Module,
  dataloader: DataLoader,
  config: TrainConfig2D,
  criterion: nn.Module,
  best_val_loss: float,
  epoch: int,
  scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> float:
  model.eval()
  val_loss = 0
  with torch.no_grad():
    for images, labels in tqdm(dataloader):
      images, labels = images.to(config.device), labels.to(config.device)

      output = model(images)
      loss = criterion(output, labels)

      val_loss += loss.item()

    val_loss /= len(dataloader)

    logger.info(
      f"Epoch {epoch} - val loss : {val_loss} -"
      f" learning rate : {scheduler.get_last_lr()[0]}"
    )

    if val_loss < best_val_loss:
      torch.save(model.state_dict(), OUTPUT_DIR / "model.pth")

  return min(val_loss, best_val_loss)


def test_model(
  model: nn.Module,
  dataloader: DataLoader,
  config: TrainConfig2D,
  criterion: nn.Module,
):
  cmap = matplotlib.colors.ListedColormap(["red", "green"])
  bounds = [0.5, 1.5, 2.5]
  norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

  os.mkdir(OUTPUT_DIR / "test_figs")
  model.eval()
  test_loss = 0
  with torch.no_grad():
    for images, labels, paths in tqdm(dataloader):
      images, labels = images.to(config.device), labels.to(config.device)

      outputs: torch.Tensor = model(images)
      loss = criterion(outputs, labels)

      test_loss += loss.item()

      for i in range(len(images)):
        images = images.cpu()
        labels = labels.cpu()
        outputs = outputs.cpu()

        image = images[i][0]
        label = labels[i]
        pred = torch.argmax(outputs[i], dim=0)
        path = paths[i]

        plt.figure(figsize=(20, 12))
        plt.subplot(1, 2, 1)
        plt.imshow(image, "gray", interpolation="none")
        plt.imshow(
          np.ma.masked_where(label == 0, label),
          cmap=cmap,
          norm=norm,
          interpolation="none",
          alpha=0.25,
        )
        plt.title("real label")
        plt.subplot(1, 2, 2)
        plt.imshow(image, "gray", interpolation="none")
        plt.imshow(
          np.ma.masked_where(pred == 0, pred),
          cmap=cmap,
          norm=norm,
          interpolation="none",
          alpha=0.25,
        )
        plt.title(
          f"our pred DSC={
            dice_score(
              outputs[i].unsqueeze(0),
              label.unsqueeze(0),
              config.num_classes,
              mode='score',
            ).item()
          }"
        )
        plt.savefig(OUTPUT_DIR / "test_figs" / f"{path}.png")

        plt.close()

    test_loss /= len(dataloader)

    logger.info(f"Test loss : {test_loss}")


def training_loop_2d(config: TrainConfig2D, create_2d_dataset: bool = False):
  if create_2d_dataset:
    Dataset(base_dir=config.scan_dataset_path).convert_CT_scans_to_images(
      output_dir=config.image_dataset_path, seed=config.seed
    )

  folds = list(range(5))
  folds_dataframe = pd.read_csv(config.image_dataset_path / "metadata.csv")

  test_dataset = Dataset2D(
    folds_dataframe=folds_dataframe,
    dataset_path=config.image_dataset_path,
    folds=[-1],
    with_path=True,
    augmentations=None,
  )
  test_dataloader = DataLoader(
    test_dataset, batch_size=config.batch_size, pin_memory=True
  )

  for fold in folds[:5]:
    train_folds = folds[0:fold] + folds[fold + 1 : len(folds)]
    logger.info(f"Using folds {train_folds} for training, fold {fold} for validation.")
    sampler = BalancedBatchSampler(
      folds_dataframe=folds_dataframe,
      folds=train_folds,
      batch_size=config.batch_size,
      ratio=0.5,
    )

    train_dataset = Dataset2D(
      folds_dataframe=folds_dataframe,
      dataset_path=config.image_dataset_path,
      folds=train_folds,
      augmentations=config.augmentations,
    )
    val_dataset = Dataset2D(
      folds_dataframe=folds_dataframe,
      dataset_path=config.image_dataset_path,
      folds=[fold],
      augmentations=None,
    )

    train_dataloader = DataLoader(
      dataset=train_dataset, sampler=sampler, pin_memory=True, num_workers=2
    )
    val_dataloader = DataLoader(
      dataset=val_dataset,
      shuffle=True,
      batch_size=config.batch_size,
      pin_memory=True,
    )

    model, optimizer, scheduler, criterion = config.get_model_optimizer_scheduler()

    train_loss_arr = []
    best_val_loss = sys.maxsize
    for epoch in range(1, config.epochs + 1):
      train_loss = 0
      model.train()
      for images, labels in tqdm(train_dataloader):
        images, labels = images.to(config.device), labels.to(config.device)

        output = model(images)

        optimizer.zero_grad()

        loss = criterion(output, labels)
        loss.backward()

        optimizer.step()

        loss_value = loss.detach().item()
        train_loss += loss_value

        train_loss_arr.append(loss_value)
        scheduler.step()

      train_loss_arr.append(train_loss / len(train_dataloader))

      logger.info(
        f"Epoch {epoch} - train loss : {train_loss / len(train_dataloader)} -"
        f" learning rate : {scheduler.get_last_lr()[0]}"
      )

      if epoch % 3 == 0:
        logger.info("------------------------------")
        logger.info("Evaluating model...")
        best_val_loss = eval_model(
          model=model,
          dataloader=val_dataloader,
          config=config,
          criterion=criterion,
          best_val_loss=best_val_loss,
          epoch=epoch,
          scheduler=scheduler,
        )
        logger.info("------------------------------")

    np.save(OUTPUT_DIR / f"train_loss_fold_{fold}.npy", np.array(train_loss_arr))

  model, _, _, criterion = config.get_model_optimizer_scheduler()
  model.load_state_dict(torch.load(OUTPUT_DIR / "model.pth", weights_only=True))
  test_model(
    dataloader=test_dataloader, model=model, config=config, criterion=criterion
  )
