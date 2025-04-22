import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ocd.config import TrainConfig2D
from ocd.dataset.dataset import Dataset
from ocd.dataset.dataset2d import BalancedBatchSampler, Dataset2D


def training_loop_2d(config: TrainConfig2D, create_2d_dataset: bool = False):
  if create_2d_dataset:
    Dataset(base_dir=config.scan_dataset_path).convert_CT_scans_to_images(
      output_dir=config.image_dataset_path, seed=config.seed
    )

  folds = list(range(5))
  folds_dataframe = pd.read_csv(config.image_dataset_path / "metadata.csv")

  for fold in folds:
    train_folds = folds[0:fold] + folds[fold + 1 : len(folds)]
    print(f"Using folds {train_folds} for training, fold {fold} for validation.")
    sampler = BalancedBatchSampler(
      folds_dataframe=folds_dataframe,
      folds=train_folds,
      batch_size=config.batch_size,
    )

    train_dataset = Dataset2D(
      folds_dataframe=folds_dataframe,
      dataset_path=config.image_dataset_path,
      folds=train_folds,
    )
    val_dataset = Dataset2D(
      folds_dataframe=folds_dataframe,
      dataset_path=config.image_dataset_path,
      folds=[fold],
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

    model = config.model
    optimizer = config.optimizer
    criterion = config.criterion
    scheduler = config.scheduler

    train_loss_arr = []
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

        train_loss += loss.detach().item()

      train_loss_arr.append(train_loss / len(train_dataloader))

      print(
        f"Epoch {epoch} - train loss : {train_loss / len(train_dataloader)} -"
        f" learning rate : {scheduler.get_last_lr()[0]}"
      )

      if epoch % 5 == 0:
        print("Evaluating model...")
        model.eval()

        val_loss = 0
        with torch.no_grad():
          for images, labels in tqdm(val_dataloader):
            images, labels = images.to(config.device), labels.to(config.device)

            output = model(images)
            loss = criterion(output, labels)

            val_loss += loss.item()

          print(
            f"Epoch {epoch} - val loss : {val_loss / len(val_dataloader)} -"
            f" learning rate : {scheduler.get_last_lr()[0]}"
          )

      scheduler.step()
