import pandas as pd
from torch.utils.data import DataLoader

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
      dataset=train_dataset,
      sampler=sampler,
      shuffle=True,
      batch_size=config.batch_size,
      pin_memory=True,
    )
    train_dataloader = DataLoader(
      dataset=val_dataset,
      shuffle=True,
      batch_size=config.batch_size,
      pin_memory=True,
    )

    model = config.model
    optimizer = config.optimizer
    criterion = config.criterion

    train_loss = []
    for epoch in range(config.epochs):
      for images, labels in train_dataloader:
        images, labels = images.to(config.device), labels.to(config.device)

        output = model(images)

        optimizer.zero_grad()

        loss = criterion(output, labels)
        loss.backward()

        optimizer.step()

      train_loss.append(loss)
