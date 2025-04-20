import random
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from ocd.dataset.dataset import SampleUtils


class BalancedBatchSampler(Sampler[list[int]]):
  def __init__(
    self,
    folds_dataframe: pd.DataFrame,
    folds: list[int],
    batch_size: int,
    ratio: float = 0.25,  # default value calculated based on the dataset
  ):
    self.batch_size = batch_size

    df = folds_dataframe[folds_dataframe["fold_id"].isin(folds)].reset_index()
    # with_labels, without_labels = (
    #  df["has_labels"].value_counts()[True].item(),
    #  df["has_labels"].value_counts()[False].item(),
    # )
    # ratio: float = with_labels / (with_labels + without_labels)  # pyright: ignore

    self.with_labels_indices: list[int] = df[df["has_labels"]].index.tolist()  # pyright: ignore
    self.without_labels_indices: list[int] = df[~df["has_labels"]].index.tolist()  # pyright: ignore

    self.n_with_labels = int(batch_size * ratio)
    self.n_without_labels = batch_size - self.n_with_labels

    self.num_batches = min(
      len(self.with_labels_indices) // self.n_with_labels,
      len(self.without_labels_indices) // self.n_without_labels,
    )

  def __iter__(self) -> Iterator[list[int]]:
    random.shuffle(self.with_labels_indices)
    random.shuffle(self.without_labels_indices)

    iter1 = iter(self.with_labels_indices)
    iter2 = iter(self.without_labels_indices)

    for _ in range(self.num_batches):
      batch_indices = []

      batch_indices.extend([next(iter1) for _ in range(self.n_with_labels)])
      batch_indices.extend([next(iter2) for _ in range(self.n_without_labels)])

      random.shuffle(batch_indices)
      yield batch_indices

  def __len__(self) -> int:
    return self.num_batches


class Dataset2D(Dataset):
  def __init__(
    self,
    folds_dataframe: pd.DataFrame,
    dataset_path: Path,
    folds: list[int],
    n_class: int = 3,
  ) -> None:
    super().__init__()

    # schema is (path, fold_id, has_labels)
    self.file_paths = folds_dataframe[
      folds_dataframe["fold_id"].isin(folds)
    ].reset_index()
    self.dataset_path = dataset_path
    self.n_class = n_class

  def __len__(self) -> int:
    return len(self.file_paths)

  def __getitems__(self, indices: list[int]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if len(indices) == 1:  # with sampler
      rows: pd.DataFrame = self.file_paths.iloc[indices[0]]
    else:
      rows: pd.DataFrame = self.file_paths.iloc[indices]
    batch = []
    for row in rows.iterrows():
      scan = np.load(self.dataset_path / "scan" / f"{row[1]['path']}.npy")
      scan = SampleUtils.normalize_ct(ct_data=scan, output_range=(0, 1)).astype(
        np.float32
      )

      # TODO: maybe implement label smoothing ?
      seg = np.load(self.dataset_path / "seg" / f"{row[1]['path']}.npy")
      seg = seg.astype(np.int16)

      batch.append((torch.from_numpy(scan).unsqueeze(0), torch.from_numpy(seg)))

    return batch
