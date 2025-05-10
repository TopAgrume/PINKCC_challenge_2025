from pathlib import Path

from ocd.dataset.dataset import Dataset

Dataset(base_dir=Path("DatasetChallengeV2")).get_labels_stats(
  overwrite_shitty_labels=True
)
