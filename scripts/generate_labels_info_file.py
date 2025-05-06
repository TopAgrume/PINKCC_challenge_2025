from pathlib import Path

from ocd.dataset.dataset import Dataset

Dataset(base_dir=Path("DatasetChallenge")).get_labels_stats()
