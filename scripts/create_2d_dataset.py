from pathlib import Path

from ocd.dataset.dataset import Dataset

print(Path("."))

Dataset(base_dir=Path("DatasetChallenge")).convert_CT_scans_to_images(
  output_dir=Path("2D_CT_SCANS")
)
