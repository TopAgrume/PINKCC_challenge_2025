from pathlib import Path

import cv2
import nibabel as nib
import numpy as np

from ocd.dataset.dataset import SampleUtils

PATH = Path("..") / "DatasetChallenge" / "CT" / "MSKCC" / "330680.nii.gz"

file = nib.loadsave.load(PATH)
data = file.get_fdata()

data = SampleUtils.normalize_ct(data, output_range=(0, 255))
output_dir = Path("images")
output_dir.mkdir(parents=True, exist_ok=True)

for i in range(data.shape[2]):
  gray_image = data[:, :, i].astype(np.float32)
  rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
  resized_rgb_image = cv2.resize(
    rgb_image, (1024, 1024), interpolation=cv2.INTER_LINEAR
  )
  cv2.imwrite(str(output_dir / f"{i}.png"), resized_rgb_image)
