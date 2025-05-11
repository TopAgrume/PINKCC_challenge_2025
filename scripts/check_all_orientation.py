import os
from pathlib import Path

import matplotlib.pyplot as plt

from ocd.dataset.dataset import SampleUtils

p = Path(
  "\\wsl.localhost\\Ubuntu\\home\\maelr\\PINKCC_challenge_2025\\nnFormer_raw_data\\Task001_MSKCC\\imagesTr"
)
o = Path("imagesss")
o.mkdir(exist_ok=True)
for f in os.listdir(p):
  _, d = SampleUtils.load_from_path(p / f)
  plt.imshow(d[:, :, 0])
  plt.savefig(o / f)
