import os
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm

from ocd.dataset.dataset import SampleUtils

if True:
  p = Path("nnUNet_raw_data") / "Dataset001_OvarianCancerDestroyer" / "imagesTr"
  o = Path("imagesss")
  o.mkdir(exist_ok=True)
  for f in tqdm(os.listdir(p)):
    _, d = SampleUtils.load_from_path(p / f)
    plt.imshow(d[:, :, 0])
    plt.savefig(o / (f.split(".")[0] + ".png"))
