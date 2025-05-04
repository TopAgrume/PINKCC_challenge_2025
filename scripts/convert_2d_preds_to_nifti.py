import os
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

SEGMENTATION_PATH = Path("DatasetChallenge") / "Segmentation"


def convert_2d_preds_to_nifti(preds_dir: Path, output_dir: Path = Path("nifti_preds")):
  output_dir.mkdir(exist_ok=True, parents=True)
  preds = os.listdir(preds_dir)
  filename_to_preds = defaultdict(list)
  [filename_to_preds[filename.split("_")[0]].append(filename) for filename in preds]
  for name in filename_to_preds:
    path = (
      SEGMENTATION_PATH
      / ("TCGA" if name.startswith("TCGA") else "MSKCC")
      / f"{name}.nii.gz"
    )
    file = nib.loadsave.load(path)
    data = file.get_fdata()  # pyright: ignore
    sorted_files = sorted(
      filename_to_preds[name], key=lambda x: int(x.split("_")[1].split(".")[0])
    )
    first_slice_index = int(sorted_files[0].split("_")[1].split(".")[0])

    pred_data = torch.cat(
      [
        torch.zeros((512, 512, first_slice_index)),
        *[
          torch.load(preds_dir / slice_name, weights_only=False).unsqueeze(-1)
          for slice_name in sorted_files
        ],
        torch.zeros(512, 512, (data.shape[2] - len(sorted_files) - first_slice_index)),
      ],
      dim=2,
    )

    pred_file = nib.nifti1.Nifti1Image(
      pred_data.numpy().astype(np.float64),
      file.affine,  # pyright: ignore
    )
    nib.loadsave.save(pred_file, output_dir / f"{name}.nii.gz")


if __name__ == "__main__":
  convert_2d_preds_to_nifti(Path("outputs_04-05-25_05h19m32s") / "pred_tensors")
