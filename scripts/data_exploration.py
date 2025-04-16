# %%
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.orientations import (
  aff2axcodes,
  apply_orientation,
  axcodes2ornt,
  inv_ornt_aff,
  ornt_transform,
)

from ocd.dataset.dataset import Dataset

PATH = Path("..") / "DatasetChallenge" / "CT" / "MSKCC" / "330680.nii.gz"

# %%
start = time()
file = nib.loadsave.load(PATH)
print(f"Loaded file in {time() - start}")
# %%
start = time()
data = file.get_fdata()
print(f"Loaded data in {time() - start}")

# %%
print(file.header)

# %%
print(file.affine)

# %%
print(file.header.get_zooms())

# %%
print(np.linalg.norm(file.affine[:3, 2]))

# %%
print(data.min(), data.max())

# %%
# L : Right to Left, R : inverse (x axis)
# P : Anterior to Posterior, A : inverse (y axis)
# I : Superior to Inferior, S : inverse
orientation = aff2axcodes(file.affine)
print(orientation)

# %%
img = data[:, :, 60]
plt.imshow(img)
plt.plot()
# %%
data.shape

# %%
img.max()

# %%
PATH = Path("..") / "DatasetChallenge" / "Segmentation" / "MSKCC" / "330680.nii.gz"

file = nib.loadsave.load(PATH)
data = file.get_fdata()

# %%
np.unique(data)
# %%

# %%
data_dict = Dataset(base_dir=Path("..") / "DatasetChallenge").data

# %%
data_dict["CT_MSKCC"][0].split("\\")[-1]
# %%
PATH.name

# %%
PATH = Path("..") / "DatasetChallenge" / "CT" / "TCGA" / "TCGA-13-0762.nii.gz"
file = nib.loadsave.load(PATH)
orientation = aff2axcodes(file.affine)
print(orientation)

# %%
PATH = Path("..") / "DatasetChallenge" / "Segmentation" / "TCGA" / "TCGA-13-0762.nii.gz"
file = nib.loadsave.load(PATH)
orientation = aff2axcodes(file.affine)
print(orientation)

# %%
for path in data_dict["CT_MSKCC"]:
  file = nib.loadsave.load(path)
  orientation = aff2axcodes(file.affine)
  if orientation != ("L", "P", "S"):
    print(path, orientation)
for path in data_dict["Segmentation_MSKCC"]:
  file = nib.loadsave.load(path)
  orientation = aff2axcodes(file.affine)
  if orientation != ("L", "P", "S"):
    print(path, orientation)
# %%
for path in data_dict["CT_TCGA"]:
  file = nib.loadsave.load(path)
  orientation = aff2axcodes(file.affine)
  if orientation != ("L", "P", "S"):
    print(path, orientation)

for path in data_dict["Segmentation_TCGA"]:
  file = nib.loadsave.load(path)
  orientation = aff2axcodes(file.affine)
  if orientation != ("L", "P", "S"):
    print(path, orientation)
# %%
#
# HERE WE CHANGE THE ORIENTATION OF THE TCGA-13-0762.nii.gz CT scan
# from ('L', 'P', 'I') to ('R', 'A', 'S')

PATH = Path("..") / "DatasetChallenge" / "Segmentation" / "TCGA" / "TCGA-13-0762.nii.gz"
img = nib.loadsave.load(PATH)
data = img.get_fdata()
affine = img.affine

axcodes = aff2axcodes(affine)

current_ornt = axcodes2ornt(axcodes)
target_ornt = axcodes2ornt(("R", "A", "S"))

transform = ornt_transform(current_ornt, target_ornt)

reoriented_data = apply_orientation(data, transform)

new_affine = affine @ inv_ornt_aff(transform, data.shape)

new_img = nib.Nifti1Image(reoriented_data, new_affine)
nib.save(new_img, "reoriented_scan.nii.gz")

# %%
img = nib.loadsave.load("reoriented_scan.nii.gz")
data = img.get_fdata()
affine = img.affine

print(aff2axcodes(affine))

# %%
PATH = Path("..") / "DatasetChallenge" / "CT" / "TCGA" / "TCGA-09-1674.nii.gz"
img = nib.loadsave.load(PATH)
data = img.get_fdata()
affine = img.affine

axcodes = aff2axcodes(affine)

current_ornt = axcodes2ornt(axcodes)
target_ornt = axcodes2ornt(("L", "P", "S"))

transform = ornt_transform(current_ornt, target_ornt)

reoriented_data = apply_orientation(data, transform)

new_affine = affine @ inv_ornt_aff(transform, data.shape)

new_img = nib.Nifti1Image(reoriented_data, new_affine)
nib.save(new_img, "reoriented_scan.nii.gz")

# %%
PATH = Path("..") / "DatasetChallenge" / "Segmentation" / "TCGA" / "TCGA-09-1674.nii.gz"
img = nib.loadsave.load(PATH)
data = img.get_fdata()
affine = img.affine

axcodes = aff2axcodes(affine)

current_ornt = axcodes2ornt(axcodes)
target_ornt = axcodes2ornt(("L", "P", "S"))

transform = ornt_transform(current_ornt, target_ornt)

reoriented_data = apply_orientation(data, transform)

new_affine = affine @ inv_ornt_aff(transform, data.shape)

new_img = nib.Nifti1Image(reoriented_data, new_affine)
nib.save(new_img, "reoriented_seg.nii.gz")
