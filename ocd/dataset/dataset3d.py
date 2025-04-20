from pathlib import Path
from typing import List, Tuple, Optional, Callable

from torch.utils.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ToTensord,
)


class CancerVolumeDataset(Dataset):
    """
    Dataset for 3D volumetric CT scans with segmentation masks
    """

    def __init__(
        self,
        data_pairs: List[Tuple[Path, Path]],
        transforms: Optional[Callable] = None,
        patch_size: Tuple[int, int, int] = (96, 96, 96),
        num_samples: int = 4,
        window_center: int = 40,
        window_width: int = 400,
        cache_rate: float = 1.0,
        phase: str = "train",
    ):
        """
        Initialize the dataset

        Args:
            data_pairs: List of tuples (ct_path, seg_path)
            transforms: Additional transforms to apply
            patch_size: Size of patches to extract (for training)
            num_samples: Number of patches to extract per volume
            window_center: Center value for CT windowing
            window_width: Width value for CT windowing
            cache_rate: Percentage of data to be cached
            phase: 'train', 'val' or 'test'
        """
        self.data_pairs = data_pairs
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.cache_rate = cache_rate
        self.phase = phase

        # normalizing
        self.window_min = window_center - window_width // 2
        self.window_max = window_center + window_width // 2

        if transforms: # TODO: in case custom transforms needs
            self.transforms = transforms
        else:
            if phase == "train":
                self.transforms = self._get_train_transforms()
            else:
                self.transforms = self._get_val_transforms()

        # complinat with monai dataset format
        self.data = []
        for ct_path, seg_path in data_pairs:
            self.data.append({
                "image": str(ct_path),
                "label": str(seg_path),
            })

    def _get_train_transforms(self):
        """
        Get transforms for training
        """
        return Compose([
            # https://docs.monai.io/en/stable/transforms.html#loadimaged
            LoadImaged(keys=["image", "label"]),

            # https://docs.monai.io/en/stable/transforms.html#ensurechannelfirstd
            EnsureChannelFirstd(keys=["image", "label"]),

            # https://docs.monai.io/en/stable/transforms.html#monai.transforms.Orientationd
            Orientationd(keys=["image", "label"], axcodes="LPS"),

            # https://docs.monai.io/en/stable/transforms.html#monai.transforms.Spacingd
            #Spacingd(
            #    keys=["image", "label"],
            #    pixdim=(1.5, 1.5, 2.0), # TODO: reduce dimensionality, need to check papers?
            #    mode=("bilinear", "nearest"),
            #,

            # https://docs.monai.io/en/stable/transforms.html#monai.transforms.ScaleIntensityRanged
            ScaleIntensityRanged(
                keys=["image"],
                a_min=self.window_min,
                a_max=self.window_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),

            # https://docs.monai.io/en/stable/transforms.html#monai.transforms.CropForegroundd
            #CropForegroundd(keys=["image", "label"], source_key="image"),

            # https://docs.monai.io/en/stable/transforms.html#monai.transforms.RandCropByPosNegLabeld
            #RandCropByPosNegLabeld(
            #    keys=["image", "label"],
            #    label_key="label",
            #    spatial_size=self.patch_size,
            #    pos=1,
            #    neg=1,
            #    num_samples=self.num_samples,
            #), # TODO: dict as output

            # https://docs.monai.io/en/stable/transforms.html#monai.transforms.RandFlipd
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),

            # https://docs.monai.io/en/stable/transforms.html#monai.transforms.RandRotate90d
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),

            # https://docs.monai.io/en/stable/transforms.html#monai.transforms.RandScaleIntensityd
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),

            # https://docs.monai.io/en/stable/transforms.html#monai.transforms.RandShiftIntensityd
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),

            # https://docs.monai.io/en/stable/transforms.html#monai.transforms.ToTensord
            ToTensord(keys=["image", "label"]),
        ])

    def _get_val_transforms(self):
        """
        Get transforms for validation/testing
        """
        return Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="LPS"),
            #Spacingd(
            #    keys=["image", "label"],
            #    pixdim=(1.5, 1.5, 2.0),
            #    mode=("bilinear", "nearest"),
            #),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=self.window_min,
                a_max=self.window_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            ToTensord(keys=["image", "label"]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_item = self.data[index]
        return self.transforms(data_item)