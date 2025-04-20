import pandas as pd
from torch.utils.data import DataLoader

from ocd.dataset.dataset import Dataset
from ocd.dataset.dataset3d import CancerVolumeDataset


def create_dataloaders(
    base_dir,
    patch_size=(96, 96, 96),
    batch_size=2,
    num_workers=4,
    cache_rate=1.0,
):
    """
    Create dataloaders for training, validation and testing

    Args:
        base_dir: Base directory containing the dataset
        patch_size: Size of patches to extract
        batch_size: Batch size
        num_workers: Number of workers for data loading
        cache_rate: Percentage of data to be cached

    Returns:
        Dictionary of dataloaders
    """
    ocds = Dataset(base_dir=base_dir)
    train_pairs, val_pairs, test_pairs = ocds.get_dataset_splits()

    # ===== datasets =====
    train_ds = CancerVolumeDataset(
        data_pairs=train_pairs,
        patch_size=patch_size,
        num_samples=4,
        cache_rate=cache_rate,
        phase="train",
    )

    val_ds = CancerVolumeDataset(
        data_pairs=val_pairs,
        patch_size=patch_size,
        cache_rate=cache_rate,
        phase="val",
    )

    test_ds = CancerVolumeDataset(
        data_pairs=test_pairs,
        patch_size=patch_size,
        cache_rate=cache_rate,
        phase="test",
    )

    # ===== dataloaders =====
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }