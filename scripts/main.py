from ocd.dataset.dataset import Dataset
from ocd.dataset.data_augmentation import DataAugmentation
from ocd.dataset.segmentation_dataset_2D import CancerSegmentationDataset


datasets = Dataset('DatasetChallenge')
train_pairs, val_pairs, test_pairs = datasets.get_dataset_splits()
print("Lengths:", len(train_pairs), len(val_pairs), len(test_pairs))

transforms = DataAugmentation.get_augmentation_pipeline(
    p=0.5,
    random_crop_scale=(0.85, 1.0),
    h_flip_prob=0.5,
    v_flip_prob=0.5,
    rotate_prob=0.5,
    mult_noise_range=(0.9, 1.1),
    blur_range=(0.5, 1.5),
    elastic_alpha_range=(1, 2),
    brightness_range=(-0.1, 0.1),
    contrast_range=(0.9, 1.1),
    gamma_range=(0.9, 1.1)
)

train_dataset = CancerSegmentationDataset(train_pairs)#, transforms=transforms)
val_dataset = CancerSegmentationDataset(val_pairs)

train_loader = CancerSegmentationDataset.get_dataloader(train_dataset, batch_size=16)
val_loader = CancerSegmentationDataset.get_dataloader(val_dataset, batch_size=16, shuffle=False)


print("Nb train slices:", len(train_dataset))
print("Nb val slices:", len(val_dataset))

data_iter = iter(train_loader)
batch = next(data_iter)
print("Batch shapes:", batch[0].shape, batch[1].shape)