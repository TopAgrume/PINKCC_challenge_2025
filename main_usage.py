from pathlib import Path
from ocd.dataset.dataset import Dataset
from full_inference import CTSAM3DSegmenter


dataset = Dataset(base_dir=Path("DatasetChallenge"))
train_pairs, val_pairs, test_pairs = dataset.get_dataset_splits()

segmenter = CTSAM3DSegmenter(checkpoint_path="ckpt_1000/params.pth")

sample_pair = test_pairs[0]
ct_image, gt_mask = segmenter.load_sample_from_dataset(sample_pair)

segmenter.visualize_slice(ct_image, gt_mask)

# Crop a patch around a region of interest
center_voxel = [100, 150, 80]  # Example coordinates
image_patch, mask_patch = segmenter.crop_patch(center_voxel)

# Define point prompts
points = [[32, 32, 32]]  # Center of the patch
labels = [1]  # Positive point

# Predict segmentation from points
predicted_mask, dsc = segmenter.predict_from_points(points, labels)
print(f"Dice score: {dsc}")

# Visualize results
segmenter.visualize_comparison(predicted_mask)

# Clear mask for new prediction
segmenter.clear_mask()

# Add more point prompts for refinement
points = [[32, 32, 32], [45, 30, 25]]
labels = [1, 0]  # One positive, one negative
predicted_mask, dsc = segmenter.predict_from_points(points, labels)
print(f"Refined dice score: {dsc}")

# Visualize refined results
segmenter.visualize_comparison(predicted_mask)