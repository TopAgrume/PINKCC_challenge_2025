# main_workflow.py
import argparse
import sys
from pathlib import Path
import os
import SimpleITK as sitk
from typing import Optional

# --- Helper function to manage environment switching (conceptual) ---
def run_in_env(env_name, script_path, args_list):
    """Conceptual function to run a script in a specific conda env."""
    # This is tricky to implement robustly across systems.
    # Usually involves calling conda activate and then python script.
    # For simplicity here, we'll assume manual execution or separate scripts.
    print(f"\n--- ACTION REQUIRED ---")
    print(f"Please activate the '{env_name}' conda environment.")
    print(f"Then run: python {script_path} {' '.join(args_list)}")
    print(f"--- WAITING FOR MANUAL EXECUTION ---")
    # In a real automated setup, you might use subprocess or job scheduling.
    # input("Press Enter after running the command in the correct environment...")


# --- Import project modules ---
# These imports might fail depending on the *current* environment,
# but we need them defined for the script structure.
from ocd.dataset.dataset import Dataset

try:
    from ocd.segmenter.nnunet_predictor import run_nnunet_prediction
except ImportError:
    print("Warning: ocd.segmenter.nnunet_predictor module not found.")
    def run_nnunet_prediction(*args, **kwargs): return None # Dummy

try:
    from ocd.visualization.visualize import visualize_prediction_vs_gt
except ImportError:
    print("Warning: ocd.visualization.visualize module not found.")
    def visualize_prediction_vs_gt(*args, **kwargs): pass # Dummy

try:
    from ocd.sampling.point_sampler import sample_points_from_prediction, sample_points_from_probabilities
except ImportError:
    print("Warning: ocd.sampling.point_sampler module not found.")
    def sample_points_from_prediction(*args, **kwargs): return [] # Dummy
    def sample_points_from_probabilities(*args, **kwargs): return [] # Dummy

try:
    from ocd.segmenter.ct_sam_segmenter import CTSAM3DSegmenter
except ImportError:
    print("Warning: ocd.segmenter.ct_sam_segmenter module not found.")
    class CTSAM3DSegmenter: # Dummy class
        def __init__(self, *args, **kwargs): pass
        def set_full_image(self, *args, **kwargs): return False
        def predict_from_points(self, *args, **kwargs): return None
        def visualize_comparison(self, *args, **kwargs): pass
        def compute_dsc(self, *args, **kwargs): return 0.0


# === Configuration ===
DATASET_BASE_DIR = Path("DatasetChallenge")

# nnUNet lib configuration
NNUNET_DATASET_ID = 1
NNUNET_CONFIGURATION = "2d" # "3d_fullres" "3d_lowres" "3d_cascade_fullres"
NNUNET_FOLD = 0  # 0 1 2 3 4 all
NNUNET_VERBOSE = False
NNUNET_VERBOSE_PREPROCESSING = False

# ctsqm3d lib configuration
CTSAMD3D_CHECKPOINT = Path("ckpt_1000/params.pth")
OUTPUT_BASE_DIR = Path("experiment_output")
SAMPLE_INDEX = 0 # Index of the sample in the test set to process
NUM_SAMPLING_POINTS = 10 # Number of points to sample for CT-SAM3D
USE_PROBABILITY_SAMPLING = False # Set to True to sample based on probabilities
PROBABILITY_THRESHOLD = 0.9 # Used if USE_PROBABILITY_SAMPLING is True

OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)


def main(run_stage: str):
    print(f"Starting workflow stage: {run_stage}")

    # --- Load Dataset ---
    print("\n--- Loading dataset ---")
    dataset = Dataset(base_dir=DATASET_BASE_DIR)
    _, _, test_pairs = dataset.get_dataset_splits()

    if not test_pairs:
        print("Error: No test samples found in the dataset.")
        sys.exit(1)
    if SAMPLE_INDEX >= len(test_pairs):
        print(f"Error: SAMPLE_INDEX {SAMPLE_INDEX} is out of bounds for test set size {len(test_pairs)}.")
        sys.exit(1)

    # select the sample
    ct_path_str, gt_path_str = test_pairs[SAMPLE_INDEX]
    ct_path = Path(ct_path_str)
    gt_path = Path(gt_path_str)
    sample_id = ct_path.stem

    print("\n--- Selecting test sample ---")
    print(f"Selected Sample ID: {sample_id}")
    print(f"CT Path: {ct_path}")
    print(f"GT Path: {gt_path}")

    # define output directories for this sample
    NNUNET_OUTPUT_DIR = OUTPUT_BASE_DIR / sample_id / "nnunet_output"
    CTSAMD3D_OUTPUT_DIR = OUTPUT_BASE_DIR / sample_id / "ctsam3d_output"


    # =============================================
    # == STAGE 1: nnUNet prediction (nnUNet env) ==
    # =============================================
    nnunet_pred_path: Optional[Path] = None
    nnunet_prob_path: Optional[Path] = None
    copied_gt_path: Optional[Path] = None

    if run_stage == "nnunet":
        if "nnunetv2" not in sys.modules:
            print("WARNING: 'nnunetv2' module not loaded. Are you in the correct environment?")
            sys.exit("Exiting: Incorrect environment for nnUNet stage.")

        print("\n" + "="*20 + " STAGE 1: nnUNet Prediction " + "="*20)

        results = run_nnunet_prediction(
            input_ct_path=ct_path,
            input_gt_path=gt_path,
            output_dir=NNUNET_OUTPUT_DIR,
            dataset_id_or_name=NNUNET_DATASET_ID,
            configuration=NNUNET_CONFIGURATION,
            fold=NNUNET_FOLD,
            save_probabilities=True,
            overwrite=True,
            verbose=NNUNET_VERBOSE,
            verbose_preprocessing=NNUNET_VERBOSE_PREPROCESSING
        )

        if results:
            nnunet_pred_path, nnunet_prob_path, copied_gt_path = results
            print(f"nnUNet prediction saved to: {nnunet_pred_path}")
            if nnunet_prob_path:
                print(f"nnUNet probabilities saved to: {nnunet_prob_path}")
            print(f"Ground truth copied to: {copied_gt_path}")

            # Visualize the nnUNet prediction vs GT
            visualize_prediction_vs_gt(
                ct_path=ct_path,
                gt_mask_path=copied_gt_path,
                pred_mask_path=nnunet_pred_path,
                title=f"nnUNet Output vs GT ({sample_id})"
            )
        else:
            print("Error: nnUNet prediction failed.")
            sys.exit(1) # Exit if nnUNet fails, as CT-SAM3D depends on it

        if run_stage == "nnunet":
             print("\nFinished nnUNet stage.")
             sys.exit(0) # Stop here if only running nnunet

    # ===============================================================
    # == STAGE 2: Point Sampling & CT-SAM3D Refinement (ct_sam3d Env) ==
    # ===============================================================

    if run_stage == "ctsam":
        print("\n" + "="*20 + " STAGE 2: CT-SAM3D Refinement " + "="*20)
        print("--> This stage requires the 'ct_sam3d' environment.")

         # Check if environment seems correct (basic check)
        if "ct_sam3d" not in sys.modules and "CTSAM3DSegmenter" not in globals(): # Check module or dummy class
             print("WARNING: 'ct_sam3d' module / CTSAM3DSegmenter not loaded. Are you in the correct environment?")
             # Decide whether to exit or continue
             # sys.exit("Exiting: Incorrect environment for CT-SAM3D stage.")

        # --- Get paths from Stage 1 ---
        # These must exist if running 'all' or 'ctsam' after 'nnunet'
        nnunet_pred_path = NNUNET_OUTPUT_DIR / (ct_path.stem + ".nii.gz")
        nnunet_prob_path = NNUNET_OUTPUT_DIR / (ct_path.stem + ".npz")
        copied_gt_path = NNUNET_OUTPUT_DIR / ("ground_truth_" + gt_path.name)

        if not nnunet_pred_path.is_file():
             print(f"Error: nnUNet prediction file not found for Stage 2: {nnunet_pred_path}")
             print("Please run the 'nnunet' stage first.")
             sys.exit(1)
        if not copied_gt_path.is_file():
             print(f"Error: Copied ground truth file not found: {copied_gt_path}")
             sys.exit(1)
        if USE_PROBABILITY_SAMPLING and (nnunet_prob_path is None or not nnunet_prob_path.is_file()):
             print(f"Error: Probability file needed but not found: {nnunet_prob_path}")
             sys.exit(1)

        # --- Sample Points ---
        if USE_PROBABILITY_SAMPLING:
            sampled_points = sample_points_from_probabilities(
                probability_map_path=nnunet_prob_path, # type: ignore
                num_points=NUM_SAMPLING_POINTS,
                probability_threshold=PROBABILITY_THRESHOLD,
                foreground_class_indices=[1, 2] # Assuming labels 1 and 2 are foreground
            )
        else:
            sampled_points = sample_points_from_prediction(
                prediction_mask_path=nnunet_pred_path,
                num_points=NUM_SAMPLING_POINTS,
                specific_label=None # Sample from any foreground label
            )

        if not sampled_points:
            print("Error: Failed to sample points from nnUNet prediction. Cannot proceed with CT-SAM3D.")
            sys.exit(1)

        # Assign positive labels to all sampled points for prompting
        point_labels = [1] * len(sampled_points)

        # --- Initialize CT-SAM3D Segmenter ---
        print("\n--- Initializing CT-SAM3D ---")
        segmenter = CTSAM3DSegmenter(checkpoint_path=CTSAMD3D_CHECKPOINT)

        # --- Load Image into CT-SAM3D ---
        # Use original CT path and the copied GT path from nnUNet output folder
        success = segmenter.set_full_image(
            ct_path=ct_path,
            gt_mask_path=copied_gt_path,
            resample_spacing=[1.5, 1.5, 1.5] # Use desired spacing for CT-SAM3D
        )

        if not success or segmenter.current_image_sitk is None:
            print("Error: Failed to load image into CT-SAM3D segmenter.")
            sys.exit(1)

        # Visualize the loaded image and sampled points
        print("\n--- Visualizing Loaded Image and Sampled Points ---")
        segmenter.visualize_slice(
             image=segmenter.current_image_sitk,
             mask=segmenter.current_mask_gt_sitk, # Show GT if loaded
             points=sampled_points,
             labels=point_labels,
             axis=2, # Show Axial view
             title_suffix=f" - Sampled Points ({len(sampled_points)})"
        )


        # --- Run CT-SAM3D Prediction ---
        print("\n--- Running CT-SAM3D Prediction ---")
        # The sampled_points are in the original image voxel space.
        # CT-SAM3D's set_full_image resamples. We need to map points
        # OR trust that predict handles this mapping internally if points are physical.
        # Let's ASSUME SamPredictor.predict expects points in the resampled image's voxel space.
        # We need to transform the points.

        # Transform sampled points (original voxel indices) to physical space
        original_ct_for_info, _ = SampleUtils.load_from_path_sitk(ct_path)
        physical_points = [original_ct_for_info.TransformIndexToPhysicalPoint(p[::-1]) for p in sampled_points]

        # Transform physical points to the resampled image's voxel space
        resampled_points_indices = [segmenter.current_image_sitk.TransformPhysicalPointToIndex(pp)[::-1] for pp in physical_points] # Get XYZ indices

        print(f"Transformed points for CT-SAM3D (resampled space):\n{resampled_points_indices}")


        prediction_result = segmenter.predict_from_points(
            points=resampled_points_indices, # Use transformed points
            labels=point_labels,
            mode="full_image" # Predict on the whole loaded image
        )

        if prediction_result:
            ctsam_pred_mask, ctsam_score = prediction_result
            print(f"CT-SAM3D prediction finished. Score (IoU/Confidence): {ctsam_score:.4f}")

            # --- Save the CT-SAM3D prediction ---
            CTSAMD3D_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            ctsam_output_filename = CTSAMD3D_OUTPUT_DIR / f"{sample_id}_ctsam3d_pred.nii.gz"
            print(f"Saving CT-SAM3D prediction to: {ctsam_output_filename}")
            sitk.WriteImage(ctsam_pred_mask, str(ctsam_output_filename))

            # --- Visualize CT-SAM3D Result ---
            print("\n--- Visualizing CT-SAM3D Result ---")
            segmenter.visualize_comparison(
                predicted_mask=ctsam_pred_mask,
                title=f"CT-SAM3D Output vs GT ({sample_id})",
                axis=2 # Axial
            )

            # Optional: Calculate final DSC if GT is available
            if segmenter.current_mask_gt_sitk:
                 final_dsc = segmenter.compute_dsc(ctsam_pred_mask, segmenter.current_mask_gt_sitk)
                 print(f"\nFinal DSC (CT-SAM3D vs GT): {final_dsc:.4f}")

        else:
            print("Error: CT-SAM3D prediction failed.")
            # No sys.exit here, maybe just log the failure


        print("\nFinished CT-SAM3D stage.")


if __name__ == "__main__":
    required_env_vars = [
        "nnUNet_raw",
        "nnUNet_preprocessed",
        "nnUNet_results"
    ]
    missing = [var for var in required_env_vars if os.environ.get(var) is None]
    if missing:
        print(f"ERROR: The following nnUNet environment variables are not set:\n{missing}")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Run nnUNet and/or CT-SAM3D workflow stages.")
    parser.add_argument(
        "stage",
        choices=["nnunet", "ctsam"],
        help="Which stage(s) to run: 'nnunet' (requires nnUNet env), 'ctsam' (requires ct_sam3d env)"
    )
    args = parser.parse_args()
    main(run_stage=args.stage)
    print("\nWorkflow finished.")