# ocd/segmenter/nnunet_predictor.py
import os
import shutil

from pathlib import Path
from typing import Tuple, Optional, Union

try:
    from nnUNet.nnunetv2.inference.predict_from_raw_data import predict_entry_point
except ImportError:
    print("Warning: nnunetv2 library not found. nnUNet prediction will not function.")
    def predict_entry_point(*args, **kwargs):
        print("Error: nnunetv2 not installed.")
        return None

def run_nnunet_prediction(
    input_ct_path: Path,
    input_gt_path: Path,
    output_dir: Path,
    dataset_id_or_name: Union[int, str],
    configuration: str = "2d",
    fold: Optional[int] = None,
    save_probabilities: bool = True,
    overwrite: bool = True,
) -> Optional[Tuple[Path, Path, Path]]:
    """
    Runs nnUNet prediction on a single CT scan using the Python API.
    Saves the prediction and optionally probabilities to the output directory.
    Also copies the ground truth mask to the output directory for convenience.

    Args:
        input_ct_path: Path to the input NIFTI CT scan.
        input_gt_path: Path to the corresponding ground truth NIFTI mask.
        output_dir: Directory to save the prediction, probabilities, and copied GT.
        dataset_id_or_name: nnUNet dataset ID (e.g., 2) or name (e.g., "Dataset002_Heart").
        configuration: Model configuration (e.g., "2d", "3d_fullres").
        fold: Specify fold to use (e.g., 0). If None, nnUNet uses models from all folds.
        model_path: Optional override path to the nnUNet results directory.
        save_probabilities: Whether to save probability maps.
        overwrite: If True, delete output_dir if it exists before running.

    Returns:
        A tuple containing paths to (prediction_file, probability_file, copied_gt_file)
        or None if prediction fails. Probability file path is None if save_probabilities is False.
    """
    print(f"\n--- Running nnUNet Prediction ---")
    print(f"Input CT: {input_ct_path}")
    print(f"Input GT: {input_gt_path}")
    print(f"Output Dir: {output_dir}")
    print(f"Dataset: {dataset_id_or_name}, Config: {configuration}, Fold: {fold}")

    if not input_ct_path.is_file():
        print(f"Error: Input CT file not found: {input_ct_path}")
        return None
    if not input_gt_path.is_file():
        print(f"Error: Input GT file not found: {input_gt_path}")
        return None

    if output_dir.exists() and overwrite:
        print(f"Overwriting existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: fix this part of the code ========================================================
    # nnUNet expects input files in a specific structure (or list of lists)
    # For a single file, provide it as a list containing a list
    list_of_lists_input = [[str(input_ct_path)]]
    output_dir_str = str(output_dir)

    # Determine trainer and plans identifier based on environment or defaults
    # These might need to be set based on trained model if not standard

    # Prepare arguments for predict_entry_point
    #import torch
    #args = {
    #    "list_of_lists_or_source_folder": list_of_lists_input,
    #    "output_folder": output_dir_str,
    #    "model_training_output_dir": str(model_path) if model_path else None,
    #    "use_folds": (fold,) if fold is not None else None,
    #    "tile_step_size": 0.5, # Standard value
    #    "use_gaussian": True, # Standard value for smoother predictions
    #    "use_mirroring": True, # Standard value, increase inference time
    #    "perform_everything_on_device": True, # If GPU allows
    #    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    #    "verbose": False,
    #    "save_probabilities": save_probabilities,
    #    "overwrite": overwrite, # nnUNet's internal overwrite check
    #    "checkpoint_name": 'checkpoint_final.pth', # Common checkpoint name
    #    "dataset_name_or_id": dataset_id_or_name,
    #    "configuration": configuration,
    #    "trainer_name": "", # Often left empty for nnUNet to find
    #    "plans_identifier": "", # Often left empty for nnUNet to find
    #    "folder_with_segs_from_prev_stage": None,
    #    "num_parts": 1,
    #    "part_id": 0
    #}


    try:
        print("Starting nnUNet predict_entry_point...")
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device('cuda', 0),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )
        predictor.initialize_from_trained_model_folder(
            join(nnUNet_results, 'Dataset004_Hippocampus/nnUNetTrainer_5epochs__nnUNetPlans__3d_fullres'),
            use_folds=(0,),
            checkpoint_name='checkpoint_final.pth',
        )

        predictor.predict_from_files(join(nnUNet_raw, 'Dataset003_Liver/imagesTs'),
    #                              join(nnUNet_raw, 'Dataset003_Liver/imagesTs_predlowres'),
    #                              save_probabilities=False, overwrite=False,
    #                              num_processes_preprocessing=2, num_processes_segmentation_export=2,
    #                              folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

        print("nnUNet prediction finished.")
    # ==================================================================================================

        # --- Find prediction and probability files ---
        # nnUNet saves prediction with the original filename stem
        pred_filename = input_ct_path.stem + ".nii.gz"
        pred_file_path = output_dir / pred_filename

        prob_file_path = None
        if save_probabilities:
            # Probabilities are saved as .npz file
            prob_filename = input_ct_path.stem + ".npz"
            prob_file_path = output_dir / prob_filename
            if not prob_file_path.is_file():
                print(f"Warning: Expected probability file not found: {prob_file_path}")
                prob_file_path = None

        if not pred_file_path.is_file():
            print(f"Error: Expected prediction file not found: {pred_file_path}")
            # Check if it was saved with _0000 suffix if input was list
            alt_pred_filename = input_ct_path.stem + "_0000.nii.gz"
            alt_pred_file_path = output_dir / alt_pred_filename
            if alt_pred_file_path.is_file():
                 print(f"Found prediction file with suffix: {alt_pred_file_path}")
                 # Rename it to the expected name
                 alt_pred_file_path.rename(pred_file_path)
            else:
                 print(f"Prediction file still not found.")
                 return None

        # --- Copy Ground Truth Mask ---
        copied_gt_filename = "ground_truth_" + input_gt_path.name
        copied_gt_path = output_dir / copied_gt_filename
        print(f"Copying ground truth to: {copied_gt_path}")
        shutil.copy(input_gt_path, copied_gt_path)

        return pred_file_path, prob_file_path, copied_gt_path # type: ignore

    except Exception as e:
        print(f"Error during nnUNet prediction: {e}")
        import traceback
        traceback.print_exc()
        return None