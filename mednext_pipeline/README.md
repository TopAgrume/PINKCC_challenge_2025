# MedNeXt Ovarian Cancer Segmentation

This repository contains the complete pipeline for training and running inference with MedNeXt tailored to ovarian cancer CT scan data using 3x3x3 (and 5x5x5 kernel configurations).

## Prerequisites

- Dataset prepared in nnUNet format as `Task001_OvarianCancer`
- **VRAM**: Minimum ~16203 MiB (approximately 15.8 GB **with batch_size = 2**)
- **Disk space**: Large amount required for preprocessing (significantly more than standard nnUNet)
- **Recommended GPU**:  A100, L40S-48G (trained on this one), ...

## Setup

The training script sets up the following environment variables:

```sh
export nnUNet_raw_data_base="/root/DATA/"
export nnUNet_preprocessed="${nnUNet_raw_data_base}/nnUNet_preprocessed"
export RESULTS_FOLDER="${nnUNet_raw_data_base}/RESULTS_FOLDER"
```

## Directory structure

```sh
/root/DATA/
├── nnUNet_preprocessed/      # Preprocessed data (requires large disk space)
├── RESULTS_FOLDER/           # Training results and model checkpoints
├── FINAL_CT_TESTS/           # Input test CT scans
└── OUTPUT_FINAL_PREDICTIONS/ # Output segmentation predictions
```

## Pipeline architecture details

- **Model**: MedNeXt Large (L)
- **Base kernel size**: 3x3x3
- **Target spacing**: 1x1x1mm (high resolution)
- **Configuration**: 3D full resolution
- **Training strategy**: 5-fold cross-validation
- **Ensemble method**: Probabilistic union of all 5 folds
- **Epochs**: 450 (sufficient for convergence)

## Usage

### Running the complete pipeline

Simply execute the provided scripts:

```bash
bash training_script.sh
bash inference_script.sh
```

This will automatically:
- Preprocess the dataset with custom 1x1x1 target spacing
- Train 5-fold cross-validation models with 3x3x3 kernels (450 epochs each)
- Find the best configuration for ensembling
- Run inference on test data

Do not use postprocessing and keep raw model predictions for maximum sensitivity (and better dice score on predictions).

### (OR) Manual Step-by-Step Execution

#### Preprocessing
```sh
mednextv1_plan_and_preprocess -t Task001_OvarianCancer -pl3d ExperimentPlanner3D_v21_customTargetSpacing_1x1x1
```

#### Training (5-fold cross-validation with 3x3x3 kernels)
```bash
# Train each fold separately (generated .npz files are required to find best configuration)
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_L_kernel3 Task001_OvarianCancer 0 -p nnUNetPlansv2.1_trgSp_1x1x1 --npz
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_L_kernel3 Task001_OvarianCancer 1 -p nnUNetPlansv2.1_trgSp_1x1x1 --npz
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_L_kernel3 Task001_OvarianCancer 2 -p nnUNetPlansv2.1_trgSp_1x1x1 --npz
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_L_kernel3 Task001_OvarianCancer 3 -p nnUNetPlansv2.1_trgSp_1x1x1 --npz
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_L_kernel3 Task001_OvarianCancer 4 -p nnUNetPlansv2.1_trgSp_1x1x1 --npz
```

#### Find Best Configuration
```sh
mednextv1_find_best_configuration -m 3d_fullres -t Task001_OvarianCancer -f 0 1 2 3 4 -pl MedNeXt_L_kernel3__nnUNetPlansv2.1_trgSp_1x1x1 --disable_postprocessing
```

#### Inference
```sh
mednextv1_predict -i /root/DATA/FINAL_CT_TESTS -o /root/DATA/OUTPUT_FINAL_PREDICTIONS -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p MedNeXt_L_kernel3__nnUNetPlansv2.1_trgSp_1x1x1 -t Task001_OvarianCancer -z
```

## To go further - UpKern training (5x5x5 kernels)

For potentially better results using, the script includes an untested training with larger kernels with transfer learning from the 3x3x3 models:

**Note**: This *UpKern* variant might require more computational resources and training time.

## Output

The final predictions will be saved in `/root/DATA/OUTPUT_FINAL_PREDICTIONS/` (.nii.gz format).