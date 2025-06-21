# MedNeXt Ovarian Cancer Segmentation

This repository contains the complete pipeline for training and running inference with MedNeXt tailored to ovarian cancer CT scan data using 3x3x3 (and 5x5x5 kernel configurations).

## Prerequisites

- CUDA-compatible GPU with at least 16203 MiB VRAM
- Dataset prepared in nnUNet format as `Task001_OvarianCancer`
- **Important**: Significant free disk space required for preprocessing!

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

## Architecture details

- **Model**: MedNeXt Large (L)
- **Base kernel size**: 3x3x3
- **Target spacing**: 1x1x1mm (high resolution)
- **Configuration**: 3D full resolution
- **Training strategy**: 5-fold cross-validation
- **Ensemble method**: Probabilistic union of all 5 folds
- **Epochs**: 450 (sufficient for convergence)

## Hardware requirements

- **VRAM**: Minimum 16203 MiB (approximately 15.8 GB **with batch_size = 2**)
- **Disk space**: Large amount required for preprocessing (significantly more than standard nnUNet)
- **Recommended GPU**:  A100, L40S-48G (trained on this one), ...

## To go further - UpKern Training (5x5x5 kernels)

For potentially better results using, the script includes an untested training with larger kernels with transfer learning from the 3x3x3 models:

**Note**: This *UpKern* variant might require more computational resources and training time.

## Output

The final predictions will be saved in `/root/DATA/OUTPUT_FINAL_PREDICTIONS/` with compressed output format for efficient storage.