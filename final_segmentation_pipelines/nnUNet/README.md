# nnUNet Ovarian Cancer Segmentation

This repository contains the complete pipeline for training and running inference with nnUNetv2 tailored to ovarian cancer CT scan data using the ResEncUNet architecture.

## Prerequisites

- CUDA-compatible GPU with at least 22943 MiB VRAM
- Dataset prepared in nnUNet raw format as `Dataset001_OvarianCancer`

## Setup

The training script sets up the following environment variables:

```sh
export nnUNet_raw_data_base="/root/DATA/nnUNet_raw_data_base"
export nnUNet_raw="${nnUNet_raw_data_base}/nnUNet_raw_data"
export nnUNet_preprocessed="${nnUNet_raw_data_base}/nnUNet_preprocessed"
export nnUNet_results="${nnUNet_raw_data_base}/nnUNet_results"
```

## Usage

### Running the Complete Pipeline

Simply execute the provided scripts:

```sh
bash training_script.sh
bash inference_script.sh
```

This will automatically:
- Preprocess the dataset with custom ResEncL planner
- Train 5-fold cross-validation models (450 epochs each)
- Find the best configuration for ensembling
- Run inference on test data

Do not use postprocessing and keep raw model predictions for maximum sensitivity (and better dice score on predictions).

## Directory Structure

```sh
/root/DATA/
├── nnUNet_raw_data_base/
│   ├── nnUNet_raw_data/
│   ├── nnUNet_preprocessed/
│   └── nnUNet_results/
├── FINAL_CT_TESTS/           # Input test CT scans
└── OUTPUT_FINAL_PREDICTIONS/ # Output segmentation predictions
```

## Architecture Details

- **Model**: ResEncUNetL (Large Residual Encoder U-Net)
- **Configuration**: 3D full resolution
- **Training strategy**: 5-fold cross-validation
- **Ensemble method**: Probabilistic union of all 5 folds
- **Epochs**: 450 (sufficient for convergence)

## Hardware Requirements

- **VRAM**: Minimum ~22943 MiB (approximately 22.4 GB **with batch_size = 2**)
- **Recommended GPU**: A100, L40S-48G (trained on this one), ...

## Advanced Usage - ResEncUNetXL model

For potentially better results, the script includes an untested ResEncUNetXL variant. You just have to replace nnUNetResEncUNetLPlans with nnUNetResEncUNetXLPlans in training commands

**Note**: This XL variant requires more computational resources and training time.

## Output

The final predictions will be saved in `/root/DATA/OUTPUT_FINAL_PREDICTIONS/` with compressed output format for efficient storage.