# nnUNet Ovarian Cancer Segmentation

This repository contains the complete pipeline for training and running inference with nnUNetv2 tailored to ovarian cancer CT scan data using the ResEncUNet architecture.

## Prerequisites

- Dataset prepared in nnUNet raw format as `Dataset001_OvarianCancer`
- **VRAM**: Minimum ~22943 MiB (approximately 22.4 GB **with batch_size = 2**)
- **Recommended GPU**: A100, L40S-48G (trained on this one), ...

## Setup

The training script sets up the following environment variables:

```sh
export nnUNet_raw_data_base="/root/DATA/nnUNet_raw_data_base"
export nnUNet_raw="${nnUNet_raw_data_base}/nnUNet_raw_data"
export nnUNet_preprocessed="${nnUNet_raw_data_base}/nnUNet_preprocessed"
export nnUNet_results="${nnUNet_raw_data_base}/nnUNet_results"
```

## Directory structure

```sh
/root/DATA/
├── nnUNet_raw_data_base/
│   ├── nnUNet_raw_data/
│   ├── nnUNet_preprocessed/
│   └── nnUNet_results/
├── FINAL_CT_TESTS/           # Input test CT scans
└── OUTPUT_FINAL_PREDICTIONS/ # Output segmentation predictions
```

## Pipeline architecture details

- **Model**: ResEncUNetL (Large Residual Encoder U-Net)
- **Configuration**: 3D full resolution
- **Training strategy**: 5-fold cross-validation
- **Ensemble method**: Probabilistic union of all 5 folds
- **Epochs**: 450 (sufficient for convergence)

## Usage

### Running the complete pipeline

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

### (OR) Manual Step-by-Step Execution

#### Preprocessing
```sh
nnUNetv2_plan_and_preprocess -d Dataset001_OvarianCancer -pl nnUNetPlannerResEncL --verify_dataset_integrity
```

#### Training (5-fold cross-validation)
```sh
# Train each fold separately (generated .npz files are required to find best configuration)
nnUNetv2_train Dataset001_OvarianCancer 3d_fullres 0 -p nnUNetResEncUNetLPlans --npz
nnUNetv2_train Dataset001_OvarianCancer 3d_fullres 1 -p nnUNetResEncUNetLPlans --npz
nnUNetv2_train Dataset001_OvarianCancer 3d_fullres 2 -p nnUNetResEncUNetLPlans --npz
nnUNetv2_train Dataset001_OvarianCancer 3d_fullres 3 -p nnUNetResEncUNetLPlans --npz
nnUNetv2_train Dataset001_OvarianCancer 3d_fullres 4 -p nnUNetResEncUNetLPlans --npz
```

#### Find Best Configuration
```sh
nnUNetv2_find_best_configuration 1 -c 3d_fullres -p nnUNetResEncUNetLPlans
```

#### Inference
```sh
nnUNetv2_predict -d Dataset001_OvarianCancer -i /root/DATA/FINAL_CT_TESTS -o /root/DATA/OUTPUT_FINAL_PREDICTIONS -f 0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetResEncUNetLPlan --save_probabilities
```


## To go further - ResEncUNetXL model

For potentially better results, the script includes an untested ResEncUNetXL variant. You just have to replace nnUNetResEncUNetLPlans with nnUNetResEncUNetXLPlans in training commands

**Note**: This *XL* variant requires more computational resources and training time.

## Output

The final predictions will be saved in `/root/DATA/OUTPUT_FINAL_PREDICTIONS/` (.nii.gz format).