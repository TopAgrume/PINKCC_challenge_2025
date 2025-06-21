export nnUNet_raw_data_base="/root/DATA/nnUNet_raw_data_base"
export nnUNet_raw="${nnUNet_raw_data_base}/nnUNet_raw_data"
export nnUNet_preprocessed="${nnUNet_raw_data_base}/nnUNet_preprocessed"
export nnUNet_results="${nnUNet_raw_data_base}/nnUNet_results"


# --- Custom nnUNet Preprocessing ---
DATASET_ID="Dataset001_OvarianCancer"
nnUNetv2_plan_and_preprocess -d $DATASET_ID -pl nnUNetPlannerResEncL --verify_dataset_integrity


# --- Training on all folds (22943MiB vRAM required) [450 epochs is enough] ---
nnUNetv2_train $DATASET_ID 3d_fullres 0 -p nnUNetResEncUNetLPlans --npz
nnUNetv2_train $DATASET_ID 3d_fullres 1 -p nnUNetResEncUNetLPlans --npz
nnUNetv2_train $DATASET_ID 3d_fullres 2 -p nnUNetResEncUNetLPlans --npz
nnUNetv2_train $DATASET_ID 3d_fullres 3 -p nnUNetResEncUNetLPlans --npz
nnUNetv2_train $DATASET_ID 3d_fullres 4 -p nnUNetResEncUNetLPlans --npz


# --- To go further: use ResEncNetXL instead of ResEncNetL (we were unable to test due to lack of time and ressources) ---
nnUNetv2_train $DATASET_ID 3d_fullres 0 -p nnUNetResEncUNetXLPlans --npz
nnUNetv2_train $DATASET_ID 3d_fullres 1 -p nnUNetResEncUNetXLPlans --npz
nnUNetv2_train $DATASET_ID 3d_fullres 2 -p nnUNetResEncUNetXLPlans --npz
nnUNetv2_train $DATASET_ID 3d_fullres 3 -p nnUNetResEncUNetXLPlans --npz
nnUNetv2_train $DATASET_ID 3d_fullres 4 -p nnUNetResEncUNetXLPlans --npz


# --- Final ensembling using the 5-fold cross-validation strategy ---
nnUNetv2_find_best_configuration 1 -c 3d_fullres -p nnUNetResEncUNetLPlans
