export nnUNet_raw_data_base="/root/DATA/"
export nnUNet_preprocessed="/root/DATA/nnUNet_preprocessed"
export RESULTS_FOLDER="/root/DATA/RESULTS_FOLDER"


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


# --- Inference using ensemble predictions with a probabilistic union ---
FINAL_CT_TESTS='/root/DATA/FINAL_CT_TESTS'
OUTPUT_FINAL_PREDICTIONS='/root/DATA/OUTPUT_FINAL_PREDICTIONS'

nnUNetv2_predict -d $DATASET_ID -i $FINAL_CT_TESTS -o $OUTPUT_FINAL_PREDICTIONS -f 0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetResEncUNetLPlan --save_probabilities
