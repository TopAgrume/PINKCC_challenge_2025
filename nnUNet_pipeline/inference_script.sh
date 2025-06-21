export nnUNet_raw_data_base="/root/DATA/nnUNet_raw_data_base"
export nnUNet_raw="${nnUNet_raw_data_base}/nnUNet_raw_data"
export nnUNet_preprocessed="${nnUNet_raw_data_base}/nnUNet_preprocessed"
export nnUNet_results="${nnUNet_raw_data_base}/nnUNet_results"


# --- Inference using ensemble predictions with a probabilistic union ---
DATASET_ID="Dataset001_OvarianCancer"
FINAL_CT_TESTS="${nnUNet_raw_data_base}/FINAL_CT_TESTS"
OUTPUT_FINAL_PREDICTIONS="${nnUNet_raw_data_base}/OUTPUT_FINAL_PREDICTIONS"

nnUNetv2_predict -d $DATASET_ID -i $FINAL_CT_TESTS -o $OUTPUT_FINAL_PREDICTIONS -f 0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetResEncUNetLPlan --save_probabilities
