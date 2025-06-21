export nnUNet_raw_data_base="/root/DATA/"
export nnUNet_preprocessed="${nnUNet_raw_data_base}/nnUNet_preprocessed"
export RESULTS_FOLDER="${nnUNet_raw_data_base}/RESULTS_FOLDER"


# --- Inference using ensemble predictions with a probabilistic union ---
DATASET_ID="Task001_OvarianCancer"
FINAL_CT_TESTS="${nnUNet_raw_data_base}/FINAL_CT_TESTS"
OUTPUT_FINAL_PREDICTIONS="${nnUNet_raw_data_base}/OUTPUT_FINAL_PREDICTIONS"

mednextv1_predict -i $FINAL_CT_TESTS -o $OUTPUT_FINAL_PREDICTIONS -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p MedNeXt_L_kernel3__nnUNetPlansv2.1_trgSp_1x1x1 -t $DATASET_ID -z
