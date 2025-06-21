export nnUNet_raw_data_base="/root/DATA/nnUNet_raw_data_base"
export nnUNet_raw="${nnUNet_raw_data_base}/nnUNet_raw_data"
export nnUNet_preprocessed="${nnUNet_raw_data_base}/nnUNet_preprocessed"
export nnUNet_results="${nnUNet_raw_data_base}/nnUNet_results"


# --- Custom MedNeXt Preprocessing (needs a lot of free disk space) ---
DATASET_ID="Task001_OvarianCancer"
mednextv1_plan_and_preprocess -t $DATASET_ID -pl3d ExperimentPlanner3D_v21_customTargetSpacing_1x1x1


# --- Training on all folds (16203MiB vRAM required)  [450 epochs is enough] ---
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_L_kernel3 $DATASET_ID 0 -p nnUNetPlansv2.1_trgSp_1x1x1 --npz
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_L_kernel3 $DATASET_ID 1 -p nnUNetPlansv2.1_trgSp_1x1x1 --npz
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_L_kernel3 $DATASET_ID 2 -p nnUNetPlansv2.1_trgSp_1x1x1 --npz
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_L_kernel3 $DATASET_ID 3 -p nnUNetPlansv2.1_trgSp_1x1x1 --npz
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_L_kernel3 $DATASET_ID 4 -p nnUNetPlansv2.1_trgSp_1x1x1 --npz


# --- To go further: use 5x5x5 kernels using UpKern (we were unable to test due to lack of time and ressources) ---
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_L_kernel5 $DATASET_ID 0 -p nnUNetPlansv2.1_trgSp_1x1x1 -pretrained_weights ${RESULTS_FOLDER}/nnUNet/3d_fullres/${DATASET_ID}/nnUNetTrainerV2_MedNeXt_L_kernel3__nnUNetPlansv2.1_trgSp_1x1x1/fold_0/model_final_checkpoint.model -resample_weights --npz
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_L_kernel5 $DATASET_ID 1 -p nnUNetPlansv2.1_trgSp_1x1x1 -pretrained_weights ${RESULTS_FOLDER}/nnUNet/3d_fullres/${DATASET_ID}/nnUNetTrainerV2_MedNeXt_L_kernel3__nnUNetPlansv2.1_trgSp_1x1x1/fold_1/model_final_checkpoint.model -resample_weights --npz
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_L_kernel5 $DATASET_ID 2 -p nnUNetPlansv2.1_trgSp_1x1x1 -pretrained_weights ${RESULTS_FOLDER}/nnUNet/3d_fullres/${DATASET_ID}/nnUNetTrainerV2_MedNeXt_L_kernel3__nnUNetPlansv2.1_trgSp_1x1x1/fold_2/model_final_checkpoint.model -resample_weights --npz
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_L_kernel5 $DATASET_ID 3 -p nnUNetPlansv2.1_trgSp_1x1x1 -pretrained_weights ${RESULTS_FOLDER}/nnUNet/3d_fullres/${DATASET_ID}/nnUNetTrainerV2_MedNeXt_L_kernel3__nnUNetPlansv2.1_trgSp_1x1x1/fold_3/model_final_checkpoint.model -resample_weights --npz
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_L_kernel5 $DATASET_ID 4 -p nnUNetPlansv2.1_trgSp_1x1x1 -pretrained_weights ${RESULTS_FOLDER}/nnUNet/3d_fullres/${DATASET_ID}/nnUNetTrainerV2_MedNeXt_L_kernel3__nnUNetPlansv2.1_trgSp_1x1x1/fold_4/model_final_checkpoint.model -resample_weights --npz


# --- Final ensembling using the 5-fold cross-validation strategy ---
mednextv1_find_best_configuration -m 3d_fullres -t $DATASET_ID -f 0 1 2 3 4 -pl MedNeXt_L_kernel3__nnUNetPlansv2.1_trgSp_1x1x1 --disable_postprocessing


# --- Inference using ensemble predictions with a probabilistic union ---
FINAL_CT_TESTS='/root/DATA/FINAL_CT_TESTS'
OUTPUT_FINAL_PREDICTIONS='/root/DATA/OUTPUT_FINAL_PREDICTIONS'

mednextv1_predict -i $FINAL_CT_TESTS -o $OUTPUT_FINAL_PREDICTIONS -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p MedNeXt_L_kernel3__nnUNetPlansv2.1_trgSp_1x1x1 -t $DATASET_ID -z
