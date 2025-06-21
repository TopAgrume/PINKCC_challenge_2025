#!/bin/bash

CT_IMAGE_BASE_DIR="/home/alex/predictions_PINKCC/CT"
SEGMENTATION_BASE_DIR="/home/alex/predictions_PINKCC/ensemble_output"

STARTING_CT_FILENAME=""

if [ ! -d "$CT_IMAGE_BASE_DIR" ]; then
    echo "Error: CT Image directory not found at $CT_IMAGE_BASE_DIR"
    exit 1
fi

if [ ! -d "$SEGMENTATION_BASE_DIR" ]; then
    echo "Error: Segmentation directory not found at $SEGMENTATION_BASE_DIR"
    exit 1
fi

echo "Starting to process files..."
echo "CT Image Directory: $CT_IMAGE_BASE_DIR"
echo "Segmentation Directory: $SEGMENTATION_BASE_DIR"
echo "Will start processing from: $STARTING_CT_FILENAME"
echo "--------------------------------------------------"

process_files_flag=true

ct_files=()
while IFS= read -r -d $'\0' file; do
    ct_files+=("$file")
done < <(find "$CT_IMAGE_BASE_DIR" -maxdepth 1 -name '*.nii.gz' -print0 | sort -zV)

if [ ${#ct_files[@]} -eq 0 ]; then
    echo "No .nii.gz files found in $CT_IMAGE_BASE_DIR"
    exit 0
fi

for ct_file_full_path in "${ct_files[@]}"; do
    ct_filename=$(basename "$ct_file_full_path")

    if [[ "$ct_filename" == "$STARTING_CT_FILENAME" ]]; then
        process_files_flag=true
    fi

    if [[ "$process_files_flag" == true ]]; then
        segmentation_file_full_path="$SEGMENTATION_BASE_DIR/$ct_filename"

        echo ""
        echo "Processing CT file: $ct_file_full_path"

        if [ -f "$segmentation_file_full_path" ]; then
            echo "Found Segmentation file: $segmentation_file_full_path"
            echo "Opening in ITK-SNAP..."

            itksnap -g "$ct_file_full_path" -s "$segmentation_file_full_path"

            echo "ITK-SNAP window closed. Proceeding to the next file if available."
        else
            echo "Warning: Segmentation file not found for $ct_filename at $segmentation_file_full_path"
            echo "Skipping this pair."
        fi
    else
        echo "Skipping file (before start point): $ct_filename"
    fi
done

if [[ "$process_files_flag" == false ]]; then
    echo "--------------------------------------------------"
    echo "Warning: The starting file '$STARTING_CT_FILENAME' was not found in $CT_IMAGE_BASE_DIR."
    echo "No files were processed."
fi

echo "--------------------------------------------------"
echo "All designated files have been processed."
