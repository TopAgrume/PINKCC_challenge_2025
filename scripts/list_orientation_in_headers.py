import os
import nibabel as nib

def get_nifti_orientation_report(input_dir, output_report_file):
    print(f"Scanning directory: '{input_dir}'")
    print(f"Report will be saved to: '{output_report_file}'")

    orientations_found = []

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".nii.gz"):
            file_path = os.path.join(input_dir, filename)
            try:
                nifti_img = nib.load(file_path)
                affine = nifti_img.affine

                orientation_codes_tuple = nib.orientations.aff2axcodes(affine)
                orientation_str = "".join(orientation_codes_tuple)

                orientations_found.append(f"{filename}: {orientation_str}")
                print(f"  Processed '{filename}': {orientation_str}")

            except Exception as e:
                error_message = f"{filename}: ERROR ({e})"
                orientations_found.append(error_message)
                print(f"  Error processing '{filename}': {e}")
        elif filename.endswith(".nii"):
            print(f"  Skipping '{filename}' (plain .nii file, expected .nii.gz). Consider including if needed.")


    if not orientations_found:
        print(f"No .nii.gz files found in '{input_dir}'.")
        with open(output_report_file, 'w') as f:
            f.write(f"No .nii.gz files found in directory: {input_dir}\n")
        return

    try:
        with open(output_report_file, 'w') as f:
            for entry in orientations_found:
                f.write(entry + "\n")
        print(f"\nReport successfully created: '{output_report_file}'")
    except IOError as e:
        print(f"Error writing report file '{output_report_file}': {e}")


if __name__ == "__main__":
    ct_scan_directory = "NouveauDatasetChallenge/Train_Phase_2/CT"
    report_file = "ct_scan_orientations_report.txt"

    get_nifti_orientation_report(ct_scan_directory, report_file)