import os

def rename_nii_files(directory):
    """
    Appends '_0000' before the .nii.gz extension for all .nii.gz files in the given directory.
    """
    for filename in os.listdir(directory):
        if filename.endswith('.nii.gz'):
            base = filename[:-7]  # remove '.nii.gz'
            new_filename = f"{base}_0000.nii.gz"
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")


if __name__ == "__main__":
    rename_nii_files("/root/DATA/nnUNet_raw_data/Task001_OvarianCancerDestroyer/imagesTr")
    rename_nii_files("/root/DATA/nnUNet_raw_data/Task001_OvarianCancerDestroyer/labelsTr")