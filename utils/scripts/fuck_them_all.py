from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


# --- NEW: Alternative for axcodes2aff for older Nibabel versions ---
def _axcodes2aff_alternative(axcodes, zooms):
  """
  Creates an affine matrix from axis codes and zooms.
  Alternative for older nibabel versions lacking nibabel.orientations.axcodes2aff.
  Voxel coordinate (0,0,0) will map to world coordinate (0,0,0).
  """
  map_char_to_idx_sign = {
    "L": (0, -1),
    "R": (0, 1),  # Canonical World X-axis (R)
    "P": (1, -1),
    "A": (1, 1),  # Canonical World Y-axis (A)
    "I": (2, -1),
    "S": (2, 1),  # Canonical World Z-axis (S)
  }

  RZS = np.zeros((3, 3))  # Rotational and scaling part

  for data_axis_idx in range(3):  # For data axes 0, 1, 2
    ax_code_char = axcodes[data_axis_idx]  # Axis code for current data axis (e.g., 'L')

    # world_axis_idx: which canonical world axis (0,1,2) this data axis corresponds to.
    # sign: direction along that canonical world axis.
    world_axis_idx, sign = map_char_to_idx_sign[ax_code_char]

    # The column of RZS for this data_axis_idx gets a non-zero value
    # at the row corresponding to the world_axis_idx.
    RZS[world_axis_idx, data_axis_idx] = sign * zooms[data_axis_idx]

  affine = np.eye(4)
  affine[:3, :3] = RZS
  return affine


def get_opposite_pole(pole_char: str) -> str:
  """Returns the opposite anatomical pole."""
  opposites = {"R": "L", "L": "R", "A": "P", "P": "A", "S": "I", "I": "S"}
  return opposites.get(pole_char.upper(), "?")


def validate_orientation_codes(codes_str: str) -> tuple[str, str, str] | None:
  """
  Validates a 3-character orientation string (e.g., "RAI").
  Returns a tuple of codes or None if invalid.
  """
  if not isinstance(codes_str, str) or len(codes_str) != 3:
    print("Error: Orientation must be a 3-character string (e.g., 'RAI', 'LPS').")
    return None

  codes = tuple(c.upper() for c in codes_str)
  valid_poles = ["R", "L", "A", "P", "I", "S"]
  seen_axes = {"RL": False, "AP": False, "IS": False}

  for code in codes:
    if code not in valid_poles:
      print(
        f"Error: Invalid character '{code}' in orientation. Must be one of {valid_poles}."
      )
      return None
    if code in ("R", "L"):
      if seen_axes["RL"]:
        print("Error: R/L axis specified more than once.")
        return None
      seen_axes["RL"] = True
    elif code in ("A", "P"):
      if seen_axes["AP"]:
        print("Error: A/P axis specified more than once.")
        return None
      seen_axes["AP"] = True
    elif code in ("I", "S"):
      if seen_axes["IS"]:
        print("Error: I/S axis specified more than once.")
        return None
      seen_axes["IS"] = True

  if not all(seen_axes.values()):
    print("Error: All three axes (R/L, A/P, I/S) must be specified.")
    return None

  return codes


def main():
  # --- Configuration ---
  input_dir_str = Path("DatasetChallengeV2") / "CT"

  output_dir_str = "output_ct_scans_corrected"

  input_dir = Path(input_dir_str)
  output_dir = Path(output_dir_str)

  if not input_dir.is_dir():
    print(f"Error: Input directory '{input_dir}' not found.")
    return
  output_dir.mkdir(parents=True, exist_ok=True)
  # --- End Configuration ---

  test_files = [
    "TCGA-13-0720.nii.gz",
    "347481.nii.gz",
    "TCGA-25-1323.nii.gz",
  ]
  #   "TCGA-13-0800.nii.gz",
  #   "TCGA-13-2066.nii.gz",
  #   "TCGA-24-0975.nii.gz",
  #   "330695.nii.gz",
  #   "330706.nii.gz",
  #   "333023.nii.gz",
  #   "333024.nii.gz",
  #   "337195.nii.gz",
  #   "TCGA-24-0966.nii.gz",
  #   "347563.nii.gz",
  #   "TCGA-13-1410.nii.gz",
  #   "333079.nii.gz",
  #   "347437.nii.gz",
  #   "333049.nii.gz",
  #   "347576.nii.gz",
  #   "TCGA-13-1505.nii.gz",
  #   "TCGA-61-1727.nii.gz",
  #   "TCGA-13-0797.nii.gz",
  #   "TCGA-13-1495.nii.gz",
  #   "TCGA-24-1103.nii.gz",
  #   "TCGA-OY-A56Q.nii.gz",
  #   "TCGA-13-1506.nii.gz",
  #   "347466.nii.gz",
  #   "TCGA-13-0795.nii.gz",
  #   "330703.nii.gz",
  #   "TCGA-25-1634.nii.gz",
  #   "333019.nii.gz",
  #   "347491.nii.gz",
  #   "TCGA-13-2059.nii.gz",
  #   "347497.nii.gz",
  #   "TCGA-61-2002.nii.gz",
  #   "TCGA-13-0807.nii.gz",
  #   "TCGA-13-1504.nii.gz",
  #   "TCGA-13-2057.nii.gz",
  #   "347526.nii.gz",
  #   "333078.nii.gz",
  #   "333059.nii.gz",
  #   "TCGA-13-2071.nii.gz",
  #   "TCGA-13-0802.nii.gz",
  #   "333040.nii.gz",
  #   "TCGA-13-0916.nii.gz",
  #   "347457.nii.gz",
  #   "347535.nii.gz",
  #   "337190.nii.gz",
  #   "347461.nii.gz",
  #   "TCGA-10-0937.nii.gz",
  #   "347575.nii.gz",
  # ]

  nifti_files = sorted(
    [
      f
      for f in input_dir.iterdir()
      if f.is_file()
      and (str(f).endswith(".nii") or str(f).endswith(".nii.gz"))
      and f.name == "TCGA-13-0802.nii.gz"
    ]
  )

  if not nifti_files:
    print(f"No NIFTI files (.nii or .nii.gz) found in '{input_dir}'.")
    return

  print(f"Found {len(nifti_files)} NIFTI files to process.")

  for file_path in nifti_files:
    print(f"\n--- Processing: {file_path.name} ---")
    try:
      img = nib.load(file_path)
      original_data = img.get_fdata()
      original_affine = img.affine
      original_header = img.header
      original_zooms = original_header.get_zooms()[:3]  # Voxel dimensions for X, Y, Z

      current_axcodes = nib.orientations.aff2axcodes(original_affine)

      # Determine slices to show, ensuring they are within bounds
      slice_indices_to_try = [0, 5, 10, 15, 20]
      max_z_index = original_data.shape[2] - 1

      slices_to_show_indices = sorted(
        list(set(s for s in slice_indices_to_try if 0 <= s <= max_z_index))
      )
      if (
        not slices_to_show_indices and max_z_index >= 0
      ):  # if none of the preferred are valid, show middle
        slices_to_show_indices = [max_z_index // 2]
      if not slices_to_show_indices:
        print(
          f"Warning: Cannot determine valid slices for {file_path.name} (shape {original_data.shape}). Skipping visualization."
        )
        slices_to_display_data = []
      else:
        slices_to_display_data = [
          (idx, original_data[:, :, idx]) for idx in slices_to_show_indices
        ]

      if slices_to_display_data:
        num_slices_to_plot = len(slices_to_display_data)
        # Adjust subplot layout based on number of slices
        if num_slices_to_plot <= 3:
          nrows, ncols = 1, num_slices_to_plot
        elif num_slices_to_plot <= 6:  # Max 5 requested, but good to be flexible
          nrows = (
            (num_slices_to_plot + 2) // 3
            if num_slices_to_plot > 3
            else (num_slices_to_plot + 1) // 2
          )
          ncols = 3 if num_slices_to_plot > 2 else num_slices_to_plot  # Max 3 columns
        else:  # More than 6, probably won't happen with Z=0,5,10,15,20
          nrows = (num_slices_to_plot + 2) // 3
          ncols = 3

        fig, axes = plt.subplots(1, 5, figsize=(5 * 8, 6), squeeze=False)
        axes_flat = axes.flatten()

        for i, (slice_idx, slice_data) in enumerate(slices_to_display_data):
          ax = axes_flat[i]
          # Using rot90 for a more standard radiological view of axial slices
          # (Anterior usually up, Right usually on viewer's left)
          # If current_axcodes = ('R','A','S'), data axis 0 is R, axis 1 is A.
          # imshow(slice) would have R along y, A along x.
          # rot90(slice) swaps these and flips one.
          # Y-axis of rot90(slice) is original data axis 1 (e.g., 'A')
          # X-axis of rot90(slice) is original data axis 0 (e.g., 'R'), but direction might be flipped.
          # Let's display without rotation and label data axes clearly.
          ax.imshow(
            slice_data.T, cmap="gray", origin="lower", aspect="equal"
          )  # Transpose and origin lower

          # Axes labels based on data array axes for the slice
          # Data axis 0 of slice -> axcodes[0]
          # Data axis 1 of slice -> axcodes[1]
          # Transposing slice_data means:
          # Plot Y-axis (rows of slice_data.T) is original data axis 1. Label: axcodes[1]
          # Plot X-axis (cols of slice_data.T) is original data axis 0. Label: axcodes[0]
          ax.set_ylabel(f"Data Axis 1 ({current_axcodes[1]})")
          ax.set_xlabel(f"Data Axis 0 ({current_axcodes[0]})")
          ax.set_title(f"Slice Z={slice_idx} (Data Axis 2: {current_axcodes[2]})")

        # Hide any unused subplots
        for j in range(num_slices_to_plot, len(axes_flat)):
          axes_flat[j].axis("off")

        fig.suptitle(
          f"File: {file_path.name}\nCurrent NIFTI axcodes: {current_axcodes}\nClose this window to enter new orientation in terminal.",
          fontsize=14,
        )
        plt.tight_layout(
          rect=[0, 0, 1, 0.96]
        )  # Adjust layout to make space for suptitle
        plt.show(block=False)  # This blocks until the plot window is closed
      else:
        print(
          f"No valid slices to display for {file_path.name}. Skipping visualization."
        )

      # Get user input for corrected orientation
      while True:
        user_input_str = (
          input(
            f"Enter desired orientation for '{file_path.name}' (e.g., RAI, LPS) or 'skip': "
          )
          .strip()
          .upper()
        )
        plt.close()
        if user_input_str == "SKIP":
          print(f"Skipping file {file_path.name} based on user input.")
          break

        corrected_axcodes = validate_orientation_codes(user_input_str)
        if corrected_axcodes:
          # Create new affine matrix based on user's desired orientation codes and original zooms
          # This creates an affine that REPRESENTS the new orientation codes.
          # The voxel data itself is NOT reordered.
          new_affine = _axcodes2aff_alternative(corrected_axcodes, zooms=original_zooms)

          # Create a new Nifti image with the original data but the new affine and original header
          # The Nifti1Image constructor will use the new_affine to set sform/qform in the header.
          new_img = nib.nifti1.Nifti1Image(original_data, new_affine)

          output_filename = output_dir / (file_path.stem.split(".")[0] + ".nii.gz")
          nib.loadsave.save(new_img, output_filename)
          print(f"Saved corrected NIFTI to: {output_filename}")
          print(f"  Old orientation: {current_axcodes}")
          print(
            f"  New orientation metadata: {corrected_axcodes} (Data array itself is NOT reordered)"
          )
          break
        else:
          print("Invalid input. Please try again.")

      if user_input_str == "SKIP":
        continue

    except Exception as e:
      print(f"Error processing file {file_path.name}: {e}")
      import traceback

      traceback.print_exc()

  print("\n--- All files processed. ---")


if __name__ == "__main__":
  main()
