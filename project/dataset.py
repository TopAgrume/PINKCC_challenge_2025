import os
import nibabel as nib
import matplotlib.pyplot as plt


class Dataset:
    @classmethod
    def loading_dataset_paths(self, base_dir: str = 'DatasetChallenge', verify: bool = True) -> dict:
        """
        Explore NIFTI files in the DatasetChallenge directory structure
        """
        datasets = {}
        CT_length = []
        segmentation_length = []

        # CT data
        ct_datasets = ['MSKCC', 'TCGA']
        for dataset in ct_datasets:
            path = os.path.join(base_dir, 'CT', dataset)
            if os.path.exists(path):
                datasets[f'CT_{dataset}'] = [os.path.join(path, f) for f in os.listdir(path)]
            CT_length.append(len(datasets[f'CT_{dataset}']))
            print(f"CT_{dataset}: {CT_length[-1]} files")
            print(f"  Sample file: {os.path.basename(datasets[f'CT_{dataset}'][0])}")

        # Segmentation data
        seg_datasets = ['MSKCC', 'TCGA']
        for dataset in seg_datasets:
            path = os.path.join(base_dir, 'Segmentation', dataset)
            if os.path.exists(path):
                datasets[f'Segmentation_{dataset}'] = [os.path.join(path, f) for f in os.listdir(path)]
            segmentation_length.append(len(datasets[f'Segmentation_{dataset}']))
            print(f"Segmentation_{dataset}: {segmentation_length[-1]} files")
            print(f"  Sample file: {os.path.basename(datasets[f'Segmentation_{dataset}'][0])}")

        if verify:
            assert CT_length == segmentation_length, "Not the same amount sample/GT"

        return datasets


    @classmethod
    def load_from_path(self, file_path):
        """Load NIFTI file content"""
        img = nib.load(file_path)
        data = img.get_fdata()
        return img, data


    @classmethod
    def display_slice(self, data, slice_idx=None, axis=2, figsize=(10, 8),
                  cmap='gray', vmin=None, vmax=None, title=None):
        """Display a single slice from a 3D volume"""
        # Determine slice if not provided
        if slice_idx is None:
            slice_idx = data.shape[axis] // 2

        # Extract the slice
        if axis == 0:
            slice_data = data[slice_idx, :, :]
        elif axis == 1:
            slice_data = data[:, slice_idx, :]
        else:
            slice_data = data[:, :, slice_idx]

        plt.figure(figsize=figsize)
        plt.imshow(slice_data, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar()

        if title:
            plt.title(title)
        else:
            plt.title(f'Slice {slice_idx} along axis {axis}')

        plt.axis('on')
        plt.tight_layout()
        plt.show()