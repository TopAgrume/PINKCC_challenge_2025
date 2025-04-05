import os
import nibabel as nib
import ipywidgets as widgets
from IPython.display import display, clear_output

from ocd.dataset.dataset import SampleUtils
from ocd.jupyter.visualizer import JupyterVisualizer


class JupyterExplorer:

    @classmethod
    def file_selector(self, dataset_dict):
        """Create an interactive widget to select files from the dataset"""
        # Create a dropdown for dataset selection
        dataset = list(dataset_dict.keys())

        if not dataset:
            print("No dataset found.")
            return None, None

        dataset_dropdown = widgets.Dropdown(
            options=dataset,
            description='Dataset:',
            disabled=False,
        )

        # Create a function to update file dropdown based on selected dataset
        file_dropdown = widgets.Dropdown(
            description='File:',
            disabled=False,
        )

        def update_files(*args):
            selected_dataset = dataset_dropdown.value
            if selected_dataset and selected_dataset in dataset_dict:
                file_list = dataset_dict[selected_dataset]
                file_dropdown.options = [(os.path.basename(f), f) for f in file_list]

        # Initialize the file dropdown
        dataset_dropdown.observe(update_files, names='value')
        update_files()  # Call once to initialize

        # Create a button to load the selected file
        load_button = widgets.Button(
            description='Load File',
            button_style='info',
            tooltip='Click to load the selected file'
        )

        output = widgets.Output()

        def on_button_click(b):
            with output:
                clear_output()
                if file_dropdown.value:
                    print(f"Selected file: {file_dropdown.value}")
                    img, data = SampleUtils.load_from_path(file_dropdown.value)
                    JupyterVisualizer.interactive_slice_viewer(data, title_prefix=f"{os.path.basename(file_dropdown.value)}: ")

        load_button.on_click(on_button_click)

        # Display widgets
        display(widgets.VBox([dataset_dropdown, file_dropdown, load_button, output]))


    @classmethod
    def paired_file_selector(self, ct_dict, seg_dict):
        """Create an interactive widget to select and compare CT and segmentation files"""
        # Create dropdowns for CT and segmentation selection
        ct_datasets = list(ct_dict.keys())
        seg_datasets = list(seg_dict.keys())

        if not ct_datasets or not seg_datasets:
            print("Either CT or segmentation dataset not found.")
            return

        ct_dataset_dropdown = widgets.Dropdown(
            options=ct_datasets,
            description='CT Dataset:',
            disabled=False,
        )

        ct_file_dropdown = widgets.Dropdown(
            description='CT File:',
            disabled=False,
        )

        seg_dataset_dropdown = widgets.Dropdown(
            options=seg_datasets,
            description='Seg Dataset:',
            disabled=False,
        )

        seg_file_dropdown = widgets.Dropdown(
            description='Seg File:',
            disabled=False,
        )

        # Update functions for file dropdowns
        def update_ct_files(*args):
            selected_dataset = ct_dataset_dropdown.value
            if selected_dataset and selected_dataset in ct_dict:
                file_list = ct_dict[selected_dataset]
                ct_file_dropdown.options = [(os.path.basename(f), f) for f in file_list]

        def update_seg_files(*args):
            selected_dataset = seg_dataset_dropdown.value
            if selected_dataset and selected_dataset in seg_dict:
                file_list = seg_dict[selected_dataset]
                seg_file_dropdown.options = [(os.path.basename(f), f) for f in file_list]

        # Initialize the file dropdowns
        ct_dataset_dropdown.observe(update_ct_files, names='value')
        seg_dataset_dropdown.observe(update_seg_files, names='value')
        update_ct_files()
        update_seg_files()

        # Create a button to load the selected files
        compare_button = widgets.Button(
            description='Compare Files',
            button_style='info',
            tooltip='Click to compare the selected CT and segmentation files'
        )

        output = widgets.Output()

        def on_button_click(b):
            with output:
                clear_output()
                if ct_file_dropdown.value and seg_file_dropdown.value:
                    print(f"CT file: {os.path.basename(ct_file_dropdown.value)}")
                    print(f"Segmentation file: {os.path.basename(seg_file_dropdown.value)}")

                    # Load the files
                    ct_img = nib.load(ct_file_dropdown.value)
                    seg_img = nib.load(seg_file_dropdown.value)

                    ct_data = ct_img.get_fdata()
                    seg_data = seg_img.get_fdata()

                    # Check if dimensions match
                    if ct_data.shape == seg_data.shape:
                        print("\nDisplaying CT with segmentation overlay:")
                        JupyterVisualizer.interactive_ct_seg_viewer(ct_data, seg_data)
                    else:
                        print(f"Warning: CT shape {ct_data.shape} doesn't match segmentation shape {seg_data.shape}")

        compare_button.on_click(on_button_click)

        # Display widgets
        display(widgets.VBox([
            widgets.HBox([ct_dataset_dropdown, seg_dataset_dropdown]),
            widgets.HBox([ct_file_dropdown, seg_file_dropdown]),
            compare_button,
            output
        ]))