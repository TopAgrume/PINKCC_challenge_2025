from project.dataset import Dataset

def main():
    dataset = Dataset.loading_dataset_paths()

    sample_ct_file = dataset['CT_MSKCC'][0]
    _, ct_data = Dataset.display_nifti_info(sample_ct_file)

    # Display interactive slice viewer
    print("\nInteractive CT Viewer:")
    Dataset.interactive_slice_viewer(ct_data, title_prefix="CT: ")



if __name__ == "__main__":
    main()