import numpy as np
from typing import List


def average_probabilities(list_of_files: List[str]) -> np.ndarray:
    """
    Average probability arrays from 3 ensembles predictions (nnUNet, MedNeXt, OVSeg).

    Args:
        list_of_files: List of 3 .npz per CT scan to predict.

    Returns:
        Averaged probability array as float32
    """
    if not list_of_files:
        raise ValueError("At least one file must be provided")

    probabilities = []

    for file_path in list_of_files:
        prob_array = _load_probability_array(file_path)
        probabilities.append(prob_array)

    stacked_probs = np.stack(probabilities, axis=0)
    return np.mean(stacked_probs, axis=0, dtype=np.float32)


def _load_probability_array(file_path: str) -> np.ndarray:
    """
    Load probability array from a single file with appropriate transformations.

    Args:
        file_path: Path to the prediction file

    Returns:
        Probability array as float32
    """
    try:
        data = np.load(file_path)
    except Exception as e:
        raise ValueError(f"Failed to load file {file_path}: {e}")

    prob_array = None
    for key in ['probabilities', 'softmax']:
        if key in data:
            prob_array = data[key]
            break

    if prob_array is None:
        available_keys = list(data.keys())
        raise ValueError(
            f"File {file_path} doesn't contain 'probabilities' or 'softmax' keys. "
            f"Available keys: {available_keys}"
        )

    # OVSeg-specific axis flipping
    if "ovseg" in file_path.lower():
        prob_array = prob_array[:, ::-1, ::-1, :]

    return prob_array.astype(np.float32)