import numpy as np
from typing import List


def average_probabilities(list_of_files: List[str]) -> np.ndarray:
    """
    Lajority vote from 3 ensembles predictions (nnUNet, MedNeXt, OVSeg).

    Args:
        list_of_files: List of 3 .npz per CT scan to predict.

    Returns:
        Averaged probability array as float32
    """
    pass