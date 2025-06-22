# Final cross-model ensemble methods

This part of the repository contains the final cross-model ensemble method for combining predictions from the three different segmentation models: nnUNet, MedNeXt, and OVSeg. Each model trained using 5-fold cross-validation, resulting in three ensembles of 5 folds each.

#### Two different ensemble strategies are implemented:
1. **Majority vote** - where at least two out of three models must agree on a voxel prediction.
2. **Probability union** - where we average the predicted probabilities from the three models.

The goal of this hybrid strategy is to offer a balance between robustness - by leveraging consensus - and flexibility, by allowing strong outlier predictions to still be considered.

```
5-fold nnUNet ──┐
5-fold MedNeXt ─┼── Majority Vote ──┐
5-fold OVSeg ───┘                   │
                                    ├── Final Merge
5-fold nnUNet ──┐                   │
5-fold MedNeXt ─┼─── Prob Union ────┘
5-fold OVSeg ───┘
```

## Files description

### `majority_vote.py`
- The most frequent class across all models is selected for each voxel
- Output is converted back to one-hot probability format

### `probability_union.py`
- Probability distributions from all models are averaged
- Maintains uncertainty information in the final predictions

## Usage instructions

Ensure you have predictions from all three model ensembles:
**nnUNet** 5-fold, **MedNeXt** 5-fold, and **OVSeg** 5-fold predictions **(with corresponding .npz files containing probability arrays)**.

### Step 1: Run majority vote ensemble

1. Replace the `average_probabilities` function in `nnunetv2/ensembling/ensemble.py` with the implementation from `majority_vote.py`

2. Execute the ensemble command:
```sh
nnUNetv2_ensemble -i INPUT_FOLDER_NNUNET INPUT_FOLDER_MEDNEXT INPUT_FOLDER_OVSEG -o OUTPUT_FOLDER_MAJORIY_VOTE -np 8
```

### Step 2: Run probability union ensemble

1. Replace the `average_probabilities` function in `nnunetv2/ensembling/ensemble.py` with the implementation from `probability_union.py`

2. Execute the ensemble command:
```bash
nnUNetv2_ensemble -i INPUT_FOLDER_NNUNET INPUT_FOLDER_MEDNEXT INPUT_FOLDER_OVSEG -o OUTPUT_FOLDER_PROBABILITY_UNION -np 8
```

### Step 3: Final prediction merge

After obtaining results from both ensemble strategies, use the provided `merge_final_predictions.py` script to combine the final predictions from both methods.
