# Overview

## `ocd/` – Ovarian Cancer Destroyer

A custom Python library designed to **open, manipulate, visualize, and convert CT scans** into different formats and orientations. It leverages **NIfTI** and **NiBabel** to ensure compatibility with standard medical imaging workflows.

Key features:

* Dataset and CT sample handling
* CT scan loading and formats conversion
* Orientation normalization
* Custom made 2d model and training/inference pipeline
* Visualization tools for 2D/3D slices
* Jupyter notebook tools

## `scripts/`

This folder contains a set of **Python and shell scripts** aimed at:

* Automating common preprocessing tasks
* Performing precise dataset transformations
* Ensuring compliance with the input requirements of deep learning libraries (nnU-Net, MedNeXt, OVSeg)
