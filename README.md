# HiArch
[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Measure the high-order genome architectures (global folding and checkerboard) from Hi-C experiment.

Citation: 

## Modules
We provide a one-click bash file to compute the strength of large-scale genome architectures, global folding and checkerboard.

Our algorithm comprises three key modules: **normalization**, **global folding**, and **checkerboard**.
1. **NormDis & CorrectMap**: Raw Hi-C maps are scaled to comparable sizes and normalized to remove distance-dependent biases. The resulting maps are subsequently utilized to calculate global folding and checkerboard scores. Notably, you can manually check the normalized maps to remove poor-assembled chromosomes, detailed in CorrectMap.
2. **Global folding**: Based on normalized maps, the computation of global folding score involves two sub-modules: detecting center anchors (GF_S1_get_center) and calculating the global folding scores (GF_S2_get_score). You can re-choose the alternative center anchors, detailed in GF_S1_get_center.
3. **Checkerboard**: Checkerboard scores are calculated based on normalized maps.

## Requirements
Python 3.7+
Seaborn
Scipy
Torch
Scikit-learn

## One-Click Pipeline Usage Guide

We provide a one-click pipeline script (`one_click_pipeline.sh`) for automated Hi-C data analysis. The core requirement for its execution is to properly organize input files within the designated `base_path` directory.

### 1. Directory Structure
The following directory tree must be created under your `base_path`:
```
base_path/
├── [species_name_1]/
│   └── sps_mtx/
├── [species_name_2]/
│   └── sps_mtx/
└── parameters.txt
```

**Steps:**
1.  Create your main `base_path` directory.
2.  Inside `base_path`, create a sub-directory for each species you wish to analyze (e.g., `human/`, `mouse/`).
3.  Inside each species directory, create a sub-sub-directory named `sps_mtx/`. This directory will contain all the input files for the samples belonging to that species.

### 2. File Preparation
Place the following two types of files for each sample inside the corresponding `sps_mtx/` directory.

#### 2.1 Sparse Matrix File (`.mtx`)
- **Purpose**: Contains the Hi-C contact data.
- **File Naming**: `<sample>_normalized.mtx`
    - `<sample>` is a unique identifier for the biological sample (e.g., `sample1_normalized.mtx`, `rep2_normalized.mtx`).
- **File Format**: A three-column, whitespace-separated text file.
    - **Column 1**: Row index (integer). Must be consistent with the index file (0-based or 1-based).
    - **Column 2**: Column index (integer).
    - **Column 3**: Contact value (float).
- **Note**: It is recommended to use pre-normalized contact matrices (e.g., using ICE or Knight-Ruiz (K) normalization) as input.

#### 2.2 Index File (`.window.bed`)
- **Purpose**: Provides the genomic coordinates for each bin (row/column) in the `.mtx` file.
- **File Naming**: `<sample>.window.bed`
    - The `<sample>` prefix must match the corresponding `.mtx` file but **without** the `_normalized` suffix.
    - Example: For `sample1_normalized.mtx`, the index file must be named `sample1.window.bed`.
- **File Format**: A four-column, whitespace-separated file in standard BED format.
    - **Column 1**: Chromosome name (string).
    - **Column 2**: Region start position (integer, **0-based**).
    - **Column 3**: Region end position (integer).
    - **Column 4**: Index (integer). This number corresponds to the row/column index in the associated `.mtx` file.
- **Generation**: This file can be created using tools like `bedtools makewindows`.

### 3. Complete Example
A correctly organized `base_path` directory will look like this:
```
base_path/
├── human/
│   └── sps_mtx/
│       ├── sample1_normalized.mtx
│       ├── sample1.window.bed
│       ├── sample2_normalized.mtx
│       └── sample2.window.bed
├── mouse/
│   └── sps_mtx/
│       ├── mouse_sample1_normalized.mtx
│       └── mouse_sample1.window.bed
└── parameters.txt   (parameter file, placed directly in base_path)
```
