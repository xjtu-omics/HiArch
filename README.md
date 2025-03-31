# HiArch
[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Measure the high-order genome architectures (global folding and checkerboard) from Hi-C experiment.

Citation: 

## Included modules
We provide a one-click bash file to compute the strength of large-scale genome architectures, global folding and checkerboard.

Our algorithm comprises three key modules: normalization, global folding, and checkerboard.
1. NormDis & CorrectMap: Raw Hi-C maps are scaled to comparable sizes and normalized to remove distance-dependent biases. The resulting maps are subsequently utilized to calculate global folding and checkerboard scores. Notably, you can manually check the normalized maps to remove poor-assembled chromosomes, detailed in CorrectMap.
2. Global folding: Based on normalized maps, the computation of global folding score involves two sub-modules: detecting center anchors (GF_S1_get_center) and calculating the global folding scores (GF_S2_get_score). You can re-choose the alternative center anchors, detailed in GF_S1_get_center.
3. Checkerboard: Checkerboard scores are calculated based on normalized maps.

## Requirements
Python 3.7+
Seaborn
Scipy
Torch
Scikit-learn

## Quick start
