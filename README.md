# HiArch
Measuring the high-order genome architectures (global folding and checkerboard) from Hi-C experiment

## Tutorial of NormDis, global folding and checkerboard
We provide a one-click bash file to compute the strength of large-scale genome architectures, global folding and checkerboard [].

Our algorithm comprises three key modules: normalization, global folding, and checkerboard.
1. NormDis & CorrectMap: Raw Hi-C maps are scaled to comparable sizes and normalized to remove distance-dependent biases. The resulting maps are subsequently utilized to calculate global folding and checkerboard scores. Notably, you can manually check the normalized maps to remove poor-assembled chromosomes, detailed in CorrectMap.
2. Global folding: Based on normalized maps, the computation of global folding score involves two sub-modules: detecting center anchors (GF_S1_get_center) and calculating the global folding scores (GF_S2_get_score). You can re-choose the alternative center anchors, detailed in GF_S1_get_center.
3. Checkerboard: Checkerboard scores are calculated based on normalized maps.


