import numpy as np
import os
import sys

from new_hic_class import ArrayHiCMtx


def get_adj_mtx(input_mtx, metric='cosine'):
    from scipy.spatial.distance import pdist, squareform
    mtx = input_mtx.matrix
    adj_mtx = squareform(pdist(mtx, metric=metric))
    adj_mtx = ArrayHiCMtx(adj_mtx, copy_mtx_paras=input_mtx, 
                          has_neg=False, col_window=input_mtx.row_window)
    return adj_mtx
