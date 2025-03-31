import copy
import numpy as np
import pandas as pd


mid_vmax_per = 66
default_vmax_per = 98


def norm_to_zero_mean(input_mtx):
    input_mtx = input_mtx.to_dense()
    new_matrix = copy.deepcopy(input_mtx)
    
    matrix_for_mean = new_matrix.matrix.copy()
    vmax = np.percentile(matrix_for_mean, mid_vmax_per)
    vmin = np.percentile(matrix_for_mean, 100 - mid_vmax_per)
    if vmin < vmax:
        sub_matrix_for_mean = matrix_for_mean[matrix_for_mean < vmax]
        sub_matrix_for_mean = sub_matrix_for_mean[sub_matrix_for_mean > vmin]
        if len(sub_matrix_for_mean) > 0:
            matrix_for_mean = sub_matrix_for_mean
    else:
        matrix_for_mean = matrix_for_mean
    new_matrix.matrix -= np.mean(matrix_for_mean)
    new_matrix.matrix /= np.std(matrix_for_mean)
    new_matrix.has_neg = True
    return new_matrix


def remove_outliers(input_mtx):
    input_mtx = input_mtx.to_dense()
    new_matrix = copy.deepcopy(input_mtx)
    
    vmax = np.percentile(new_matrix.matrix[new_matrix.matrix > 0], default_vmax_per)
    vmin = np.percentile(new_matrix.matrix[new_matrix.matrix < 0], 100 - default_vmax_per)
    min_value = np.min([vmax, -vmin])
    vmax, vmin = min_value, -min_value
    new_matrix.matrix[new_matrix.matrix > vmax] = vmax
    new_matrix.matrix[new_matrix.matrix < vmin] = vmin
    return new_matrix


def norm_de_mtx(input_mtx):
    new_matrix = norm_to_zero_mean(input_mtx)
    new_matrix = remove_outliers(new_matrix)
    return new_matrix
