import numpy as np
import sys
sys.path.append("/data/home/cheyizhuo/project/centr_loc/mycode/")
sys.path.append("/data/home/cheyizhuo/mycode/hi-c/")
from new_hic_class import ArrayHiCMtx


def histplot_high_cons(new_data):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    n_bins = 30
    
    nonzero_arr = new_data[new_data > 0]
    sns.histplot(x=nonzero_arr, bins=n_bins)
    plt.show()
    plt.close()


def keep_high_cons(input_mtx, per=100/3, reverse=False, 
                   norm=True, plot_hist=False):
    new_data = input_mtx.matrix.copy()
    
    if not reverse:
        min_value = np.percentile(new_data, 100 - per)
        new_data[input_mtx.matrix < min_value] = min_value
        new_data -= min_value
    else:
        max_value = np.percentile(new_data, per)
        new_data[input_mtx.matrix > max_value] = max_value
        new_data *= -1
        new_data += max_value
    
    if norm:
        nonzero_arr = new_data[new_data > 0]
        new_data /= np.mean(nonzero_arr)
    
    if plot_hist:
        histplot_high_cons(new_data)
    
    return ArrayHiCMtx(new_data, copy_mtx_paras=input_mtx, has_neg=False)
