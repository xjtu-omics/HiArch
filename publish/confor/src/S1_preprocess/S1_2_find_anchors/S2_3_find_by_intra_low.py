import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.append("/data/home/cheyizhuo/project/centr_loc/mycode/")
sys.path.append("/data/home/cheyizhuo/mycode/hi-c/")
from new_hic_class import Values, concat_values
from confor.src.S1_preprocess.S1_2_find_anchors.U2_keep_top_cons import keep_high_cons
from confor.src.S1_preprocess.S1_2_find_anchors.U1_plot_anchor import plot_anchors
from confor.src.S1_preprocess.S1_2_find_anchors.S2_1_find_by_inter_rowsum import find_anchors_by_inter_rowsum, smooth_rowsum


##################
# Get intra_low row sum
intra_rowsum_name = 'intra_rowsum'
def get_intra_rowsum(input_mtx):
    intra_rowsum = np.zeros(input_mtx.shape[1])
    for chro in input_mtx.row_window.chros:
        chr_map = input_mtx.get_region_mtx(chro)
        
        chr_sep = input_mtx.row_window.chr_seps[chro]
        intra_rowsum[chr_sep[0]: chr_sep[1] + 1] = chr_map.get_row_sum(axis=1) / chr_map.shape[1]
    intra_rowsum = pd.DataFrame({intra_rowsum_name: intra_rowsum})
    intra_rowsum = Values(intra_rowsum, input_mtx.row_window)
    return intra_rowsum


##################
# Main function for anchors
def main_find_anchors_by_intra_low(sps_map, do_plot=False, fig_outfile=None,
                                      return_map_hist_for_plot=False):
    sps_map = sps_map.to_dense()
    sps_map.to_all(inplace=True)

    high_map = keep_high_cons(sps_map, reverse=True, plot_hist=False)
    intra_rowsum = get_intra_rowsum(high_map)
    smoothed_intra_rowsum = smooth_rowsum(intra_rowsum)
    
    anchor_regions = find_anchors_by_inter_rowsum(smoothed_intra_rowsum)
    
    plot_maps = {'Filt': sps_map, 'Low': high_map}
    value_dicts = {'Intra rowsum': intra_rowsum,
                    'Smoothed rowsum': smoothed_intra_rowsum}
    if do_plot or fig_outfile is not None:
        plot_anchors(plot_maps, 
                     value_dicts,
                     anchor_regions, 
                     fig_outfile)
    
    if not return_map_hist_for_plot:
        return anchor_regions
    else:
        return anchor_regions, plot_maps, value_dicts

