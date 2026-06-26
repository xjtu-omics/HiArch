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
from iutils.read_matrix import Anchors


def get_kernal_len(chr_rowsum, kp, mkl, only_odd=False):
    kernal_len = int(chr_rowsum.shape[0] * kp)
    kernal_len = mkl if kernal_len < mkl else kernal_len
    if only_odd:
        kernal_len = (kernal_len // 2) * 2 + 1
    return kernal_len


##################
# Get row sum
inter_rowsum_name = 'inter_rowsum'
def get_inter_rowsum(input_mtx):
    inter_rowsum = np.zeros(input_mtx.shape[1])
    for chro in input_mtx.row_window.chros:
        chr_inter_map = input_mtx.get_region_mtx(chro, col_is_inter=True)
        if chr_inter_map is None:
            continue
        
        chr_sep = input_mtx.row_window.chr_seps[chro]
        inter_rowsum[chr_sep[0]: chr_sep[1] + 1] = chr_inter_map.get_row_sum(axis=1) / chr_inter_map.shape[1]
    inter_rowsum = pd.DataFrame({inter_rowsum_name: inter_rowsum})
    inter_rowsum = Values(inter_rowsum, input_mtx.row_window)
    return inter_rowsum


kernal_per = .1
min_kernal_len = 5
def smooth_rowsum(rowsum):
    from scipy.ndimage import convolve
    smoothed_rowsum = []
    for _, chr_rowsum in rowsum.yield_by_chro():
        kernal_len = get_kernal_len(chr_rowsum, kernal_per, min_kernal_len, only_odd=True)
        smoothed_chr_rowsum = convolve(chr_rowsum.values, 
                                       weights=np.ones((kernal_len, 1)) / kernal_len,
                                       mode='reflect')
        smoothed_chr_rowsum = pd.DataFrame(smoothed_chr_rowsum, 
                                           index=chr_rowsum.values.index,
                                           columns=chr_rowsum.values.columns)
        smoothed_chr_rowsum = Values(smoothed_chr_rowsum, chr_rowsum.gen_index)
        smoothed_rowsum.append(smoothed_chr_rowsum)
    return concat_values(smoothed_rowsum, rowsum.gen_index)


##################
# Get anchor points
# Steps to find anchors:
#   1. Anchors must be nearest highest points (This range cannot be too small or too large). (per: .1)
#   2. Anchors must be obviously higher (?) in density than its boundaries. (max absolute diff value > 0.03, min diff value > 0.01), Keep value strength & domain size.
#   3. Anchors must not be at telomeres (diatance?). (decrease_per=.2)


def get_anchor_regions(rowsum_arr, search_per=.1, min_search_len=5):
    search_len = get_kernal_len(rowsum_arr, search_per, min_search_len)
    
    nearest_higher_point = {}
    region_assign = {}
    regions = pd.DataFrame([])
    
    def _get_nearest_higher_points():
        nonlocal nearest_higher_point
        for ip in np.arange(len(rowsum_arr)):
            has_nearest_higher_point = False
            for step in np.arange(search_len):
                # The nearest higher point is found
                if ip - step >= 0:
                    if rowsum_arr[ip - step] > rowsum_arr[ip]:
                        nearest_higher_point[ip] = ip - step
                        has_nearest_higher_point = True
                        break
                    
                if ip + step <= len(rowsum_arr) - 1:
                    if rowsum_arr[ip + step] > rowsum_arr[ip]:
                        nearest_higher_point[ip] = ip + step
                        has_nearest_higher_point = True
                        break
            
            if not has_nearest_higher_point:
                nearest_higher_point[ip] = ip

    def _get_anchors():
        center = {}
        nonlocal region_assign
        cluster_num = 1
        for ip in nearest_higher_point:
            region_assign[ip] = 0
            if nearest_higher_point[ip] == ip and rowsum_arr[ip] > 0:
                region_assign[ip] = cluster_num
                center[cluster_num] = ip
                cluster_num += 1
        nonlocal regions
        regions = pd.DataFrame({'rs': center, 're': center, 'cen': center})
    
    def _assign_bins():
        new_join = 1
        while new_join != 0:
            new_join = 0
            for ip in list(nearest_higher_point.keys()):
                # Two situations could happen here:
                # ONE: ip is a center or already been assigned, do not need assigned
                # TWO: If nearest higher point is not found and ip is not a center,
                #       then ip is too sparse to assigned to any cluster.
                if region_assign[ip] != 0 or nearest_higher_point[ip] == ip: continue

                # Center points are filtered in last step, so cluster_assign[ip] should be 0.
                # If nearest higher point is a center and center den_vali is larger than ip den_vali:
                possible_center = nearest_higher_point[ip]

                if region_assign[possible_center] != 0:
                    new_join += 1

                    # Check if ip belonged to any cluster
                    ip_class = regions[(ip > regions['rs']) & (ip < regions['re'])].values

                    # If ip is already in a cluster, assign to the first one
                    if len(ip_class) != 0:
                        region_assign[ip] = region_assign[ip_class[0][2]]

                    # if not, assign ip to possible_center
                    else:
                        region_assign[ip] = region_assign[possible_center]
                        # Change the cluster boundaries
                        if ip < regions.loc[region_assign[ip], 'rs']:
                            regions.loc[region_assign[ip], 'rs'] = ip
                        if ip > regions.loc[region_assign[ip], 're']:
                            regions.loc[region_assign[ip], 're'] = ip
    
    _get_nearest_higher_points()
    _get_anchors()
    _assign_bins()
    
    regions = regions.sort_values(by='cen')
    return regions


def mark_anchor_state(rowsum_arr, chr_anchor_regions, update_idxes=None):
    marked_cols = ['before_min', 'before_diff',
                   'after_min', 'after_diff',
                   'cen_rowsum']
    for markes_col in marked_cols:
        if markes_col not in chr_anchor_regions.columns:
            chr_anchor_regions[markes_col] = np.nan
    
    update_idxes = chr_anchor_regions.index if update_idxes is None else update_idxes
    for region_idx in update_idxes:
        chr_anchor_region = chr_anchor_regions.loc[region_idx, :]
        
        rs = int(chr_anchor_region['rs'])
        re = int(chr_anchor_region['re'])
        cen = int(chr_anchor_region['cen'])
        
        cen_rowsum = rowsum_arr[cen]
        chr_anchor_regions.loc[region_idx, 'cen_rowsum'] = rowsum_arr[cen]
        
        before_cen_rowsum = rowsum_arr[rs: cen]
        if len(before_cen_rowsum) > 0:
            chr_anchor_regions.loc[region_idx, 'before_min'] = np.min(before_cen_rowsum)
            chr_anchor_regions.loc[region_idx, 'before_diff'] = cen_rowsum - np.min(before_cen_rowsum)
        
        after_cen_rowsum = rowsum_arr[cen + 1: re + 1]
        if len(after_cen_rowsum) > 0:
            chr_anchor_regions.loc[region_idx, 'after_min'] = np.min(after_cen_rowsum)
            chr_anchor_regions.loc[region_idx, 'after_diff'] = cen_rowsum - np.min(after_cen_rowsum)

    return chr_anchor_regions


min_for_larger_diff = .03
min_for_smaller_diff = .01
def remove_small_anchors(chr_anchor_regions, rowsum_arr):
    def barplot_regions(rowsum_arr, anchor_regions):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.barh(np.arange(len(rowsum_arr)), rowsum_arr,
                color='gray')
        
        region_colors = sns.color_palette('tab10', n_colors=anchor_regions.shape[0])
        i_region = 0
        for _, region in anchor_regions.iterrows():
            region_range = np.arange(int(region['rs']), int(region['re']) + 1)
            region_inter_rowsum = rowsum_arr[region_range]
            ax.barh(region_range, region_inter_rowsum, color=region_colors[i_region])
            i_region += 1
        
        ax.set_title('Inter row-sum')
        # ax.set_yticks([])
        ax.set_ylim([0, len(rowsum_arr)])
        ax.invert_yaxis()
        
        plt.show()
        plt.close()

    def merge_region(region_idx):
        new_chr_anchor_regions = chr_anchor_regions.copy()
        
        merge_diff = {}
        
        index_list = list(chr_anchor_regions.index)
        region_arr_idx = index_list.index(region_idx)
        
        last_arr_idx = region_arr_idx - 1
        if last_arr_idx >= 0:
            last_region_idx = index_list[last_arr_idx]
            merge_diff[last_region_idx] = chr_anchor_regions.loc[region_idx, 'before_diff']
        
        next_arr_idx = region_arr_idx + 1
        if next_arr_idx < len(index_list):
            next_region_idx = index_list[next_arr_idx]
            merge_diff[next_region_idx] = chr_anchor_regions.loc[region_idx, 'after_diff']
        
        if len(merge_diff) == 0:
            new_chr_anchor_regions.drop(region_idx, axis=0, inplace=True)
            return new_chr_anchor_regions

        merge_idx = min(merge_diff, key=merge_diff.get)
        
        new_region_rs = np.min([new_chr_anchor_regions.loc[merge_idx, 'rs'],
                                new_chr_anchor_regions.loc[region_idx, 'rs']])
        new_region_re = np.max([new_chr_anchor_regions.loc[merge_idx, 're'],
                                new_chr_anchor_regions.loc[region_idx, 're']])

        merge_cen = new_chr_anchor_regions.loc[merge_idx, 'cen']
        region_cen = int(chr_anchor_regions.loc[region_idx, 'cen'])
        merge_cen_rowsum = new_chr_anchor_regions.loc[merge_idx, 'cen_rowsum']
        region_cen_rowsum = new_chr_anchor_regions.loc[region_idx, 'cen_rowsum']
        new_region_cen = merge_cen if merge_cen_rowsum > region_cen_rowsum else region_cen
        
        new_chr_anchor_regions.drop(region_idx, axis=0, inplace=True)
        new_chr_anchor_regions.loc[merge_idx, 'rs'] = new_region_rs
        new_chr_anchor_regions.loc[merge_idx, 're'] = new_region_re
        new_chr_anchor_regions.loc[merge_idx, 'cen'] = new_region_cen
        
        new_chr_anchor_regions = mark_anchor_state(rowsum_arr, 
                                                   new_chr_anchor_regions, 
                                                   update_idxes=[merge_idx])
        
        return new_chr_anchor_regions

    has_changed_regions = True
    while has_changed_regions:
        has_changed_regions = False
        
        # barplot_regions(rowsum_arr, chr_anchor_regions)
        
        for region_idx, chr_anchor_region in chr_anchor_regions.iterrows():
            rowsum_diff = []
            if chr_anchor_region['rs'] != 0:
                rowsum_diff.append(chr_anchor_region['before_diff'])
            if chr_anchor_region['re'] != len(rowsum_arr) - 1:
                rowsum_diff.append(chr_anchor_region['after_diff'])
            rowsum_diff = np.array(rowsum_diff)[~np.isnan(rowsum_diff)]
            
            # If it is the only anchor region, keep this region regardless of strength.
            if chr_anchor_region['rs'] == 0 and chr_anchor_region['re'] == len(rowsum_arr) - 1:
                continue
            
            if len(rowsum_diff) == 0:
                has_changed_regions = True
                chr_anchor_regions = merge_region(region_idx)
                break
            
            # For these, merge into higher cluster of nearby two and its own.
            if np.max(rowsum_diff) < min_for_larger_diff:
                has_changed_regions = True
                chr_anchor_regions = merge_region(region_idx)
                break
            # For these, merge into higher cluster of nearby two.
            if np.min(rowsum_diff) < min_for_smaller_diff:
                has_changed_regions = True
                chr_anchor_regions = merge_region(region_idx)
                break
    
    return chr_anchor_regions


decrease_per = 1 / 3
def mark_telo_anchors(chr_anchor_regions):
    chr_anchor_regions['type'] = 'normal'
    
    if chr_anchor_regions.shape[0] == 0:
        return chr_anchor_regions
    
    chr_anchor_regions = chr_anchor_regions.sort_values('rs')
    
    possible_telo_region_idxes = [0, -1]
    
    for possible_telo_region_idx in possible_telo_region_idxes:
        region_idx = chr_anchor_regions.index[possible_telo_region_idx]
        
        before_min = chr_anchor_regions.loc[region_idx, 'before_diff']
        after_min = chr_anchor_regions.loc[region_idx, 'after_diff']
        
        if np.isnan(before_min) or np.isnan(after_min):
            chr_anchor_regions.loc[region_idx, 'type'] = 'telo'
            continue
        
        if possible_telo_region_idx == 0:
            if before_min < after_min * decrease_per:
                chr_anchor_regions.loc[region_idx, 'type'] = 'telo'
                
        if possible_telo_region_idx == -1:
            if after_min < before_min * decrease_per:
                chr_anchor_regions.loc[region_idx, 'type'] = 'telo'
    return chr_anchor_regions


best_value_col = 'cen_rowsum'
def mark_best_regions(chr_anchor_regions):
    non_telo_regions = chr_anchor_regions[chr_anchor_regions['type'] != 'telo']
    if non_telo_regions.shape[0] == 0:
        return chr_anchor_regions
    non_telo_regions = non_telo_regions.sort_values(best_value_col, ascending=False)
    best_region_idx = non_telo_regions.index[0]
    chr_anchor_regions.loc[best_region_idx, 'type'] = 'used'
    return chr_anchor_regions


default_cols = ['cen', 'rs', 're']
def reindex_n_add_cs_ce(regions, chr_rowsum, cols=default_cols):
    chr_start_idx = np.min(chr_rowsum.values.index)
    chr_end_idx = np.max(chr_rowsum.values.index)
    
    for col in cols:
        regions[col] += chr_start_idx

    regions['cs'] = regions['cen']
    regions['ce'] = regions['cen'] + 1

    # In case anchor is at end of chromosome.
    exceed_indexes = regions[regions['ce'] > chr_end_idx].index
    if len(exceed_indexes) > 0:
        regions.loc[exceed_indexes, 'cs'] = regions.loc[exceed_indexes, 'cen'] - 1
        regions.loc[exceed_indexes, 'ce'] = regions.loc[exceed_indexes, 'cen']
    
    return regions


def find_anchors_by_inter_rowsum(rowsum):
    anchor_regions = []
    for _, chr_rowsum in rowsum.yield_by_chro():
        chr_rowsum_arr = np.squeeze(np.array(chr_rowsum.values))
        chr_anchor_regions = get_anchor_regions(chr_rowsum_arr)
        chr_anchor_regions = mark_anchor_state(chr_rowsum_arr, chr_anchor_regions)
        chr_anchor_regions = remove_small_anchors(chr_anchor_regions, chr_rowsum_arr)
        chr_anchor_regions = mark_telo_anchors(chr_anchor_regions)
        chr_anchor_regions = mark_best_regions(chr_anchor_regions)
        
        chr_anchor_regions = reindex_n_add_cs_ce(chr_anchor_regions, chr_rowsum)
        anchor_regions.append(chr_anchor_regions)
    
    anchor_regions = pd.concat(anchor_regions, axis=0, ignore_index=True)
    anchor_regions = Anchors(anchor_regions, rowsum.gen_index, keep_first=False)
    return anchor_regions


##################
# Main function for anchors
def main_find_anchors_by_inter_rowsum(sps_map, do_plot=False, fig_outfile=None,
                                      return_map_hist_for_plot=False):
    sps_map = sps_map.to_dense()
    sps_map.to_all(inplace=True)

    high_map = keep_high_cons(sps_map, plot_hist=False)
    inter_rowsum = get_inter_rowsum(high_map)
    smoothed_inter_rowsum = smooth_rowsum(inter_rowsum)
    
    anchor_regions = find_anchors_by_inter_rowsum(smoothed_inter_rowsum)
    
    plot_maps = {'Filt': sps_map, 'High': high_map}
    value_dicts = {'Inter rowsum': inter_rowsum,
                    'Smoothed rowsum': smoothed_inter_rowsum}
    if do_plot or fig_outfile is not None:
        plot_anchors(plot_maps, 
                     value_dicts,
                     anchor_regions, 
                     fig_outfile)
    
    if not return_map_hist_for_plot:
        return anchor_regions
    else:
        return anchor_regions, plot_maps, value_dicts

