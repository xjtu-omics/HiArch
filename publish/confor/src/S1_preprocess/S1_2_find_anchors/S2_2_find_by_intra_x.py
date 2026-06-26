import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.append("/data/home/cheyizhuo/project/centr_loc/mycode/")
sys.path.append("/data/home/cheyizhuo/mycode/hi-c/")
from new_hic_class import Values, concat_values
from confor.src.S1_preprocess.S1_2_find_anchors.S2_1_find_by_inter_rowsum import reindex_n_add_cs_ce, get_anchor_regions
from confor.src.S1_preprocess.S1_2_find_anchors.U1_plot_anchor import plot_anchors
from iutils.read_matrix import Anchors


def get_kernal_len(chr_rowsum, kp, mkl, only_odd=False):
    kernal_len = int(chr_rowsum.shape[0] * kp)
    kernal_len = mkl if kernal_len < mkl else kernal_len
    if only_odd:
        kernal_len = (kernal_len // 2) * 2 + 1
    return kernal_len


###################
# Get intra-chromosome X-shape
def get_intra_x_kernal(cen_pos, mtx_size, k_len):   
    # Get intra-X-shape kernal.
    # By distance to line marked by telomere and centromere.
    # Points on line strength is 1, while k_len distance is 0.
    
    telo_point = [0, mtx_size]
    cen_point = [cen_pos, cen_pos]
    
    if cen_point[0] != telo_point[0]:
        k = (cen_point[1]  - telo_point[1]) / (cen_point[0] - telo_point[0])
        b = telo_point[1]
        
        def get_dis_to_line(x, y):
            return np.abs(k * x - y + b) / np.sqrt(1 + k ** 2)
    else:
        def get_dis_to_line(x, y):
            return np.abs(x - cen_point[0])
    
    intra_kernal = np.zeros((mtx_size, mtx_size))
    i_mtx, j_mtx = np.meshgrid(range(mtx_size), range(mtx_size))
    dis = get_dis_to_line(i_mtx, j_mtx)
    intra_kernal = 1 / k_len * (k_len - dis)
    intra_kernal[intra_kernal < 0] = 0

    intra_kernal = np.tril(intra_kernal).T + np.tril(intra_kernal, k=-1)
    return intra_kernal


def plot_intra_kernal_sep():
    mtx_size = 25
    k_len = 3
    cen_poses = range(0, 25, 3)
    
    n_fig = len(cen_poses)
    n_col = 3
    n_row = int(n_fig / n_col)
    
    fig = plt.figure(figsize=(5, 5.5))
    i_fig = 0
    for cen_pos in cen_poses:
        i_fig += 1
        
        ax = fig.add_subplot(n_row, n_col, i_fig)
        
        intra_kernal = get_intra_x_kernal(cen_pos, mtx_size, k_len)
        sns.heatmap(intra_kernal, ax=ax, cbar=False,
                    xticklabels=False, yticklabels=False)
        ax.set_title(f'Cen. pos. {int(np.round(cen_pos / (mtx_size - 1), 2) * 100)}%')
        ax.axis('equal')
    
    plt.tight_layout()
    plt.show()
    plt.close()
# plot_intra_kernal_sep()


def plot_intra_kernal_all():
    from conformation.src.S3_anc_contact.U1_plot_anc_con import plot_heatmap_axis
    
    mtx_size = 50
    k_len = 3
    cen_poses = range(10, 41, 10)
    
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111)
    
    for cen_pos in cen_poses:
        intra_kernal = get_intra_x_kernal(cen_pos, mtx_size, k_len)
        intra_kernal[intra_kernal == 0] = np.nan
    
        sns.heatmap(intra_kernal, cbar=False,
                    xticklabels=False, yticklabels=False,
                    cmap='Reds', ax=ax, alpha=1)
    plt.axis('equal')
    plot_heatmap_axis(ax, color='black')
    
    xs, xe = ax.get_xlim()
    ys, ye = ax.get_ylim()
    x = [xs, xe]
    y = [ye, ys]
    ax.plot(x,y, color='black')
    
    plt.tight_layout()
    plt.show()
    plt.close()
# plot_intra_kernal_all()

############
# Get intra-chromosome x strength
intra_x_kp = .05
intra_x_mkl = 3
def get_intra_x(chr_mtx):
    kernal_len = get_kernal_len(chr_mtx, intra_x_kp, intra_x_mkl)
    
    chr_intra_x = np.zeros(chr_mtx.shape[0])
    for idx in range(chr_mtx.shape[0]):
        intra_x_kernal = get_intra_x_kernal(idx, chr_mtx.shape[0],
                                            kernal_len)
        idx_intra_x = np.multiply(chr_mtx.matrix, intra_x_kernal)
        idx_intra_x = np.sum(idx_intra_x) / np.sum(intra_x_kernal)
        chr_intra_x[idx] = idx_intra_x
    return chr_intra_x


intra_x_name = 'intra_x'
def main_get_intra_x(input_mtx):
    intra_x = np.zeros(input_mtx.shape[1])
    for chro, chr_mtx in input_mtx.yield_by_chro(only_intra=True):
        chr_intra_x = get_intra_x(chr_mtx)
        
        chr_sep = input_mtx.row_window.chr_seps[chro]
        intra_x[chr_sep[0]: chr_sep[1] + 1] = chr_intra_x
    intra_x = pd.DataFrame({intra_x_name: intra_x})
    intra_x = Values(intra_x, input_mtx.row_window)
    return intra_x


############
# Get regions
def mark_anchor_state(intra_x_arr, chr_anchor_regions, update_idxes=None):
    marked_cols = ['cen_intrax']
    for markes_col in marked_cols:
        if markes_col not in chr_anchor_regions.columns:
            chr_anchor_regions[markes_col] = np.nan
    
    update_idxes = chr_anchor_regions.index if update_idxes is None else update_idxes
    for region_idx in update_idxes:
        chr_anchor_region = chr_anchor_regions.loc[region_idx, :]
        cen = int(chr_anchor_region['cen'])
        chr_anchor_regions.loc[region_idx, 'cen_intrax'] = intra_x_arr[cen]
        
    return chr_anchor_regions


best_value_col = 'cen_intrax'
def mark_best_regions(chr_anchor_regions):
    chr_anchor_regions['type'] = 'normal'
    chr_anchor_regions = chr_anchor_regions.sort_values(best_value_col, ascending=False)
    best_region_idx = chr_anchor_regions.index[0]
    chr_anchor_regions.loc[best_region_idx, 'type'] = 'used'
    return chr_anchor_regions


search_per = .05
def find_anchors_by_intra_x(intra_x):
    anchor_regions = []
    for _, chr_intra_x in intra_x.yield_by_chro():
        chr_intra_x_arr = np.squeeze(np.array(chr_intra_x.values))
        chr_intra_x_arr[chr_intra_x_arr < 0] = 0
        
        chr_anchor_regions = get_anchor_regions(chr_intra_x_arr, search_per)
        chr_anchor_regions = mark_anchor_state(chr_intra_x_arr, chr_anchor_regions)
        chr_anchor_regions = mark_best_regions(chr_anchor_regions)
        
        chr_anchor_region = reindex_n_add_cs_ce(chr_anchor_regions, chr_intra_x)
        anchor_regions.append(chr_anchor_region)
    
    anchor_regions = pd.concat(anchor_regions, axis=0, ignore_index=True)
    anchor_regions = Anchors(anchor_regions, intra_x.gen_index, keep_first=False)
    return anchor_regions


##################
# Main function for denoise
def main_find_anchors_by_x(sps_map, do_plot=False, fig_outfile=None, return_map_hist_for_plot=False):
    sps_map = sps_map.to_dense()
    sps_map.to_all(inplace=True)

    # high_map = keep_high_cons(sps_map, plot_hist=False)
    intra_x = main_get_intra_x(sps_map)
    anchor_regions = find_anchors_by_intra_x(intra_x)
    
    plot_maps = {'Filt': sps_map}
    value_dicts = {'Intra x': intra_x}
    if do_plot or fig_outfile is not None:
        plot_anchors(plot_maps,
                     value_dicts,
                     anchor_regions, 
                     fig_outfile)
    
    if not return_map_hist_for_plot:
        return anchor_regions
    else:
        return anchor_regions, plot_maps, value_dicts


# if __name__ == '__main__':
#     data_dir = "/data/home/cheyizhuo/project/centr_loc/data/oe_map"
#     file_list = ['Saccharomyces_cerevisiae',
#                  'Pbr', 'HN1']
#     # file_list = ['Acropora_millepora']
    
#     ###########
#     from iutils.read_matrix import read_hic_files
#     for hic_struct in read_hic_files(data_dir, file_list,
#                                      mtx_suffix='.filt_ode.mtx'):
#         new_mtx = main_find_anchors_by_x(hic_struct['sps_mtx'], do_plot=True)
