import os
import numpy as np
import matplotlib.pyplot as plt

import sys

from new_hic_class import concat_mtxes
from oe_map.S3_denoise import denoise_one_method
from oe_map.S4_norm_value import norm_to_zero_mean, remove_outliers

##################
# filter methods
def used_filter(sps_mtx):
    filtered_mtx = denoise_one_method(sps_mtx, 'mean', None, .1)
    filtered_mtx = norm_to_zero_mean(filtered_mtx)
    return filtered_mtx


##################
# Main function for denoise
def main_filter(sps_map):
    sps_map = sps_map.to_dense()
    sps_map.to_all(inplace=True)

    new_chr_mtxes = []
    for _, _, chr_mtx in sps_map.yield_by_chro():
        new_matrix = used_filter(chr_mtx)
        new_chr_mtxes.append(new_matrix)
    
    new_mtx = concat_mtxes(new_chr_mtxes, copy_mtx_paras=sps_map)
    new_mtx = remove_outliers(new_mtx)
    return new_mtx.to_dense()


def filter_io(sps_file, index_file, output=None, fig_output=None, use_exist_filt=False):
    from iutils.read_matrix import read_matrix
    from iutils.plot_heatmap import heatmap_plot
    if os.path.exists(f'{output}.filt_ode.mtx') and use_exist_filt:
        filtered_mtx = read_matrix(f'{output}.filt_ode.mtx', index_file)
        filtered_mtx.to_all(inplace=True)
    else:
        if sps_file is None:
            raise ValueError('Plz input sparse matrix.')
        sps_mtx = read_matrix(sps_file, index_file)
        filtered_mtx = main_filter(sps_mtx)
        if output is not None:
            filtered_mtx.to_sps().to_output(f'{output}.filt_ode.mtx')
        if fig_output is not None:
            heatmap_plot(filtered_mtx, out_file=f'{fig_output}.filtered_map.png')
            print(f'Plotting filtered mtx DONE.')
    print(f'Filtering mtx DONE.')
    return filtered_mtx
    

##################
# Plot for comparing denoising method
plot_modes = ['mean', 'median']
sigmas = [None, None]
sp_sigmas = [.1, .1]
def plot_denoise(sps_mtx, spe_name, output=None, n_chro=4, intra=True):
    from iutils.plot_heatmap import heatmap_plot
    n_rows = n_chro * 2 - 1
    n_cols = len(plot_modes) + 1
    
    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
    i_chro = 0
    
    if intra:
        plot_chros = sps_mtx.row_window.chros[:n_chro]
    else:
        chro1 = sps_mtx.row_window.chros[0]
        chro1 = sps_mtx.row_window.chros[1:n_chro + 1]
    
    for chro in plot_chros:
        i_chro += 1
        
        ax = fig.add_subplot(n_rows, n_cols, (i_chro - 1) * n_cols + 1)
        
        if intra:
            chr_raw_mtx = sps_mtx.get_region_mtx(chro)
        else:
            chr_raw_mtx = sps_mtx.get_region_mtx(chro1, chro2=chro)
        
        # chr_raw_mtx = tid_off_outliers(chr_raw_mtx)
        
        heatmap_plot(chr_raw_mtx, input_ax=ax)
        if i_chro == 1:
            ax.set_title('Raw')
        ax.set_ylabel(f'{chro.replace("scaffold_", "")}')
        
        i_plot = 0
        for plot_mode, sigma, sp_sigma in zip(plot_modes, sigmas, sp_sigmas):
            i_plot += 1
            
            denoised_mtx = denoise_one_method(chr_raw_mtx, mode=plot_mode, 
                                                sigma=sigma, 
                                                spatial_sigma_ratio=sp_sigma)
            
            denoised_mtx = norm_to_zero_mean(denoised_mtx)
            denoised_mtx = remove_outliers(denoised_mtx)
                
            ax = fig.add_subplot(n_rows, n_cols, (i_chro - 1) * n_cols + 1 + i_plot)
            heatmap_plot(denoised_mtx, input_ax=ax)
            
            if i_chro == 1:
                ax.set_title(plot_mode)
            # print(plot_mode)
    
    plt.suptitle(spe_name, size=16)
    plt.tight_layout()
    if output is None:
        plt.show()
    else:
        plt.savefig(f'{output}.denoise_plot.png')
    plt.close()

