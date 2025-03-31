import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import skimage.restoration as restoration
from scipy.ndimage import uniform_filter, median_filter

import sys
from new_hic_class import ArrayHiCMtx, concat_mtxes


##################
# Denoise methods
def low_pass_filter(matrix, sigma_ratio=.2):
    def gaussianLP(sigma_ratio, imgShape):
        sigma = int(np.sqrt(np.min(imgShape) * sigma_ratio))
        sigma = 1 if sigma < 1 else sigma
        base = np.zeros(imgShape[:2])
        rows, cols = imgShape[:2] 
        center = (rows/2, cols/2)
        for x in range(cols):
            for y in range(rows):
                distance = np.linalg.norm((y - center[0], x - center[1]))
                base[y,x] = np.exp((-distance ** 2) / (2 * (sigma ** 2))) 
        return base
    
    # heatmap_plot(ArrayHiCMtx(matrix, row_window=row_window))
    
    f = np.fft.fft2(matrix)
    fshift = np.fft.fftshift(f)
    fshift_filtered = fshift * gaussianLP(sigma_ratio, matrix.shape)
    ffiltered = np.fft.ifftshift(fshift_filtered)
    new_matrix = np.abs(np.fft.ifft2(ffiltered))

    return new_matrix


def denoise_one_method(sps_mtx, mode='wavelet', 
                       sigma=None, spatial_sigma_ratio=None):
    dense_mtx = sps_mtx.to_dense()
    dense_arr = dense_mtx.matrix
    
    if mode =='wavelet': 
        sigma = 0.2 if sigma is None else sigma
        dense_arr = restoration.denoise_wavelet(dense_arr,
                                                wavelet='db3',
                                                method="VisuShrink")
    elif mode == 'median':
        spatial_sigma_ratio = .1 if spatial_sigma_ratio is None else spatial_sigma_ratio
        size = int(np.mean(np.array(list(sps_mtx.row_window.chr_lens.values())) * spatial_sigma_ratio))
        size = 2 if size < 2 else size
        dense_arr = median_filter(dense_arr, size=size, mode='reflect')
    elif mode == 'mean':
        spatial_sigma_ratio = .1 if spatial_sigma_ratio is None else spatial_sigma_ratio
        size = int(np.mean(np.array(list(sps_mtx.row_window.chr_lens.values())) * spatial_sigma_ratio))
        dense_arr = uniform_filter(dense_arr, size=size,
                                    mode='reflect')
    elif mode == 'low-pass':
        spatial_sigma_ratio = 10 if spatial_sigma_ratio is None else spatial_sigma_ratio
        dense_arr = low_pass_filter(dense_arr, sigma_ratio=spatial_sigma_ratio)
    elif mode == 'bilateral':
        sigma = 0.05 if sigma is None else sigma
        spatial_sigma_ratio = 10 if spatial_sigma_ratio is None else spatial_sigma_ratio
        # spatial_sigma = int(min(np.array(sps_mtx.shape) * spatial_sigma_ratio))
        dense_arr =restoration.denoise_bilateral(dense_arr,
                                                 sigma_color=sigma, 
                                                 sigma_spatial=spatial_sigma_ratio,
                                                 mode='reflect')
    else:
        raise ValueError('Unrecognized denoise mode.')

    dense_mtx = ArrayHiCMtx(dense_arr, copy_mtx_paras=dense_mtx)
    return dense_mtx.to_sps()


def used_denoise(sps_mtx):
    new_matrix = denoise_one_method(sps_mtx, 'wavelet', None, None)
    new_matrix = denoise_one_method(new_matrix, 'median', None, .01)
    return new_matrix


##################
# Main function for denoise
default_vmax_per = 98
def tid_off_outliers(sps_map, vamx_per=default_vmax_per):
    sps_map = sps_map.to_dense()
    
    vmax = np.percentile(sps_map.matrix, vamx_per)
    vmin = np.percentile(sps_map.matrix, 100 - vamx_per)
    sps_map.matrix[sps_map.matrix > vmax] = vmax
    sps_map.matrix[sps_map.matrix < vmin] = vmin
    return sps_map


def main_denoise(sps_map):
    sps_map = sps_map.to_dense()
    sps_map.to_all(inplace=True)
    
    sps_map = tid_off_outliers(sps_map)

    new_chr_mtxes = []
    for _, _, chr_mtx in sps_map.yield_by_chro():
        new_matrix = used_denoise(chr_mtx)
        new_chr_mtxes.append(new_matrix)
    
    new_mtx = concat_mtxes(new_chr_mtxes, copy_mtx_paras=sps_map)
    return new_mtx.to_dense()


##################
# Plot for comparing denoising method
# plot_modes = ['mean', 'median', 'bilateral', 'low-pass', 'wavelet']
# sigmas = [None, None, 1, None, None]
# sp_sigmas = [.05, .05, .8, .4, None]
plot_modes = ['wavelet', 'used']
sigmas = [None, None]
sp_sigmas = [None, None]
def plot_denoise(sps_mtx, spe_name, output=None, n_chro=4, intra=True):
    from iutils.plot_heatmap import heatmap_plot
    from oe_map.S4_norm_value import norm_de_mtx
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
        
        chr_raw_mtx = tid_off_outliers(chr_raw_mtx)
        
        heatmap_plot(norm_de_mtx(chr_raw_mtx), input_ax=ax)
        if i_chro == 1:
            ax.set_title('Raw')
        ax.set_ylabel(f'{chro.replace("scaffold_", "")}')
        
        i_plot = 0
        for plot_mode, sigma, sp_sigma in zip(plot_modes, sigmas, sp_sigmas):
            i_plot += 1
            
            if plot_mode == 'used':
                denoised_mtx = used_denoise(chr_raw_mtx)
            else:
                denoised_mtx = denoise_one_method(chr_raw_mtx, mode=plot_mode, 
                                                  sigma=sigma, 
                                                  spatial_sigma_ratio=sp_sigma)
                
            ax = fig.add_subplot(n_rows, n_cols, (i_chro - 1) * n_cols + 1 + i_plot)
            heatmap_plot(norm_de_mtx(denoised_mtx), input_ax=ax)
            
            if i_chro == 1:
                ax.set_title(plot_mode)
            # print(plot_mode)
    
    # chro1 = plot_chros[0]
    # for chro in plot_chros[1:]:
    #     i_chro += 1

    #     ax = fig.add_subplot(n_rows, n_cols, (i_chro - 1) * n_cols + 1)
    #     chr_raw_mtx = sps_mtx.get_region_mtx(chro1, chro2=chro)
    #     heatmap_plot(chr_raw_mtx, input_ax=ax)
    #     ax.set_ylabel(f'{chro.replace("scaffold_", "")} vs {chro1.replace("scaffold_", "")}')
        
    #     i_plot = 0
    #     for plot_mode, sigma, sp_sigma in zip(plot_modes, sigmas, sp_sigmas):
    #         i_plot += 1
    #         if use_main_denoise_func:
    #             denoised_mtx = main_denoise(chr_raw_mtx)
    #         else:
    #             denoised_mtx = denoise(chr_raw_mtx, mode=plot_mode, 
    #                                 sigma=sigma, spatial_sigma_ratio=sp_sigma)
    #         ax = fig.add_subplot(n_rows, n_cols, (i_chro - 1) * n_cols + 1 + i_plot)
    #         heatmap_plot(denoised_mtx, input_ax=ax)
    
    plt.suptitle(spe_name, size=16)
    plt.tight_layout()
    if output is None:
        plt.show()
    else:
        plt.savefig(f'{output}.denoise_plot.png')
    plt.close()
