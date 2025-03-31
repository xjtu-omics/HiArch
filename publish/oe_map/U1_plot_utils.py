import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import sys
from iutils.plot_heatmap import heatmap_plot, get_vmax_vmin


def heatmap_diff_ob_methods(sps_mtx, centr_loc=None, n_chro=5, out_file=None):
    fig = plt.figure(figsize=(10, 10))
    
    i_chro = 1
    if centr_loc is not None:
        centr_chros = set(list(centr_loc['chr'].unique()))
        mtx_chros = set(sps_mtx.row_window.chros)
        chros = list(centr_chros & mtx_chros)
    else:
        chros = sps_mtx.row_window.chros
        
    for chro in chros:
        if i_chro > n_chro:
            break
        
        ax = fig.add_subplot(n_chro, 5, (i_chro - 1) * 5 + 1)
        chr_sps_mtx = sps_mtx.get_region_mtx(chro).to_dense()
        heatmap_plot(chr_sps_mtx, centr_loc, input_ax=ax, cmap='vlag', 
                     show_mtx_size=False, no_margin=False)
        ax.set_ylabel(chro)
        if i_chro == 1:
            ax.set_title('Raw HiC map')
        
        fit_ode_mtx, fit_counts = chr_sps_mtx.obs_d_exp(mode='fit', return_exp_counts=True)
        diag_ode_mtx, diag_counts = chr_sps_mtx.obs_d_exp(mode='diag_mean', return_exp_counts=True,
                                                         counts_must_decrese=False)
        sgd_ode_mtx, sgd_counts = chr_sps_mtx.obs_d_exp(mode='sgd_mean', return_exp_counts=True)
        
        ax = fig.add_subplot(n_chro, 5, (i_chro - 1) * 5 + 2)
        dises = np.arange(chr_sps_mtx.shape[0])
        sns.scatterplot(x=dises, y=diag_counts, ax=ax)
        sns.lineplot(x=dises, y=fit_counts, ax=ax)
        sns.lineplot(x=dises, y=sgd_counts, ax=ax)
        ax.set_yscale('log')
        ax.set_xscale('log')
        if i_chro == 1:
            ax.set_title(r"Counts vs Distance")
        ax.set_ylabel('Hi-C counts')
        if i_chro == n_chro:
            ax.set_xlabel('Distance (bins)')
        
        ax = fig.add_subplot(n_chro, 5, (i_chro - 1) * 5 + 3)
        heatmap_plot(fit_ode_mtx, centr_loc, input_ax=ax, 
                     cmap='vlag', show_mtx_size=False)
        if i_chro == 1:
            ax.set_title('Obs/exp map by\n' + r"$\bf{linear\ fitting}$")
        
        ax = fig.add_subplot(n_chro, 5, (i_chro - 1) * 5 + 4)
        heatmap_plot(diag_ode_mtx, centr_loc, input_ax=ax, 
                     cmap='vlag', show_mtx_size=False)
        if i_chro == 1:
            ax.set_title('Obs/exp map by\n' + r"$\bf{diagonal\ mean}$")
        
        ax = fig.add_subplot(n_chro, 5, (i_chro - 1) * 5 + 5)
        heatmap_plot(sgd_ode_mtx, centr_loc, input_ax=ax, 
                     cmap='vlag', show_mtx_size=False)
        if i_chro == 1:
            ax.set_title('Obs/exp map by\n' + r"$\bf{SGD\ mean}$")
        
        i_chro += 1
    
    plt.tight_layout()
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)
    plt.close()


default_n_sep = 16
def histplot_oe_values(oe_mtx, out_file=None):
    oe_mtx = oe_mtx.to_dense()

    vmax, vmin = get_vmax_vmin(oe_mtx)
    
    data_arr = oe_mtx.matrix.reshape((-1,))
    data_arr[data_arr > vmax] = vmax
    data_arr[data_arr < vmin] = vmin
    
    sns.histplot(x=data_arr,
                 stat='probability',
                 bins=default_n_sep)
    
    # plt.xlim([vmin, vmax])
    plt.axvline(x=0, color='maroon')
    plt.grid(axis='y')
    
    plt.title(f'Mean: {np.mean(oe_mtx.matrix.reshape((-1,)))}')
    
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)
    plt.close()
