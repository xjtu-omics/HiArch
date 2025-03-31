import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import sys
import plot_utils.my_polar as mp
from iutils.plot_heatmap import heatmap_plot, plot_centr_loc, get_fontsize_ratio
from confor.utils import Confor, default_con_col, confor_suffix, filt_mtx_suffix, intra_ave_order, inter_ave_order
from confor.confor_class import get_x, to_sqaure
from new_hic_class import ArrayHiCMtx


n_intra_plot = 3
n_inter_plot = 4

def ran_get_plot_chros(input_data, plot_all=False):
    chros = input_data.row_window.chros
    if plot_all:
        plot_intra_chros = chros
    else:
        if len(chros) > n_intra_plot:
            plot_intra_chros = np.random.choice(chros, n_intra_plot, replace=False)
        else:
            plot_intra_chros = chros
    
    if len(chros) <= 1:
        plot_inter_pairs = []
    else:
        from itertools import combinations
        inter_chro_pairs = [f'{chr1}-{chr2}' for chr1, chr2 in combinations(list(chros), 2)]
        if plot_all:
            plot_inter_pairs = inter_chro_pairs
        else:
            if len(inter_chro_pairs) > n_inter_plot:
                plot_inter_pairs = np.random.choice(inter_chro_pairs, n_inter_plot, replace=False)
            else:
                plot_inter_pairs = inter_chro_pairs
    return plot_intra_chros, plot_inter_pairs


###############
# Plot
# TODO: Set vmax and vmin for xs plot
def radar_boxplot_confor(confor, type_order=None, title=None, input_ax=None):
    xname = 'type'
    yname = default_con_col
    
    confor = confor.values
    confor.loc[:, yname] = confor[yname].fillna(0)
    
    if input_ax is None:
        fig = plt.figure(figsize=(3, 4))
        ax = fig.add_subplot(111, polar=True)
    else:
        ax = input_ax
    
    ax = mp.polar_boxplot(confor[xname], confor[yname], cate_order=type_order,
                          ax=ax, plot_median=True, outliers=False)
    # ax = mp.polar_boxplot(confor[xname], confor[yname],
    #                       ax=ax, plot_median=True, outliers=False)
    fontsize_ratio = get_fontsize_ratio(ax, axsize_to_fontsize_ratio=2.5)
    
    confor_median = confor.groupby(xname).agg({yname: 'median'})
    median_x = [ax.cate_angles[cate] for cate in confor_median.index]
    median_y = np.array(confor_median)
    plt.polar(median_x, median_y, markeredgecolor="black", marker=".",
              markerfacecolor="white", linewidth=0, 
              markersize=10 * fontsize_ratio, zorder=3)
    
    # transferred_x = confor[xname].apply(lambda x: ax.cate_angles[x])
    # sns.swarmplot(x=transferred_x, y=confor[yname], ax=ax,
    #               hue=transferred_x, legend=False, palette='tab20',
    #               native_scale=True)
    
    xtick_labels = [t.replace('-', '-\n').capitalize() for t in ax.cates]
    # xtick_labels = [t for t in ax.cates]
    ax.set_xticks(ax.x_angles[:-1], xtick_labels, size=10 * fontsize_ratio)
    ax.set_ylabel('')
    ax.set_xlabel('')
    
    ylims, ylime = ax.get_ylim()
    # ylims = -3 if ylims > -3 else ylims
    ylims = -2
    ylime = 1.6 if ylime < 1.6 else ylime
    ax.set_ylim([ylims, ylime])
    yticks = [-1, 0, 1]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontdict={'fontsize': 9 * fontsize_ratio})
    special_mark_ytick = 0
    for ytick in yticks:
        color = 'maroon' if ytick == special_mark_ytick else 'gray'
        alpha = 1 if ytick == special_mark_ytick else .6
        mp.polar_axhline(ax=ax, y=ytick, 
                         color=color, lw=1, alpha=alpha,
                         zorder=2)
    ax.grid(axis='y')
    
    ax.set_title(title, size=12 * fontsize_ratio)
    
    if input_ax is None:
        # plt.tight_layout()
        plt.show()
        plt.close()


def plot_confor_chro(input_mtx, x, confor, type_order=None, cen_loc=None,
                     out_file=None, map_kwargs=None):
    fig_height_ratio = .005
    min_fig_height = 5

    n_col = 3
    gs = plt.GridSpec(1, n_col)
    
    fig_height = input_mtx.shape[0] * fig_height_ratio
    fig_height = fig_height if fig_height > min_fig_height else min_fig_height
    fig = plt.figure(figsize=(fig_height * n_col, fig_height))

    axes = {}
    #########
    # Heatmap plot map
    map_kwargs = {} if map_kwargs is None else map_kwargs
    
    ax = fig.add_subplot(gs[:, 0])
    axes['map'] = ax
    heatmap_plot(input_mtx, input_ax=ax, **map_kwargs)
    if cen_loc is not None:
        if cen_loc[0] is None and cen_loc[1] is None:
            pass
        elif cen_loc[0] is None:
            cen_loc[0] = 0
        elif cen_loc[1] is None:
            cen_loc[1] = 0
            
        sns.scatterplot(x=[cen_loc[1]], y=[cen_loc[0]], 
                        s=100, ax=ax, color='black')
    
    #########
    # Xs plot
    ax = fig.add_subplot(gs[:, 1])
    axes['x'] = ax
    x_mtx = ArrayHiCMtx(x, has_neg=True)
    heatmap_plot(x_mtx, input_ax=ax)
    
    #########
    # Radar plot types
    ax = fig.add_subplot(gs[:, 2], polar=True)
    axes['radar'] = ax
    radar_boxplot_confor(confor, input_ax=ax, type_order=type_order)
 
    plt.tight_layout()
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)
    plt.close()
    return fig, axes


def plot_confor(input_mtx, confor, cen_loc=None,
                out_file=None, map_kwargs=None):
    plot_confors = ['Intra-chro', 'Inter-chro']
    
    fig_height_ratio = .005
    min_fig_height = 5
    
    n_col = 3
    
    gs = plt.GridSpec(1, 3)
    
    fig_height = input_mtx.shape[0] * fig_height_ratio
    fig_height = fig_height if fig_height > min_fig_height else min_fig_height
    fig = plt.figure(figsize=(fig_height * n_col, fig_height))

    axes = {}
    #########
    # Heatmap plot map
    map_kwargs = {} if map_kwargs is None else map_kwargs
    
    ax = fig.add_subplot(gs[:, 0])
    axes['map'] = ax
    heatmap_plot(input_mtx, input_ax=ax, show_mtx_size=False, **map_kwargs)
    if cen_loc is not None:
        plot_centr_loc(cen_loc.keep_used_anchors(), ax, 
                    facecolor='black')
    
    #########
    # Plot confor
    i_plot = 0
    for plot_confor in plot_confors:
        if plot_confor == 'Intra-chro':
            sub_confor = confor.get_intra_sub()
            type_order = intra_ave_order
        elif plot_confor == 'Inter-chro':
            sub_confor = confor.get_intra_sub(intra=False)
            type_order = inter_ave_order
        else:
            sub_confor = confor
        
        if len(sub_confor.values) == 0:
            continue
        
        i_plot += 1
        ax = fig.add_subplot(gs[:, i_plot], polar=True)
        axes[f'radar_{plot_confor}'] = ax
        
        type_order = intra_ave_order if plot_confor == 'Intra-chro' else inter_ave_order
        radar_boxplot_confor(sub_confor, type_order=type_order, input_ax=ax)
    
    plt.tight_layout()
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)
    plt.close()
    return fig, axes


def main_plot(input_data, confor, species, output=None, plot_all=False):
    plot_intra_chros, plot_inter_pairs = ran_get_plot_chros(input_data, plot_all=plot_all)
    
    for chro in plot_intra_chros:
        chr_mtx = input_data.get_region_mtx(chro)
        
        xs = get_x(species, chro)
        if xs is None:
            continue
        xs = to_sqaure(xs)
        # xs = np.zeros((16, 16))
        
        chr_confor = confor[(confor['chr1'] == chro) & 
                            (confor['chr2'] == chro)]
        chr_confor = Confor(chr_confor)
        
        fig_outfile = f'{output}.{chro}.chr_confor.png' if output is not None else None
        plot_confor_chro(chr_mtx, xs, chr_confor, 
                         out_file=fig_outfile,
                         type_order=intra_ave_order)
    
    for chro1_chro2 in plot_inter_pairs:
        chro1, chro2 = chro1_chro2.split('-')
        chr_mtx = input_data.get_region_mtx(chro1, chro2=chro2)
        
        xs = get_x(species, chro1, chro2)
        if xs is None:
            continue
        xs = to_sqaure(xs)
        
        chr_confor = confor[(confor['chr1'] == chro1) & 
                            (confor['chr2'] == chro2)]
        chr_confor = Confor(chr_confor)
        
        fig_outfile = f'{output}.{chro1}_{chro2}.chr_confor.png' if output is not None else None
        plot_confor_chro(chr_mtx, xs, chr_confor, 
                         out_file=fig_outfile,
                         type_order=inter_ave_order)
    
    fig_outfile = f'{output}.confor.png' if output is not None else None
    plot_confor(input_data, confor, out_file=fig_outfile)

