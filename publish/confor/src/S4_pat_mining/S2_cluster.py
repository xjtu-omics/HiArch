import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import manifold
import matplotlib as mpl
from scipy.io import loadmat, savemat
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

import sys
sys.path.append("/data/home/cheyizhuo/project/centr_loc/mycode/confor/")
import confor_class as cc
import src.S4_pat_mining.S1_tsne_trans as tt
sys.path.append("/data/home/cheyizhuo/project/centr_loc/mycode/")
from iutils.plot_heatmap import remove_margin
import iutils.save_fig as sf

intra_clu_file = f'{cc.intra_dir}/clu.npy'
inter_clu_file = f'{cc.inter_dir}/clu.npy'


# eps_inter, ms_inter = .8, 580
eps_inter, ms_inter = 1, 620 # for diag 1
# eps_inter, ms_inter = 1.2, 120
eps_intra, ms_intra = 1.2, 130
def clustering(tsne_vs, eps, min_samples):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    clu = model.fit_predict(tsne_vs)
    return clu


def get_clu_ave_maps(para, clu):
    ave_plots = {}
    for one_clu in np.unique(clu):
        if one_clu < 0:
            continue
        ave_xs = np.mean(para.xs[clu == one_clu], axis=0)
        # ave_xs = to_sqaure(ave_xs)
        ave_plots[one_clu] = ave_xs
    return ave_plots


#########
# Plot
colors = sns.palettes.color_palette('Dark2', n_colors=7)
colors += [(104 / 255, 36 / 255, 135 / 255),
           (204 / 255, 51 / 255, 0 / 255)]

def tsne_scatter_set(tsne_vs, clu=None, input_ax=None, 
                     is_intra=True, label_name='T-SNE', **kwargs):
    if input_ax is None:
        fig = plt.figure(figsize=(4, 4), dpi=100)
        ax = fig.add_subplot(111)
    else:
        ax = input_ax
    
    s = 10 if is_intra else 2
    kwargs['s'] = kwargs['s'] if 's' in kwargs else s
        
    sns.scatterplot(x=tsne_vs[:, 0], y=tsne_vs[:, 1], 
                    hue=clu, **kwargs)
    sns.despine(trim=True)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_xlabel(f'{label_name} axis-1')
    # ax.set_ylabel(f'{label_name} axis-2')
    ax.set_axis_off()
    return ax


def scatter_clu(tsne_vs, clu, is_intra=True, label_name='T-SNE'):
    clu_only_idxes = np.where(clu >= 0)
    clu_only_tsne_vs = tsne_vs[clu_only_idxes]
    clu_only_clu = clu[clu_only_idxes]
    
    uni_clu = np.unique(clu_only_clu)
    
    ax = tsne_scatter_set(tsne_vs, is_intra=is_intra, 
                          color='lightgray', label_name=label_name)
    palette = {uni_clu[i_map]: colors[i_map] 
               for i_map in range(len(uni_clu))}
    tsne_scatter_set(clu_only_tsne_vs, clu_only_clu,
                     is_intra=is_intra, palette=palette, input_ax=ax,
                     legend=False, label_name=label_name)
    # plt.show()
    # plt.close()
    return colors


def to_sqaure(arr):
    sqaure_size = np.sqrt(arr.size)
    if sqaure_size % 1 != 0:
        raise ValueError(f'Array size {arr.size} is not square.')
    sqaure_size = int(sqaure_size)
    return arr.reshape((sqaure_size, sqaure_size))


def plot_kernal(kernal, line_color=None, dpi=100, large_ratio=15):
    line_width = 20
    
    ax = sns.heatmap(to_sqaure(kernal),
                     cmap='vlag', center=0, vmax=1.5, vmin=-1.5,
                     cbar=False)
    
    ori_xlim = ax.get_xlim()
    ori_ylim = sorted(list(ax.get_ylim()))
    
    if line_color is not None:
        line_pos_change = line_width / (ori_xlim[1] - ori_xlim[0]) / 1.5
        plt.axvline(x=ori_xlim[0] - line_pos_change,
                    color=line_color, lw=line_width)
        plt.axvline(x=ori_xlim[1] + line_pos_change,
                    color=line_color, lw=line_width)
        plt.axhline(y=ori_ylim[0] - line_pos_change,
                    color=line_color, lw=line_width)
        plt.axhline(y=ori_ylim[1] + line_pos_change,
                    color=line_color, lw=line_width)
        
        ax.set_xlim([ori_xlim[0] - line_pos_change,
                 ori_xlim[1] + line_pos_change])
        ax.set_ylim([ori_ylim[1] + line_pos_change,
                     ori_ylim[0] - line_pos_change])
    
    xlim = ax.get_xlim()
    height = np.abs(xlim[1] - xlim[0])
    ylim = ax.get_ylim()
    width = np.abs(ylim[1] - ylim[0])  
    
    ax.figure.set_size_inches(width/dpi * large_ratio, 
                              height/dpi * large_ratio)
    ax.figure.set_dpi(dpi)
    
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    plt.axis('off')
    plt.margins(0, 0)
    
    ax.figure.patch.set_alpha(0.)
    
    # plt.show()
    # plt.close()


# fig_out_dir = f'{sf.default_out_dir}/figure2_sub'
fig_out_dir = f'{sf.default_out_dir}/figureS_GF_repro_sub'
def main_plot_clu(para, tsne_vs, clu, kept_map_idxes, fig_output=None, is_intra=True):
    tsne_scatter_set(tsne_vs, is_intra=is_intra, color='lightgray')
    plt.show()
    plt.close()
    
    colors = scatter_clu(tsne_vs, clu, is_intra)
    if fig_output is not None:
        sf.save_plot_png(f'{fig_output}_tsne.png', fig_out_dir)
    else:
        plt.show()
        plt.close()
    
    ave_maps = get_clu_ave_maps(para, clu)
    for i_map in range(len(ave_maps)):
        if i_map in kept_map_idxes:
            print('## Keep this')
        color = colors[i_map]
        plot_kernal(list(ave_maps.values())[i_map], color)
        if fig_output is not None:
            sf.save_plot_jpg(f'{fig_output}_kernal{i_map}.txt', fig_out_dir)
        else:
            plt.show()
            plt.close()


if __name__ == '__main__':
    save_file = True
    
    ######
    # Intra
    intra_para = cc.ConforPara(cc.intra_para_file)
    intra_tsne_vs = np.load(tt.intra_tsne_file)
    
    if save_file:
        intra_clu = clustering(intra_tsne_vs, eps_intra, ms_intra)
        np.save(intra_clu_file, intra_clu)
    intra_clu = np.load(intra_clu_file)
    
    kept_map_idxes = [0, 1, 2, 3, 4]
    # kept_map_idxes = None
    if save_file:
        ave_maps = get_clu_ave_maps(intra_para, intra_clu)
        if kept_map_idxes is None:
            kept_map_idxes = [i for i in range(len(ave_maps))]
        savemat(cc.intra_ave_map_file, {str(i): ave_maps[i] for i in kept_map_idxes})
    
    main_plot_clu(intra_para, intra_tsne_vs, 
                  intra_clu, kept_map_idxes,
                  fig_output='Intra',
                #   fig_output=None,
                  is_intra=True)
    
    ######
    # Inter
    # inter_para = cc.ConforPara(cc.inter_para_file)
    # inter_tsne_vs = np.load(tt.inter_tsne_file)
    
    # # inter_para = cc.ConforPara(tt.inter_sub_para_file)
    # # inter_tsne_vs = np.load(tt.inter_sub_tsne_file)

    # if save_file:
    #     inter_clu = clustering(inter_tsne_vs, eps_inter, ms_inter)
    #     np.save(inter_clu_file, inter_clu)
    # inter_clu = np.load(inter_clu_file)
    
    # # kept_map_idxes = [0, 1, 2, 3, 4]
    # kept_map_idxes = None
    # if save_file:
    #     ave_maps = get_clu_ave_maps(inter_para, inter_clu)
    #     if kept_map_idxes is None:
    #         kept_map_idxes = [i for i in range(len(ave_maps))]
    #     savemat(cc.inter_ave_map_file, {str(i): ave_maps[i] for i in kept_map_idxes})
    
    # main_plot_clu(inter_para, inter_tsne_vs, inter_clu, kept_map_idxes, 
    #               fig_output='Inter',
    #             #   fig_output=None,
    #               is_intra=False)
