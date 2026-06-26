import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import manifold
import matplotlib as mpl
from sklearn.cluster import DBSCAN

import sys
sys.path.append("/data/home/cheyizhuo/project/centr_loc/mycode/confor/")
from confor_class import ConforPara, intra_para_file, inter_para_file, to_sqaure
from src.S4_pat_mining.S1_tsne_trans import intra_tsne_file, inter_tsne_file, inter_sub_para_file, inter_sub_tsne_file
from src.S4_pat_mining.S2_cluster import intra_clu_file, inter_clu_file, tsne_scatter_set


def plot_ave_maps_of_clus(para, tsne_vs, clu, is_intra=True):
    for one_clu in np.unique(clu):
        if one_clu < 0:
            continue
        print('#################\n\n')
        clu_vs = pd.DataFrame(para.vs[clu == one_clu])
        clu_vs = pd.melt(clu_vs, var_name='V', value_name='value')
        sns.boxplot(data=clu_vs, x='V', y='value')
        plt.show()
        plt.close()
        
        ave_xs = np.mean(para.xs[clu == one_clu], axis=0)
        ave_xs = to_sqaure(ave_xs)
        sns.heatmap(ave_xs, cmap='vlag', center=0)
        plt.show()
        plt.close()
        
        n_plot_x = 5
        plot_x_idxes = np.random.choice(np.where(clu == one_clu)[0], n_plot_x)
        
        ax = tsne_scatter_set(tsne_vs, color='lightgray', is_intra=is_intra)
        clu_tsne_vs = tsne_vs[clu == one_clu]
        tsne_scatter_set(clu_tsne_vs, color='maroon', input_ax=ax, 
                         is_intra=is_intra)
        tsne_scatter_set(tsne_vs[plot_x_idxes, :], color='black', 
                         input_ax=ax, s=5)
        plt.show()
        plt.close()
        
        for plot_x_idx in plot_x_idxes:
            print(para.names[plot_x_idx])
            plot_x = to_sqaure(para.xs[plot_x_idx, :])
            sns.heatmap(plot_x, cmap='vlag', center=0)
            plt.show()
            plt.close()


if __name__ == '__main__':
    intra_para = ConforPara(intra_para_file)
    intra_para.remove_zero_w_samples()
    intra_clu = np.load(intra_clu_file)
    intra_tsne_vs = np.load(intra_tsne_file)
    plot_ave_maps_of_clus(intra_para, intra_tsne_vs, intra_clu, is_intra=True)
    
    # inter_para = ConforPara(inter_para_file)
    # inter_para.remove_zero_w_samples()
    # inter_clu = np.load(inter_clu_file)
    # inter_tsne_vs = np.load(inter_tsne_file)
    # plot_ave_maps_of_clus(inter_para, inter_tsne_vs, inter_clu, is_intra=False)
    
    # inter_para = ConforPara(inter_sub_para_file)
    # inter_para.remove_zero_w_samples()
    # inter_clu = np.load(inter_clu_file)
    # inter_tsne_vs = np.load(inter_sub_tsne_file)
    # plot_ave_maps_of_clus(inter_para, inter_tsne_vs, inter_clu, is_intra=False)
