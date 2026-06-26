import os
import sys

from iutils.read_matrix import Anchors, get_species_names, read_matrix
import iutils.read_matrix as utl
from confor.confor_class import ConforData
import numpy as np
from scipy.io import savemat
import seaborn as sns
import matplotlib.pyplot as plt
from confor.src.S2_to_train_mtx.preprocess import norm_mtx, pool_mtx, symmetry, remove_circle



def yield_mtx(mtx, centers, out_type=None, n_chro=None):
    i_chro = 0
    for chro1, chro2, chr_mtx in mtx.yield_by_chro():
        if out_type is not None:
            if out_type == 'intra' and chro1 != chro2:
                continue
            elif out_type == 'inter' and chro1 == chro2:
                continue
            elif out_type != 'intra' and out_type != 'inter':
                raise ValueError('Unrecognized out_type. Shoule be intra or inter.')
        
        i_chro += 1
        if n_chro is not None:
            if i_chro > n_chro:
                break

        chr_mtx.to_all(inplace=True)
        chr_mtx = np.array(chr_mtx.matrix.todense())

        print(chro1)
        print(centers['chr'].unique())
        center_idx1 = centers.get_used_chro_loc(chro1, chr_idx=True)
        print(center_idx1)
        center_idx2 = centers.get_used_chro_loc(chro2, chr_idx=True)
        
        yield chro1, chro2, chr_mtx, center_idx1, center_idx2


##################
# Plot
def plot_pre_mtx(maps, out_file=None, vmaxs=None, col_wrap=2):
    vmaxs = {} if vmaxs is None else vmaxs
    
    fig_size = 2
    n_map = len(maps)
    n_row = int(np.ceil(n_map / col_wrap))
    fig = plt.figure(figsize=(n_row * fig_size,
                              col_wrap * fig_size))
    i_ax = 0
    for map_name in maps:
        i_ax += 1
        ax = fig.add_subplot(n_row, col_wrap, i_ax)
        
        one_map = maps[map_name]
        vmax = np.percentile(one_map, 98) if map_name not in vmaxs else vmaxs[map_name]

        heatmap_kwargs = {'cmap': 'vlag', 'vmax': vmax, 'vmin': -vmax, 'center': 0, 
                          'xticklabels': False, 'yticklabels': False, 'cbar': False}
        
        sns.heatmap(one_map, ax=ax, **heatmap_kwargs)
        
        ax.set_title(map_name)

    if out_file is not None:
        plt.savefig(out_file)
    else:
        plt.show()
    plt.close()


def to_train_mtx(sps_file, window_file, center_file, spe_name,
                 do_plot=False, fig_out_dir=None,
                 n_plot_intra=2, n_plot_inter=3, is_circle=False):
    mtx = read_matrix(sps_file, window_file)
    mtx.to_all(inplace=True)
    centers = Anchors(center_file, window_file)

    X, X_isym, theName = [], [], []
    X_diag, X_isym_diag, theName_diag = [], [], []
    i_intra, i_inter = 0, 0
    for chro1, chro2, chr_mtx, c1, c2 in yield_mtx(mtx, centers):
        intra = True if chro1 == chro2 else False
        
        pooled_img = pool_mtx(chr_mtx, c1, c2)
        norm_img = norm_mtx(pooled_img)
        sym_img = symmetry(norm_img, intra=intra)
        # sym_img = remove_outliers(sym_img)
        # print(sym_img)
        
        if is_circle and intra:
            sym_img = remove_circle(sym_img)
            print(sym_img)
        
        now_name = f'{spe_name}:{chro1}' if intra else f'{spe_name}:{chro1}:{chro2}'
        print(now_name)

        save_name = now_name.ljust(100)
        if intra:
            X_diag.append(sym_img)
            X_isym_diag.append(norm_img)
            theName_diag.append(np.array([save_name]))
        else:
            X.append(sym_img)
            X_isym.append(norm_img)
            theName.append(np.array([save_name]))

        if do_plot:
            if intra:
                i_intra += 1
                if i_intra > n_plot_intra:
                    continue
            else:
                i_inter += 1
                if i_inter > n_plot_inter:
                    continue
            os.system(f'mkdir -p {fig_out_dir}/{spe_name}')
            fig_outfile = f'{fig_out_dir}/{spe_name}/{now_name}.pre_learn.png' if fig_out_dir is not None else None
            plot_pre_mtx({'Raw': chr_mtx, 'Pooled': pooled_img,
                            'Norm': norm_img, 'Sym': sym_img},
                            vmaxs={'Norm': 2, 'Sym': 2},
                            out_file=fig_outfile)
    
    intra_data = ConforData([theName_diag, X_diag, X_isym_diag])
    inter_data = ConforData([theName, X, X_isym])
    return intra_data, inter_data





