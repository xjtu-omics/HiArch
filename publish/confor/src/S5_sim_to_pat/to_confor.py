import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

import sys
from confor.confor_class import intra_ave_names, inter_ave_names, intra_ave_map_file, inter_ave_map_file
from confor.src.S2_to_train_mtx.x_size import X_SIZE
from confor.utils import confor_suffix


def get_n_norm_ave_maps(ave_maps):
    """ Normalize average maps.
        Aim: Sign not change.
    """
    norm_ave_maps = {}
    for clu in ave_maps:
        if clu.startswith('_'):
            continue
        ave_map = ave_maps[clu]
        
        # Norm ave map
        norm_ave_map = ave_map / np.sum(np.abs(ave_map))
        norm_ave_maps[clu] = norm_ave_map.reshape((-1,))
    return norm_ave_maps


def change_map_name(ave_maps, ave_names):
    u_names = [int(float(i)) for i in ave_maps if not i.startswith('_')]
    u_names.sort()
    new_ave_maps = {ave_names[i]: ave_maps[str(u_names[i])] for i in range(len(u_names))}
    return new_ave_maps


def to_confor(xs, ave_maps, do_trans=False):
    if xs.xs is None:
        return pd.DataFrame([])
    
    ave_maps_arr = np.vstack(list(ave_maps.values()))
    
    scores = np.matmul(xs.xs, ave_maps_arr.T)
    if do_trans:
        trans_idx = np.arange(X_SIZE ** 2).reshape((X_SIZE, X_SIZE)).T.reshape((-1,))
        x_trans = xs.xs[:, trans_idx]
        trans_scores = np.matmul(x_trans, ave_maps_arr.T)
        change_idx = scores < trans_scores
        scores[change_idx] = trans_scores[change_idx]
    
    scores = pd.DataFrame(scores, xs.out_names(), 
                          columns=list(ave_maps.keys()))
    scores.reset_index(inplace=True, names=['name'])
    
    scores['name'] = scores['name'].str.rstrip()
    a = scores['name'].str.split(':', expand=True)
    scores['species'] = a.iloc[:, 0]
    scores['chr1'] = a.iloc[:, 1]
    if a.shape[1] == 2:
        scores['chr2'] = a.iloc[:, 1]
    else:
        scores['chr2'] = a.iloc[:, 2]
    
    scores = pd.melt(scores, id_vars=['name', 'species', 'chr1', 'chr2'],
                     var_name='type', value_name='score')
    return scores


def out_confor(confor, out_dir):
    confor = confor.sort_values(['species', 'chr1', 'chr2'])
    for species in confor['species'].unique():
        print(species)
        spe_file = f'{out_dir}/{species}/{species}{confor_suffix}'
        if os.path.exists(f'{out_dir}/{species}'):
            spe_confor = confor[confor['species'] == species]
            spe_confor.to_csv(spe_file, sep="\t", index=False)


def main_to_confor(intra_xs, inter_xs, out_dir,
                   intra_ave_file=intra_ave_map_file,
                   inter_ave_file=inter_ave_map_file):
    ######
    # Intra
    intra_ave_maps = loadmat(intra_ave_file)
    intra_ave_maps = change_map_name(get_n_norm_ave_maps(intra_ave_maps), intra_ave_names)
    intra_confor = to_confor(intra_xs, intra_ave_maps)

    ######
    # Inter
    inter_ave_maps = loadmat(inter_ave_file)
    inter_ave_maps = change_map_name(get_n_norm_ave_maps(inter_ave_maps), inter_ave_names)
    inter_confor = to_confor(inter_xs, inter_ave_maps, do_trans=True)
    
    all_confor = pd.concat([intra_confor, inter_confor], axis=0)
    out_confor(all_confor, out_dir)
