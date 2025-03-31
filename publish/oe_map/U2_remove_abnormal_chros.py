import pandas as pd

import sys
from new_hic_class import SpsHiCMtx
cur_dir = os.path.abspath(os.path.dirname(__file__))
abnormal_chro_file = cur_dir + '/abnormal_chr_list.txt'
abnormal_chros = pd.read_csv(abnormal_chro_file, names=['species', 'chr_idx'])


def remove_abnormal_chros(input_mtx, spe_name):
    input_mtx.to_all(inplace=True)
    
    spe_ab_chros = abnormal_chros[abnormal_chros['species'] == spe_name]
    if spe_ab_chros.shape[0] == 0:
        return input_mtx
    else:
        raw_chros = input_mtx.row_window.chros
        kept_chros = raw_chros.copy()
        for ab_chro_idx in spe_ab_chros['chr_idx']:
            if ab_chro_idx > 0:
                ab_chro_idx -= 1
            ab_chro = raw_chros[ab_chro_idx]
            kept_chros.remove(ab_chro)
        return input_mtx.get_multi_chros_mtx(kept_chros)


def reorder_chros(input_mtx):
    return input_mtx.reorder_chros_by_len()
