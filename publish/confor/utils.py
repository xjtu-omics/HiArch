import os
import numpy as np
import pandas as pd

import sys
from iutils.read_matrix import read_kwargs_file


filt_mtx_suffix = '.filt_ode.mtx'
anchor_suffix = '.anchors.txt'
confor_suffix = '.confor.txt'

label_for_confor = 'Global folding score'

min_confor_value = .5
min_normed_confor_value = .43
# min_normed_confor_value = .8
max_normed_confor_value = .97


default_con_col = 'score'
class Confor:
    def __init__(self, input_confor):
        if isinstance(input_confor, str):
            self.values = pd.read_table(input_confor,
                                        dtype={'type': str})
        elif isinstance(input_confor, pd.DataFrame):
            self.values = input_confor
        elif isinstance(input_confor, Confor):
            self.values = input_confor.values
        else:
            raise ValueError('Unrecognized input type.')
        self.values.dropna(inplace=True)

    def __getitem__(self, key):
        return self.values[key]

    def filt(self, key, value, inplace=False):
        new_values = self.values[self.values[key] == value].copy()
        if inplace:
            self.values = new_values
        else:
            return Confor(new_values)
    
    def get_intra_sub(self, intra=True):
        if intra:
            new_confor_df = self.values[self.values['chr1'] == self.values['chr2']].copy()
        else:
            new_confor_df = self.values[self.values['chr1'] != self.values['chr2']].copy()
        return Confor(new_confor_df)

    def get_strength(self):
        confors = self.values.groupby('type').agg({default_con_col: 'mean'})
        return  confors[default_con_col].max()


###############
# Utils
def norm_confor_among_species(confors):
    """ Normalization only by positive values.
        Norm by values = values / np.std(nonzero_values)
    """
    mid_vmax_per = 66
    
    for confor_type in confors['type'].unique():
        type_confor_idx = confors['type'] == confor_type
        
        type_arr = np.array(confors.loc[type_confor_idx, default_con_col])
        
        nonzero_arr = type_arr[~np.isnan(type_arr)]
        vmax = np.percentile(nonzero_arr, mid_vmax_per)
        vmin = np.percentile(nonzero_arr, 100 - mid_vmax_per)
        if vmin < vmax:
            sub_arr_for_mean = nonzero_arr[nonzero_arr < vmax]
            sub_arr_for_mean = sub_arr_for_mean[sub_arr_for_mean > vmin]
            if len(sub_arr_for_mean) > 0:
                arr_for_mean = sub_arr_for_mean
            else:
                arr_for_mean = nonzero_arr
        else:
            arr_for_mean = nonzero_arr

        nonzero_arr = (nonzero_arr - np.mean(arr_for_mean)) / np.std(nonzero_arr)

        type_arr[~np.isnan(type_arr)] = nonzero_arr
        confors.loc[type_confor_idx, default_con_col] = type_arr
    return confors


def keep_pos(values):
    value_arr = np.array(values[default_con_col])
    value_arr[value_arr < 0] = 0
    values.loc[:, default_con_col] = value_arr
    return values


def get_global_strength(intra=True, confors=None):
    if confors is None:
        confors = read_spe_confor(intra=intra, norm=True, only_keep_pos=False)
    
    confors = confors.groupby(by=['file_name']).agg({default_con_col: 'max'})
    confors.reset_index(inplace=True)
    confors.index = confors['file_name']
    return confors


####################
# Species average confor
def get_ave_confor(input_dir):
    intra_ave_confors = []
    inter_ave_confors = []
    for hic_struct in read_kwargs_file(input_dir, confor_suffix):
        confors = Confor(hic_struct['main_file']).values
        
        intra_confors = confors[confors['chr1'] == confors['chr2']]
        intra_confors = intra_confors.groupby('type').agg({default_con_col: 'mean'})
        intra_confors.reset_index(inplace=True)
        intra_confors['file_name'] = hic_struct['hic_file_name']
        intra_ave_confors.append(intra_confors)
        
        inter_confors = confors[confors['chr1'] != confors['chr2']]
        inter_confors = inter_confors.groupby('type').agg({default_con_col: 'mean'})
        inter_confors.reset_index(inplace=True)
        inter_confors['file_name'] = hic_struct['hic_file_name']
        inter_ave_confors.append(inter_confors)
        
    intra_ave_confor = pd.concat(intra_ave_confors, axis=0)
    inter_ave_confor = pd.concat(inter_ave_confors, axis=0)
    return intra_ave_confor, inter_ave_confor


intra_ave_order = ['Center', 'Large-center', 'Center-end-axis', 
                   'Center-whole', 'End-whole']
inter_ave_order = ['Center-center', 'End-end', 'Center-end-axis',
                   'Center-whole', 'End-whole']
def read_spe_confor(input_dir, intra=True, only_keep_pos=False, norm=False):
    if intra:
        if os.path.exists(f'{input_dir}/intra_ave_confor.txt'):
            confors = pd.read_table(f'{input_dir}/intra_ave_confor.txt',
                                    dtype={'type': str})
        else:
            confors, _ = get_ave_confor(input_dir)
    else:
        if os.path.exists(f'{input_dir}/inter_ave_confor.txt'):
            confors =  pd.read_table(f'{input_dir}/inter_ave_confor.txt',
                                    dtype={'type': str})
        else:
            _, confors = get_ave_confor(input_dir)

    if norm:
        confors = norm_confor_among_species(confors)

    if only_keep_pos:
        confors = keep_pos(confors)

    type_order = intra_ave_order if intra else inter_ave_order
    confors['type_order'] = confors['type'].apply(lambda x: type_order.index(x) if x in type_order else np.nan)
    confors.dropna(subset=['type_order'], inplace=True)
    return confors


if __name__ == '__main__':
    get_ave_confor(out_file=True)
