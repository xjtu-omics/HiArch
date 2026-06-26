import os
import glob
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sps

import sys
from run_hic_dataset import get_cell_file_name
from new_hic_class import GenomeIndex, read_sps_file, SpsHiCMtx


def chro_compare(chro1, chro2):
    chro1 = chro1.capitalize()
    chro2 = chro2.capitalize()
    return chro1 == chro2


#################
# Read File
def read_matrix(mtx_file, index_file=None, thread=1):
    if mtx_file.endswith('.mtx'):
        sps_mtx = read_sps_file(mtx_file, index_file)
        sps_mtx.mtx_name = mtx_file
        sps_mtx.mtx_type = 'triu'
        if 'ode' in mtx_file.split('/')[-1].split('.')[-2]:
            sps_mtx.has_neg = True
    else:
        raise ValueError('Unrecognized input matrix file.')
    return sps_mtx


def get_spe_file(spe_name, data_dir, suffix):
    spe_file_this_dir = get_cell_file_name(
        spe_name, data_dir, suffix, verbose=False)

    sub_dir = f'{data_dir}/{spe_name}'
    spe_file_sub_dir = get_cell_file_name('*', sub_dir, suffix, verbose=False)

    spe_files = [spe_file_this_dir, spe_file_sub_dir]
    not_none_idx = [f is not None for f in spe_files]
    not_none_num = np.count_nonzero(not_none_idx)

    if not_none_num == 0:
        return None
    elif not_none_num == 1:
        return spe_files[np.where(not_none_idx)[0][0]]
    else:
        warnings.warn(
            f'Multiple files are found for {spe_name}. Will use {spe_files[np.where(not_none_idx)[0][0]]}')
        return spe_files[np.where(not_none_idx)[0][0]]


raw_hic_data_suffix = '.sps_normalized.mtx'
raw_window_suffix = '.window.bed'

result_dir = 'project/centr_loc/result'
oe_dir = f"{result_dir}/oe_map"
oe_mtx_suffix = '.clean_de_ode.mtx'
oe_window_suffix = '.clean_window.bed'
def read_hic_files(input_dir, file_list=None, mtx_suffix=oe_mtx_suffix, window_suffix=oe_window_suffix,
                   read_sps_mtx=True, skip_none=True, spe_is_sep_dir=True,
                   n_file=None, shuffle=False, verbose=True, **other_dir_kwargs):
    """ other_dir_kwargs: {file_type: [dir, suffix]}. None for dir is default to data_dir.
                        For example, {'centr_file': [None, '_centromere.bed']}.
    """
    if file_list is None:
        hic_file_names = get_species_names(input_dir, mtx_suffix, spe_is_sep_dir=spe_is_sep_dir)
    elif isinstance(file_list, list):
        hic_file_names = file_list
    else:
        raise ValueError('Unrecognized input file_list type')

    if shuffle:
        np.random.shuffle(hic_file_names)
    if n_file is not None:
        if len(hic_file_names) > n_file:
            hic_file_names = hic_file_names[:n_file]

    used_hic_file_names = []
    for hic_file_name in hic_file_names:
        hic_file_name = hic_file_name.strip("\n")
        species = hic_file_name

        hic_file = get_spe_file(species, input_dir, mtx_suffix)
        if hic_file is None and skip_none:
            continue
        index_file = get_spe_file(species, input_dir, window_suffix)
        # if index_file is None and skip_none:
        #     continue

        if read_sps_mtx:
            sps_mtx = read_matrix(hic_file, index_file)
        else:
            sps_mtx = None

        hic_struct = {'hic_file_name': hic_file_name,
                      'hic_file': hic_file,
                      'sps_mtx': sps_mtx,
                      'index_file': index_file}

        for file_type in other_dir_kwargs:
            file_type_dir, file_type_suffix = other_dir_kwargs[file_type]
            file_type_dir = input_dir if file_type_dir is None else file_type_dir
            file_type_file = get_spe_file(
                species, file_type_dir, file_type_suffix)
            if file_type_file is None and skip_none:
                break
            hic_struct[file_type] = file_type_file
        # else in for only runs when for-process complete (no break)
        else:
            if verbose:
                print("#######################")
                print(hic_file_name)
                print(f'###hic_file: {hic_file}')
                print(f'###index_file: {index_file}')
                for file_type in other_dir_kwargs:
                    print(f'###{file_type}: {hic_struct[file_type]}')
            used_hic_file_names.append(hic_file_name)
            yield hic_struct
    
    if verbose:
        print(used_hic_file_names)


def read_kwargs_file(input_dir, file_suffix, window_suffix=oe_window_suffix, file_list=None, skip_none=True,
                     n_file=None, shuffle=False, verbose=True, **other_dir_kwargs):
    if file_list is None:
        hic_file_names = get_species_names(input_dir, file_suffix)
    elif isinstance(file_list, list):
        hic_file_names = file_list
    else:
        raise ValueError('Unrecognized input file_list type')

    if shuffle:
        np.random.shuffle(hic_file_names)
    if n_file is not None:
        if len(hic_file_names) > n_file:
            hic_file_names = hic_file_names[:n_file]

    used_hic_file_names = []
    for hic_file_name in hic_file_names:
        hic_file_name = hic_file_name.strip("\n")
        species = hic_file_name

        main_file = get_spe_file(species, input_dir, file_suffix)
        index_file = get_spe_file(species, input_dir, window_suffix)

        hic_struct = {'hic_file_name': hic_file_name,
                      'main_file': main_file,
                      'index_file': index_file}

        for file_type in other_dir_kwargs:
            file_type_dir, file_type_suffix = other_dir_kwargs[file_type]
            file_type_dir = input_dir if file_type_dir is None else file_type_dir
            file_type_file = get_spe_file(
                species, file_type_dir, file_type_suffix)
            if file_type_file is None and skip_none:
                break
            hic_struct[file_type] = file_type_file
        # else in for only runs when for-process complete (no break)
        else:
            if verbose:
                print("#######################")
                print(hic_file_name)
                print(f'###main_file: {main_file}')
                print(f'###index_file: {index_file}')
                for file_type in other_dir_kwargs:
                    print(f'###{file_type}: {hic_struct[file_type]}')
            used_hic_file_names.append(hic_file_name)
            yield hic_struct
    if verbose:
        print(used_hic_file_names)


def get_species_names(input_dir, mtx_suffix=oe_mtx_suffix,
                      spe_is_sep_dir=True, verbose=False):
    hic_file_names = []
    if spe_is_sep_dir:
        hic_files = glob.glob(f'{input_dir}/*/*{mtx_suffix}')
        print(hic_files)
        hic_file_names += [f.split('/')[-2] for f in hic_files]
    else:
        hic_files = glob.glob(f'{input_dir}/*{mtx_suffix}')
        hic_file_names += [f.split('/')[-1].split(mtx_suffix)[0]
                           for f in hic_files]

    if verbose:
        print(hic_file_names)
    return hic_file_names


#################
# Centr loc
class Anchors:
    def __init__(self, input_anchors, input_index=None, keep_first=False):
        if isinstance(input_anchors, str):
            if input_anchors.endswith('_centromere.bed'):
                self.loc_df = pd.read_table(
                    input_anchors, names=['chr', 'start', 'end'],
                    dtype={'chr': str})
            elif input_anchors.endswith('.anchors.txt'):
                self.loc_df = pd.read_table(input_anchors, header=0,
                                            dtype={'chr': str})
            else:
                raise ValueError('Unrecognized centromere input')

        elif isinstance(input_anchors, Anchors):
            self.loc_df = input_anchors.loc_df
        elif isinstance(input_anchors, pd.DataFrame):
            self.loc_df = input_anchors
        else:
            raise ValueError('Unrecognized input for centr_loc.')

        if input_index is None:
            self.gen_index = None
        else:
            self.gen_index = GenomeIndex(input_index)

        # if reset_col_names:
        #     if self.loc_df.shape[1] == 2:
        #         self.loc_df.columns = ['cs', 'ce']
        #         self.find_centr_gen_pos()
        #     elif self.loc_df.shape[1] == 3:
        #         self.loc_df.columns = ['chr', 'start', 'end']
        #         self.find_centr_index()
        #     elif self.loc_df.shape[1] == 5:
        #         self.loc_df.columns = ['chr', 'start', 'end', 'cs', 'ce']
        #     else:
        #         raise ValueError('Number of columns are wrong. If it is centr loc, please use reset_col_names=False.')

        if 'start' not in self.loc_df.columns:
            self.find_anchor_gen_pos()

        if 'cs' not in self.loc_df.columns:
            self.find_anchor_index()

        self.chros = list(self.loc_df['chr'].unique())

        if keep_first:
            self.keep_first_centr()

        self.shape = self.loc_df.shape

    def __getitem__(self, key):
        return self.loc_df[key]

    def __setitem__(self, key, value):
        self.loc_df[key] = value

    def find_anchor_gen_pos(self):
        if self.gen_index is None:
            raise ValueError(
                'Plz input gen_index if want to find centr index.')

        if 'chr' not in self.loc_df.columns:
            self.loc_df['chr'] = 'nan'
        if 'start' not in self.loc_df.columns:
            self.loc_df['start'] = np.nan
        if 'end' not in self.loc_df.columns:
            self.loc_df['end'] = np.nan

        for idx, row in self.loc_df.iterrows():
            chro1 = self.gen_index.window_df.loc[row['cs'], 'chr']
            chro2 = self.gen_index.window_df.loc[row['ce'], 'chr']
            if chro1 != chro2:
                raise ValueError('Position of same chromosome are not same.')
            self.loc_df.loc[idx, 'chr'] = chro1
            self.loc_df.loc[idx,
                            'start'] = self.gen_index.window_df.loc[row['cs'], 'start']
            self.loc_df.loc[idx,
                            'end'] = self.gen_index.window_df.loc[row['cs'], 'end']

    def find_anchor_index(self):
        if self.gen_index is None:
            raise ValueError(
                'Plz input gen_index if want to find centr index.')

        if 'cs' not in self.loc_df.columns:
            self.loc_df['cs'] = np.nan
        if 'ce' not in self.loc_df.columns:
            self.loc_df['ce'] = np.nan

        for idx, centr in self.loc_df.iterrows():
            chro, start, end = centr['chr'], centr['start'], centr['end']

            chro_idx = [chro_compare(chro, chro1)
                        for chro1 in self.gen_index.chros]
            if np.count_nonzero(chro_idx) == 0:
                continue
            if np.count_nonzero(chro_idx) > 1:
                raise ValueError('Multiple chros matched.')
            chro = self.gen_index.chros[np.where(chro_idx)[0][0]]

            cen_s, cen_e = self.gen_index.get_region_startend_idx(chro, start, end,
                                                                  is_mtx_idx=False)
            self.loc_df.loc[idx, 'cs'] = cen_s
            self.loc_df.loc[idx, 'ce'] = cen_e

        if self.loc_df.isnull().any().any():
            warnings.warn('Nan in centr loc. Will remove.')
        self.loc_df.dropna(inplace=True)
        self.loc_df['cs'] = self.loc_df['cs'].astype(int)
        self.loc_df['ce'] = self.loc_df['ce'].astype(int)
        
        self.loc_df['cen'] = ((self.loc_df['cs'] + self.loc_df['ce']) / 2).astype(int)

    def keep_used_anchors(self):
        if 'type' not in self.loc_df.columns:
            return self

        used_loc_df = self.loc_df.copy()
        used_loc_df = used_loc_df[used_loc_df['type'] == 'used']
        return Anchors(used_loc_df, self.gen_index, keep_first=False)

    def keep_first_centr(self):
        self.loc_df = self.loc_df.groupby(by=['chr']).first()
        self.loc_df.reset_index(inplace=True)

    def get_used_chro_loc(self, chro, chr_idx=False):
        chr_loc_df = self.loc_df[self.loc_df['chr'] == chro]
        # print(chr_loc_df.to_string())
        if 'type' in chr_loc_df:
            chr_loc_df = chr_loc_df[chr_loc_df['type'] == 'used']
        
        if chr_loc_df.shape[0] == 0:
            return None
        elif chr_loc_df.shape[0] == 1:
            cen_loc = chr_loc_df.iloc[0, :].loc['cen']
        else:
            warnings.warn(f'Multiple cen_loc found in {chro}, will use first.')
            cen_loc = chr_loc_df.iloc[0, :].loc['cen']
        
        if chr_idx:
            if self.gen_index is None:
                raise ValueError('Plz input gen_index for centr_loc.')
            if cen_loc is not None:
                cen_loc -= self.gen_index.chr_seps[chro][0]
        
        print(f'{chro}: {cen_loc}')
        
        return cen_loc
