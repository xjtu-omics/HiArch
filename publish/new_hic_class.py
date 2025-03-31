import copy
import json
import os
import warnings
import statistics
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.sparse as sps
import matplotlib.pyplot as plt


def change_suffix(file_name, changed_suffix):
    return f"{'/'.join(file_name.split('/')[:-1])}/{file_name.split('/')[-1].split('.')[0]}{changed_suffix}"


##############################
# Genome Index
class GenomeIndex:
    def __init__(self, input_window, corr_index=None, read_corr=True):
        self.is_diplo, self.corr_index = None, None

        if isinstance(input_window, str):
            self._read_index_from_file(input_window, read_corr=read_corr)
        elif isinstance(input_window, pd.DataFrame):
            self.window_df = input_window
        elif isinstance(input_window, GenomeIndex):
            self.window_df = input_window.window_df
            self.corr_index = input_window.corr_index
        else:
            raise ValueError(f'Unrecognized input window type: {input_window.type}')

        if corr_index is not None:
            if self.corr_index is not None:
                warnings.warn(f'Corr index will be replaced by input corr index.')
            self.corr_index = GenomeIndex(corr_index, read_corr=False)
            self.get_is_diplo()

        self.window_df.index = self.window_df['index']
        self.window_df = self.window_df.sort_index()

        self.bin_size = None

        self.chros = None
        self.chr_seps = None
        self.chr_lens = None
        self.chr_abs_lens = None

        self.min_idx = None
        self.max_idx = None
        self.all_len = None

        self.ori_start_idx = None # for indexing the start of sub index of original index

        self.get_index_features()

    def __getitem__(self, item):
        return self.window_df[item]

    def copy(self):
        return copy.copy(self)

    def _read_index_from_file(self, input_file, read_corr=True):
        self.window_df = pd.read_table(input_file,
                                       names=['chr', 'start', 'end', 'index'],
                                       dtype={'chr': str, 'start': int,
                                              'end': int, 'index': int},
                                       comment="#")

        if read_corr:
            file_handle = open(input_file, 'r')
            for line in file_handle.readlines():
                if not line.startswith("#"):
                    break
                if line.startswith('#corr'):
                    line = line.split('#corr:')[1].strip()
                    self.corr_index = GenomeIndex(line, read_corr=False)
            file_handle.close()

            if self.corr_index is not None:
                self.get_is_diplo()

    def get_index_features(self):
        # self.window_df.index = self.window_df['index']

        self.min_idx = self.window_df['index'].min()
        self.max_idx = self.window_df['index'].max()
        self.all_len = self.max_idx - self.min_idx + 1

        self.chros = list(set(self.window_df['chr']))

        self.chr_seps, self.chr_lens, self.chr_abs_lens = {}, {}, {}
        for chro in self.chros:
            chr_indexs = self.window_df[self.window_df['chr'] == chro]
            min_chr_idx = np.min(chr_indexs['index'])
            max_chr_idx = np.max(chr_indexs['index'])
            self.chr_seps[chro] = [min_chr_idx, max_chr_idx]
            self.chr_lens[chro] = chr_indexs.shape[0]
            self.chr_abs_lens[chro] = np.max(chr_indexs['end'])

        chros = sorted(self.chr_seps.items(), key=lambda x: x[1][0])
        self.chros = [chro[0] for chro in chros]
        
        bin_sizes = np.array(self.window_df['end'].iloc[:10] - self.window_df['start'].iloc[:10])
        self.bin_size = statistics.mode(bin_sizes)

    #########
    # choose sub
    def get_sub_corr_index(self, indexes):
        if self.corr_index is None:
            return None

        dip_hap_trans, dip_index = self.dip_hap_trans()

        if self.is_diplo:
            dip_hap_trans.index = dip_hap_trans['index']
            dip_hap_trans = dip_hap_trans.loc[indexes, :]

            sub_corr_index = dip_hap_trans['hap_index'].drop_duplicates()
            sub_corr_index = self.corr_index.window_df.loc[sub_corr_index, :]
            return GenomeIndex(sub_corr_index)
        else:
            dip_hap_trans.index = dip_hap_trans['hap_index']
            dip_hap_trans = dip_hap_trans.loc[indexes, :]

            sub_corr_index = dip_hap_trans['index']
            sub_corr_index = self.corr_index.window_df.loc[sub_corr_index, :]
            return GenomeIndex(sub_corr_index)

    def get_region_startend_idx(self, chros, starts=None, ends=None, is_mtx_idx=True):
        """
        chro: start-end (start, end are matrix index, by default)
        is_mtx_idx: input start, end is mtx_index. If not, start, end =/ bin_size.
        
        Either input one chro: start-end. Or input chros, starts and ends.
        
        return:
        end_idx is the ceil of input end. Use [start_idx:end_idx] for index.
        """
        from numbers import Number
        
        if starts is None:
            if not isinstance(chros, str):
                raise ValueError('chros must be one specific chros if starts and ends is None.')
            starts = 0
            ends = self.chr_lens[chros]
        
        return_number = False
        if isinstance(starts, Number):
            starts, ends = pd.Series([starts]), pd.Series([ends])
            return_number = True
        if isinstance(chros, str):
            chros = pd.Series([chros])
        
        uni_chros = list(set(list(chros)))

        start_idxes, end_idxes = starts.copy(), ends.copy()
        for chro in uni_chros:
            chr_index = chros[chros == chro].index
            chr_starts, chr_ends = starts[chr_index], ends[chr_index]
            
            if chro not in self.chros:
                return None, None
        
            if is_mtx_idx:
                chr_gen_index = self.window_df[self.window_df['chr'] == chro]
                chr_gen_index = chr_gen_index.sort_values(['start'])
                
                new_starts, new_ends = [], []
                for start, end in zip(chr_starts, chr_ends):
                    start = 0 if start is None else start
                    new_starts.append(start)
                    end = self.chr_lens[chro] - 1 if end is None else end
                    end = self.chr_lens[chro] - 1 if end > self.chr_lens[chro] - 1 else end
                    new_ends.append(end)
                
                start_idxes.loc[chr_index] = list(chr_gen_index.iloc[new_starts, :]['index'])
                end_idxes.loc[chr_index] = list(chr_gen_index.iloc[new_ends, :]['index'])
            
            else:
                window_df_copy = self.window_df.copy()
                window_df_copy.index = window_df_copy['chr'] + '_' + window_df_copy['start'].astype(str)
                
                chr_start_window_df = self.window_df[self.window_df['chr'] == chro].copy()
                chr_start_window_df.index = chr_start_window_df['chr'] + '_' + chr_start_window_df['start'].astype(str)
                sub_starts = ((chr_starts / self.bin_size).astype(int) * self.bin_size).astype(int)
                try:
                    start_idxes.loc[chr_index] = list(window_df_copy.loc[chro + '_' + sub_starts.astype(str), 'index'])
                except KeyError:
                    def _get_start_idxes(s):
                        starts = np.sort(np.array(chr_start_window_df['start']))
                        start_idx = starts.searchsorted(s, side='right')
                        s_idx = chro + '_' + str(starts[start_idx - 1])
                        return chr_start_window_df.loc[s_idx, 'index']
                        # while s >= 0:
                        #     s_idx = chro + '_' + str(s)
                        #     if s_idx in window_df_copy.index:
                        #         return window_df_copy.loc[s_idx, 'index']
                        #     if s <= 0:
                        #         return self.chr_seps[chro][0]
                        #     s -= self.bin_size
                    start_idxes.loc[chr_index] = sub_starts.apply(_get_start_idxes)
                
                chr_end_window_df = self.window_df[self.window_df['chr'] == chro].copy()
                chr_end_window_df.index = chr_end_window_df['chr'] + '_' + chr_end_window_df['end'].astype(str)
                sub_ends = ((chr_ends / self.bin_size).astype(int) * self.bin_size).astype(int)
                try:
                    end_idxes.loc[chr_index] = list(window_df_copy.loc[chro + '_' + sub_ends.astype(str), 'index'])
                except KeyError:
                    def _get_end_idxes(e):
                        ends = np.sort(np.array(chr_end_window_df['end']))
                        end_idx = ends.searchsorted(e)
                        e_idx = chro + '_' + str(ends[end_idx])
                        return chr_end_window_df.loc[e_idx, 'index']
                        # while e <= self.chr_seps[chro][1] * self.bin_size:
                        #     e_idx = chro + '_' + str(e)
                        #     if e_idx in window_df_copy.index:
                        #         return window_df_copy.loc[e_idx, 'index']
                        #     if e >= self.chr_seps[chro][1] * self.bin_size:
                        #         return self.chr_seps[chro][1]
                        #     e += self.bin_size
                    end_idxes.loc[chr_index] = sub_ends.apply(_get_end_idxes)
        
        if return_number:
            return start_idxes[0], end_idxes[0]
        else:
            return start_idxes, end_idxes

    def get_sub_gen_index(self, start=None, end=None, idxes=None, reset_index=True):
        """get sub gen_index by list of indexes"""
        if idxes is None:
            sub_window = self.window_df.loc[start: end, :].copy()
        elif start is None and end is None:
            sub_window = self.window_df.loc[idxes, :].copy()
        else:
            raise ValueError(f'Unrecognized input')

        if reset_index:
            sub_window.loc[:, 'ori_index'] = sub_window["index"].copy()
            sub_window.reset_index(inplace=True, drop=True)
            sub_window.loc[:, "index"] = sub_window.index
            
        sub_window = GenomeIndex(sub_window)
        sub_window.ori_start_idx = np.min(idxes)

        if self.corr_index is not None:
            sub_window.corr_index = self.get_sub_corr_index(idxes)

        return sub_window

    def get_region_gen_index(self, chro=None, start=None, end=None, reset_index=True):
        start_idx, end_idx = self.get_region_startend_idx(chro, start, end)
        chr_window = self.get_sub_gen_index(start=start_idx, end=end_idx, reset_index=reset_index)
        return chr_window

    def keep_chros(self, chros, return_old_index=False):
        if len(chros) == 0:
            if return_old_index:
                return None, None
            else:
                return None
        
        new_window_df = []
        for chro in chros:
            chr_window_df = self.window_df[self.window_df['chr'] == chro]
            new_window_df.append(chr_window_df)
        
        new_window_df = pd.concat(new_window_df, axis=0)
        
        new_window_df.reset_index(inplace=True, drop=True)
        old_index = np.array(new_window_df['index'])
        new_window_df.drop(['index'], axis=1, inplace=True)
        new_window_df.reset_index(inplace=True)
        
        if return_old_index:
            return GenomeIndex(new_window_df), old_index
        else:
            return GenomeIndex(new_window_df)

    def keep_chros_by_len(self, larger_than_last_ratio=.1,
                          larger_than_max_ratio=.01,
                          return_old_index=False, verbose=True):
        """ Only keep chromosomes larger than larger_than_last_ratio of last larger chromosome.
            and larger than larger_than_max_ratio of largest chromosome.
            So that tid off small contigs.
        """
        sorted_chros = sorted(self.chr_lens.items(), key=lambda x: x[1], reverse=True)
        max_len = sorted_chros[0][1]
        
        last_len = max_len
        kept_chros = []
        tid_off_chros = []
        for chro, chr_len in sorted_chros:
            if chr_len >= last_len * larger_than_last_ratio and \
               chr_len >= max_len * larger_than_max_ratio:
                kept_chros.append(chro)
            else:
                tid_off_chros.append(chro)

        if verbose:
            if len(tid_off_chros) > 0:
                print(f'{" ".join(tid_off_chros)} is too short.')
                
        return self.keep_chros(kept_chros, return_old_index)

    def keep_long_chros(self, min_chro_len=5, return_old_index=False, verbose=True):
        """ Only keep chromosomes with length larger than min_chro_len (bin number).
            So that tid off small contigs.

        Args:
            min_chro_len (int, optional): Minimum chromosome length, in bin number. Defaults to 5.
        """
        chros = []
        tid_off_chros = []
        for chro in self.chros:
            chr_window_df = self.window_df[self.window_df['chr'] == chro]
            
            if chr_window_df.shape[0] > min_chro_len:
                chros.append(chro)
            else:
                tid_off_chros.append(chro)
        if verbose:
            if len(tid_off_chros) > 0:
                print(f'{" ".join(tid_off_chros)} is too short.')
        
        return self.keep_chros(chros, return_old_index)

    #########
    # hap dip
    def get_is_diplo(self):
        if self.window_df.shape[0] > self.corr_index.window_df.shape[0]:
            self.is_diplo = True
        elif self.window_df.shape[0] < self.corr_index.window_df.shape[0]:
            self.is_diplo = False
        else:
            raise ValueError(f'Index shape {self.window_df.shape[0]} is not compiled with'
                             f'corr index {self.corr_index.window_df.shape[0]}')

    def dip_hap_trans(self, trans_index=None, mat_str='(mat)', pat_str='(pat)'):
        if self.is_diplo:
            dip_index = self.window_df.copy()
            if trans_index is not None:
                hap_index = trans_index.copy()
            else:
                if self.corr_index is not None:
                    hap_index = self.corr_index.window_df.copy()
                else:
                    raise ValueError('Corr index is not assigned.')
            new_index = hap_index
        else:
            hap_index = self.window_df.copy()
            if trans_index is not None:
                dip_index = trans_index.copy()
            else:
                if self.corr_index is not None:
                    dip_index = self.corr_index.window_df.copy()
                else:
                    raise ValueError('Corr index is not assigned.')
            new_index = dip_index

        dip_index['hap_chr'] = dip_index['chr'].str.replace(mat_str, '', regex=False)
        dip_index['is_mat'] = dip_index['chr'] != dip_index['hap_chr']
        dip_index['hap_chr'] = dip_index['hap_chr'].str.replace(pat_str, '', regex=False)

        if not dip_index['hap_chr'][0].startswith('chr'):
            dip_index['hap_chr'] = 'chr' + dip_index['hap_chr']
        if not hap_index['chr'][0].startswith('chr'):
            hap_index['chr'] = 'chr' + hap_index['chr']

        hap_index.index = pd.MultiIndex.from_arrays([hap_index['chr'],
                                                     hap_index['start']])

        dip_index_mat = dip_index[dip_index['is_mat']].copy()
        dip_index_mat.index = pd.MultiIndex.from_arrays([dip_index_mat['hap_chr'],
                                                         dip_index_mat['start']])
        shared_index = hap_index.index.intersection(dip_index_mat.index)
        dip_index_mat['hap_index'] = hap_index.loc[shared_index, 'index']

        dip_index_pat = dip_index[~dip_index['is_mat']].copy()
        dip_index_pat.index = pd.MultiIndex.from_arrays([dip_index_pat['hap_chr'],
                                                         dip_index_pat['start']])
        shared_index = hap_index.index.intersection(dip_index_pat.index)
        dip_index_pat['hap_index'] = hap_index.loc[shared_index, 'index']

        dip_index = pd.concat([dip_index_mat, dip_index_pat], axis=0)
        dip_index['hap_index'] = dip_index['hap_index'].astype(int)
        dip_index.reset_index(inplace=True, drop=True)
        return dip_index[['index', 'is_mat', 'hap_index']], new_index

    #########
    # other utils
    def get_idx_chro(self, idx):
        return self.window_df.loc[idx, 'chr']
    
    def intersect(self, inter_df):
        inter_idx_df = {'start_idx': [], 'end_idx': []}

        for idx, row in inter_df.iterrows():

            chr_window = find_chro_match_df(self.window_df, row['chr'])

            if chr_window.shape[0] == 0:
                warnings.warn(f"Cannot find {row['chr']} in genome index.")

            start_idx = np.max(chr_window.loc[chr_window['start'] <= row['start'], 'start'])
            end_idx = np.min(chr_window.loc[chr_window['end'] >= row['end'], 'end'])

            inter_idx_df['start_idx'].append(start_idx)
            inter_idx_df['end_idx'].append(end_idx)

        return pd.DataFrame(inter_idx_df)

    def sort_by_chro(self):
        new_window_df = self.window_df.copy()
        new_window_df.sort_values(by=['chr', 'start'], inplace=True)

        return GenomeIndex(new_window_df)

    def to_output(self, out_file):
        self.window_df.to_csv(out_file, header=False, index=False, sep="\t",
                              columns=['chr', 'start', 'end', 'index'])

    def condense(self, condense_fold):
        if 'ori_index' in self.window_df.columns:
            print('Warning: Condense in truncted (might by mtx.tid_off_low_bins()) index might cause index error.')
        
        new_window_df = self.window_df.copy()
        new_window_df['new_index'] = 0

        current_idx = 0
        for chro in self.chros:
            min_idx, max_idx = self.chr_seps[chro][0], self.chr_seps[chro][1]

            # Change window index
            chr_index = new_window_df['chr'] == chro
            new_index = (new_window_df.loc[chr_index, 'index'] - min_idx) // condense_fold + current_idx
            new_window_df.loc[chr_index, 'new_index'] = new_index

            current_idx = np.max(new_index) + 1

            # print(np.max(new_index))
            # print(current_idx - 1)
            if np.max(new_index) != current_idx - 1:
                raise ValueError('Index Wrong!')

        index_trans = new_window_df['new_index'].copy()
        
        window_start = new_window_df.groupby(by='new_index').aggregate('min')
        window_start = window_start[['chr', 'start']]

        window_end = new_window_df.groupby(by='new_index').aggregate('max')
        window_end = window_end['end']

        new_window_df = pd.concat([window_start, window_end], axis=1)
        new_window_df['index'] = new_window_df.index

        new_window = GenomeIndex(input_window=new_window_df)
        return new_window, index_trans


class Values:
    def __init__(self, input_value, input_index=None):
        drop_cols = ['chr', 'start', 'end', 'index']

        if isinstance(input_value, str):
            self.values = pd.read_table(input_value, comment="#")

            if 'index' not in self.values.columns:
                warnings.warn('Index col not be found in header.')
                self.values['index'] = np.arange(self.values.shape[0])

            self.values.index = self.values['index']
            self.value_names = list(self.values.columns.values)

        elif isinstance(input_value, pd.DataFrame):
            self.values = input_value
            if 'index' in self.values.columns:
                self.values.index = self.values['index']
            self.value_names = list(self.values.columns)

        elif isinstance(input_value, pd.Series):
            self.values = input_value.to_frame()
            self.value_names = [input_value.name]

        else:
            raise ValueError('Input value type unrecognized.')

        self.shape = self.values.shape

        for col in drop_cols:
            if col in self.value_names:
                self.value_names.remove(col)
                self.values.drop(columns=col, inplace=True)

        if input_index is None:
            self.gen_index = None
            self.bin_size = None
        else:
            self.gen_index = GenomeIndex(input_index)
            self.bin_size = self.gen_index.bin_size

        self.tid_off_mark_suffix()

    def copy(self):
        return copy.deepcopy(self)

    def __getitem__(self, item):
        return self.values[item]

    def __setitem__(self, key, value):
        self.values[key] = value

    def to_output(self, out_file, out_window=False,
                  float_format="%.4f", compression='infer'):
        if out_window:
            self.gen_index.to_output(change_suffix(out_file, '.window.bed'))

        self.values.to_csv(out_file, sep="\t",
                           float_format=float_format,
                           compression=compression)

    def tid_off_mark_suffix(self):
        tid_off_suffixes = ['-human', '-mouse']

        changed_cols = {}
        new_value_names = []
        for value_name in self.value_names:
            if not isinstance(value_name, str):
                # print(value_name)
                continue
            
            add_name = False
            for tid_off_suffix in tid_off_suffixes:
                if value_name.endswith(tid_off_suffix):
                    new_value_name = value_name.removesuffix(tid_off_suffix)
                    changed_cols[value_name] = new_value_name
                    new_value_names.append(new_value_name)
                    add_name = True
                    break
            if not add_name:
                new_value_names.append(value_name)
        
        self.value_names = new_value_names
        self.values.rename(columns=changed_cols, inplace=True)

    def select_values(self, include_pattern):
        value_names = [value_name for value_name in self.value_names if include_pattern in value_name]
        values = self.values[value_names]
        return Values(input_value=values)

    def select_chro(self, chro):
        new_values = copy.deepcopy(self)
        start_idx, end_idx = self.gen_index.get_region_startend_idx(chro)
        new_values.values = new_values.values.loc[start_idx: end_idx, :]
        return new_values

    def get_sub_by_condition(self, index):
        new_values = copy.deepcopy(self)
        new_values.values = new_values.values.loc[index]
        return new_values
    
    def get_region_values(self, chro='chr1', start=None, end=None, ip_idxes=None):
        region_start, region_end = self.gen_index.get_region_startend_idx(chro, start, end)
        if region_start is None:
            return None
        if ip_idxes is None:
            idxes = np.arange(region_start, region_end + 1, dtype=int)
            idx_label = f'{chro}:{start}-{end}'
        else:
            idxes = ip_idxes
            idx_label = 'Genomic loci'
        
        new_values = self.values.loc[self.values.index.intersection(idxes), :].copy()
        new_gen_index = self.gen_index.get_region_gen_index(chro, start, end, reset_index=False)
        new_values = Values(new_values, new_gen_index)
        new_values.idx_label = idx_label
        
        return new_values
    
    def yield_by_chro(self):
        for chro in self.gen_index.chros:
            yield chro, self.get_region_values(chro)
    
    def normalize_values(self, method='z-score'):
        """
        method: ['z-score', 'max-min']
        """
        for col in self.value_names:
            da_ar = self.values[col]
            if method == 'z-score':
                self.values[col] = (da_ar - np.mean(da_ar)) / np.std(da_ar)
            elif method == 'max-min':
                self.values[col] = (da_ar - np.min(da_ar)) / (np.max(da_ar) - np.min(da_ar))
            else:
                raise ValueError('Unrecogenized method.')

    def smooth_values(self, k=5):
        from scipy.ndimage import convolve1d
        for value_name in self.value_names:
            self.values[value_name] = convolve1d(self.values[value_name],
                                                 weights=np.ones(k) / k,
                                                 mode='reflect')

    def replace_outliers_with_nan(self, tid_off_per=None):
        """
        None for 3 * sd tid-off.
        tid_off_per for percentile tid-off.
        """
        
        for col in self.value_names:
            da_ar = self.values[col].copy()
            
            nan_index = self.values[col].copy().astype(bool)
            nan_index[:] = False

            if tid_off_per is None:
                nan_index[da_ar < np.mean(da_ar) - 3 * np.std(da_ar)] = True
                nan_index[da_ar > np.mean(da_ar) + 3 * np.std(da_ar)] = True
            else:
                min_value = np.percentile(da_ar, tid_off_per)
                max_value = np.percentile(da_ar, 100 - tid_off_per)
                if min_value == max_value:
                    warnings.warn(f'Constant col {col} detected. No tid off applied')
                    continue
                nan_index[da_ar <= min_value] = True
                nan_index[da_ar >= max_value] = True
                
                # print(np.count_nonzero(remain_idxes))
            
            a = self.values[col]
            a[nan_index] = np.nan
            self.values[col] = a

    ###########
    # compare
    def to_hap(self, concat_func='mean'):
        if self.gen_index is None:
            raise ValueError('Genome index is not defined.')

        if not self.gen_index.is_diplo:
            return self

        else:
            dip_hap_trans, hap_index = self.gen_index.dip_hap_trans()

            new_values = self.values[self.value_names].copy()

            dip_hap_trans.index = dip_hap_trans['index']
            shared_index = dip_hap_trans.index.intersection(new_values.index)

            new_values['hap_index'] = dip_hap_trans.loc[shared_index, 'hap_index']
            new_values = new_values.groupby(by='hap_index').apply(concat_func)

            return Values(new_values, self.gen_index.corr_index)

    def to_dip(self):
        if self.gen_index is None:
            raise ValueError('Genome index is not defined.')

        if self.gen_index.is_diplo:
            return self

        else:
            dip_hap_trans, dip_index = self.gen_index.dip_hap_trans()

            mat_dip_hap_trans = dip_hap_trans[dip_hap_trans['is_mat']]
            new_values_mat = self.values[self.value_names].copy()
            mat_dip_hap_trans.index = mat_dip_hap_trans['hap_index']
            shared_index = new_values_mat.index.intersection(mat_dip_hap_trans.index)
            new_values_mat.index = mat_dip_hap_trans.loc[shared_index, 'index']

            pat_dip_hap_trans = dip_hap_trans[~dip_hap_trans['is_mat']]
            new_values_pat = self.values[self.value_names].copy()
            pat_dip_hap_trans.index = pat_dip_hap_trans['hap_index']
            shared_index = new_values_pat.index.intersection(pat_dip_hap_trans.index)
            new_values_pat.index = pat_dip_hap_trans.loc[shared_index, 'index']

            new_values = pd.concat([new_values_pat, new_values_mat], axis=0)

            return Values(new_values, self.gen_index.corr_index)

    def concat_w_other_data(self, other_values, preferred_hap='dip'):
        if self.gen_index.window_df.equals(other_values.gen_index.window_df):
            values1 = self.values
            values2 = other_values.values
            gen_index = self.gen_index
        else:
            if self.gen_index.is_diplo is None:
                self.gen_index.get_is_diplo()
            if other_values.gen_index.is_diplo is None:
                other_values.gen_index.get_is_diplo()

            if self.gen_index.is_diplo == other_values.gen_index.is_diplo:
                values1 = self.values
                values2 = other_values.values
                gen_index = self.gen_index
            else:
                if preferred_hap == 'dip':
                    if self.gen_index.is_diplo:
                        values1 = self.values
                        values2 = other_values.to_dip().values
                        gen_index = self.gen_index
                    else:
                        values1 = self.to_dip().values
                        values2 = other_values.values
                        gen_index = other_values.gen_index
                else:
                    if not self.gen_index.is_diplo:
                        values1 = self.values
                        values2 = other_values.to_hap().values
                        gen_index = self.gen_index
                    else:
                        values1 = self.to_hap().values
                        values2 = other_values.values
                        gen_index = other_values.gen_index

        concat_values = pd.concat([values1, values2], axis=1, join='inner')
        concat_values = Values(concat_values, gen_index)
        return concat_values

    def condense(self, condense_fold=10, inplace=False):
        new_index, index_trans = self.gen_index.condense(condense_fold)
        # print(index_trans.head(10))

        new_values = self.values.copy()
        new_values['new_index'] = index_trans.loc[self.values.index]

        new_values = new_values.groupby('new_index').mean()
        new_values.index.rename('index', inplace=True)

        if inplace:
            self.values = new_values
            self.gen_index = new_index
        else:
            return Values(new_values, new_index)


def concat_values(value_lists, gen_window):
    con_values = [v.values for v in value_lists]
    con_values = pd.concat(con_values, axis=0)
    return Values(con_values, gen_window)


def find_chro_match_df(df, chro, case=False):
    chr_df = df[df['chr'].str.match(chro, case=case)]

    if chr_df.shape[0] == 0:
        chr_df = df[df['chr'].str.match(f'chr{chro}', case=case)]

    if chr_df.shape[0] == 0 and 'chr' == chro[:3]:
        chr_df = df[df['chr'].str.match(chro[3:], case=case)]

    return chr_df


def window_from_chrom_size(chrom_size_file, window_size):
    ran_num = np.random.rand()

    os.system(f"bedtools makewindows -g {chrom_size_file} -w {window_size} | "
              "awk -v OFS=\"\\t\" 'BEGIN{i=0}{print $1, $2, $3, i; i+=1}' > "
              f"tmp{ran_num}.window.bed")

    window = GenomeIndex(f"tmp{ran_num}.window.bed")

    os.system(f"rm tmp{ran_num}.window.bed")

    return window


def save_plot_pdf(out_file):
    import matplotlib as mpl

    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    import matplotlib.pyplot as plt

    if out_file[-4:] != '.pdf':
        out_file = out_file + '.pdf'

    plt.tight_layout()
    plt.savefig(out_file, transparent=True)
    plt.close()


##############################
# HiC Data Class
class ScaledGenomicDistance:
    """
    Resize the genomic distances, so that group long-distance bins.
    In case that there are incontinuous size label. Reordered Size will be used for plot.
    BUT raw size will be showed in plot as ticks.
    for example,
    Genomic distance    Raw size    Reordered Size
    1                   1           0
    2                   3           1
    3                   5           2
    4                   5           2
    5                   6           3
    6                   6           3
    7                   6           3
    ...                 ...         ...
    """
    def __init__(self, k=.5, s=2e4, e=6e7, bin_size=2e4, 
                 auto_change_end=True, input_dict=None):
        self.paras = {'k': k, 's': s, 'e': e, 'bin_size': bin_size,
                      'auto_change_end': auto_change_end}
        
        if isinstance(input_dict, dict):
            for key in input_dict:
                self.paras[key] = input_dict[key]
        elif isinstance(input_dict, str):
            input_dict = json.loads(input_dict.split('#sgd: ')[1])
        elif input_dict is None:
            input_dict = {}
        else:
            raise ValueError(f'Unrecognized type of input_dict: '
                             f'{input_dict.type}')
        
        for key in input_dict:
            self.paras[key] = input_dict[key]
        
        self.size_transform_df = None
        self.size_dict = None
        self.max_dis = None
        self.min_dis = None
        self.n_bins_of_sizes = None
        self.max_size = None
        self.size_ranges = None
        
        self.get_sgd()
    
    def __getitem__(self, dis, force_mode=None):
        """Get corresponding sgd from raw genomic distance or matrix distance (raw distance / bin_size)

        Args:
            dis (int): raw distance or matrix distance.
            Will automatically checking if it's raw distance or matrix distance, by
                matrix distance is normally smaller than bin_size.
            One can force to be raw or matrix by force_mode.
            force_mode (str, optional): Force to be raw or matrix distance. Defaults to None.
                'matrix' for matrix distance.
                'raw' for raw genomic distance.
        """
        if isinstance(dis, np.ndarray):
            if force_mode is None:
                if np.min(dis[dis > 0]) >= self.bin_size:
                    dis = dis / self.bin_size
            else:
                if force_mode == 'raw':
                    dis = dis / self.bin_size
            
            dis[(dis <= self.max_dis) & (dis >= self.min_dis)] = dis[(dis <= self.max_dis) & (dis >= self.min_dis)].astype(int)
            dis[dis <= 0] = 1
            
            trans_dis = dis[(dis <= self.max_dis) & (dis >= self.min_dis)]
            out_trans_sgd = np.vectorize(self.size_dict.get)(trans_dis)
            out_sgd = dis.copy().astype(float)
            out_sgd[dis > self.max_dis] = np.nan
            out_sgd[dis < self.min_dis] = np.nan
            out_sgd[(dis <= self.max_dis) & (dis >= self.min_dis)] = out_trans_sgd.astype(int)
            return out_sgd
        
        else:
            if force_mode is None:
                if not dis < self.bin_size:
                    dis = dis / self.bin_size
            else:
                if force_mode == 'raw':
                    dis = dis / self.bin_size
                
            if dis in self.size_dict:
                return self.size_dict[dis]
            else:
                return None

    def plot_sgd(self):
        x = np.array(list(self.size_dict.keys()))
        y = np.array(list(self.size_dict.values()))
        
        sns.set_style('ticks')
        plt.figure(figsize=(4, 3))
        
        # plt.axvline(x=35, c='maroon', alpha=.5)
        # plt.text(x=100, y=9.4, s='x=35', c='maroon')
        
        sns.lineplot(x=x, y=y)
        sns.scatterplot(x=x, y=y)
        
        plt.xlabel(r'Genomic distance ($\times$20kb)')
        plt.ylabel('Scaled Genomic distance')
        
        plt.grid(axis='y')
        
        sns.despine(trim=True)
        plt.show()
        plt.close()

    def get_sgd(self):
        self.auto_change_end = self.paras['auto_change_end']
        self.e = self.paras['e']
        self.s = self.paras['s']
        self.k = self.paras['k']
        self.bin_size = self.paras['bin_size']
        
        """        
        ################
        # If auto_change_end, will assure that the last SGD has enough genomic bins.
        # If max distance (self.e) is nearer to smaller separation (np.floor((np.log2(self.e) + self.k) / self.k))
        # rather than larger one (np.ceil((np.log2(self.e) + self.k) / self.k)), the max distance will be set to smaller sep.
        if self.auto_change_end:
            cur_max_size = (np.log2(self.e) + self.k) / self.k
            smaller_sep = np.power(2, self.k * np.floor(cur_max_size) - self.k)
            larger_sep = np.power(2, self.k * np.ceil(cur_max_size) - self.k)
            
            if np.abs(self.e - smaller_sep) < np.abs(self.e - larger_sep):
                e = smaller_sep
            else:
                e = self.e
        else:
            e = self.e
        
        self.max_dis = e / self.bin_size
        """
        
        ################
        # If auto_change_end, the max size will trimed to smaller sep.
        if self.auto_change_end:
            cur_max_size = (np.log2(self.e) + self.k) / self.k
            smaller_sep = np.power(2, self.k * np.floor(cur_max_size) - self.k)
            e = smaller_sep
        else:
            e = self.e
        
        self.min_dis = self.s / self.bin_size
        self.max_dis = e / self.bin_size
        
        ################
        # get size_dict
        size_max = int(np.floor((np.log2(e) + self.k) / self.k)) + 1

        size_dict = {}
        for diff in range(int(self.s), int(e) + 1, int(self.bin_size)):
            size = np.floor((np.log2(diff) + self.k) / self.k)

            if size >= size_max:
                break

            mtx_diff = int(diff / self.bin_size)
            size_dict[mtx_diff] = size
            
        ################
        # get a continuous size number
        self.size_transform_df = {'raw_size': []}
        
        raw_sizes = sorted(list(set(list(size_dict.values()))))
        reordered_sizes = [i for i in range(len(raw_sizes))]
        reordered_size_dict = {raw_sizes[i_size]: i_size 
                               for i_size in range(len(raw_sizes))}
        
        self.size_dict = {mtx_diff: reordered_size_dict[size_dict[mtx_diff]] 
                          for mtx_diff in size_dict}

        ################
        # get reverse dicts for output
        re_size_dict = {'dis': [], 'size': []}
        for dis in self.size_dict:
            re_size_dict['dis'].append(int(dis))
            re_size_dict['size'].append(int(self.size_dict[dis]))

        re_size_dict = pd.DataFrame(re_size_dict)
        re_size_dict = re_size_dict.groupby('size').mean()['dis']
        re_size_dict = re_size_dict.to_dict()

        tick_labels = [f'{np.round(re_size_dict[size]  * self.bin_size / 1e6, 2)}M'
                       for size in reordered_sizes]
        
        ################
        # get size transformation
        self.size_transform_df = pd.DataFrame({'raw_size': raw_sizes,
                                               'reordered_size': reordered_sizes,
                                               'median_dis': [re_size_dict[s] for s in reordered_sizes],
                                               'median_gen_dis': [re_size_dict[s] * self.bin_size for s in reordered_sizes],
                                               'ticks': tick_labels})
        
        self.max_size = np.max(list(self.size_dict.values()))
        
        
        size_dict_values = np.array(list(self.size_dict.values()))
        # Bins located at two directions.
        self.n_bins_of_sizes = {size: np.count_nonzero(size_dict_values==size) * 2 for size in reordered_sizes}
        
        size_dict_keys = np.array(list(self.size_dict.keys()))
        self.size_ranges = {}
        for size in reordered_sizes:
            size_keys = size_dict_keys[size_dict_values==size]
            self.size_ranges[size] = [np.min(size_keys), np.max(size_keys)]
        
        # print(f'####### Sgd GOT.')
        return size_dict

    def out_paras_to_string(self):
        return f'#sgd: {json.dumps(self.paras)}\n'


class HiCMtx:
    def __init__(self):
        self.mtx_file = None
        self.mtx_name = None
        self.matrix = None

        self.row_window = None
        self.col_window = None

        # self.equal_window = None

        self.mtx_type = None
        self.bin_size = None

        self.intra = None
        self.has_neg = False

        self.sub_all_mtx = None

    def __getitem__(self, item):
        return self.matrix[item]

    def __setitem__(self, key, item):
        self.matrix[key] = item

    def copy(self):
        return copy.copy(self)

    def get_row_sum(self, axis=0):
        if sps.issparse(self.matrix):
            return np.squeeze(np.array(self.matrix.sum(axis=axis)))
        else:
            return self.matrix.sum(axis=axis)
    
    def get_nonzero_ratio(self, only_intra=False):
        if not only_intra:
            shape = self.matrix.shape
            
            if self.mtx_type is None:
                self.get_mtx_type()
                
            if self.mtx_type == 'triu' or self.mtx_type == 'tril':
                total_num = int((shape[0] * shape[1] - shape[0]) / 2 + shape[0])
            else:
                total_num = shape[0] * shape[1] 
            
            if total_num <= 0:
                return 0
            
            if isinstance(self, ArrayHiCMtx):
                nonzero_num = np.count_nonzero(self.matrix.reshape((-1,)))
            else:
                nonzero_num = np.count_nonzero(self.matrix.data)
            return nonzero_num / total_num
        else:
            nonzero_ratios = []
            for _, chr_mtx in self.yield_by_chro(only_intra=True):
                # print(chr_mtx.get_nonzero_ratio())
                nonzero_ratios.append(chr_mtx.get_nonzero_ratio())
            return np.mean(nonzero_ratios)
        
    def get_fre_counts(self, sgd=None, max_size=None, zero_included=False):
        """
        Get frequency counts by different size.

        :param sgd: ScaledGenomicDistance
        :param zero_included: Included zero contacts when computing averaged Hi-C counts.
        :return:
        """
        coo_mtx = self.to_all(inplace=False).matrix

        
        if sps.issparse(coo_mtx):
            coo_mtx = coo_mtx.tocoo()
        else:
            coo_mtx = sps.csr_matrix(coo_mtx).tocoo()
        
        row, col, data = coo_mtx.row, coo_mtx.col, coo_mtx.data

        dises = np.abs(row - col)

        row, data = row[dises > 0], data[dises > 0]
        dises = dises[dises > 0]

        if sgd is None:
            max_size = 100 if max_size is None else max_size
            row, data = row[dises < max_size], data[dises < max_size]
            sizes = dises[dises < max_size]
        else:
            max_size = max(list(sgd.size_dict.keys()))
            row, data = row[dises < max_size], data[dises < max_size]
            dises = dises[dises < max_size]
            sizes = sgd[dises * self.row_window.bin_size]

        fre_counts = pd.DataFrame({'size': sizes, 'fre': data, 'row': row})
        
        if zero_included and sgd is not None:
            fre_counts = fre_counts.groupby(by=['row', 'size']).sum()
        else:
            fre_counts = fre_counts.groupby(by=['row', 'size']).mean()
        
        fre_counts.reset_index(inplace=True)
        fre_counts = fre_counts.pivot(index='row', columns='size', values='fre')
        fre_counts.fillna(value=0, inplace=True)
        
        if zero_included and sgd is not None:
            for col in fre_counts.columns:
                fre_counts[col] /= sgd.n_bins_of_sizes[col]

        fre_counts.columns = fre_counts.columns.astype(int)
    
        fre_counts = FreCounts(fre_counts, sgd=sgd, input_index=self.row_window)
        
        return fre_counts
    
    def obs_d_exp(self, mode='sgd_mean', k=.2, verbose=False, only_intra=False,
                  return_exp_counts=False, counts_must_decrese=True, min_dis_mean=1):
        """ Generate Observed / Expected Hi-C counts map.
            Yield for every chro, three modes for intra-chromosome.

        Args:
            mode (str, optional): Modes for intra-chromosome Obs/Exp. Defaults to 'diag_mean'.
                "diag_mean": Exp counts are the mean of each diagonal.
                "sgd_mean":  Exp counts are the mean of diagonals in each sgd.
                                For distance larger than max_size in sgd,
                                will use the last sgd_mean as mean.
                "fit":       Fit a counts_vs_distance line for exp counts. 
        """
        def _diag_mean(chr_mtx):
            diag_means = []
            for _, diag_index in chr_mtx.yield_diag_idxes():
                data_line = chr_mtx.matrix[diag_index]
                diag_mean = data_line.sum() / len(diag_index[0])
                diag_mean = 1 if diag_mean == 0 else diag_mean
                diag_means.append(diag_mean)
            return np.array(diag_means)
        
        def _get_exp_from_fit(chr_mtx, fit_range=30, tid_off_zero=False,
                              plot_curve=False):
            fit_x, fit_y = [], []
            fit_range = fit_range if fit_range <= chr_mtx.shape[0] else chr_mtx.shape[0]
            for i, diag_index in chr_mtx.yield_diag_idxes():
                if i == 0:
                    continue
                if i > fit_range:
                    break

                fit_x.append(i)
                if tid_off_zero:
                    data_line = chr_mtx.matrix[diag_index]
                    fit_y.append(np.mean(data_line[data_line > 0]))
                else:
                    fit_y.append(np.mean(chr_mtx.matrix[diag_index]))

            fit_y = np.log(np.array(fit_y))
            fit_x = np.log(np.array(fit_x))
            k, b = np.polyfit(fit_x, fit_y, 1)

            if plot_curve:
                plt.plot(fit_x, fit_y, 'rx')
                plt.plot(fit_x, k * fit_x + b, 'k-')
                plt.xlabel('Log(Genomic distance)')
                plt.ylabel('Log(Average Hi-C counts)')
                sns.despine(trim=True, offset=2)
                plt.show()
            
            exp_counts = []
            for idx in range(chr_mtx.shape[0]):
                if idx == 0:
                    dis_mean = np.mean(chr_mtx.matrix[np.diag_indices_from(chr_mtx.matrix)])
                else:
                    dis_mean = np.power(np.e, b + k * np.log(idx))
                    dis_mean = dis_mean if dis_mean >= min_dis_mean else min_dis_mean
                exp_counts.append(dis_mean)
            return np.array(exp_counts)
        
        def _sgd_mean(chr_mtx, sgd):
            chr_mean_fre_counts = {}
            cur_sgd = None
            data_lines, data_num = [], 0
            for idx, diag_index in chr_mtx.yield_diag_idxes():
                i_sgd = sgd[idx]
                if i_sgd is None:
                    continue
                
                if cur_sgd is None:
                    data_lines += list(chr_mtx.matrix[diag_index])
                    data_num += len(diag_index[0])
                    cur_sgd = i_sgd
                elif cur_sgd == i_sgd:
                    data_lines += list(chr_mtx.matrix[diag_index])
                    data_num += len(diag_index[0])
                else:
                    chr_mean_fre_counts[cur_sgd] = np.sum(data_lines) / data_num
                    data_lines = list(chr_mtx.matrix[diag_index])
                    data_num = len(diag_index[0])
                    cur_sgd = i_sgd

            sgd_means = []
            for idx in range(chr_mtx.shape[0]):
                i_sgd = sgd[idx]
                if i_sgd is None or i_sgd not in chr_mean_fre_counts:
                    if idx == 0:
                        sgd_mean = np.mean(chr_mtx.matrix[np.diag_indices_from(chr_mtx.matrix)])
                    else:
                        sgd_mean = chr_mean_fre_counts[np.max(list(chr_mean_fre_counts.keys()))]
                else:
                    sgd_mean = 1 if chr_mean_fre_counts[i_sgd] == 0 else chr_mean_fre_counts[i_sgd]
                sgd_means.append(sgd_mean)
            
            return sgd_means
        
        new_chr_mtxes = []
        for chro1, chro2, chr_mtx in self.yield_by_chro():
            if chro1 == chro2:
                if verbose:
                    print(f'Ode {chro1} vs {chro2}')
                
                if isinstance(chr_mtx, SpsHiCMtx):
                    chr_mtx = chr_mtx.to_dense()
                
                chr_mtx.matrix = chr_mtx.matrix.astype(float)
                
                if chr_mtx.shape[0] <= 1:
                    chr_mtx.matrix = chr_mtx.matrix / np.mean(chr_mtx.matrix)
                    new_chr_mtxes.append(chr_mtx)
                    continue
                
                if np.sum(chr_mtx.matrix) == 0:
                    new_chr_mtxes.append(chr_mtx)
                    continue
                
                if mode == 'diag_mean':
                    exp_counts = _diag_mean(chr_mtx)
                elif mode == "sgd_mean":
                    sgd = ScaledGenomicDistance(s=chr_mtx.row_window.bin_size,
                                                e=chr_mtx.row_window.bin_size * chr_mtx.shape[0],
                                                bin_size=chr_mtx.row_window.bin_size,
                                                k=k)
                    exp_counts = _sgd_mean(chr_mtx, sgd)
                elif mode == "fit":
                    exp_counts = _get_exp_from_fit(chr_mtx)
                else:
                    raise ValueError('Unrecognized mode')
                
                if counts_must_decrese:
                    cur_counts = None
                    for idx in range(len(exp_counts)):
                        if cur_counts is None:
                            if exp_counts[idx] != 0:
                                cur_counts = exp_counts[idx]
                        elif exp_counts[idx] > cur_counts:
                            exp_counts[idx] = cur_counts
                        else:
                            if exp_counts[idx] != 0:
                                cur_counts = exp_counts[idx]
                
                for i, diag_idx in chr_mtx.yield_diag_idxes():
                    if exp_counts[i] <= 0:
                        if verbose:
                            warnings.warn(f'Exp count is below 0 for idx {i}. will skip for ode.')
                        continue
                    chr_mtx.matrix[diag_idx] = chr_mtx.matrix[diag_idx] / exp_counts[i]

                new_chr_mtxes.append(chr_mtx)
            
            elif not only_intra:
                if verbose:
                    print(f'Ode {chro1} vs {chro2}')
                if isinstance(chr_mtx, SpsHiCMtx):
                    chr_mtx = chr_mtx.to_dense()
                
                nonzero_index = np.nonzero(chr_mtx.matrix)
                if len(nonzero_index[0]) > 0:
                    chr_mtx.matrix = chr_mtx.matrix / np.mean(chr_mtx.matrix[nonzero_index])
                
                new_chr_mtxes.append(chr_mtx)
        
        new_mtx = concat_mtxes(new_chr_mtxes, copy_mtx_paras=self)
        new_mtx = new_mtx.to_dense()
        
        if return_exp_counts:
            return new_mtx, exp_counts
        else:
            return new_mtx

    ##############
    # convert mtx type
    def get_mtx_type(self):
        if sps.issparse(self.matrix):
            triu_func, tril_func = sps.triu, sps.tril
        elif isinstance(self.matrix, np.ndarray):
            triu_func, tril_func = np.triu, np.tril
        else:
            raise ValueError(f'Unrecognized matrix type: {self.matrix.type}')

        if self.matrix.sum() <= 0:
            warnings.warn('Sum of input matrix less than 0.')
            mtx_type = 'all'

        if self.matrix.shape[0] != self.matrix.shape[1]:
            mtx_type = 'all'

        diag = self.matrix.diagonal().sum()
        tril_sum = tril_func(self.matrix, k=-1).sum()
        triu_sum = triu_func(self.matrix, k=1).sum()

        if diag != 0 and tril_sum == 0 and triu_sum == 0:
            mtx_type = 'triu'
        elif tril_sum != 0 and triu_sum == 0:
            mtx_type = 'tril'
        elif tril_sum == 0 and triu_sum != 0:
            mtx_type = 'triu'
        else:
            mtx_type = 'all'

        self.mtx_type = mtx_type

    def get_has_neg(self):
        pass

    def to_all(self, inplace=False):
        pass

    def to_triu(self, k=0, inplace=False):
        pass
    
    def if_obey_mtx_type(self, chro1, chro2):
        if self.mtx_type is None:
            self.get_mtx_type()

        if self.mtx_type == 'all':
            return True

        elif self.mtx_type == 'triu':
            if self.row_window.chr_seps[chro1] > self.col_window.chr_seps[chro2]:
                return False
            else:
                return True

        elif self.mtx_type == 'tril':
            if self.row_window.chr_seps[chro1] < self.col_window.chr_seps[chro2]:
                return False
            else:
                return True

        else:
            raise ValueError(f'Unrecognized mtx type: {self.mtx_type}')

    def transpose(self):
        row_window, col_window = self.col_window, self.row_window
        matrix = self.matrix.T
        
        if isinstance(self, ArrayHiCMtx):
            return ArrayHiCMtx(matrix, copy_mtx_paras=self, 
                                row_window=row_window, 
                                col_window=col_window,
                                mtx_type=None)
        else:
            return SpsHiCMtx(matrix, copy_mtx_paras=self, 
                                row_window=row_window, 
                                col_window=col_window,
                                mtx_type=None)

    ##############
    # get sub regions (including chro yield)
    def yield_by_chro(self, only_intra=False, triu=False):
        for chro1 in self.row_window.chros:
            for chro2 in self.col_window.chros:

                if triu and self.row_window.chr_seps[chro1][0] > self.col_window.chr_seps[chro2][0]:
                    continue

                if chro1 != chro2 and only_intra:
                    continue

                chr_mtx = self.get_region_mtx(chro1, chro2=chro2)
                chr_mtx.row_window.ori_start_idx = self.row_window.chr_seps[chro1][0]
                chr_mtx.col_window.ori_start_idx = self.col_window.chr_seps[chro2][0]

                chr_mtx.intra = True if chro1 == chro2 else False
                chr_mtx.mtx_type = self.mtx_type if chro1 == chro2 else 'all'
                chr_mtx.has_neg = self.has_neg

                if triu and chro1 == chro2:
                    chr_mtx.to_triu(inplace=True)
                
                if only_intra:
                    yield chro1, chr_mtx
                else:
                    yield chro1, chro2, chr_mtx

    def get_region_mtx(self, chro, start=None, end=None, 
                       chro2=None, start2=None, end2=None,
                       col_is_inter=False, col_is_all=False, inplace=False):
        """Get sub-region of matrix

        Args:
            Row is chro:start-end.
            Col is default to the same of row if None, else chro2:start2-end2.
            If col_is_inter, col is all indexes not including row.

        Returns:
            HiCMtx: _description_
        """
        start_idx, end_idx = self.row_window.get_region_startend_idx(chro, starts=start, ends=end)

        chr_row_window = self.row_window.window_df.loc[start_idx: end_idx, :].copy()
        chr_row_window.loc[:, 'ori_index'] = chr_row_window["index"].copy()
        chr_row_window.loc[:, "index"] = chr_row_window["index"].apply(lambda x: x - np.min(chr_row_window['index']))
        chr_row_window = GenomeIndex(chr_row_window)
        chr_row_window.ori_start_idx = start_idx

        if col_is_inter:
            chr_len = end_idx - start_idx + 1
            before_mtx = self.matrix[start_idx: end_idx + 1, :start_idx] if start_idx > 0 else np.empty((chr_len, 0))
            before_mtx = np.array(before_mtx.todense()) if sps.issparse(before_mtx) else before_mtx
            after_mtx = self.matrix[start_idx: end_idx + 1, end_idx + 1:] if end_idx + 1 < self.matrix.shape[0] else np.empty((chr_len, 0))
            after_mtx = np.array(after_mtx.todense()) if sps.issparse(after_mtx) else after_mtx
            new_matrix = np.hstack((before_mtx, after_mtx))
            
            chr_col_before_window = self.col_window.window_df.loc[: start_idx, :].copy()
            chr_col_before_window.drop(index=[start_idx], inplace=True)
            chr_col_after_window = self.col_window.window_df.loc[end_idx:, :].copy()
            chr_col_after_window.drop(index=[end_idx], inplace=True)
            chr_col_window = pd.concat([chr_col_before_window, chr_col_after_window], axis=0)
            
            if chr_col_window.shape[0] == 0:
                return None
            
            chr_col_window.loc[:, 'ori_index'] = chr_col_window["index"].copy()
            chr_col_window.loc[:, "index"] = chr_col_window["index"].apply(lambda x: x - np.min(chr_col_window['index']))
            chr_col_window = GenomeIndex(chr_col_window)
            chr_col_window.ori_start_idx = self.col_window.ori_start_idx
            
            intra = False
            mtx_type = 'all'
        
        elif col_is_all:
            chr_len = end_idx - start_idx + 1
            new_matrix = self.matrix[start_idx: end_idx + 1, :]
            chr_col_window = self.col_window
            intra = False
            mtx_type = 'all'
        
        else:
            chro2 = chro if chro2 is None else chro2
            start2 = start if start2 is None else start2
            end2 = end if end2 is None else end2
            start_idx2, end_idx2 = self.col_window.get_region_startend_idx(chro2, starts=start2, ends=end2)

            new_matrix = self.matrix[start_idx: end_idx + 1, start_idx2: end_idx2 + 1]

            chr_col_window = self.col_window.window_df.loc[start_idx2: end_idx2, :].copy()
            chr_col_window.loc[:, 'ori_index'] = chr_col_window["index"].copy()
            chr_col_window.loc[:, "index"] = chr_col_window["index"].apply(lambda x: x - np.min(chr_col_window['index']))
            chr_col_window = GenomeIndex(chr_col_window)
            chr_col_window.ori_start_idx = start_idx2

            intra = True if chro == chro2 and start == start2 and end == end2 else False
            mtx_type = 'all' if chro != chro2 else self.mtx_type

        if inplace:
            self.row_window = chr_row_window
            self.col_window = chr_col_window
            self.intra = intra
            self.mtx_type = mtx_type
            self.matrix = new_matrix
        else:
            if sps.issparse(new_matrix):
                return SpsHiCMtx(new_matrix,
                                 mtx_file=self.mtx_file,
                                 row_window=chr_row_window,
                                 col_window=chr_col_window, 
                                 mtx_type=mtx_type,
                                 intra=intra,
                                 copy_mtx_paras=self)
            else:
                return ArrayHiCMtx(new_matrix,
                                 mtx_file=self.mtx_file,
                                 row_window=chr_row_window,
                                 col_window=chr_col_window, 
                                 mtx_type=mtx_type,
                                 intra=intra,
                                 copy_mtx_paras=self)

    def get_multi_chros_mtx(self, chros):
        new_row_window, kept_index = self.row_window.keep_chros(chros, return_old_index=True)
        new_matrix = self.matrix[kept_index, :]
        
        new_col_window, kept_index = self.col_window.keep_chros(chros, return_old_index=True)
        new_matrix = new_matrix[:, kept_index]
        
        if isinstance(self, ArrayHiCMtx):
            return ArrayHiCMtx(new_matrix, copy_mtx_paras=self,
                               row_window=new_row_window, 
                               col_window=new_col_window)
        else:
            return SpsHiCMtx(new_matrix, copy_mtx_paras=self, 
                             row_window=new_row_window, 
                             col_window=new_col_window)

    def get_one_row(self, idx, length=None, return_idx=False):
        """Return row with length: 2 * length or mtx.shape[0] if length is None."""
        if self.mtx_type == 'triu' or self.mtx_type == 'all':
            if sps.issparse(self.matrix):
                row = np.squeeze(np.array(self.matrix[idx, :].todense()))
                up_row = np.squeeze(np.array(self.matrix[:, idx].todense()))
            else:
                row = self.matrix[idx, :]
                up_row = self.matrix[:, idx]
            row[:idx] = up_row[:idx]
        else:
            if sps.issparse(self.matrix):
                row = np.squeeze(np.array(self.matrix[:, idx].todense()))
                up_row = np.squeeze(np.array(self.matrix[idx, :].todense()))
            else:
                row = self.matrix[:, idx]
                up_row = self.matrix[idx, :]
            row[idx:] = up_row[idx:]

        start, end = 0, len(row)
        if length is not None:
            start = idx - length if idx > length else 0
            end = idx + length if idx + length < len(row) else len(row)
            row = row[start:end]
        
        if not return_idx:
            return row
        else:
            return row, np.arange(start, end)

    def yield_diag_idxes(self, mtx_type=None, stick_out_zero=False):
        if self.mtx_type is None:
            self.get_mtx_type()
        
        rows, cols = np.diag_indices_from(self.matrix)
        mtx_type = self.mtx_type if mtx_type is None else mtx_type

        for i in range(self.matrix.shape[0]):
            # Get the index of lines parallel to diagonal
            if mtx_type == 'all':
                row = np.append(rows[i:], cols[:-i])
                col = np.append(cols[:-i], rows[i:])
            elif mtx_type == 'triu':
                row = rows[:-i]
                col = cols[i:]
            elif mtx_type == 'tril':
                row = rows[i:]
                col = cols[:-i]
            else:
                raise ValueError('Unrecognized matrix type.')

            diag_index = row, col
            if i == 0:
                diag_index = rows, cols
                row, col = rows, cols

            if stick_out_zero:
                # Stick out zeros
                diag_nonzero_index = np.nonzero(np.squeeze(np.array(self.matrix[diag_index])))
                if len(diag_nonzero_index[0]) <= 0:
                    continue

                # Normalize
                diag_index = row[diag_nonzero_index], col[diag_nonzero_index]

            yield i, diag_index

    def keep_chros_by_len(self, larger_than_last_ratio=.1, larger_than_max_ratio=.01):
        """ Only keep chromosomes larger than larger_than_last_ratio of last larger chromosome.
            and larger than larger_than_max_ratio of largest chromosome.
            So that tid off small contigs.
        """
        new_row_window, kept_index = self.row_window.keep_chros_by_len(larger_than_last_ratio, 
                                                                     larger_than_max_ratio,
                                                                     return_old_index=True,
                                                                     verbose=True)
        new_matrix = self.matrix[kept_index, :]
        
        new_col_window, kept_index = self.col_window.keep_chros_by_len(larger_than_last_ratio, 
                                                                     larger_than_max_ratio,
                                                                     return_old_index=True,
                                                                     verbose=False)
        new_matrix = new_matrix[:, kept_index]
        
        if isinstance(self, ArrayHiCMtx):
            return ArrayHiCMtx(new_matrix, copy_mtx_paras=self,
                               row_window=new_row_window, 
                               col_window=new_col_window)
        else:
            return SpsHiCMtx(new_matrix, copy_mtx_paras=self, 
                             row_window=new_row_window, 
                             col_window=new_col_window)

    def keep_long_chros(self, min_chro_len=5):
        """ Only keep chromosomes with length larger than min_chro_len (bin number).
            So that tid off small contigs.

        Args:
            min_chro_len (int, optional): Minimum chromosome length, in bin number. Defaults to 5.
        """
        new_row_window, kept_index = self.row_window.keep_long_chros(min_chro_len, 
                                                                     return_old_index=True,
                                                                     verbose=True)
        if new_row_window is None:
            warnings.warn(f'No chros left by min_chro_len {min_chro_len}.')
            return None
        new_matrix = self.matrix[kept_index, :]
        
        new_col_window, kept_index = self.col_window.keep_long_chros(min_chro_len, 
                                                                     return_old_index=True,
                                                                     verbose=False)
        if new_row_window is None:
            warnings.warn(f'No chros left by min_chro_len {min_chro_len}.')
            return None
        new_matrix = new_matrix[:, kept_index]
        
        if isinstance(self, ArrayHiCMtx):
            return ArrayHiCMtx(new_matrix, copy_mtx_paras=self,
                               row_window=new_row_window, 
                               col_window=new_col_window)
        else:
            return SpsHiCMtx(new_matrix, copy_mtx_paras=self, 
                             row_window=new_row_window, 
                             col_window=new_col_window)

    def tid_off_zero_chros(self, tid_off_zero_ratio=.99):
        """ Only keep chromosomes with intra sum larger than 0.
        """
        kept_chros = []
        tid_off_chros = []
        for chro, chr_mtx in self.yield_by_chro(only_intra=True):
            nonzero_ratio = chr_mtx.get_nonzero_ratio()
            if 1 - nonzero_ratio <= tid_off_zero_ratio:
                kept_chros.append(chro)
            else:
                tid_off_chros.append(chro)
        
        if len(tid_off_chros) > 0:
            print(f'{" ".join(tid_off_chros)} contains too much zero.')
        
        new_row_window, kept_index = self.row_window.keep_chros(kept_chros, return_old_index=True)
        new_matrix = self.matrix[kept_index, :]
        
        new_col_window, kept_index = self.col_window.keep_chros(kept_chros, return_old_index=True)
        new_matrix = new_matrix[:, kept_index]
        
        if isinstance(self, ArrayHiCMtx):
            return ArrayHiCMtx(new_matrix, copy_mtx_paras=self, 
                               row_window=new_row_window, 
                               col_window=new_col_window)
        else:
            return SpsHiCMtx(new_matrix, copy_mtx_paras=self, 
                             row_window=new_row_window, 
                             col_window=new_col_window)

    def reorder_chros_by_len(self):
        chr_lens = self.row_window.chr_lens
        sorted_chros = sorted(chr_lens.items(), key=lambda x:x[1], reverse=True)
        sorted_chros = [i[0] for i in sorted_chros]
        return self.get_multi_chros_mtx(sorted_chros)

    ##############
    # tid off bins or values
    def get_high_coverage_bins(self, ratio=10):
        def _get_axis_sum(mtx, axis):
            bin_sum = mtx.sum(axis=axis)
            if sps.issparse(bin_sum):
                bin_sum = np.squeeze(np.array(bin_sum.todense()))
            elif isinstance(bin_sum, np.matrix):
                bin_sum = np.squeeze(np.array(bin_sum))
            return bin_sum

        bin_sum = _get_axis_sum(self.matrix, axis=0) + _get_axis_sum(self.matrix, axis=1)
        bin_window = self.row_window
        remain_idxes = bin_window.window_df.iloc[bin_sum > np.percentile(bin_sum, ratio)]['index']
        return remain_idxes

    def tid_off_low_bins(self, ratio=10):
        remain_idxes = self.get_high_coverage_bins(ratio)
        print(f'### Tid off bins. Remain: {len(remain_idxes)}/{self.matrix.shape[0]}')
    
        new_mtx = self.matrix[remain_idxes, :]
        new_mtx = new_mtx[:, remain_idxes]
        
        new_row_window = self.row_window.get_sub_gen_index(idxes=remain_idxes)
        new_col_window = self.col_window.get_sub_gen_index(idxes=remain_idxes)
    
        if isinstance(self, ArrayHiCMtx):
            return ArrayHiCMtx(new_mtx, copy_mtx_paras=self, 
                                row_window=new_row_window, 
                                col_window=new_col_window)
        else:
            return SpsHiCMtx(new_mtx, copy_mtx_paras=self,
                                row_window=new_row_window, 
                                col_window=new_col_window)

    def get_sep_by_near_num(self, k, adjust=True, n_ran=5000, sep_change_ratio=.5):
        def _get_k_change(i_k, k):
            max_change_ratio = .5
            min_k_change = 3

            k_change = sep_change_ratio * np.abs(i_k - k)
            # print(f'Ori_k_change:{k_change}', end=' ')

            k_change = k_change if k_change < i_k * max_change_ratio else i_k * max_change_ratio
            k_change = min_k_change if k_change < min_k_change else k_change
            # print(f'Final:{k_change}')
            return k_change

        def _estimate_sep(t, type='n_near'):
            if type == 'per':
                per = t
            elif type == 'n_near':
                per = (1 - t / self.sub_all_mtx.shape[1]) * 100
            else:
                raise ValueError('Unrecognized per type.')

            seps = np.percentile(self.sub_all_mtx, q=per, axis=1)
            sep = np.mean(seps)
            return sep
        
        def _get_ndarray_sub_all_mtx(self, n_ran=5000):
            mtx = self.copy()
            mtx.to_all(inplace=True)

            non_zero_cols = np.where(self.get_row_sum() > 0)[0]

            if n_ran is not None and mtx.matrix.shape[0] > n_ran:
                ran_idxes = np.random.choice(non_zero_cols, n_ran, replace=False)
                sub_mtx_all = mtx.matrix[ran_idxes, :]

                if sps.issparse(sub_mtx_all):
                    sub_mtx_all = np.array(sub_mtx_all.todense())

                self.sub_all_mtx = sub_mtx_all
            else:
                self.sub_all_mtx = mtx.matrix[non_zero_cols, :]

        def _get_near_num(self, sep, n_ran=5000):
            if self.sub_all_mtx is None:
                self._get_ndarray_sub_all_mtx(n_ran)

            sub_all_mtx = self.sub_all_mtx.copy()

            sub_all_mtx[sub_all_mtx < sep] = 0
            near_nums = np.count_nonzero(sub_all_mtx, axis=1)
            return int(np.mean(near_nums))
        
        self._get_ndarray_sub_all_mtx(n_ran)

        sep = _estimate_sep(k)
        est_k = self._get_near_num(sep)
        print(f'#ori_near_num:{est_k}')
        print(f'#k:{k}')
        if not adjust:
            return sep

        tol_diff = 2
        n_sep = 30

        # To find best k
        if np.abs(k - est_k) < tol_diff:
            print(f'#end_near_num:{est_k}')
            return sep

        if est_k > k:
            lower_sep = sep
            i_k = k - _get_k_change(est_k, k)
            upper_sep = None

            while i_k > 0:
                # print(f'i_k:{i_k}. est_k:{est_k}')
                can_upper_sep = _estimate_sep(i_k)
                est_k = self._get_near_num(can_upper_sep)
                if est_k <= k:
                    upper_sep = can_upper_sep
                    break
                lower_sep = can_upper_sep
                i_k = i_k - _get_k_change(est_k, k)

            if upper_sep is None:
                warnings.warn(f'Cannot find upper_sep for seps.')
                upper_sep = np.mean(np.max(self.sub_all_mtx, axis=1))

        else:
            upper_sep = sep
            i_k = k + _get_k_change(est_k, k)
            lower_sep = None

            while i_k < self.sub_all_mtx.shape[0]:
                can_lower_sep = _estimate_sep(i_k)
                est_k = self._get_near_num(can_lower_sep)
                if can_lower_sep == 0:
                    break
                if est_k >= k:
                    lower_sep = can_lower_sep
                    break
                upper_sep = can_lower_sep
                i_k = i_k + _get_k_change(est_k, k)

            if lower_sep is None:
                warnings.warn(f'Cannot find lower_sep for seps.')
                lower_sep = np.min(self.sub_all_mtx[self.sub_all_mtx > 0])

        seps = np.linspace(upper_sep, lower_sep, n_sep)
        for candidate_sep in seps:
            est_k = self._get_near_num(candidate_sep)
            # print(f'sep: {sep}, near_num: {most_near_num}')
            if est_k >= k:
                sep = candidate_sep
                break

        print(f'#end_near_num:{est_k}')
        print(f'#sep:{sep}')
        return sep

    def tid_off_low_value(self, near_num=100, reverse=False):
        sep = self.get_sep_by_near_num(near_num)
        if reverse:
            self.matrix = self.matrix.multiply(self.matrix < sep)
        else:
            self.matrix = self.matrix.multiply(self.matrix > sep)

    ##############
    # plot
    def plot(self, input_ax=None, fig_size=(10, 10), cmap="Reds", vmax_per=None, out_file=None,
             chro=None, start=None, end=None, chro2=None, start2=None, end2=None):
        max_plot_shape_size = 5000
        
        if self.row_window is not None:
            plot_mtx = self.get_region_mtx(chro, start, end, chro2, start2, end2)
        else:
            plot_mtx = self
            
        plot_mtx = plot_mtx.to_all()
        
        if np.max(plot_mtx.matrix.shape) > max_plot_shape_size:
            x_size = plot_mtx.matrix.shape[0] if plot_mtx.matrix.shape[0] < max_plot_shape_size else max_plot_shape_size
            y_size = plot_mtx.matrix.shape[1] if plot_mtx.matrix.shape[1] < max_plot_shape_size else max_plot_shape_size
            plot_mtx.matrix = plot_mtx.matrix[:x_size, :y_size]
            plot_mtx = plot_mtx.to_dense()
        
        # print(np.percentile(plot_mtx.matrix, 99))

        if input_ax is None:
            fig = plt.figure(figsize=fig_size)
            ax = fig.add_subplot(111)
        else:
            ax = input_ax

        if vmax_per is None:
            sns.heatmap(plot_mtx.matrix, ax=ax, cmap=cmap, cbar=False)
        else:
            vmax = np.percentile(plot_mtx.matrix, 100-vmax_per)
            vmin = np.percentile(plot_mtx.matrix, vmax_per)
            print(f'{vmax} {vmin}')
            sns.heatmap(plot_mtx.matrix, ax=ax, cmap=cmap, vmax=vmax, vmin=vmin, cbar=False)

        plt.title(f'{chro}:{start}-{end} vs. \n'
                    f'{chro2}:{start2}-{end2}')

        if input_ax is None:
            if out_file is None:
                plt.show()
            else:
                plt.savefig(out_file)
            plt.close()

    def triangle_plot(self):
        # ndimage.rotate
        # Not used for now.
        pass

    ##############
    # shared functions
    def expand(self, expand_radius=3):
        """expand hic counts by gaussian filter.
            This is actually a methods for smoothening,
            especially for sparse regions (long-range or inter-chromosome).
            
            ratio = k * Gaussian(mean=[0, 0], cov=sigma(this value is sqaured s, s ** 2)).
            Fitting k and s, so that at expand_radius ratio is .1, while at center (0, 0) ratio is 1.

        Args:
            expand_radius (int, optional): maximum radius for gaussian kernal. Defaults to 3.
        """
        
        from scipy.optimize import curve_fit

        ratio_for_longest_filter = .1
        
        """
        Fitting curve: k * Gaussian(mean=[0, 0], cov=sigma)
                       k * (1 / 2 / pi / sigma * np.exp(- x ** 2 / 2 / sigma))
        
        1.  Put (0, 1) in curve.
            Then get k = 2 * pi * sigma
        2.  Put (expand_radius, ratio_for_longest_filter) in curve.
            Then get sigma = - expand_radius ** 2 / 2 / np.log(ratio_for_longest_filter)
        """
        
        x = np.array([-expand_radius, expand_radius, 0])
        y = np.array([ratio_for_longest_filter, ratio_for_longest_filter, 1])

        sigma = - expand_radius ** 2 / 2 / np.log(ratio_for_longest_filter)
        return sigma

    def log(self):
        """Logrithm each element in matrix
        """
        pass

    def normalize(self, mode='percentile', k=90):
        """Generally normalize hic counts.
           Aim for comparing between different Hi-C experiments.

        Args:
            mode (str, optional): Mode for normalize. Defaults to 'percentile'.
            k (int, optional): Paras for norm. Defaults to 90.
        """
        pass

    def condense(self, condense_fold, mean=False):
        """Concatenate the nearby bins to get lower-resolution map.

        Args:
            condense_fold (int): the fold of condense.
            mean (bool, optional): whether to take the average counts. Defaults to False.
        """
        pass


class ArrayHiCMtx(HiCMtx):
    def __init__(self, matrix, mtx_file=None,
                 row_window=None, col_window=None, 
                 mtx_type=None, intra=None, has_neg=None,
                 copy_mtx_paras=None):
        super().__init__()
        self.matrix = matrix
        self.shape = self.matrix.shape

        self.has_neg = None
        if copy_mtx_paras is not None:
            self.mtx_file = copy_mtx_paras.mtx_file
            self.row_window = copy_mtx_paras.row_window
            self.col_window = copy_mtx_paras.col_window
            self.mtx_type = copy_mtx_paras.mtx_type
            self.intra = copy_mtx_paras.intra
            self.has_neg = copy_mtx_paras.has_neg
        
        if row_window is None and col_window is not None:
            self.row_window, self.col_window = col_window, col_window
        elif col_window is None and row_window is not None:
            self.row_window, self.col_window = row_window, row_window
        elif col_window is None and row_window is None:
            if copy_mtx_paras is None:
                self.row_window, self.col_window = None, None
        else:
            self.row_window, self.col_window = row_window, col_window

        if self.row_window is not None:
            self.bin_size = self.row_window.bin_size

        if mtx_file is not None:
            self.mtx_file = mtx_file
        if mtx_type is not None:
            self.mtx_type = mtx_type
        if intra is not None:
            self.intra = intra
        
        if has_neg is not None:
            self.has_neg = has_neg
        elif self.has_neg is None:
            self.get_has_neg()

    ######################
    # Matrix transformation
    def to_sps(self):
        return SpsHiCMtx(sps.csr_matrix(self.matrix), copy_mtx_paras=self)

    def to_dense(self):
        return copy.deepcopy(self)

    def to_all(self, inplace=False):
        if self.mtx_type is None:
            self.get_mtx_type()

        if self.mtx_type == 'all':
            new_mtx = self.matrix
        elif self.mtx_type == 'triu':
            new_mtx = np.triu(self.matrix, k=0).T + np.triu(self.matrix, k=1)
        elif self.mtx_type == 'tril':
            new_mtx = np.tril(self.matrix, k=0).T + np.tril(self.matrix, k=1)
        else:
            raise ValueError(f'Unrecognized mtx type: {self.mtx_type}')

        if inplace:
            self.mtx_type = 'all'
            self.matrix = new_mtx
        else:
            return ArrayHiCMtx(new_mtx, copy_mtx_paras=self, mtx_type='all')

    def to_triu(self, k=0, inplace=False):
        if self.mtx_type is None:
            self.get_mtx_type()

        if self.mtx_type == 'all':
            new_mtx = np.triu(self.matrix, k=k)
        elif self.mtx_type == 'triu':
            new_mtx = np.triu(self.matrix, k=k)
        elif self.mtx_type == 'tril':
            new_mtx = np.triu(self.matrix.T, k=k)
        else:
            raise ValueError(f'Unrecognized mtx type: {self.mtx_type}')

        if inplace:
            self.mtx_type = 'triu'
            self.matrix = new_mtx
        else:
            return ArrayHiCMtx(new_mtx, copy_mtx_paras=self, mtx_type='triu')

    def to_output(self, out_file, out_window=False, out_int=False,
                  float_format="%.4f", compression='infer', only_row_window=True):
        if out_window:
            if only_row_window:
                self.row_window.to_output(change_suffix(out_file, '.window.bed'))
            else:
                self.row_window.to_output(change_suffix(out_file, '.row_window.bed'))
                self.col_window.to_output(change_suffix(out_file, '.col_window.bed'))

        matrix = pd.DataFrame(self.matrix)

        if out_int:
            matrix = matrix.astype(int)
            float_format = '%d'

        matrix.to_csv(out_file,
                      header=False, index=False, sep="\t",
                      float_format=float_format,
                      compression=compression)

    def get_has_neg(self):
        if np.min(self.matrix) < 0:
            self.has_neg = True
        else:
            self.has_neg = False

    ######################
    # Operation
    def expand(self, expand_radius=3):
        sigma = super().expand(expand_radius)
        
        if self.mtx_type is None:
            self.get_mtx_type()
        new_matrix = self.to_all(inplace=False)
        
        from scipy.ndimage import gaussian_filter
        new_matrix = gaussian_filter(new_matrix.matrix, sigma, truncate=(expand_radius + .1) / sigma)
        
        if self.mtx_type == 'triu':
            new_matrix = np.triu(new_matrix, k=0)
        elif self.mtx_type == 'tril':
            new_matrix = np.tril(new_matrix, k=0)
        else:
            pass
        
        return ArrayHiCMtx(new_matrix, copy_mtx_paras=self)
    
    def log(self, inplace=False, verbose=True):
        if self.has_neg:
            warnings.warn('Contain negative values, cannot do log.')
            new_matrix = self.matrix.copy()
        
        else:
            pos_idx = self.matrix > 0
            new_matrix = self.matrix.copy()
            new_matrix[pos_idx] = np.log(new_matrix[pos_idx] + 1)
            
            if len(np.where(self.matrix <= 0)[0]) > 0:
                if verbose:
                    warnings.warn('Include non-positive values in matrix.')
            new_matrix[self.matrix <= 0] = np.min(new_matrix[pos_idx])
        
        if inplace:
            self.matrix = new_matrix
        else:
            return ArrayHiCMtx(new_matrix, copy_mtx_paras=self)
    
    def normalize(self, mode='percentile', k=None):
        new_matrix = self.matrix.copy()
        if mode == 'percentile':
            k = 90 if k is None else k
            new_matrix = new_matrix / np.percentile(new_matrix[new_matrix > 0], k)

        elif mode == 'coverage':
            k = 1e8 if k is None else k
            new_matrix = new_matrix / np.sum(new_matrix) * 1e8

        elif mode == 'max-min':
            pos_arr = new_matrix[new_matrix > 0]
            division = np.max(pos_arr) - np.min(pos_arr)
            if division > 0:
                new_matrix[new_matrix > 0] = (pos_arr - np.min(pos_arr)) / division

        else:
            raise ValueError(f'Unrecognized mode: {mode}.')
        
        return ArrayHiCMtx(new_matrix, copy_mtx_paras=self)

    def condense(self, condense_fold, mean=False):
        def _condense_row(arr_mtx, condense_fold, mean=False):
            new_window, index_trans = arr_mtx.row_window.condense(condense_fold)
            
            row_idx = np.arange(arr_mtx.matrix.shape[0])
            new_row_idx = index_trans.loc[row_idx]
            
            # new_row_idx = np.zeros_like(row_idx)
            # for chro in arr_mtx.row_window.chros:
            #     min_idx, max_idx = arr_mtx.row_window.chr_seps[chro][0], arr_mtx.row_window.chr_seps[chro][1]
            #     new_min_idx = new_window.chr_seps[chro][0]

            #     # Change matrix index
            #     chr_row_idx = (row_idx >= min_idx) & (row_idx <= max_idx)
            #     new_row_idx[chr_row_idx] = (row_idx[chr_row_idx] - min_idx) // condense_fold + new_min_idx
            
            new_matrix = pd.DataFrame(arr_mtx.matrix)
            if mean:
                new_matrix = new_matrix.groupby(by=new_row_idx).mean()
            else:
                new_matrix = new_matrix.groupby(by=new_row_idx).sum()

            return ArrayHiCMtx(np.array(new_matrix),
                               mtx_file=arr_mtx.mtx_file, 
                               row_window=new_window,
                               col_window=arr_mtx.col_window)

        new_matrix = _condense_row(self, condense_fold, mean=mean)
        new_matrix = new_matrix.transpose()

        new_matrix = _condense_row(new_matrix, condense_fold, mean=mean)
        new_matrix = new_matrix.transpose()
        return new_matrix


class SpsHiCMtx(HiCMtx):
    def __init__(self, sps_mtx, mtx_file=None, 
                 row_window=None, col_window=None, 
                 mtx_type=None, intra=None, has_neg=None,
                 copy_mtx_paras=None):
        super().__init__()
        if not isinstance(sps_mtx, sps.csr_matrix):
            self.matrix = sps_mtx.tocsr()
        else:
            self.matrix = sps_mtx
        del sps_mtx
        # self.matrix.data = self.matrix.data.astype(float)
        self.shape = self.matrix.shape

        self.has_neg = None
        if copy_mtx_paras is not None:
            self.mtx_file = copy_mtx_paras.mtx_file
            self.row_window = copy_mtx_paras.row_window
            self.col_window = copy_mtx_paras.col_window
            self.mtx_type = copy_mtx_paras.mtx_type
            self.intra = copy_mtx_paras.intra
            self.has_neg = copy_mtx_paras.has_neg
            
        if row_window is None and col_window is not None:
            self.row_window, self.col_window = col_window, col_window
        elif col_window is None and row_window is not None:
            self.row_window, self.col_window = row_window, row_window
        elif col_window is None and row_window is None:
            if copy_mtx_paras is None:
                self.row_window, self.col_window = None, None
        else:
            self.row_window, self.col_window = row_window, col_window

        if self.row_window is not None:
            self.bin_size = self.row_window.bin_size

        if mtx_file is not None:
            self.mtx_file = mtx_file
        if mtx_type is not None:
            self.mtx_type = mtx_type
        if intra is not None:
            self.intra = intra
        
        if has_neg is not None:
            self.has_neg = has_neg
        elif self.has_neg is None:
            self.get_has_neg()

    def tid_off_zeros(self):
        new_matrix = self.matrix.tocoo()
        new_matrix.row = new_matrix.row[new_matrix.data > 0]
        new_matrix.col = new_matrix.col[new_matrix.data > 0]
        new_matrix.data = new_matrix.data[new_matrix.data > 0]
        self.matrix = new_matrix.tocsr()

    ######################
    # Matrix transformation
    def to_dense(self, zero_fill='min'):
        if not self.has_neg:
            dense_mtx = np.array(self.matrix.todense())
        else:
            if not isinstance(self.matrix, sps.coo_matrix):
                coo_mtx = self.matrix.tocoo()
            else:
                coo_mtx = self.matrix
            min_value = np.min(coo_mtx.data)
            if zero_fill == 'min':
                dense_mtx = np.ones(coo_mtx.shape) * min_value
            else:
                dense_mtx = np.ones(coo_mtx.shape) * zero_fill
            dense_mtx[coo_mtx.row, coo_mtx.col] = coo_mtx.data
        
        return ArrayHiCMtx(dense_mtx, copy_mtx_paras=self)

    def to_sps(self):
        return copy.deepcopy(self)

    def to_all(self, inplace=False):
        if self.mtx_type is None:
            self.get_mtx_type()
        
        if self.mtx_type == 'all':
            new_mtx = self.matrix
        elif self.mtx_type == 'triu':
            new_mtx = sps.triu(self.matrix, k=0).T + sps.triu(self.matrix, k=1)
        elif self.mtx_type == 'tril':
            new_mtx = sps.tril(self.matrix, k=0).T + sps.tril(self.matrix, k=1)
        else:
            raise ValueError(f'Unrecognized mtx type: {self.mtx_type}')

        if inplace:
            self.mtx_type = 'all'
            self.matrix = new_mtx
        else:
            return SpsHiCMtx(new_mtx, copy_mtx_paras=self, mtx_type='all')

    def to_triu(self, k=0, inplace=False):
        if self.mtx_type is None:
            self.get_mtx_type()

        if self.mtx_type == 'all':
            new_mtx = sps.triu(self.matrix, k=k)
        elif self.mtx_type == 'triu':
            new_mtx = sps.triu(self.matrix, k=k)
        elif self.mtx_type == 'tril':
            new_mtx = sps.triu(self.matrix.T, k=k)
        else:
            raise ValueError(f'Unrecognized mtx type: {self.mtx_type}')

        if inplace:
            self.mtx_type = 'triu'
            self.matrix = new_mtx
        else:
            return SpsHiCMtx(new_mtx, copy_mtx_paras=self, mtx_type='triu')

    def to_output(self, out_file, out_window=True, out_near_num=None,
                  out_triu=True, out_int=False, only_out_intra=False, only_row_window=True,
                  fmt=None):
        if not sps.issparse(self.matrix):
            raise ValueError('Please read matrix first, by HiCData.get_sps_all_mtx()')

        if out_near_num is not None:
            self.tid_off_low_value(near_num=out_near_num)

        if out_triu:
            self.to_triu(inplace=True)

        if not only_out_intra:
            matrix = self.matrix.tocoo()
            rows = np.array(matrix.row)
            cols = np.array(matrix.col)
            values = np.array(matrix.data)
        else:
            rows, cols, values = [], [], []
            for chro, chr_sps_mtx in self.yield_by_chro(only_intra=True, triu=True):
                chr_sps_mtx = chr_sps_mtx.matrix.tocoo()
                rows += list(chr_sps_mtx.row + self.row_window.chr_seps[chro][0])
                cols += list(chr_sps_mtx.col + self.row_window.chr_seps[chro][0])
                values += list(chr_sps_mtx.data)

        result_array = np.vstack([rows, cols, values])
        if out_int:
            if fmt is None:
                warnings.warn('fmt is not used if output is int.')
            np.savetxt(out_file, result_array.T, fmt="%d\t%d\t%d")
        else:
            fmt = "%d\t%d\t%.2f" if fmt is None else fmt
            np.savetxt(out_file, result_array.T, fmt=fmt)

        if out_window:
            if only_row_window:
                self.row_window.to_output(change_suffix(out_file, '.window.bed'))
            else:
                self.row_window.to_output(change_suffix(out_file, '.row_window.bed'))
                self.col_window.to_output(change_suffix(out_file, '.col_window.bed'))

        print("Window file and sparse matrix file have been outputted.")
    
    def get_has_neg(self):
        if np.min(self.matrix.data) < 0:
            self.has_neg = True
        else:
            self.has_neg = False
    
    ######################
    # Operation
    def expand(self, expand_radius=3, min_exp_value=0.3):
        from scipy.stats import multivariate_normal  
        sigma = super().expand(expand_radius)
        
        expand_range = np.arange(-expand_radius, expand_radius + 1)
        expand_rows, expand_cols = np.meshgrid(expand_range, expand_range)
        expand_rows, expand_cols = expand_rows.reshape(-1,), expand_cols.reshape(-1,)
        expand_poses = np.vstack([expand_rows, expand_cols]).T
        expand_values = multivariate_normal(mean=[0, 0], cov=sigma).pdf(expand_poses)

        coo_matrix = self.matrix.tocoo()
        exp_sps_rows, exp_sps_cols, exp_sps_values = [], [], []

        for row, col, value in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
            sub_exp_sps_rows = row + expand_rows
            sub_exp_sps_cols = col + expand_cols
            sub_exp_sps_values = value * expand_values

            exp_sps_rows += list(sub_exp_sps_rows)
            exp_sps_cols += list(sub_exp_sps_cols)
            exp_sps_values += list(sub_exp_sps_values)

        exp_sps_cols, exp_sps_rows = np.array(exp_sps_cols), np.array(exp_sps_rows)
        exp_sps_values = np.array(exp_sps_values)

        filter_index = np.nonzero((0 < exp_sps_rows < self.matrix.shape[0]) &
                                  (0 < exp_sps_cols < self.matrix.shape[1]))

        exp_sps_cols, exp_sps_rows = exp_sps_cols[filter_index], exp_sps_rows[filter_index]
        exp_sps_values = exp_sps_values[filter_index]

        new_matrix = sps.coo_matrix((exp_sps_values, (exp_sps_rows, exp_sps_cols)),
                                     shape=self.matrix.shape).tocsr()
        new_matrix = SpsHiCMtx(new_matrix, copy_mtx_paras=self)
        return new_matrix

    def expand_only_intra(self, expand_radius=3):
        """Expand through dense matrix.
        Only for intra-chromosome HiC map.

        Args:
            sigma (int, optional): _description_. Defaults to 2.
            truncate (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        new_row, new_col, new_data = [], [], []
        for chro, chr_mtx in self.yield_by_chro(only_intra=True, triu=False):
            chr_mtx.to_all(inplace=True)
            chr_mtx = chr_mtx.to_dense()

            chr_mtx.expand(expand_radius)
            chr_sps_mtx = sps.csr_matrix(chr_mtx.matrix)
            chr_sps_mtx = chr_sps_mtx.tocoo()

            new_row += list(chr_sps_mtx.row + self.row_window.chr_seps[chro][0])
            new_col += list(chr_sps_mtx.col + self.row_window.chr_seps[chro][0])
            new_data += list(chr_sps_mtx.data)

        return SpsHiCMtx(sps.coo_matrix((new_data, (new_row, new_col)),
                                        shape=self.matrix.shape),
                         copy_mtx_paras=None)

    def log(self, inplace=False):
        if self.has_neg:
            warnings.warn('Contain negative values, cannot do log.')
            new_matrix = self.matrix.copy()
        
        else:
            matrix_size = self.matrix.shape[0] * self.matrix.shape[1]
            if self.matrix.data.size < matrix_size and np.min(self.matrix.data) <= 1:
                warnings.warn('Both include empty values (0 before and after log),\n'
                            'and <=1 value (<=0 after log).')
            
            pos_idx = self.matrix.data > 0
            nonpos_idx = np.logical_not(pos_idx)
            if np.any(nonpos_idx):
                warnings.warn('Non-positive value in matrix.')
            
            new_matrix = self.matrix.copy()
            new_matrix.data[pos_idx] = np.log(new_matrix.data[pos_idx] + 1)
            # new_matrix.data[nonpos_idx] = np.min(new_matrix.data[pos_idx])
        
        if inplace:
            self.matrix.data = new_matrix.data
        else:
            return SpsHiCMtx(new_matrix, copy_mtx_paras=self)

    def normalize(self, mode='percentile', k=90):
        if mode == 'percentile':
            print(np.percentile(self.matrix.data, k))
            self.matrix.data = self.matrix.data / np.percentile(self.matrix.data, k)

        elif mode == 'coverage':
            self.matrix.data = self.matrix.data / np.sum(self.matrix.data) * 1e8

        elif mode == 'max-min':
            self.matrix.data = (self.matrix.data - np.min(self.matrix.data)) / \
                               (np.max(self.matrix.data) - np.min(self.matrix.data))

        elif mode == 'z-score':
            self.matrix.data = (self.matrix.data - np.mean(self.matrix.data)) / \
                               np.std(self.matrix.data)

        else:
            raise ValueError(f'Unrecognized mode: {mode}.')

    def condense(self, condense_fold, mean=False):
        def _condense_row(sps_mtx, condense_fold, mean=False):
            new_window, index_trans = sps_mtx.row_window.condense(condense_fold)
            
            sps_coo_mtx = sps_mtx.matrix.tocoo()
            row, col, data = sps_coo_mtx.row, sps_coo_mtx.col, sps_coo_mtx.data

            new_row = index_trans.loc[row]
            
            # new_row = np.zeros_like(row)
            # for chro in sps_mtx.row_window.chros:
            #     min_idx, max_idx = sps_mtx.row_window.chr_seps[chro][0], sps_mtx.row_window.chr_seps[chro][1]
            #     new_min_idx = new_window.chr_seps[chro][0]

            #     # Change matrix index
            #     chr_row_idx = (row >= min_idx) & (row <= max_idx)
            #     new_row[chr_row_idx] = (row[chr_row_idx] - min_idx) // condense_fold + new_min_idx

            new_matrix = sps.coo_matrix((data, (new_row, col)),
                                        shape=(new_window.all_len, sps_coo_mtx.shape[1]))
            new_matrix = new_matrix.tocsr()

            if mean:
                count_matrix = sps.coo_matrix((np.ones_like(new_row), (new_row, col)),
                                              shape=(new_window.all_len, sps_coo_mtx.shape[1]))
                count_matrix = count_matrix.tocsr()

                new_matrix.data = new_matrix.data / count_matrix.data

            return SpsHiCMtx(new_matrix, 
                             mtx_file=sps_mtx.mtx_file,
                             row_window=new_window,
                             col_window=sps_mtx.col_window)

        new_matrix = _condense_row(self, condense_fold, mean=mean)
        new_matrix = new_matrix.transpose()

        new_matrix = _condense_row(new_matrix, condense_fold, mean=mean)
        new_matrix = new_matrix.transpose()
        return new_matrix


class FreCounts(Values):
    def __init__(self, fre_counts_input, sgd=None, input_index=None):
        super().__init__(fre_counts_input, input_index=input_index)

        if isinstance(fre_counts_input, str):
            self._read_fre_counts(fre_counts_input)
        else:
            # if sgd is None:
                # raise ValueError('Plz input sgd')
            self.sgd = sgd

    def out_to_file(self, out_file):
        if os.path.exists(out_file):
            os.remove(out_file)

        out_file_handle = open(out_file, 'a')

        if self.sgd is not None:
            out_file_handle.write(self.sgd.out_paras_to_string())
            out_file_handle.write('\n')

        self.fre_counts.to_csv(out_file_handle, sep="\t", float_format="%.4f")
        out_file_handle.close()

    def _read_fre_counts(self, fre_counts_file):
        sgd = None
        file_handle = open(fre_counts_file, 'r')
        for line in file_handle.readlines():
            if line.startswith('#sgd'):
                sgd = ScaledGenomicDistance(input_dict=line)
                break
        file_handle.close()

        # if sgd is None:
        #     raise ValueError(f'Cannot find sgd in {fre_counts_file}.')

        self.sgd = sgd
    

##############################
# utils
def is_mtx(mtx):
    if isinstance(mtx, ArrayHiCMtx) or isinstance(mtx, SpsHiCMtx):
        return True
    else:
        return False


def concat_mtxes(mtxes, row_window=None, col_window=None, **other_mtx_kwargs):
    if len(mtxes) == 1:
        new_matrix = mtxes[0].to_sps().matrix
    else:
        new_data, new_row, new_col = [], [], []
        for mtx in mtxes:
            if is_mtx(mtx):
                sps_mtx = mtx.to_sps().matrix.tocoo()
            else:
                sps_mtx = mtx.tocoo()
            new_data += list(sps_mtx.data)
            new_row += list(sps_mtx.row + mtx.row_window.ori_start_idx)
            new_col += list(sps_mtx.col + mtx.col_window.ori_start_idx)
        
        if row_window is None and col_window is None:
            if 'copy_mtx_paras' in other_mtx_kwargs:
                max_idx = other_mtx_kwargs['copy_mtx_paras'].row_window.all_len
            else:
                max_idx = np.max([np.max(new_row), np.max(new_col)])
            shape = (max_idx, max_idx)
        elif row_window is not None and col_window is None:
            max_idx = row_window.all_len
            shape = (max_idx, max_idx)
        elif row_window is None and col_window is not None:
            max_idx = row_window.all_len
            shape = (max_idx, max_idx)
        else:
            shape = (row_window.all_len, col_window.all_len)
        
        new_matrix = sps.coo_matrix((new_data, (new_row, new_col)),
                                    shape=shape)
    
    return SpsHiCMtx(new_matrix, 
                     row_window=row_window,
                     col_window=col_window,
                     **other_mtx_kwargs)


def straw_one_chro(hic_file, chro1, chro2, bin_size=20000):
    #chro1 = chro1[3:] if 'chr' == chro1[:3] and chro1[3:].isdigit() else chro1
    #chro2 = chro2[3:] if 'chr' == chro2[:3] and chro2[3:].isdigit() else chro2
    chro1 = chro1[3:] if 'chr' == chro1[:3]  else chro1
    chro2 = chro2[3:] if 'chr' == chro2[:3]  else chro2
    juicer_chro1, row, juicer_chro2, col, value = mystraw.straw('NONE', hic_file, chro1, chro2, 'BP', bin_size)
    # row, col, value = mtstraw.straw('NONE', hic_file, chro1, chro2, 'BP', bin_size)
    # juicer_chro1 = chro1
    # juicer_chro2 = chro2

    if juicer_chro1 == -1:
        return None, None, None

    row, col = np.array(row) / bin_size, np.array(col) / bin_size
    row, col = row.astype(int), col.astype(int)

    if juicer_chro1 == chro1 and juicer_chro2 == chro2:
        return row, col, value
    else:
        return col, row, value


def read_sps_file(sps_file, row_window_file=None, col_window_file=None, 
                  mtx_type=None, intra=None, has_neg=None):
    # sps_df = pd.read_table(sps_file, names=['row', 'col', 'value'],
    #                        dtype={'row': int, 'col': int})
    sps_df = pd.read_table(sps_file, names=['row', 'col', 'value'])
    sps_df.dropna(inplace=True)

    if row_window_file is None and col_window_file is None:
        row_window, col_window = None, None
        mtx_shape = (np.max(sps_df['row']) + 1, np.max(sps_df['col']) + 1)
    elif row_window_file is not None and col_window_file is None:
        row_window, col_window = GenomeIndex(row_window_file), None
        mtx_shape = (row_window.max_idx + 1, row_window.max_idx + 1)
    elif row_window_file is None and col_window_file is not None:
        row_window, col_window = None, GenomeIndex(col_window_file)
        mtx_shape = (col_window.max_idx + 1, col_window.max_idx + 1)
    else:
        row_window, col_window = GenomeIndex(row_window_file), GenomeIndex(col_window_file)
        mtx_shape = (row_window.max_idx + 1, col_window.max_idx + 1)

    sps_mtx = sps.coo_matrix((sps_df['value'], (sps_df['row'], sps_df['col'])), shape=mtx_shape)

    sps_mtx = SpsHiCMtx(sps_mtx, mtx_file=sps_file, 
                     row_window=row_window, col_window=col_window, 
                     mtx_type=mtx_type, intra=intra, has_neg=has_neg)
    sps_mtx.mtx_name = sps_file.split('/')[-1].split('.sps.mtx')[0]
    return sps_mtx


class HiCFile:
    def __init__(self, hic_file, window_file=None, chrom_size_file=None, bin_size=None):
        self.hic_file = hic_file

        self.bin_size = bin_size

        if window_file is not None:
            self.window = GenomeIndex(window_file)

            if self.bin_size is not None and self.bin_size != self.window.bin_size:
                warnings.warn(f"Input bin_size is not compatible with window bin_size.")

            self.bin_size = self.window.bin_size

        elif chrom_size_file is not None:
            if bin_size is None:
                raise ValueError('Input bin_size for chrom_size_file.')

            self.window = window_from_chrom_size(chrom_size_file, bin_size)
            self.bin_size = bin_size

        else:
            self.window = None

    def get_region_mtx(self, chro, start=None, end=None, chro2=None, start2=None, end2=None):
        if start is not None and self.window is None:
            raise ValueError('Please input window for position.')

        chro2 = chro if chro2 is None else chro2

        ###########
        # Get matrix
        row, col, value = straw_one_chro(self.hic_file, chro, chro2, self.bin_size)

        if row is None:
            return None

        if self.window is not None:
            mtx_shape = (self.window.chr_lens[chro], self.window.chr_lens[chro2])
        else:
            mtx_shape = (np.max(row) + 1, np.max(col) + 1)

        matrix = sps.coo_matrix((value, (row, col)), shape=mtx_shape)

        ###########
        # Get window
        if self.window is None:
            row_window, col_window = None, None
        else:
            row_window = GenomeIndex(self.window[self.window['chr'] == chro])
            col_window = GenomeIndex(self.window[self.window['chr'] == chro2])

        intra = True if chro == chro2 else False
        mtx_type = 'all' if not intra else 'triu'
        matrix = SpsHiCMtx(matrix,
                            mtx_file=self.hic_file, 
                            row_window=row_window, 
                            col_window=col_window, 
                            mtx_type=mtx_type, 
                            intra=intra)

        if start is None:
            return matrix

        else:
            if chro == chro2 and (start != start2 or end != end2):
                matrix.to_all(inplace=True)

            return matrix.get_region_mtx(chro, start, end, chro2, start2, end2)

    def get_whole_mtx(self):
        if self.window is None:
            raise ValueError(f"Please input window for whole mtx reading.")

        all_rows, all_cols, all_value = [], [], []

        sort_gen_index = self.window.sort_by_chro()

        for chro1_idx in range(len(sort_gen_index.chros)):

            for chro2_idx in range(chro1_idx, len(sort_gen_index.chros)):

                chro1, chro2 = sort_gen_index.chros[chro1_idx], sort_gen_index.chros[chro2_idx]

                if sort_gen_index.chr_seps[chro1][0] >= sort_gen_index.chr_seps[chro2][0]:
                    chro1, chro2 = chro2, chro1

                # print(f'Test Bin size: {sort_gen_index.bin_size}')
                row, col, value = straw_one_chro(self.hic_file, chro1, chro2, sort_gen_index.bin_size)

                if row is None:
                    continue

                all_rows += list(row + sort_gen_index.chr_seps[chro1][0])
                all_cols += list(col + sort_gen_index.chr_seps[chro2][0])
                all_value += list(value)
                
                print(f'Read {chro1} vs {chro2} done.')

        matrix = sps.coo_matrix((all_value, (all_rows, all_cols)),
                                shape=(sort_gen_index.all_len, sort_gen_index.all_len))

        return SpsHiCMtx(matrix,
                        mtx_file=self.hic_file, 
                        row_window=sort_gen_index, 
                        col_window=sort_gen_index, 
                        mtx_type='triu', 
                        intra=False)

    def yield_by_chro(self, only_intra=False, triu=False):
        if self.window is None:
            raise ValueError(f"Please input window for whole mtx reading.")

        for chro1 in self.window.chros:
            for chro2 in self.window.chros:

                if triu and self.window.chr_seps[chro1][0] > self.window.chr_seps[chro2][0]:
                    continue

                if chro1 != chro2 and only_intra:
                    continue

                chr_mtx = self.get_region_mtx(chro1, chro2=chro2)

                if chr_mtx is None:
                    warnings.warn(f"Cannot get matrix: {chro1}.")
                    continue

                if triu and chro1 == chro2:
                    chr_mtx.to_triu()

                if only_intra:
                    yield chro1, chr_mtx
                else:
                    yield chro1, chro2, chr_mtx
