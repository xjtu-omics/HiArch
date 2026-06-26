import os
import glob
import json
import sys
import warnings
import inspect
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.sparse as sps
import multiprocessing as mt
from scipy.stats import pearsonr, spearmanr
import new_hic_class as hic


#################################
# Utilities
def read_mtx(input_file, index_file=None, chro=None, resolution=None):
    from new_hic_class import read_sps_file, SpsHiCMtx, GenomeIndex

    if input_file.endswith('sps.mtx') or input_file.endswith('.sps.mtx.gz') or \
            input_file.endswith('.sps_normalized.mtx'):
        if index_file is None:
            raise ValueError(f'Please input index_file for sps mtx.')
        mtx = read_sps_file(input_file, index_file, mtx_type='triu')

    elif input_file.endswith('.mcool'):
        import cooler
        
        if resolution is None:
            raise ValueError(f'Please input resolution for mcool file.')
        
        cool_mtx = cooler.Cooler(input_file)
        pixel_data = cool_mtx.pixels()[:]
        max_size = np.max([pixel_data['bin1_id'].max(),
                           pixel_data['bin2_id'].max()]) + 1
        sps_mtx = sps.coo_matrix((pixel_data['count'], (pixel_data['bin1_id'], pixel_data['bin2_id'])),
                                shape=(max_size, max_size))
        
        # Get index file
        index_file_name = input_file.split('.mcool')[0] + f'_{resolution}.window.bed'
        cool_mtx.chromsizes.to_csv('tmp.chrom.sizes', sep="\t", header=False)
        os.system(f"bedtools makewindows -g tmp.chrom.sizes -w {resolution} | awk -v OFS='\t' 'BEGIN{{i=0}}{{print $1, $2, $3, i; i+=1}}' - > {index_file_name}")
        gen_index = GenomeIndex(index_file_name)
        os.system(f'rm tmp.chrom.sizes')
        
        mtx = SpsHiCMtx(sps_mtx, input_file, gen_index)

    ##########
    # Warning: the pair files and 3dg files are transformed to sps mtx before using.
    #           And the input file names are recorded in HiC class now.
    ##########
    # elif input_file.endswith('.pairs.txt') or input_file.endswith('.pairs.txt.gz'):
    #     if index_file is None:
    #         raise ValueError(f'Please input index_file for pair file.')
    #     from bD2_git.precrocess.pair2sps_mtx import pair2mtx
    #     mtx = pair2mtx(input_file, index_file, chro)
    #
    # elif input_file.endswith('.3dg.txt') or input_file.endswith('.3dg.txt.gz'):
    #     if index_file is None:
    #         raise ValueError(f'Please input index_file for pair file.')
    #     from bD2_git.precrocess.compute_dis_mtx import compute_dis_mtx
    #     mtx = compute_dis_mtx(input_file, index_file, chro, thread=thread)

    ##########
    # Warning: the dis files and pair files are transformed to sps mtx before using.
    #           And the input file names are recorded in HiC class now.
    ##########
    # elif input_file.endswith('.dis_mtx.npz'):
    #     mtx = np.load(input_file)['dis_mtx']
    #     mtx[mtx > 0] = np.power(mtx[mtx > 0], -1 / alpha)
    #     if chro is not None:
    #         raise ValueError('Chro selection for numpy format is not finished.')
    #     mtx = _array_to_triu(mtx)

    # elif input_file.endswith('.dis_mtx.gz'):
    #     mtx = np.loadtxt(input_file, delimiter='\t')
    #     mtx[mtx > 0] = np.power(mtx[mtx > 0], -1 / alpha)
    #     if chro is not None:
    #         raise ValueError('Chro selection for matrix format is not finished.')
    #     mtx = _array_to_triu(mtx)

    # elif input_file.endswith('.pair_mtx.gz'):
    #     mtx = np.loadtxt(input_file, delimiter='\t')
    #     if chro is not None:
    #         raise ValueError('Chro selection for matrix format is not finished.')
    #     mtx = _array_to_triu(mtx)

    else:
        raise ValueError('Unrecognized input file format.')

    return mtx


def read_true_value(input_file, index_file, chro=None, read_type='den'):
    gen_index = hic.GenomeIndex(index_file)

    if input_file.endswith('.spread.txt'):
        spread = pd.read_table(input_file)
        spread.index = spread['index']
        if f'{read_type}_mean' in spread.columns:
            true_value = spread[f'{read_type}_mean']
        else:
            return None

    elif input_file.endswith('.den_dtp.txt') or input_file.endswith('.den_clu.txt'):
        den_clu = pd.read_table(input_file, comment="#", usecols=[0, 1, 2, 3, 8, 9],
                                names=['chr', 'start', 'end', 'index', 'den', 'dtp'])
        den_clu.index = den_clu['index']
        if read_type in den_clu.columns:
            true_value = den_clu[read_type].rename(f'{read_type}_mean')
        else:
            return None

    elif input_file.endswith('.values.txt') or input_file.endswith('.rna.txt'):
        values = hic.Values(input_file)
        if read_type == 'all':
            remain_read_type = values.value_names
            true_value = values.values[remain_read_type].copy()
        elif isinstance(read_type, list):
            remain_read_type = list(set(read_type) & set(values.value_names))
            print(f'Input: {read_type}. Remain: {remain_read_type}')
            true_value = values.values[remain_read_type].copy()
        elif read_type in values.values.columns:
            true_value = values.values[read_type].copy()
        else:
            return None
    else:
        raise ValueError('Unrecognized spread file format.')

    true_value = hic.Values(true_value, gen_index)

    if chro is not None:
        true_value = true_value.select_chro(chro)

    return true_value


def get_cell_file_name(cell_name, input_dir, suffix="", iter=True,
                       include_pat=None, verbose=True):
    default_pat = 'mat'

    cell_name = cell_name.split('.')[0]
    file_names = glob.glob(f'{input_dir}/*{cell_name}*{suffix}')
    if include_pat is not None:
        file_names = [fn for fn in file_names if include_pat in fn]

    if len(file_names) == 0:
        sub_cell_name = cell_name.split('_')[0]
        if sub_cell_name != '' and iter:
            return get_cell_file_name(sub_cell_name, input_dir, suffix, iter=False, verbose=verbose)
        else:
            if verbose:
                warnings.warn(f'Cannot find file for {cell_name}')
            return None

    if len(file_names) == 1:
        return file_names[0]

    if len(file_names) > 1:
        new_file_names = [fn for fn in file_names if default_pat in fn]
        if len(new_file_names) == 1:
            if verbose:
                warnings.warn(f'Find multiple file for {cell_name}. Use the chro MAT version')
            return new_file_names[0]
        else:
            if verbose:
                warnings.warn(f'Find multiple file for {cell_name}. Use the first one')
            return file_names[0]


def get_paras_product(func_kwargs, no_iter_list=None):
    """Only LIST will be iterated!!"""
    non_iter_values = {}
    iter_values = {}
    for func_key in func_kwargs:
        func_values = func_kwargs[func_key]
        if isinstance(func_values, list):
            if no_iter_list is not None and func_key in no_iter_list:
                non_iter_values[func_key] = func_values
            else:
                iter_values[func_key] = func_values
        else:
            non_iter_values[func_key] = func_values

    func_kwargs_list = []
    if len(iter_values) >= 1:
        from itertools import product
        # print(list(iter_values.values()))
        for iter_value in product(*list(iter_values.values())):
            one_para = non_iter_values.copy()

            iter_value = {list(iter_values.keys())[i]: iter_value[i] for i in range(len(iter_value))}
            one_para.update(iter_value)

            func_kwargs_list.append(one_para)
    # elif len(iter_values) == 1:
    #     for value in list(iter_values.values())[0]:
    #         one_para = non_iter_values.copy()
    #
    #         iter_value = {list(iter_values.keys())[0]: value}
    #
    #         one_para.update(iter_value)
    #         func_kwargs_list.append(one_para)
    else:
        func_kwargs_list = [non_iter_values]

    return func_kwargs_list


def get_static(true_value, pre_value, ratio=100 / 3):
    true_value.rename('true', inplace=True)
    pre_value.rename('pre', inplace=True)

    concat_value = pd.concat([true_value, pre_value],
                             axis=1, join='inner')

    class_result = {}

    ##############
    # pear and spear
    pear = pearsonr(concat_value['true'], concat_value['pre'])[0]
    class_result['pear'] = pear

    spear = spearmanr(concat_value['true'], concat_value['pre'])[0]
    class_result['spear'] = spear

    ##############
    # TP, FP, TN, FN
    def get_idxes(series, sep):
        low_idxes = series[series <= sep].index
        high_index = series.index.difference(low_idxes)
        return set(np.array(low_idxes)), set(np.array(high_index))

    if ratio is not None:
        true_sep = np.percentile(concat_value['true'], ratio)
        true_low_idxes, true_high_idxes = get_idxes(concat_value['true'], true_sep)

        pre_sep = np.percentile(concat_value['pre'], ratio)
        pre_low_idxes, pre_high_idxes = get_idxes(concat_value['pre'], pre_sep)

        all_num = concat_value.shape[0]

        true_pos = len(true_low_idxes & pre_low_idxes) / all_num
        class_result['true_pos'] = true_pos
        false_pos = len(pre_low_idxes - true_low_idxes) / all_num
        class_result['false_pos'] = false_pos
        true_neg = len(pre_high_idxes & true_high_idxes) / all_num
        class_result['true_neg'] = true_neg
        false_neg = len(pre_high_idxes - true_high_idxes) / all_num
        class_result['false_neg'] = false_neg
        accuracy = true_pos + true_neg
        class_result['accuracy'] = accuracy

    return class_result


def yield_mp_idxes(size, thread):
    cur_idx = 0
    n_sub_idx = int(size / thread) if size > thread else 1
    while cur_idx < size:
        if cur_idx + n_sub_idx < size:
            sub_idxes = np.arange(cur_idx, cur_idx + n_sub_idx).astype(int)
        else:
            sub_idxes = np.arange(cur_idx, size).astype(int)
        cur_idx += n_sub_idx

        yield sub_idxes


def read_dir_files(input_dir, include_pats=None, exclude_pats=None, suffix=""):
    mtx_files = glob.glob(f'{input_dir}/*{suffix}')

    if include_pats is not None:
        new_mtx_files = mtx_files.copy()
        for mtx_file in mtx_files:
            for include_pat in include_pats:
                if include_pat not in mtx_file:
                    new_mtx_files.remove(mtx_file)
                    break
        mtx_files = new_mtx_files

    if exclude_pats is not None:
        new_mtx_files = mtx_files.copy()
        for mtx_file in mtx_files:
            for exclude_pat in exclude_pats:
                if exclude_pat in mtx_file:
                    new_mtx_files.remove(mtx_file)
                    break
        mtx_files = new_mtx_files

    print(len(mtx_files))
    return mtx_files


#################################
# Parallel function
data_dir = ""
ref_dir = f'{data_dir}/ref'


def _read_kwargs(file_dir):
    kwargs_file = f'{file_dir}/kwargs.json.txt'
    if not os.path.exists(kwargs_file):
        return None
    kwargs_json = open(kwargs_file, 'r').readline()
    kwargs_json = kwargs_json.replace('$PWD', file_dir)
    kwargs_json = kwargs_json.replace('$REFDIR', ref_dir)
    kwargs = json.loads(kwargs_json)
    return kwargs


class CaptureParas:
    def __init__(self, ori_out, init_dict=None, callable_paras=None):
        self.ori_out = ori_out

        self.paras = {}

        if init_dict is None:
            self.init_paras = {}
        else:
            self.init_paras = init_dict.copy()

            if callable_paras is not None:
                for para in callable_paras:
                    del self.init_paras[para]

    def update_init_paras(self, new_dict):
        self.init_paras.update(new_dict)

    def write(self, string):
        # TODO: para write must start with #para-
        if string.startswith('#para-'):
            para_name = string.split(':')[0][len('#para-'):]
            if para_name not in self.paras:
                self.paras[para_name] = []
            # self.ori_out.write(string)
            self.paras[para_name].append(string.split(':')[1].strip())

        else:
            self.ori_out.write(string)

    def out_paras(self):
        if len(self.paras) > 0:
            para_len = len(list(self.paras.values())[0])
            out_paras = self.paras.copy()
            for para_name in self.init_paras:
                out_paras[para_name] = [self.init_paras[para_name]] * para_len
            return pd.DataFrame(out_paras)
        else:
            if len(self.init_paras) > 0:
                out_paras = self.init_paras.copy()
                return out_paras
            else:
                return pd.DataFrame([])


class ParallelFunc:
    """
    Provide parallel analysis for mtx process.

    Usage:

        para = ParallelFunc('mouse_mm10', 'concat_dis', func,
                            spread_type='D2_result/D2_hap', read_type='den',
                            include_pats='_12.')

        para.update_para_output(output, output_para='fig_output')
        para.update_para_true_value(para_name='value_col', do_static=False)

        para.vali_parallel(out_file, thread=10)

    """

    def __init__(self, species, dataset_name, func, spread_type=None, read_type='den',
                 n_cell=None, include_pats=None, exclude_pats=None,
                 spread_include_pat=None, out_concat_pre_value=False):
        mtx_dir = f'{data_dir}/{species}/mtx/{dataset_name}'
        spread_dir = f'{data_dir}/{species}/values/{spread_type}'

        self.read_mtx_func = read_mtx
        self.mtx_kwargs = _read_kwargs(mtx_dir)
        self.file_names, self.mtx_files = self.get_mtx_files(None, include_pats, exclude_pats)

        self.func = func
        self.func_kwargs = {}
        self.true_value_para = None
        self.concat_pre_value = None
        self.lock = None

        if out_concat_pre_value:
            self.prepare_for_out_concat_pre_value()

        if spread_type is not None:
            self.spread_kwargs = _read_kwargs(spread_dir)
            self.spread_files = self.get_spread_files(spread_include_pat)
            if read_type is not None:
                self.spread_kwargs['read_type'] = read_type
        else:
            self.spread_kwargs = None
            self.spread_files = None

        if n_cell is not None:
            self.file_names = self.file_names[:n_cell]
            self.mtx_files = {f: self.mtx_files[f] for f in self.file_names}
            if spread_type is not None:
                self.spread_files = {f: self.spread_files[f] for f in self.file_names}

    def update_mtx_kwargs(self, new_dict):
        self.mtx_kwargs.update(new_dict)

    def update_spread_kwargs(self, new_dict):
        self.spread_kwargs.update(new_dict)

    def update_func_kwargs(self, new_dict):
        """ func_kwargs could be function with cell_name as input.
            for example, "output": lambda cell_name: f'{output_to_path}/{cell_name}'"""
        self.func_kwargs.update(new_dict)

    def prepare_for_out_concat_pre_value(self):
        self.concat_pre_value = []
        self.lock = mt.Lock()

    def get_mtx_files(self, n_cell, include_pats=None, exclude_pats=None):
        mtx_files = read_dir_files(self.mtx_kwargs['mtx_dir'],
                                   include_pats,
                                   exclude_pats,
                                   suffix=self.mtx_kwargs['suffix'])
        del self.mtx_kwargs['mtx_dir']
        del self.mtx_kwargs['suffix']

        if n_cell is not None:
            if n_cell < len(mtx_files):
                mtx_files = mtx_files[:n_cell]

        mtx_file_dict = {}
        file_names = []
        for mtx_file in mtx_files:
            file_name = mtx_file.split("/")[-1]
            file_name = file_name.removesuffix('.sps.mtx')
            file_name = file_name.removesuffix('.sps.mtx.gz')
            file_names.append(file_name)
            mtx_file_dict[file_name] = mtx_file

        return file_names, mtx_file_dict

    def get_spread_files(self, include_pat=None):
        spread_input = self.spread_kwargs['spread_input']

        new_file_names = []
        spread_files = {}
        for file_name in self.file_names:
            if os.path.isfile(spread_input):
                spread_file = spread_input
            elif os.path.isdir(spread_input):
                spread_file = get_cell_file_name(file_name, spread_input,
                                                 include_pat=include_pat)
            else:
                raise ValueError('Spread input not exists.')

            # print(f'Spread for {file_name}: {spread_file}')

            if spread_file is None:
                continue

            new_file_names.append(file_name)
            spread_files[file_name] = spread_file

        self.file_names = new_file_names
        return spread_files

    def get_true_value(self, file_name, new_read_type=None):
        spread_file = self.spread_files[file_name]
        spread_kwargs_copy = self.spread_kwargs.copy()

        if 'spread_input' in spread_kwargs_copy:
            del spread_kwargs_copy['spread_input']

        if new_read_type is not None:
            spread_kwargs_copy['read_type'] = new_read_type

        if isinstance(spread_kwargs_copy['read_type'], list):
            true_values = []
            for read_type in spread_kwargs_copy['read_type']:
                sub_spread_kwargs_copy = spread_kwargs_copy.copy()
                sub_spread_kwargs_copy['read_type'] = read_type
                true_value = read_true_value(spread_file, **sub_spread_kwargs_copy)
                if true_value is not None:
                    true_values.append(true_value)
            return true_values
        else:
            true_value = read_true_value(spread_file, **spread_kwargs_copy)
            return true_value

    """
    # def update_para_output(self, output=None, output_para='output', sep="_", suffix=""):
    #     if output is None:
    #         if output_para not in self.func_kwargs:
    #             raise ValueError(f'{output_para} is not assigned in func_kwargs.')
    #         output = self.func_kwargs[output_para]

    #     output_updator = lambda cell_name: f'{output}{sep}{cell_name}{suffix}'
    #     self.func_kwargs.update({output_para: output_updator})
    
    # def update_para_cell_name(self, cellname_para='cell_name'):
    #     output_updator = lambda cell_name: f'{cell_name}'
    #     self.func_kwargs.update({cellname_para: output_updator})
    """
    
    def update_para_true_value(self, para_name='true_value', do_static=True):
        self.func_kwargs.update({para_name: self.get_true_value})
        if do_static:
            self.true_value_para = para_name

    def _vali_multi(self, mtx_files, all_paras, capture_paras=True):
        i_file = 0

        for file_name in mtx_files:
            mtx_file = self.mtx_files[file_name]
            mtx = self.read_mtx_func(mtx_file, **self.mtx_kwargs)
            mtx.mtx_name = file_name

            func_kwargs = self.func_kwargs.copy()
            callable_paras = []
            for func_key in func_kwargs:
                if callable(func_kwargs[func_key]):
                    new_para = func_kwargs[func_key](file_name)
                    if new_para is None:
                        continue
                    func_kwargs[func_key] = new_para
                    callable_paras.append(func_key)

            func_kwargs_list = get_paras_product(func_kwargs) if func_kwargs is not None else [None]

            for one_func_kwargs in func_kwargs_list:
                print(mtx_file)
                if self.spread_files is not None:
                    print(self.spread_files[file_name])
                print(one_func_kwargs)

                paras_cap = CaptureParas(sys.stdout,
                                         init_dict=one_func_kwargs,
                                         callable_paras=callable_paras)
                paras_cap.update_init_paras({'file_name': file_name})
                if capture_paras:
                    sys.stdout = paras_cap

                if one_func_kwargs is not None:
                    pre_value = self.func(mtx.copy(), **one_func_kwargs)
                else:
                    pre_value = self.func(mtx.copy())

                if capture_paras:
                    sys.stdout = paras_cap.ori_out
                paras = paras_cap.out_paras()

                if self.true_value_para is not None:
                    class_result = get_static(one_func_kwargs[self.true_value_para],
                                              pre_value)
                    paras.update(class_result)

                if self.concat_pre_value is not None:
                    self.lock.acquire()
                    self.concat_pre_value.append(pre_value.rename(file_name))
                    self.lock.release()

                all_paras.append(paras)

            i_file += 1
            print(f'{i_file}/{len(mtx_files)} Done.')

    def vali_parallel(self, out_file=None, thread=2, capture_paras=True):
        print(f'File names: {self.file_names}')
        print(f'Mtx Files: {self.mtx_files}')
        print(f'Mtx Kwargs: {self.mtx_kwargs}')
        if self.spread_files is not None:
            print(f'Spread files: {self.spread_files}')
            print(f'Spread kwargs: {self.spread_kwargs}')
        print(f'Func kwargs: {self.func_kwargs}')

        all_paras = mt.Manager().list()
        threads = []
        for sub_idxes in yield_mp_idxes(len(self.file_names), thread):
            sub_cells = np.array(self.file_names)[sub_idxes].tolist()
            sub_mtx_files = {cell: self.mtx_files[cell] for cell in sub_cells}
            threads.append(mt.Process(target=self._vali_multi,
                                      args=(sub_mtx_files, all_paras, capture_paras)))

        for p in threads:
            p.start()

        for p in threads:
            p.join()

        if out_file is not None:
            all_paras = pd.concat(all_paras, axis=0)
            all_paras.to_csv(out_file, sep="\t", index=False)

        if self.concat_pre_value is not None:
            self.concat_pre_value = pd.concat(self.concat_pre_value,
                                              axis=1, join='inner')
            return self.concat_pre_value


#################################
# Read paras result
class ValidResult:
    def __init__(self, out_file, parallel_func):
        self.para_df = pd.read_table(out_file)
        self.cell_names = list(set(self.para_df['cell_name']))

        self.mtx_kwargs = parallel_func.mtx_kwargs
        self.mtx_files = parallel_func.mtx_files

        self.func = parallel_func.func
        self.func_kwargs = parallel_func.func_kwargs

        self.spread_kwargs = parallel_func.spread_kwargs
        self.get_true_value = parallel_func.get_true_value

        self.best_paras = None

    def get_tissue(self):
        self.para_df['tissue'] = self.para_df['cell_name'].str.split('.', expand=True).iloc[:, 0]
        self.para_df['tissue'] = self.para_df['tissue'].str.split('_', expand=True).iloc[:, -1]

    def get_full_cell_name(self, input_cell_name):
        found_cell_names = [cell_name for cell_name in self.cell_names if input_cell_name in cell_name]
        if len(found_cell_names) == 0:
            raise ValueError(f'Cannot found full cell name for {input_cell_name}')
        elif len(found_cell_names) > 1:
            warnings.warn(f'Multiple full cell names are found. Will use first one.')
            return found_cell_names[0]
        else:
            return found_cell_names[0]

    def get_best_para_of_each_cell(self, sort_by='pear', reverse=False):
        self.best_paras = []

        cells = list(set(self.para_df['cell_name']))
        for cell_name in cells:
            cell_den_corr = self.para_df[self.para_df['cell_name'] == cell_name].copy()
            if not reverse:
                cell_den_corr = cell_den_corr[cell_den_corr[sort_by] == np.max(cell_den_corr[sort_by])]
            else:
                cell_den_corr = cell_den_corr[cell_den_corr[sort_by] == np.min(cell_den_corr[sort_by])]

            self.best_paras.append(cell_den_corr.iloc[0, :])

        self.best_paras = pd.concat(self.best_paras, axis=1).T
        self.best_paras.index = self.best_paras['cell_name']

    def plot_best_para_distribution(self, para_name, sort_by='pear', reverse=False):
        if self.best_paras is None:
            self.get_best_para_of_each_cell(sort_by, reverse)

        best_paras_set = np.sort(np.array(list(set(self.best_paras[para_name]))))
        binwidth = np.min(np.diff(best_paras_set))
        # print(binwidth)

        sns.histplot(data=self.best_paras, x=para_name, binwidth=binwidth,
                     stat='probability', common_norm=False, element="step", fill=False)
        # plt.axvline(x=39, c='pink', linewidth=4)
        sns.despine(offset=2, trim=True)
        plt.show()
        plt.close()

    def lineplot_para_vs_pear(self, para_name, y_name='pear'):
        sns.lineplot(data=self.para_df, x=para_name, y=y_name, ci='sd')
        sns.despine(offset=2, trim=True)
        plt.show()
        plt.close()

    def boxplot_pear(self, y_name='pear'):
        if self.best_paras is None:
            self.get_best_para_of_each_cell(y_name)

        f, ax = plt.subplots(figsize=(2, 4))

        ax.yaxis.grid(True)

        sns.boxplot(data=self.best_paras, x='tissue', y=y_name, width=.6, showfliers=False)
        sns.stripplot(x="tissue", y=y_name, data=self.best_paras,
                      size=2, color=".3", linewidth=0)

        ax.set(xlabel="")
        plt.xticks(rotation=60)

        plt.ylim([0, 1])
        plt.ylabel(y_name)

        sns.despine(trim=True, left=True)
        plt.tight_layout()

        plt.show(dpi=300)
        plt.close()

    def plot_best_vs_col(self, col, y_name='pear', col_to_int=False, hue=None):
        # tissue_to_int is used for concat_dis or concat_pair
        # denoting number of concatenated cells.
        if self.best_paras is None:
            self.get_best_para_of_each_cell(y_name)

        if col_to_int:
            self.best_paras[col] = self.best_paras[col].str.replace(u'([^\u0030-\u0039])', '', regex=True)
            self.best_paras[col] = self.best_paras[col].astype(int)

        sns.lineplot(data=self.best_paras, x=col, y=y_name, hue=hue)
        plt.show()
        plt.close()

    # TODO: delete thread input
    def get_pre_value(self, input_cell_name, cus_paras_dict=None, sort_by='pear'):
        cell_name = self.get_full_cell_name(input_cell_name)

        mtx_kwargs = self.mtx_kwargs.copy()
        mtx_file = get_cell_file_name(cell_name, mtx_kwargs['mtx_dir'], mtx_kwargs['suffix'])
        del mtx_kwargs['mtx_dir']
        del mtx_kwargs['suffix']
        mtx = read_mtx(mtx_file, **mtx_kwargs)

        if self.best_paras is None:
            self.get_best_para_of_each_cell(sort_by)

        # wrong names
        func_para_names = set(inspect.getfullargspec(self.func).args)
        cols = set(list(self.best_paras.columns))
        shared_paras = list(func_para_names & cols)
        paras_dict = {}
        for para_name in shared_paras:
            paras_dict[para_name] = self.best_paras.loc[cell_name, para_name]

        if cus_paras_dict is not None:
            for cus_paras_name in cus_paras_dict:
                paras_dict[cus_paras_name] = cus_paras_dict[cus_paras_name]

        print(paras_dict)
        pre_value = self.func(mtx, **paras_dict)
        return pre_value

    def get_concat_value(self, cell_name, join='inner', **pre_kwargs):
        true_value = self.get_true_value(cell_name)
        if 'true_value' in pre_kwargs['cus_paras_dict']:
            if pre_kwargs['cus_paras_dict']['true_value']:
                pre_kwargs['cus_paras_dict']['true_value'] = true_value
        pre_value = self.get_pre_value(cell_name, **pre_kwargs)

        true_value.rename('true', inplace=True)
        pre_value.rename('pre', inplace=True)

        concat_value = pd.concat([true_value, pre_value],
                                 axis=1, join=join)
        return concat_value

    def scatter_pre_ture(self, cell_name, heat_value_name=None, plot_ratio=None, **pre_kwargs):
        if self.spread_kwargs is None:
            raise ValueError('Please input spread_kwargs for true value.')

        concat_value = self.get_concat_value(cell_name, **pre_kwargs)

        if heat_value_name is not None:
            heat_value = self.get_true_value(cell_name, new_read_type=heat_value_name)
            heat_value.rename('heat', inplace=True)

            concat_value = pd.concat([concat_value, heat_value],
                                     axis=1, join='inner')

        #############
        # Plot
        sns.set_style('whitegrid')

        x_name = 'true'
        y_name = 'pre'
        heat_name = 'heat'

        if heat_value_name is None:
            sns.kdeplot(data=concat_value, x=x_name, y=y_name, color="#1f78b4")
            sns.scatterplot(data=concat_value, x=x_name, y=y_name, s=10, color='#969696')
        else:
            sns.scatterplot(data=concat_value, x=x_name, y=y_name, s=3,
                            hue=heat_name, palette='Reds_r')

        if plot_ratio is not None:
            plt.axhline(y=np.percentile(concat_value['pre'], plot_ratio), color='maroon')
            plt.axvline(x=np.percentile(concat_value['true'], plot_ratio), color='maroon')

        plt.xlabel(x_name)
        plt.ylabel(y_name)

        class_result = get_static(concat_value['pre'], concat_value['true'], plot_ratio)
        class_result_out = "\n".join([f'{key}:{class_result[key]}' for key in class_result])

        plt.title(f'Cell name: {cell_name}\n'
                  f'{class_result_out}')
        plt.tight_layout()
        plt.show()
        plt.close()

    def scatter_corr_pre_true(self, sort_by='pear', ascending=True, n_cell=5,
                              scatter_kwargs=None, pre_kwargs=None):
        if self.best_paras is None:
            self.get_best_para_of_each_cell(sort_by)

        sub_best_paras = self.best_paras.copy()
        sub_best_paras.sort_values(by=sort_by, inplace=True, ascending=ascending)

        for i_cell in range(n_cell):
            cell_name = sub_best_paras.iloc[i_cell, :].loc['cell_name']
            scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs
            pre_kwargs = {'sort_by': sort_by} if pre_kwargs is None else pre_kwargs
            self.scatter_pre_ture(cell_name, **scatter_kwargs, **pre_kwargs)

    def scatter_2D_with_pre_true_value(self, cell_name, dg_dir, index_file,
                                       start=10, end=90, n_slice=6, **pre_kwargs):
        def _read_dg(dg_file, index_file):
            dg = pd.read_table(dg_file, names=['chr', 'start', 'end', 'x', 'y', 'z'],
                               index_col=False)

            dg.index = dg['chr'] + '_' + dg['start'].astype(str)
            dg.drop(columns=['chr', 'start', 'end'], inplace=True)

            window = hic.GenomeIndex(index_file).window_df
            window.index = window['chr'] + '_' + window['start'].astype(str)

            dg = pd.concat([window, dg], join='inner', axis=1)
            dg.index = dg['index']
            return dg

        dg_file = get_cell_file_name(cell_name, dg_dir)
        dg = _read_dg(dg_file, index_file)
        dg = dg[['x', 'y', 'z']]

        concat_value = self.get_concat_value(cell_name, join='outer', **pre_kwargs)
        concat_value = pd.concat([concat_value, dg], axis=1, join='outer')

        values = ['pre', 'true']
        # 0-1 normalization
        per = 5
        for value in values:
            value_array = np.array(concat_value[value])
            value_array = value_array[~np.isnan(value_array)]
            value_min = np.percentile(value_array, per)
            print(value_min)
            value_max = np.percentile(value_array, 100 - per)
            print(value_max)
            concat_value[value] = (concat_value[value] - value_min) / \
                                  (value_max - value_min)

        concat_value.fillna(value=0, inplace=True)

        x_range = np.linspace(start, end, n_slice)
        x_range = [np.percentile(concat_value['x'], i) for i in x_range]

        result_df = {'i_x': [], 'y': [], 'z': [], 'value': [], 'value_name': []}
        for x_idx in range(len(x_range) - 1):
            x_dg = concat_value[(concat_value['x'] >= x_range[x_idx]) &
                                (concat_value['x'] <= x_range[x_idx + 1])]

            for value in values:
                result_df['i_x'] += [x_idx] * x_dg.shape[0]
                result_df['y'] += list(x_dg['y'])
                result_df['z'] += list(x_dg['z'])
                result_df['value'] += list(x_dg[value])
                result_df['value_name'] += [value] * x_dg.shape[0]

        result_df = pd.DataFrame(result_df)

        g = sns.FacetGrid(data=result_df, row='i_x', col='value_name')
        g.map_dataframe(sns.scatterplot, x='y', y='z', hue='value',
                        edgecolor="none", s=3, hue_norm=(0, 1))
        # legend=False, edgecolor="none", s=3)

        plt.show()
        plt.close()
