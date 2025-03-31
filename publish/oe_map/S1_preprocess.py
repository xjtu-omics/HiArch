import re
import numpy as np


def filt_chr_by_name(sps_mtx):    
    chr_lens = sps_mtx.row_window.chr_lens
    max_len_chr = max(chr_lens, key=chr_lens.get)
    max_len_chr_prefix = re.search(r'([A-Z]*)\d', max_len_chr).group(1)
    
    chros = sps_mtx.row_window.chros
    kept_chros = [chro for chro in chros if chro.startswith(max_len_chr_prefix)]
    
    new_mtx = sps_mtx.get_multi_chros_mtx(kept_chros)
    return new_mtx


def rough_concat(sps_mtx, max_bin_num=1000, log_base=10):
    max_chr_len = np.max(list(sps_mtx.row_window.chr_lens.values()))
    concat_times = np.log(max_chr_len / max_bin_num) / np.log(log_base)
    concat_times = 0 if concat_times < 0 else concat_times
    concat_times = np.power(log_base, int(concat_times))
    print(f'# Will concat raw max len ({int(max_chr_len)}) {concat_times} times')
    if concat_times > 1:
        sps_mtx = sps_mtx.condense(concat_times)
    # print(f'# Matrix shape after concat: {sps_mtx.shape}. Ave len: {int(np.mean(list(sps_mtx.row_window.chr_lens.values())))}')
    return sps_mtx


min_nonzero_ratio = .95
max_searching_n_con = 10
def estimate_concat_time(nonzero_ratio, min_nonzero_ratio=min_nonzero_ratio):
    """ concat time can be estimated from nonzero_ratio.
        0.8 (min_nonzero_ratio) / .2 (current nonzero) = 4 = 2 ^ 2 (the shape is two-dimension). concat times is to
    """
    time = np.sqrt(min_nonzero_ratio / nonzero_ratio)
    print(f'Estimated concat time is {int(time) + 1} ({time}).')
    return int(time) + 1


larger_than_last_ratio = .1
larger_than_max_ratio = .01

min_len_for_kept_chro = 20 # TODO: minimum length for chromosome??
tid_off_low_bin_ratio = 5

max_bin_num_for_chros = 600
log_base = 2
def preprocess_mtx(sps_mtx, do_filt_chr_by_name=False):
    sps_mtx = sps_mtx.to_all()
    
    trun_mtx = sps_mtx.tid_off_zero_chros()
    if do_filt_chr_by_name:
        trun_mtx = filt_chr_by_name(trun_mtx)
    
    trun_mtx = rough_concat(trun_mtx, max_bin_num=max_bin_num_for_chros, log_base=log_base)
    
    trun_mtx = trun_mtx.tid_off_low_bins(tid_off_low_bin_ratio)
    
    nonzero_ratio = trun_mtx.get_nonzero_ratio()
    print(f'Non-zero ratio is {nonzero_ratio}.')
    if nonzero_ratio > min_nonzero_ratio:
        # trun_mtx = trun_mtx.tid_off_low_bins(tid_off_low_bin_ratio)
        trun_mtx = trun_mtx.keep_long_chros(min_len_for_kept_chro)
        return trun_mtx
    
    print(f'### Non-zero ratio is smaller than {min_nonzero_ratio}. Will try to concat.')
    est_n_con = estimate_concat_time(nonzero_ratio)
    
    for n_con in range(est_n_con, est_n_con + max_searching_n_con):
        con_mtx = trun_mtx.condense(n_con)
        
        con_mtx = con_mtx.tid_off_low_bins(tid_off_low_bin_ratio)
        con_mtx = con_mtx.keep_long_chros(min_len_for_kept_chro)
        
        if con_mtx is None:
            return None
        
        new_nonzero_ratio = con_mtx.get_nonzero_ratio()
        if new_nonzero_ratio > min_nonzero_ratio:
            print(f'Found mtx with nonzero {new_nonzero_ratio} > {min_nonzero_ratio} with condense fold {n_con}')
            return con_mtx
    print(f'### Warning: Can not find mtx with nonzero > {min_nonzero_ratio}. Max nonzero_ratio is {new_nonzero_ratio} with {n_con}')
    return None
