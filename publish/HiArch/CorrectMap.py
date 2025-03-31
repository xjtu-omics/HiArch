import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from new_hic_class import read_sps_file
from iutils.plot_heatmap import heatmap_plot


def remove_abnormal_chros(input_mtx, ab_chrs):
    input_mtx.to_all(inplace=True)
    
    if ab_chrs == None:
        return input_mtx
    else:
        raw_chros = input_mtx.row_window.chros
        kept_chros = raw_chros.copy()
        for ab_chro_idx in ab_chrs:
            if ab_chro_idx > 0:
                ab_chro_idx -= 1
            ab_chro = raw_chros[ab_chro_idx]
            kept_chros.remove(ab_chro)
        return input_mtx.get_multi_chros_mtx(kept_chros)


def reorder_chros(input_mtx):
    return input_mtx.reorder_chros_by_len()


def correct_map(sps_file, index_file, output, ab_chrs=None,
                fig_output=None, do_reorder_chrs=True):
    fig_output = output if fig_output is None else fig_output
    
    sps_mtx = read_sps_file(sps_file, index_file)
    sps_mtx.mtx_name = sps_file
    sps_mtx.mtx_type = 'triu'
    
    new_mtx = remove_abnormal_chros(sps_mtx, ab_chrs)
    if do_reorder_chrs:
        new_mtx = reorder_chros(new_mtx)
    
    new_mtx.to_sps().to_output(f'{output}.clean_de_ode.mtx', out_window=False)
    new_mtx.row_window.to_output(f'{output}.clean_window.bed')
    heatmap_plot(new_mtx, cmap='vlag',
                    out_file=f'{fig_output}.clean_denor_map.png')    
    
    return new_mtx


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(usage='CorrectMap.py -f sps_file -w index_file -o out_file -ac 1 -1',
                                     add_help=False, formatter_class=argparse.RawDescriptionHelpFormatter)

    group = parser.add_argument_group("Required Parameters")
    group.add_argument('-f', '--sps_file', type=str, required=True)
    group.add_argument('-w', '--index_file', type=str, required=True)
    group.add_argument('-o', '--output', type=str, required=True)

    group1 = parser.add_argument_group("Optional Parameters")
    group1.add_argument('-fo', '--fig_output', default=None)
    group1.add_argument('-ac', '--ab_chrs', nargs='*', type=int, default=None,const=None)
    group1.add_argument('-drc', '--do_reorder_chrs', default=True)

    args = vars(parser.parse_args())

    correct_map(**args)
