import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from new_hic_class import read_sps_file
from oe_map.main_get_oe import oe_map_io


def norm_dis(sps_file, index_file, output, fig_output=None,
             do_filt_chr_by_name=True, counts_must_decrese=True):
    fig_output = output if fig_output is None else fig_output
    
    oe_map_io(sps_file, output, fig_output,
              index_file=index_file,
              do_filt_chr_by_name=do_filt_chr_by_name,
              counts_must_decrese=counts_must_decrese)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(usage='NormDis.py -f sps_file -w index_file -o output',
                                     add_help=False, formatter_class=argparse.RawDescriptionHelpFormatter)

    group = parser.add_argument_group("Required Parameters")
    group.add_argument('-f', '--sps_file', type=str, required=True)
    group.add_argument('-w', '--index_file', type=str, required=True)
    group.add_argument('-o', '--output', type=str, required=True)

    group1 = parser.add_argument_group("Optional Parameters")
    group1.add_argument('-fo', '--fig_output', default=None)
    group1.add_argument('-df', '--do_filt_chr_by_name', default=True)
    group1.add_argument('-cmd', '--counts_must_decrese', default=True)

    args = vars(parser.parse_args())

    norm_dis(**args)
