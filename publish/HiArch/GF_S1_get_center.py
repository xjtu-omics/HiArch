import numpy as np
import pandas as pd

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from confor.src.S1_preprocess.S1_1_filtering import filter_io
from confor.src.S1_preprocess.S1_2_find_anchor_main import main_find_anchor

def anchors_io(index_file, output, sps_file=None, fig_output=None, use_exist_filt=True,
               anchor_method='inter_rowsum', anchor_change=None):
    """ Find the center anchor for global folding.
    
    Two steps:
        1. Get spatially filtered matrix.
        2. Find anchor by one of the following methods:
            "inter_rowsum", "intra_x", "intra_low"
    
    Anchors can be mannually chosen. By inputting "chro1:anchor_num,chro2:anchor_num, ..."
    Example: "1:2,3:1" as the second anchor as center anchor for the first chromosome, and first for third chro.
    """
   
    filtered_mtx = filter_io(sps_file, index_file, output, fig_output, use_exist_filt)
    main_find_anchor(filtered_mtx, output, fig_output,
                     anchor_method=anchor_method, anchor_change=anchor_change)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(usage='GF_S1_get_anchor.py -f sps_file -w index_file -o out_file -ac 1:2,3:1',
                                     add_help=False, formatter_class=argparse.RawDescriptionHelpFormatter)

    group = parser.add_argument_group("Required Parameters")
    group.add_argument('-f', '--sps_file', type=str, required=True)
    group.add_argument('-w', '--index_file', type=str, required=True)
    group.add_argument('-o', '--output', type=str, required=True)

    group1 = parser.add_argument_group("Optional Parameters")
    group1.add_argument('-fo', '--fig_output', default=None)
    group1.add_argument('-am', '--anchor_method', type=str, default='inter_rowsum')
    group1.add_argument('-ac', '--anchor_change', default=None, const=None, nargs='?')
    group1.add_argument('-ue', '--use_exist_filt', default=True)

    args = vars(parser.parse_args())

    anchors_io(**args)