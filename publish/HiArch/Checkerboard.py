import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from complex.src.main_local import local_io


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(usage='Checkerboard.py -f sps_file -w index_file -o out_file',
                                     add_help=False, formatter_class=argparse.RawDescriptionHelpFormatter)

    group = parser.add_argument_group("Required Parameters")
    group.add_argument('-f', '--sps_file', type=str, required=True)
    group.add_argument('-w', '--index_file', type=str, required=True)
    group.add_argument('-o', '--out_file', type=str, required=True)

    group1 = parser.add_argument_group("Optional Parameters")
    group1.add_argument('-fo', '--fig_output', default=None)
    group1.add_argument('-sd', '--short_dis_max_per',type=float, default=.15)

    args = vars(parser.parse_args())

    local_io(**args)
