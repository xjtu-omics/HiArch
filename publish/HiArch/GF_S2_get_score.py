import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from confor.src.S2_to_train_mtx.to_train_mtx import to_train_mtx
from confor.src.S5_sim_to_pat.to_confor import main_to_confor
# from confor.src.S5_sim_to_pat.plot_confor import main_plot

# TODO: is_circle
def get_confor(sps_file, index_file, anchor_file, out_dir, fig_outdir=None):
    fig_outdir = out_dir if fig_outdir is None else fig_outdir
    
    spe_name = sps_file.split('/')[-1].split('.')[0]
    
    intra_data, inter_data = to_train_mtx(sps_file, index_file, anchor_file, spe_name,
                                          do_plot=True, fig_out_dir=fig_outdir)
    os.system(f'mv {fig_outdir}/{spe_name}/* {fig_outdir}')
    os.system(f'rmdir {fig_outdir}/{spe_name}')
    
    os.system(f'mkdir -p {out_dir}/{spe_name}')
    main_to_confor(intra_data, inter_data, out_dir=out_dir)
    os.system(f'mv {out_dir}/{spe_name}/* {out_dir}')
    os.system(f'rmdir {out_dir}/{spe_name}')
    
    # main_plot()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(usage='GF_S2_get_confor.py -f sps_file -w index_file -o out_dir',
                                     add_help=False, formatter_class=argparse.RawDescriptionHelpFormatter)

    group = parser.add_argument_group("Required Parameters")
    group.add_argument('-f', '--sps_file', type=str, required=True)
    group.add_argument('-w', '--index_file', type=str, required=True)
    group.add_argument('-af', '--anchor_file', type=str, required=True)
    group.add_argument('-o', '--out_dir', type=str, required=True)

    group1 = parser.add_argument_group("Optional Parameters")
    group1.add_argument('-fo', '--fig_outdir', default=None)
    args = vars(parser.parse_args())

    get_confor(**args)
    