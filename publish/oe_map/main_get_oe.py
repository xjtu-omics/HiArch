import numpy as np

import sys
from iutils.plot_heatmap import heatmap_plot
from iutils.read_matrix import Anchors
from oe_map.U1_plot_utils import histplot_oe_values, heatmap_diff_ob_methods


#################
def oe_map_main(sps_mtx, output=None, fig_output=None, centr_loc=None, 
                do_filt_chr_by_name=False, counts_must_decrese=True):
    from oe_map.S1_preprocess import preprocess_mtx
    pro_mtx = preprocess_mtx(sps_mtx, do_filt_chr_by_name)
    if pro_mtx is None:
        print(f'Warning: Not found proper matrix for {sps_mtx.mtx_file}')
        return None
    print(f'Processed matrix size is {pro_mtx.shape}')
    chr_lens_list = list(pro_mtx.row_window.chr_lens.values())
    print(f'Chr num is {len(chr_lens_list)}. Max chr size is {np.max(chr_lens_list)}, min is {np.min(chr_lens_list)}.')
    
    # if fig_output is not None:
        # heatmap_diff_ob_methods(pro_mtx, centr_loc, n_chro=5,
        #                         out_file=f'{fig_output}.nor_method.png')
        # print(f'Plotting normalization methods DONE.')
        
    if fig_output is not None:
        heatmap_plot(pro_mtx, centr_loc,
                     out_file=f'{fig_output}.ori_map.png')
        print(f'Plotting raw mtx DONE.')
    
    from oe_map.S2_oe import oe_map_core
    ode_mtx = oe_map_core(pro_mtx, counts_must_decrese=counts_must_decrese)
    print(f'Matrix normalization DONE.')
    if output is not None:
        ode_mtx.to_sps().to_output(f'{output}.ode.mtx')
    
    if fig_output is not None:
        heatmap_plot(ode_mtx, centr_loc, cmap='vlag',
                     out_file=f'{fig_output}.nor_map.png')
        histplot_oe_values(ode_mtx, out_file=f'{fig_output}.nor_hist.png')
        print(f'Plotting norm mtx DONE.')
    
    from oe_map.S3_denoise import main_denoise
    de_ode_mtx = main_denoise(ode_mtx)
    
    from oe_map.S4_norm_value import norm_de_mtx
    de_ode_mtx = norm_de_mtx(de_ode_mtx)

    print(f'Denoise normalized matrix DONE.')
    if output is not None:
        de_ode_mtx.to_sps().to_output(f'{output}.de_ode.mtx',
                                      out_window=False)
    
    if fig_output is not None:
        heatmap_plot(de_ode_mtx, centr_loc, cmap='vlag',
                     out_file=f'{fig_output}.denor_map.png')
        histplot_oe_values(de_ode_mtx, out_file=f'{fig_output}.denor_hist.png')
        print(f'Plotting denoised norm mtx DONE.')

    return de_ode_mtx


#################
# Main
def oe_map_io(mtx_file, output, fig_output=None,
              do_filt_chr_by_name=False, counts_must_decrese=True,
              index_file=None, input_centr_loc=None, thread=1):
    if input_centr_loc is not None:
        centr_loc = Anchors(input_centr_loc)
    else:
        centr_loc = None
    
    from iutils.read_matrix import read_matrix
    sps_mtx = read_matrix(mtx_file, index_file, thread=thread)
    
    de_ode_mtx = oe_map_main(sps_mtx, output, fig_output, centr_loc, 
                             do_filt_chr_by_name, counts_must_decrese)
    
    return de_ode_mtx
