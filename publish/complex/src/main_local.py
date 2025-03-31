import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sys
from iutils.plot_heatmap import heatmap_plot
import complex.src.U1_plot_local as pl


adj_cmap = sns.light_palette("#555555", as_cmap=True, reverse=True)
adj_kwargs = {'cmap': adj_cmap, 'vmin': 0, 'vmax': 1}
map_kwargs = {'Adj': adj_kwargs}

n_col = 3
min_chro_len = 50
min_n_short_sims = 200

def get_local(input_mtx, do_plot=False, fig_output=None,
              chro_str=None, input_axes=None, short_dis_max_per=.15):
    if np.min(input_mtx.shape) < min_chro_len:
        return None
    
    from complex.src.S1_get_adj_mtx import get_adj_mtx
    adj_mtx = get_adj_mtx(input_mtx)
    
    import complex.src.S2_get_entro as s2
    short_dis_max = s2.get_min_len(adj_mtx, short_dis_max_per, 10)
    short_dis_min = s2.get_min_len(adj_mtx, .05, 5)
    
    # short_sims = s2.get_short_sims(adj_mtx, short_dis_max, short_dis_min)
    # if short_sims is None:
    #     return None
    # entro = s2.get_accordance(short_sims)
    
    # Calculate entropy separately at each diagonal,
    # to remove distance effect of large pattern.
    entro = []
    short_sims = []
    last_short_sims = []
    for i in range(short_dis_min, short_dis_max):
        sub_short_sims = s2.get_short_sims(adj_mtx, i, i)
        if sub_short_sims is None:
            short_sims += []
        else:
            short_sims += list(sub_short_sims)
        
        if len(short_sims) > min_n_short_sims:
            sub_entro = s2.get_accordance(short_sims)
            entro.append(sub_entro)
            last_short_sims = short_sims
            short_sims = []
    if len(entro) == 0:
        return None
    entro = np.mean(entro)
    
    if do_plot:
        short_sims = last_short_sims
        plot_maps = {'Raw': input_mtx, 'Adj': adj_mtx}
        
        if input_axes is None:
            fig_height = pl.get_fig_height(input_mtx)
            fig = plt.figure(figsize=(fig_height * n_col, fig_height))

        i_fig = 0
        axes = {}
        for map_name in plot_maps:
            i_fig += 1
            ax = fig.add_subplot(1, n_col, i_fig) if input_axes is None else input_axes[i_fig - 1]
            axes[map_name] = ax
            
            plot_map = plot_maps[map_name]
            one_map_kwargs = map_kwargs[map_name] if map_name in map_kwargs else {}
            heatmap_plot(plot_map, input_ax=ax, show_mtx_size=False, **one_map_kwargs)
        
        axes['Raw'].set_title(f'Normed map\nMatrix shape: {plot_maps["Raw"].shape}')
        axes['Adj'].set_title(f'Adjacency map')
        
        s2.plot_short_dis_bound(plot_maps['Adj'], axes['Adj'], short_dis_max, short_dis_min)

        i_fig += 1
        ax = fig.add_subplot(1, n_col, i_fig) if input_axes is None else input_axes[i_fig - 1]
        s2.histplot_short_adj(short_sims, entro, ax=ax)
        
        parameters = {"axes.labelsize": 20, "axes.titlesize": 24,
                      'xtick.labelsize': 16, 'ytick.labelsize': 16}
        plt.rcParams.update(parameters)
        
        if chro_str is not None:
            plt.suptitle(chro_str, fontsize=24)
        plt.tight_layout()
        if input_axes is None:
            if fig_output is None:
                plt.show()
            else:
                plt.savefig(f'{fig_output}_{chro_str}.accord.png')
            plt.close()
            
        # pl.plot_sim_vs_dis(adj_mtx)
        
       
        # heatmap_plot(input_mtx, out_file=f'{result_dir}/{fig_output.split("/")[-1]}_{chro_str}.map.tiff')
        # heatmap_plot(input_mtx, out_file=f'{result_dir}/{fig_output.split("/")[-1]}_{chro_str}.map.png')
        
    return entro


def main_local(input_data, out_file=None, do_plot=False, fig_output=None, short_dis_max_per=.15):
    plot_chros = pl.ran_get_plot_chros(input_data)
    
    complexes = {'accord_chro': [], 'other_chro': [], 'accord': []}
    for chro1, chro2, chr_mtx in input_data.yield_by_chro(triu=True):
        
        chr_mtx = chr_mtx.to_dense()
        chr_mtx.to_all(inplace=True)

        chro_str = f'{chro1}.{chro2}' if chro1 != chro2 else chro1
        chr_do_plot = True if do_plot and (chro_str in plot_chros[0] or chro_str in plot_chros[1]) else False
        accord = get_local(chr_mtx, chr_do_plot, fig_output, chro_str=chro_str, short_dis_max_per=short_dis_max_per)
        if accord is None:
            continue
        complexes['accord_chro'].append(chro1)
        complexes['other_chro'].append(chro2)
        complexes['accord'].append(accord)
        
        if chro1 != chro2:
            chr_mtx_T = chr_mtx.copy()
            chr_mtx_T.matrix = chr_mtx.matrix.T
            chro_str = f'{chro2}.{chro1}'
            chr_do_plot = True if do_plot and chro_str in plot_chros[1] else False
            accord = get_local(chr_mtx_T, chr_do_plot, fig_output, chro_str=chro_str, short_dis_max_per=short_dis_max_per)
            if accord is None:
                continue
            complexes['accord_chro'].append(chro2)
            complexes['other_chro'].append(chro1)
            complexes['accord'].append(accord)
    
    complexes = pd.DataFrame(complexes)
    if out_file is not None:
        complexes.to_csv(out_file, sep="\t", index=False)
    return complexes


def local_io(sps_file, index_file, out_file, do_plot=False, fig_output=None, short_dis_max_per=.15):
    from new_hic_class import read_sps_file
    sps_mtx = read_sps_file(sps_file, index_file)
    main_local(sps_mtx, out_file=out_file, fig_output=fig_output, do_plot=do_plot,
               short_dis_max_per=short_dis_max_per)

