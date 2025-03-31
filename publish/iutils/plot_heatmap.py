import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle


##############
# Main plot heatmap
def get_fontsize_ratio(ax, axsize_to_fontsize_ratio=5):
    _, _, width, height = ax.get_position().bounds
    fig_width, fig_height = ax.figure.get_size_inches()
    ax_width, ax_height = width * fig_width, height * fig_height
    fontsize_ratio = np.min([ax_width, ax_height]) / axsize_to_fontsize_ratio
    return fontsize_ratio


default_vmax_per = 95
def get_vmax_vmin(input_mtx, vmax_per=default_vmax_per, vmax=None, vmin=None):
    input_mtx = input_mtx.to_dense()
    
    out_vmax = vmax
    out_vmin = vmin
    
    if out_vmax is not None and out_vmin is not None:
        return out_vmax, out_vmin
    
    if input_mtx.has_neg:
        vmax = np.percentile(input_mtx.matrix[input_mtx.matrix > 0], vmax_per)
        vmin = np.percentile(input_mtx.matrix[input_mtx.matrix < 0], 100 - vmax_per)
        # vmax, vmin = np.max(input_mtx.matrix), np.min(input_mtx.matrix)
        min_value = np.min(np.abs([vmax, -vmin]))
        vmax, vmin = min_value, -min_value
    else:
        vmax = np.percentile(input_mtx.matrix[input_mtx.matrix > 0], vmax_per)
        vmin = np.percentile(input_mtx.matrix[input_mtx.matrix > 0], 100 - vmax_per)
    
    out_vmax = vmax if out_vmax is None else out_vmax
    out_vmin = vmin if out_vmin is None else out_vmin
    
    return out_vmax, out_vmin


def plot_chr_seps(new_mtx, ax, side='both'):
    if new_mtx.row_window is not None:
        linewidth = 1
        if len(new_mtx.row_window.chros) > 1:
            for chro in new_mtx.row_window.chr_seps:
                sep1, _ = new_mtx.row_window.chr_seps[chro]
                if sep1 == 0:
                    continue
                
                if side == 'both':
                    ax.axhline(sep1, color='black', lw=linewidth)
                    ax.axvline(sep1, color='black', lw=linewidth)
                elif side == 'v':
                    ax.axvline(sep1, color='black', lw=linewidth)
                elif side == 'h':
                    ax.axhline(sep1, color='black', lw=linewidth)


def plot_doms(doms, ax, both_side=False):
    from matplotlib.patches import Rectangle
    colors = ['gold', 'limegreen']

    x_min, x_max = ax.get_xlim()
    x_width = x_max - x_min
    y_min, y_max = ax.get_ylim()
    y_width = y_max - y_min

    cur_color_idx = 0
    for syn_dom in doms:
        min_idx = np.min(syn_dom)
        syn_len = np.max(syn_dom) - min_idx + 1
        cur_color_idx = 1 if cur_color_idx == 0 else 0
        ax.add_patch(Rectangle((x_min, min_idx),
                                x_width, syn_len,
                                edgecolor=None, alpha=.2,
                                facecolor=colors[cur_color_idx]))
        if both_side:
            ax.add_patch(Rectangle((min_idx, y_min),
                                    syn_len, y_width,
                                    edgecolor=None, alpha=.2,
                                    facecolor=colors[cur_color_idx]))


def get_min_len(ax, chr_num, side='both',
                min_len_chr_per=.1, min_len_whole_per=.01):
    x_shape = np.abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    y_shape = np.abs(ax.get_ylim()[1] - ax.get_ylim()[0])
    if side == 'both':
        min_chr_len = int(np.min([x_shape, y_shape]) / chr_num * min_len_chr_per)
        min_whole_len = int(np.min([x_shape, y_shape]) * min_len_whole_per)
    elif side == 'v':
        min_chr_len = int(x_shape / chr_num * min_len_chr_per)
        min_whole_len = int(x_shape * min_len_whole_per)
    elif side == 'h':
        min_chr_len = int(y_shape / chr_num * min_len_chr_per)
        min_whole_len = int(y_shape * min_len_whole_per)
    return np.max([min_chr_len, min_whole_len])


def plot_centr_loc(centr_loc, ax, side='both', plot_type='point', 
                   facecolor=None, edgecolor=None, alpha=None, zorder=2,
                   min_len_chr_per=.1, min_len_whole_per=.01):
    loc_df = centr_loc.loc_df.copy()
    if loc_df.shape[0] == 0:
        warnings.warn('No valid centromeres are found.')
        return 0
    
    if plot_type == 'point':
        if side != 'both':
            warnings.warn('Side must be both for points.')
        side = 'both'
    
    min_len = get_min_len(ax, loc_df.shape[0], side,
                          min_len_chr_per, min_len_whole_per)
    
    if plot_type == 'point':
        alpha = 1 if alpha is None else alpha
        facecolor = 'black' if facecolor is None else facecolor
        edgecolor = 'white' if edgecolor is None else edgecolor
        for _, loc1 in loc_df.iterrows():
            for _, loc2 in loc_df.iterrows():
                center_x = (loc1['cs'] + loc1['ce']) / 2
                len_x = loc1['ce'] - center_x
                len_x = len_x if len_x > min_len else min_len
                
                center_y = (loc2['cs'] + loc2['ce']) / 2
                len_y = loc2['ce'] - center_y
                len_y = len_y if len_y > min_len else min_len
                
                ell = Ellipse((center_x, center_y),
                              width=len_x, height=len_y,
                              facecolor=facecolor, edgecolor=edgecolor,
                              zorder=zorder, lw=2)
                ax.add_patch(ell)
    else:
        alpha = .2 if alpha is None else alpha
        facecolor = 'black' if facecolor is None else facecolor
        
        x_min, x_max = ax.get_xlim()
        x_width = np.abs(x_max - x_min)
        y_min, y_max = ax.get_ylim()
        y_min, y_max = [y_max, y_min] if y_min > y_max else [y_min, y_max]
        y_width = np.abs(y_max - y_min)

        for _, loc in loc_df.iterrows():
            center_x = (loc['cs'] + loc['ce']) / 2
            len_x = loc['ce'] - center_x
            len_x = len_x if len_x > min_len else min_len
            cs = center_x - len_x / 2
            if centr_loc.gen_index is not None:
                chr_start = centr_loc.gen_index.chr_seps[loc['chr']][0]
                chr_end = centr_loc.gen_index.chr_seps[loc['chr']][1]
            else:
                chr_start = 0
                chr_end = np.max([x_width, y_width])
            cs = chr_start if cs < chr_start else cs
            len_x = chr_end - cs if cs + len_x > chr_end else len_x
            
            if side == 'v' or side == 'both':
                ax.add_patch(Rectangle((x_min, cs),
                                        x_width, len_x,
                                        edgecolor=edgecolor, alpha=alpha,
                                        facecolor=facecolor,
                                        zorder=zorder))
            if side == 'h' or side == 'both':
                ax.add_patch(Rectangle((cs, y_min),
                                        len_x, y_width,
                                        edgecolor=edgecolor, alpha=alpha,
                                        facecolor=facecolor,
                                        zorder=zorder))


def heatmap_plot(new_mtx, centr_loc=None, out_file=None, show_mtx_size=True,
                 input_ax=None, cmap=None, vmax_per=95, vmin=None, vmax=None, center=None,
                 no_margin=True):
    fig_width_ratio = 1 / 1000
    min_fig_width = 5
    
    new_mtx = new_mtx.to_dense()
    new_mtx.to_all(inplace=True)
    
    if input_ax is None:
        fig_width = int(new_mtx.shape[0] * fig_width_ratio)
        fig_width = min_fig_width if fig_width < min_fig_width else fig_width
        fig_height = int(new_mtx.shape[1] * fig_width_ratio)
        fig_height = min_fig_width if fig_height < min_fig_width else fig_height
        fig = plt.figure(figsize=(fig_width, fig_height))
        ax = fig.add_subplot(111)
    else:
        fig = None
        ax = input_ax
    
    vmax, vmin = get_vmax_vmin(new_mtx, vmax_per=vmax_per, vmax=vmax, vmin=vmin)
    if new_mtx.has_neg:
        cmap = 'vlag' if cmap is None else cmap
        center = 0 if center is None else center
    else:
        cmap = 'Reds' if cmap is None else cmap
        center = None if center is None else center
    sns.heatmap(new_mtx.matrix, vmax=vmax, vmin=vmin, center=center, cmap=cmap, ax=ax,
                cbar=False, xticklabels=False, yticklabels=False)
    
    if new_mtx.row_window is not None:
        linewidth = 1
        if len(new_mtx.row_window.chros) > 1:
            for chro in new_mtx.row_window.chr_seps:
                sep1, _ = new_mtx.row_window.chr_seps[chro]
                if sep1 == 0:
                    continue
                ax.axhline(sep1, color='black', lw=linewidth)
                ax.axvline(sep1, color='black', lw=linewidth)
    
    if centr_loc is not None:
        if centr_loc.gen_index is None:
            centr_loc.gen_index = new_mtx.row_window
            centr_loc.find_centr_index()
        
        centr_loc.loc_df = centr_loc.loc_df[centr_loc.loc_df['chr'].isin(new_mtx.row_window.chros)]
        plot_centr_loc(centr_loc, ax, plot_type='point')
    
    fontsize_ratio = get_fontsize_ratio(ax)
    if show_mtx_size:
        ax.set_title(f'Chro shape: {new_mtx.shape}',
                    size=12 * fontsize_ratio)
    if new_mtx.shape[0] == new_mtx.shape[1]:
        ax.set_aspect('equal')
    
    if fig is not None:
        fig.set_dpi(int(1 / fig_width_ratio / 3))
    if no_margin:
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(0, 0, 1, 1, 0, 0)
        plt.margins(0, 0)
        plt.axis('off')
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    if input_ax is None:
        if out_file is None:
            plt.show()
        else:
            plt.savefig(out_file)
        plt.close()


def remove_margin(ax, dpi=100, large_ratio=1,
                  width=None, height=None):
    if height is None:
        xlim = ax.get_xlim()
        height = np.abs(xlim[1] - xlim[0])
    else:
        height = height * dpi
        
    if width is None:
        ylim = ax.get_ylim()
        width = np.abs(ylim[1] - ylim[0])
    else:
        width = width * dpi
    
    ax.figure.set_size_inches(width/dpi * large_ratio, 
                              height/dpi * large_ratio)
    ax.figure.set_dpi(dpi)
    ax.figure.patch.set_alpha(0.)
    
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    plt.axis('off')
    plt.margins(0, 0)


##############
# Heatmap utils
def plot_whole_maps(plot_maps, do_show=True, out_file=None,
                    n_col=None, map_kwargs=None, map_to_ax_ratio=1,
                    show_mtx_size=True):
    fig_height_ratio = .001
    min_fig_height = 5
    
    n_col = len(plot_maps) if n_col is None else n_col
    
    n_gs_col = map_to_ax_ratio * len(plot_maps) + n_col - len(plot_maps)
    gs = plt.GridSpec(map_to_ax_ratio, n_gs_col)
    
    fig_height = list(plot_maps.values())[0].shape[0] * fig_height_ratio
    fig_height = fig_height if fig_height > min_fig_height else min_fig_height
    fig = plt.figure(figsize=(fig_height * n_gs_col / map_to_ax_ratio, fig_height))

    map_kwargs = {} if map_kwargs is None else map_kwargs
    axes = {}
    
    i_fig = 0
    cur_col_gs = 0
    for map_name in plot_maps:
        ax = fig.add_subplot(gs[:, cur_col_gs: cur_col_gs + map_to_ax_ratio])
        cur_col_gs += map_to_ax_ratio
        axes[map_name] = ax
        i_fig += 1
        
        plot_map = plot_maps[map_name]
        one_map_kwargs = map_kwargs[map_name] if map_name in map_kwargs else {}
        heatmap_plot(plot_map, input_ax=ax, 
                     show_mtx_size=show_mtx_size, 
                     **one_map_kwargs)
    
    while i_fig < n_col:
        axes[f'a{i_fig - len(plot_maps)}'] = fig.add_subplot(gs[:, cur_col_gs: cur_col_gs + 1])
        cur_col_gs += 1
        i_fig += 1
    
    if do_show or out_file is not None:
        if out_file is None:
            plt.show()
        else:
            plt.savefig(out_file)
        plt.close()
    
    return fig, axes


def yield_chro_maps(plot_maps, yield_map_name=None, 
                    do_show=False, out_file=None,
                    n_chro=3, n_col=None, map_kwargs=None):
    if do_show or out_file is not None:
        if n_col is not None:
            if n_col > len(plot_maps):
                raise ValueError('If add another column to plot, must not do show.')
    
    n_col = len(plot_maps) if n_col is None else n_col
    n_col = len(plot_maps) if n_col < len(plot_maps) else n_col
    n_row = n_chro
    
    map_kwargs = {} if map_kwargs is None else map_kwargs
    
    maps = list(plot_maps.values())
    inter_chros = set(maps[0].row_window.chros)
    inter_chros = inter_chros.intersection(*[set(m.row_window.chros) for m in maps])
    plot_chros = np.random.choice(list(inter_chros), n_chro, replace=False)
    
    fig = plt.figure(figsize=(n_col * 4.2, n_row * 4))

    i_chro = 0
    for chro in plot_chros:
        i_chro += 1
        i_fig = (i_chro - 1) * n_col
        
        for map_name in plot_maps:
            i_fig += 1
            ax = fig.add_subplot(n_row, n_col, i_fig)
            
            plot_chr_map = plot_maps[map_name].get_region_mtx(chro)
            one_map_kwargs = map_kwargs[map_name] if map_name in map_kwargs else {}
            heatmap_plot(plot_chr_map, input_ax=ax, **one_map_kwargs)
            
            if (i_fig - 1) // n_col == 0:
                ax.set_title(map_name)
            
            if i_fig % n_col == 1:
                ax.set_ylabel(chro)
        
        yield_axes = []
        for i_fig in range(i_fig + 1, i_chro * n_col + 1):
            ax = fig.add_subplot(n_row, n_col, i_fig)
            yield_axes.append(ax)
        
        if yield_map_name is not None:
            yield yield_axes, plot_maps[map_name].get_region_mtx(chro)
        else:
            yield yield_axes
    
    if do_show or out_file is not None:
        if out_file is None:
            plt.show()
        else:
            plt.savefig(out_file)
        plt.close()
    return fig, plot_chros

