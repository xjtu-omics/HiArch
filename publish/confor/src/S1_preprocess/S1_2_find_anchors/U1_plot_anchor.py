import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sys
from iutils.plot_heatmap import plot_chr_seps, plot_whole_maps, plot_centr_loc


########
# Plot
def barplot_value(input_value, ax, anchor_regions=None,
                  plot_value_name=None):
    if plot_value_name is not None:
        pass
    elif len(input_value.value_names) == 1:
        plot_value_name = input_value.value_names[0]
    else:
        raise ValueError('Plz input value_name for values with mutiple columns.')
    
    #######
    # Bar plot of rowsum and regions
    value_arr = np.array(input_value[plot_value_name])
    ax.barh(np.arange(len(value_arr)), value_arr,
            color='gray')
    
    if anchor_regions is not None:
        if 'rs' in anchor_regions.loc_df.columns:
            region_colors = sns.color_palette('tab10', n_colors=anchor_regions.shape[0])
            i_region = 0
            for _, region in anchor_regions.loc_df.iterrows():
                region_range = np.arange(region['rs'], region['re'] + 1)
                region_inter_rowsum = input_value.values.loc[region_range, plot_value_name]
                ax.barh(region_range, region_inter_rowsum, color=region_colors[i_region])
                i_region += 1
    
    ax.set_title('Intra x strength')
    ax.set_yticks([])
    ax.set_ylim([0, len(value_arr)])
    ax.invert_yaxis()
    
    #######
    # Scatter plot of anchors
    s_per = .03
    
    if anchor_regions is not None:
        if 'type' in anchor_regions.loc_df.columns:
            type_colors = {'normal': 'orange',
                        'telo': 'gray',
                        'used': 'red'}
            anchor_type = np.array(anchor_regions.loc_df['type'])
            color = None
        else:
            type_colors = None
            anchor_type = None
            color = 'red'
    
        y_pos = np.array(anchor_regions.loc_df['cen']).astype(int)
        x_value = input_value.values.loc[y_pos, plot_value_name]
        s = np.abs(ax.get_ylim()[1] - ax.get_ylim()[0]) * s_per
        sns.scatterplot(x=x_value, y=y_pos, ax=ax, s=s,
                        color=color, legend=False,
                        hue=anchor_type, palette=type_colors)


def plot_anchors(plot_maps, value_dicts, anchor_regions, out_file=None):
    n_col = len(plot_maps) + len(value_dicts) * 2

    one_map = list(plot_maps.values())[0]

    ###########
    # Plot maps
    _, axes = plot_whole_maps(plot_maps, do_show=False, 
                                n_col=n_col, map_to_ax_ratio=3)
    for map_name in plot_maps:
        plot_centr_loc(anchor_regions.keep_used_anchors(), axes[map_name], 
                       plot_type='rec', side='both')
    
    ###########
    # Plot values
    i_hist_fig = 0
    for value_name in value_dicts:
        ax = axes[f'a{i_hist_fig}']
        i_hist_fig += 1
        barplot_value(value_dicts[value_name], ax)
        ax.set_title(value_name)
        plot_chr_seps(one_map, ax, side='h')
    
        ax = axes[f'a{i_hist_fig}']
        i_hist_fig += 1
        barplot_value(value_dicts[value_name], ax,
                      anchor_regions)
        ax.set_title(value_name)
        plot_chr_seps(one_map, ax, side='h')
    
    ###########
    # Plot
    plt.suptitle(one_map.mtx_name, size=16)
    plt.tight_layout()
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)
    plt.close()

