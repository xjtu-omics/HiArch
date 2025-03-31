import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

##########
# Random get plot chros
def ran_get_plot_chros(input_data, n_intra_plot=4, n_inter_plot=2):
    chros = input_data.row_window.chros
    if len(chros) > n_intra_plot:
        plot_intra_chros = np.random.choice(chros, n_intra_plot, replace=False)
    else:
        plot_intra_chros = chros

    if len(chros) <= 1:
        plot_inter_pairs = []
    else:
        from itertools import combinations
        inter_chro_pairs = [f'{chr1}.{chr2}' for chr1, chr2 in combinations(list(chros), 2)]
        if len(inter_chro_pairs) > n_inter_plot:
            plot_inter_pairs = np.random.choice(inter_chro_pairs, n_inter_plot, replace=False)
        else:
            plot_inter_pairs = inter_chro_pairs
    return plot_intra_chros, plot_inter_pairs


##########
# Plot utils
def get_fig_height(input_map):
    fig_height_ratio = .005
    min_fig_height = 5
    fig_height = input_map.shape[0] * fig_height_ratio
    fig_height = fig_height if fig_height > min_fig_height else min_fig_height
    return fig_height


##########
# Plot
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

