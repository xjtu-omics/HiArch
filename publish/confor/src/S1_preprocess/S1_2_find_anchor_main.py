import pandas as pd
import os
import sys
from confor.utils import anchor_suffix
from confor.src.S1_preprocess.S1_2_find_anchors.U1_plot_anchor import plot_anchors


def main_find_anchor(filtered_mtx, output, fig_output,
                     anchor_method=None, anchor_change=None):
    ########
    # Find anchors 
    if anchor_method == 'inter_rowsum' or anchor_method == 'no_anchor':
        from confor.src.S1_preprocess.S1_2_find_anchors.S2_1_find_by_inter_rowsum import main_find_anchors_by_inter_rowsum
        print('Anchor method: Inter rowsum')
        anchor_regions, plot_maps, value_dicts = main_find_anchors_by_inter_rowsum(filtered_mtx, return_map_hist_for_plot=True)
    elif anchor_method == 'intra_x':
        from confor.src.S1_preprocess.S1_2_find_anchors.S2_2_find_by_intra_x import main_find_anchors_by_x
        print('Anchor method: Intra x-shape')
        anchor_regions, plot_maps, value_dicts = main_find_anchors_by_x(filtered_mtx, return_map_hist_for_plot=True)
    elif anchor_method == 'intra_low':
        from confor.src.S1_preprocess.S1_2_find_anchors.S2_3_find_by_intra_low import main_find_anchors_by_intra_low
        print('Anchor method: Intra low')
        anchor_regions, plot_maps, value_dicts = main_find_anchors_by_intra_low(filtered_mtx, return_map_hist_for_plot=True)
    else:
        raise ValueError('Unrecognized anchor method.')

    if anchor_change is not None:
        if isinstance(anchor_change, str):
            anchor_change = anchor_change.split(',')
            anchor_change = pd.DataFrame({'raw': anchor_change})
            anchor_change['chr'] = anchor_change['raw'].str.split(':', expand=True).iloc[:, 0].astype(int)
            anchor_change['changed_to_anchor'] = anchor_change['raw'].str.split(':', expand=True).iloc[:, 1].astype(int)
            print(f"anchor_change:{anchor_change}")
        from confor.src.S1_preprocess.S1_2_find_anchors.U3_change_anchor import change_anchor
        anchor_regions = change_anchor(anchor_regions, anchor_change)
    
    if anchor_method == 'no_anchor':
        from confor.src.S1_preprocess.S1_2_find_anchors.U3_change_anchor import remove_all_used
        anchor_regions = remove_all_used(anchor_regions)
    
    if output is not None:
        anchor_regions.loc_df.to_csv(f'{output}{anchor_suffix}', sep="\t", index=False)
    
    anchor_fig_outfile = f'{fig_output}.anchor.png' if fig_output is not None else None
    plot_anchors(plot_maps, value_dicts, anchor_regions, anchor_fig_outfile)
