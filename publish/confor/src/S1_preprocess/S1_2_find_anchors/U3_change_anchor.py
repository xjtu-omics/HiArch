import numpy as np
import pandas as pd


def change_anchor(anchor_regions, spe_anchor_change):
    new_anchor_region_df = anchor_regions.loc_df.copy()
    
    for _, row in spe_anchor_change.iterrows():
        chro = anchor_regions.gen_index.chros[row['chr'] - 1]
        chr_anchors = new_anchor_region_df[new_anchor_region_df['chr'] == chro]
        
        chr_anchors = chr_anchors.sort_values('start')
        
        old_used_idx = chr_anchors[chr_anchors['type'] == 'used']
        if old_used_idx.shape[0] > 0:
            old_used_idx = old_used_idx.index[0]
            new_anchor_region_df.loc[old_used_idx, 'type'] = 'normal'
        
        if row['changed_to_anchor'] != 0:
            changed_to_idx = chr_anchors.index[row['changed_to_anchor'] - 1]
            new_anchor_region_df.loc[changed_to_idx, 'type'] = 'used'
    
    anchor_regions.loc_df = new_anchor_region_df
    return anchor_regions


def remove_all_used(anchor_regions):
    new_anchor_region_df = anchor_regions.loc_df.copy()
    old_used_idx = anchor_regions[anchor_regions['type'] == 'used']
    if old_used_idx.shape[0] > 0:
        new_anchor_region_df.loc[old_used_idx.index, 'type'] = 'normal'
    anchor_regions.loc_df = new_anchor_region_df
    return anchor_regions


def main_change_anchor(mtx_name, anchor_regions, anchor_change):
    anchor_change_species = list(anchor_change['species'].unique())
    
    if mtx_name not in anchor_change_species:
        return anchor_regions
    
    spe_anchor_change = anchor_change[anchor_change['species'] == mtx_name]
    anchor_regions = change_anchor(anchor_regions, spe_anchor_change)
    return anchor_regions
