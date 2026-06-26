import warnings
import numpy as np
import pandas as pd
import os
import sys
from scipy.io import loadmat
cur_dir = os.path.abspath(os.path.dirname(__file__))

##############
# Ave map class
intra_ave_map_file = f"{cur_dir}/intra_ave_maps.mat"
inter_ave_map_file = f'{cur_dir}/inter_ave_maps.mat'
intra_ave_names = ['End-whole', 'Large-center', 'Center',
                   'Center-end-axis', 'Center-whole']
inter_ave_names = ['Center-center', 'Center-end-axis',
                   'Center-whole', 'End-whole', 'End-end']

##############
# Data class
def loadmats(loadmat_prefix):
    import glob
    loadmat_files = glob.glob(f'{loadmat_prefix}*.mat')
    names, imgs, isym_imgs = [], [], []
    for loadmat_file in loadmat_files:
        sub_data = loadmat(loadmat_file)
        names.append(np.squeeze(sub_data['theName']))
        imgs.append(np.squeeze(sub_data['X']))
        isym_imgs.append(np.squeeze(sub_data['X_isym']))
    names = np.hstack(names)
    print(f"Name shape: {names.shape}")
    imgs = np.vstack(imgs)
    print(f"Img shape: {imgs.shape}")
    isym_imgs = np.vstack(isym_imgs)
    print(f"Isym img shape: {isym_imgs.shape}")
    return names, imgs, isym_imgs

class ConforData:
    def __init__(self, input_data):
        if isinstance(input_data, str):
            names, imgs, isym_imgs = loadmats(input_data)
        else:
            names, imgs, isym_imgs = input_data
            def _list2arr(item):
                if isinstance(item, list):
                    if len(item) == 0:
                        item = None
                    elif len(item) == 1:
                        item = np.array(item)
                    else:
                        item = np.stack(item, axis=0)
                return item
            names = _list2arr(names)
            imgs = _list2arr(imgs)
            isym_imgs = _list2arr(isym_imgs)
        
        self.names = names
        self.is_intra = self.check_intra()
        
        self.xs = imgs
        if self.xs is not None:
            self.xs = self.xs.reshape((self.xs.shape[0], -1))
            
        self.isym_xs = isym_imgs
        if self.isym_xs is not None:
            self.isym_xs = self.isym_xs.reshape((self.xs.shape[0], -1))
    
    def check_intra(self):
        if self.names is None:
            return None
        one_name = str(self.names[0]).strip()
        if len(one_name.split(':')) == 2:
            return True
        elif len(one_name.split(':')) == 3:
            return False
        else:
            raise ValueError('Unrecognized name.')

    def concat_other(self, other_confor):
        if other_confor.names is None:
            return 0
        if self.names is None:
            self.names = other_confor.names
            self.xs = other_confor.xs
            self.isym_xs = other_confor.isym_xs
            return 0
        self.names = np.concatenate((self.names, other_confor.names), axis=0)
        self.xs = np.concatenate((self.xs, other_confor.xs), axis=0)
        self.isym_xs = np.concatenate((self.isym_xs, other_confor.isym_xs), axis=0)

    def out_names(self):
        if self.names.size > 1:
            return [str(i) for i in np.squeeze(self.names)]
        else:
            return [str(self.names[0, 0])]

class AveMap:
    def __init__(self, ave_map_file):
        self.ave_maps = loadmat(ave_map_file)
        self.clus = self.get_clus()
        self.norm_ave_maps = self.norm_ave_maps()

    def get_clus(self):
        clus = []
        for clu in self.ave_maps:
            if not clu.startswith('_'):
                clus.append(clu)
        return clus
    
    def norm_ave_maps(self):
        norm_ave_maps = {}
        for clu in self.clus:
            ave_map = self.ave_maps[clu]
            norm_ave_map = ave_map / np.sum(np.abs(ave_map))
            norm_ave_maps[clu] = norm_ave_map.reshape((-1,))
        return norm_ave_maps
