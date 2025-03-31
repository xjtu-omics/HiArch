import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import os
import torch
from scipy.io import loadmat, savemat

import sys
sys.path.append("/data/home/cheyizhuo/project/centr_loc/mycode/confor/src/")
from S2_to_train_mtx.x_size import X_SIZE
from S3_dim_reduc.iter_dim_reduc import trans_x_by_UV
import confor_class as cc


##############################
# Iterations
def update(U, W, X, beta):
    V_tmp2 = torch.matmul(U.T, X) + beta * W # 8*10000
    V_tmp1 = torch.linalg.inv(torch.matmul(U.T, U) + beta * torch.eye(W.shape[0]).cuda()) # 8*8
    new_V = torch.matmul(V_tmp1, V_tmp2) # 8*10000

    new_W = torch.relu(new_V - lamb/(2*beta)) # 8*10000
    return new_V, new_W


def iter(X, U, beta=1e1, lamb=2e0, itermax=200, do_trans=False):
    # X 256*10000, U 256*8, V 8*10000, W 8*10000
    V = torch.matmul(U.T, X)
    V = V + torch.randn_like(V)
    W = torch.relu(V - lamb/beta)

    for _ in tqdm(range(itermax)):
        X_tmp = trans_x_by_UV(X, U, V) if do_trans else X
        V, W = update(U, W, X_tmp, beta)
    return V, W, X_tmp


##############################
# Main
def main_learn(save_path, loadmat_prefix, para_path, do_trans, lamb, itermax):
    all_names, all_imgs, _ = cc.loadmats(loadmat_prefix)

    para = loadmat(para_path)
    us = para['U'].T
    us = torch.from_numpy(us).cuda()

    inputs = torch.from_numpy(all_imgs).cuda()
    inputs = inputs.flatten(1).T.float()
    print(inputs.shape)

    V, W, X = iter(inputs, us, itermax=itermax, lamb=lamb, do_trans=do_trans)

    savemat(os.path.join(save_path, 'all_vs.mat'), {'theName':all_names, 
                                                     'X': X.detach().cpu().numpy(),
                                                     'V': V.detach().cpu().numpy().T,
                                                     'W': W.detach().cpu().numpy().T})


if __name__ == '__main__':
    save_path = cc.intra_dir
    para_path = cc.intra_para_file
    loadmat_prefix = cc.intra_prefix
    do_trans = False
    lamb = 5
    itermax = 200
    
    # save_path = cc.inter_dir
    # para_path = cc.inter_para_file
    # loadmat_prefix = cc.inter_prefix
    # do_trans = True
    # lamb = 5
    # itermax = 200
    
    main_learn(save_path, loadmat_prefix, para_path, do_trans, lamb, itermax)
