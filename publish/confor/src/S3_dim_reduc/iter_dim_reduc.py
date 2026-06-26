import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import os
import torch
from scipy.io import loadmat, savemat

import sys
from S3_dim_reduc.filter_low_sig import keep_top_imgs

from S2_to_train_mtx.to_train_mtx import pool_mtx
from S2_to_train_mtx.x_size import X_SIZE
import confor_class as cc


##############################
# Kernals
def get_rand_kernal():
    half_size = int(X_SIZE / 2)
    rand_k = np.zeros((X_SIZE, X_SIZE))
    k_0 = np.random.randn(half_size, half_size)
    rand_k[:half_size, :half_size] = k_0
    rand_k[:half_size, -half_size:] = np.fliplr(k_0)
    rand_k[-half_size:, :half_size] = np.flipud(k_0)
    rand_k[-half_size:, -half_size:] = np.fliplr(np.flipud(k_0))
    return rand_k.reshape((-1,))
 

def load_kernel(pre_kernal=None, ker_num=8):
    all_kernels = []
    i_kernal = 0
    
    for _ in range(i_kernal, ker_num):
        all_kernels.append(get_rand_kernal())
    
    all_kernels = np.stack(all_kernels, axis=-1)
    all_kernels = torch.from_numpy(all_kernels).cuda().float()
    return all_kernels


def norm_kernal(kernels):
    kernels = kernels / (torch.sqrt(torch.sum(kernels ** 2, dim=0, keepdim=True)))
    return kernels


def plot_kernals(kernels, output):
    for i, U_img in enumerate(kernels.T.detach().cpu().numpy()):
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(U_img.reshape(X_SIZE, X_SIZE), cmap=plt.get_cmap("coolwarm"))
        plt.savefig(f'{output}{i}.png', bbox_inches='tight')
        plt.close(fig)


##############################
# Iterations
def trans_x_by_UV(X, U, V):
    trans_idx = np.arange(X_SIZE ** 2).reshape((X_SIZE, X_SIZE)).T.reshape((-1,))
    X_trans = X[trans_idx, :]
    
    X_pre = torch.matmul(U, V)
    X_diff = torch.sum((X_pre - X) ** 2, dim=0)
    X_trans_diff = torch.sum((X_pre - X_trans) ** 2, dim=0)
    
    X_tmp = X.clone()
    change_idxes = X_trans_diff < X_diff
    X_tmp[:, change_idxes] = X_trans[:, change_idxes]
    return X_tmp


def update(U, W, X, beta, gamma):
    V_tmp2 = torch.matmul(U.T, X) + beta * W # 8*10000
    V_tmp1 = torch.linalg.inv(torch.matmul(U.T, U) + beta * torch.eye(W.shape[0]).cuda()) # 8*8
    new_V = torch.matmul(V_tmp1, V_tmp2) # 8*10000

    new_W = torch.relu(new_V - lamb/(2*beta)) # 8*10000

    VV = torch.linalg.inv(torch.matmul(new_V, new_V.T) + gamma * torch.eye(new_W.shape[0]).cuda()) # 8*8
    U_tmp = torch.matmul(X, torch.matmul(new_V.T, VV)) # 256*8
    new_U = norm_kernal(U_tmp)
    return new_U, new_V, new_W


def iter(X, U, beta=1e1, lamb=2e0, gamma=1e-5, itermax=200, do_trans=False):
    # X 256*10000, U 256*8, V 8*10000, W 8*10000
    V = torch.matmul(U.T, X)
    V = V + torch.randn_like(V)
    W = torch.relu(V - lamb/beta)

    for _ in tqdm(range(itermax)):
        X_tmp = trans_x_by_UV(X, U, V) if do_trans else X
        U, V, W = update(U, W, X_tmp, beta, gamma)
    return U, V, W, X_tmp


##############################
# Main
def main_learn(save_path, loadmat_prefix, do_trans, pre_kernal, 
               nU, lamb, filt_per, itermax):
    enc_kernels = load_kernel(pre_kernal, nU)
    enc_kernels = norm_kernal(enc_kernels)

    plot_kernals(enc_kernels, f'{save_path}/U_raw_')

    all_names, all_imgs, all_isym_imgs = cc.loadmats(loadmat_prefix)

    all_names = keep_top_imgs(all_imgs, all_isym_imgs, all_names, filt_per)
    all_imgs = keep_top_imgs(all_imgs, all_isym_imgs, bot_per=filt_per)

    inputs = torch.from_numpy(all_imgs).cuda()
    inputs = inputs.flatten(1).T.float()
    print(inputs.shape)

    U, V, W, X = iter(inputs, enc_kernels.float(), itermax=itermax,
                      lamb=lamb, do_trans=do_trans)

    plot_kernals(U, f'{save_path}/U_')

    savemat(os.path.join(save_path, 'all_pre.mat'), {'theName':all_names, 
                                                     'X': X.detach().cpu().numpy(),
                                                     'V': V.detach().cpu().numpy().T,
                                                     'U': U.detach().cpu().numpy().T, 
                                                     'W': W.detach().cpu().numpy().T})


if __name__ == '__main__':
    save_path = cc.intra_dir
    loadmat_prefix = cc.intra_prefix
    do_trans = False
    pre_kernal = None
    nU = 6
    lamb = 5
    filt_per = 20
    itermax = 200
    
    # save_path = cc.inter_dir
    # loadmat_prefix = cc.inter_prefix
    # do_trans = True
    # pre_kernal = None
    # nU = 6
    # lamb = 5
    # filt_per = 5
    # itermax = 200
    
    main_learn(save_path, loadmat_prefix, do_trans, pre_kernal, 
               nU, lamb, filt_per, itermax)
