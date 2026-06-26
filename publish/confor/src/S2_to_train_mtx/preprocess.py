import numpy as np
from torch import nn
import torch
import sys
from confor.src.S2_to_train_mtx.x_size import X_SIZE


def pool_mtx(chr_mtx, center_idx1=None, center_idx2=None):
    pool = nn.AdaptiveAvgPool2d(int(X_SIZE / 2))
    pool_all = nn.AdaptiveAvgPool2d(X_SIZE)
    
    if center_idx1 is None or center_idx2 is None:
        now_img = pool_all(torch.tensor(chr_mtx).unsqueeze(0).unsqueeze(0))
    
    # TODO: if center_idx to boundaries < .05 * chro_length, do symmetry pool.
    elif center_idx1 == 0 or center_idx1 >= chr_mtx.shape[0] - 1:
        now_img = pool_all(torch.tensor(chr_mtx).unsqueeze(0).unsqueeze(0))
        
    elif center_idx2 == 0 or center_idx2 >= chr_mtx.shape[1] - 1:
        now_img = pool_all(torch.tensor(chr_mtx).unsqueeze(0).unsqueeze(0))
        
    else:
        img_tensor = torch.tensor(chr_mtx).unsqueeze(0).unsqueeze(0)
        img_tensor11 = pool(img_tensor[...,:center_idx1,:center_idx2])
        img_tensor12 = pool(img_tensor[..., :center_idx1, center_idx2:])
        img_tensor21 = pool(img_tensor[..., center_idx1:, :center_idx2])
        img_tensor22 = pool(img_tensor[..., center_idx1:, center_idx2:])
        now_img = torch.cat([torch.cat([img_tensor11,img_tensor12],dim=-1),torch.cat([img_tensor21,img_tensor22],dim=-1)],dim=-2)
    return np.squeeze(np.array(now_img))


def symmetry(input_x, intra=True):
    if intra:
        sym_x = (input_x + np.flipud(np.fliplr(input_x))) / 2
    else:
        sym_x = input_x.copy()
        sym_x += np.flipud(input_x)
        sym_x += np.fliplr(input_x)
        sym_x += np.flipud(np.fliplr(input_x))
        sym_x /= 4
    return sym_x


def norm_mtx(input_x):
    if np.std(input_x) > 0:
        return (input_x - np.mean(input_x)) / np.std(input_x)
    else:
        return input_x - np.mean(input_x)


def remove_circle(input_x):
    circle_len = 5
    
    circle_idx_x = []
    circle_idx_y = []
    # Upper right
    for i in range(circle_len):
        circle_idx_x += [i] * (circle_len - i)
        circle_idx_y += [j - 1 for j in range(X_SIZE, X_SIZE - circle_len + i, -1)]
    
    for i in range(circle_len):
        circle_idx_x += [X_SIZE - i - 1] * (circle_len - i)
        circle_idx_y += [j for j in range(circle_len - i)]
    
    input_x[circle_idx_x, circle_idx_y] = 0
    return input_x
