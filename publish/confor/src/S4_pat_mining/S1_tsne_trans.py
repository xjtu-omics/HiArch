import numpy as np
import pandas as pd
from sklearn import manifold

import sys
sys.path.append("/data/home/cheyizhuo/project/centr_loc/mycode/confor/")
import confor_class as cc

intra_tsne_file = f'{cc.intra_dir}/tsne_vs.npy'
inter_tsne_file = f'{cc.inter_dir}/tsne_vs.npy'

inter_sub_para_file = f'{cc.inter_dir}/sub_pre.mat'
inter_sub_tsne_file = f'{cc.inter_dir}/sub_tsne_vs.npy'


def tsne_trans(para, tsne_file=None, perplexity=800,
               n_jobs=1, n_sub=None, sub_save_file=None):
    vs = pd.DataFrame(para.vs)

    if n_sub is not None:
        if sub_save_file is None:
            raise ValueError('Plz input save file for sub paras.')
        sub_para = para.get_sub(n_sub)
        sub_para.save_mat(sub_save_file)
        vs = sub_para.vs

    tsne = manifold.TSNE(n_components=2, init='pca',
                         perplexity=perplexity,
                         n_jobs=n_jobs)
    tsne_vs = tsne.fit_transform(vs)

    if tsne_file is not None:
        np.save(tsne_file, tsne_vs)
    return tsne_vs


if __name__ == '__main__':
    ##############
    # Intra
    # intra_para = cc.ConforPara(cc.intra_para_file)
    # intra_para.replace_vs(cc.intra_v_file)
    
    # tsne_trans(intra_para, perplexity=700,
    #            tsne_file=intra_tsne_file,
    #            n_jobs=40)

    ##############
    # Inter
    inter_para = cc.ConforPara(cc.inter_para_file)
    inter_para.replace_vs(cc.inter_v_file)
    
    # tsne_trans(inter_para, n_jobs=40,
    #            perplexity=4000, tsne_file=inter_tsne_file)
            
    #         #    perplexity=800, tsne_file=inter_sub_tsne_file, 
    #         #    n_sub=4000, sub_save_file=inter_sub_para_file)
