import sys
from new_hic_class import SpsHiCMtx


mode = 'sgd_mean'
k = .2
counts_must_decrese = True


def oe_map_core(sps_mtx, counts_must_decrese=True):
    ode_mtx = sps_mtx.obs_d_exp(mode=mode,
                                k=k,
                                counts_must_decrese=counts_must_decrese)
    ode_mtx = ode_mtx.log(verbose=False)
    return ode_mtx.to_dense()
