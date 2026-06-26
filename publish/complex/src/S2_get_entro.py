import numpy as np
from scipy.stats import entropy

##########
# Get accordance
def get_min_len(input_mtx, per, min_len):
    # min_len = int(np.min(input_mtx.matrix.shape) * per)
    min_len = int(np.mean(list(input_mtx.row_window.chr_lens.values())) * per)
    min_len = min_len if min_len < min_len else min_len
    return min_len


# def get_short_sims(adj_mtx, short_dis_max, short_dis_min, per=2):
#     i_sims = []
#     for i in range(0, adj_mtx.shape[0]):
#         start = i + short_dis_min
#         end = i + short_dis_max + 1
#         end = adj_mtx.shape[1] if end > adj_mtx.shape[1] else end
#         if end >= start:
#             i_sims += list(adj_mtx[i, start: end])
        
#         start = i - short_dis_max
#         start = 0 if start < 0 else start
#         end = i - short_dis_min + 1
#         if end >= start:
#             i_sims += list(adj_mtx[i, start: end])
    
#     i_sims = np.array(i_sims)
#     # Filt out outliers
#     if len(i_sims) < 1:
#         return None
#     min_thresh = np.percentile(i_sims, per)
#     max_thresh = np.percentile(i_sims, 100 - per)
#     i_sims = i_sims[(i_sims < max_thresh) & 
#                     (i_sims > min_thresh)]
    
#     return i_sims


def get_short_sims(adj_mtx, short_dis_max, short_dis_min, per=2):
    i_sims = []
    for i, diag_index in adj_mtx.yield_diag_idxes():
        if i > short_dis_max:
            break
        if i < short_dis_min:
            continue
        i_sims += list(adj_mtx[diag_index])
    
    i_sims = np.array(i_sims)
    # Filt out outliers
    if len(i_sims) < 1:
        return None
    min_thresh = np.percentile(i_sims, per)
    max_thresh = np.percentile(i_sims, 100 - per)
    i_sims = i_sims[(i_sims < max_thresh) & 
                    (i_sims > min_thresh)]
    
    return i_sims


bins = np.linspace(0, 1.6, 30)
def get_accordance(short_sims):
    bin_pers, _ = np.histogram(short_sims, bins=bins)
    bin_pers = bin_pers / np.sum(bin_pers)
    return entropy(bin_pers)


##########
# Plot
import seaborn as sns
import matplotlib.pyplot as plt
def histplot_short_adj(short_sims, entro, ax):
    sns.histplot(x=short_sims,
                 stat='probability',
                 color='gray',
                 bins=bins,
                 ax=ax)
    ax.grid(axis='x')
    
    ax.set_title(f'Histogram of distance\nEntropy: {np.round(entro, 4)}')
    ax.set_xlabel('Distance')


def plot_short_dis_bound(input_mtx, ax, short_dis_max, short_dis_min):
    # Upper bounds
    ax.plot([short_dis_max, input_mtx.shape[0]], 
            [0, input_mtx.shape[1] - short_dis_max],
            color='maroon')
    ax.plot([short_dis_min, input_mtx.shape[0]], 
            [0, input_mtx.shape[1] - short_dis_min],
            color='maroon')
    
    xs = np.arange(0, input_mtx.shape[0])
    ymax = xs + short_dis_max
    ymax[ymax > input_mtx.shape[1]] = input_mtx.shape[1]
    ymin = xs + short_dis_min
    ymin[ymin > input_mtx.shape[1]] = input_mtx.shape[1]
    ax.fill_between(xs, ymin, ymax, 
                    color='maroon', alpha=.2)
    
    # Lower bounds
    ax.plot([0, input_mtx.shape[1] - short_dis_max], 
            [short_dis_max, input_mtx.shape[0]],
            color='maroon')
    ax.plot([0, input_mtx.shape[1] - short_dis_min], 
            [short_dis_min, input_mtx.shape[0]],
            color='maroon')
    
    xs = np.arange(0, input_mtx.shape[0])
    ymax = xs - short_dis_min
    ymax[ymax < 0] = 0
    ymin = xs - short_dis_max
    ymin[ymin < 0] = 0
    ax.fill_between(xs, ymin, ymax, 
                    color='maroon', alpha=.2)

