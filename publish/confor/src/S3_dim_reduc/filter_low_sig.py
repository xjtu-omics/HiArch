import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import loadmat


###########
# Main
# def get_median_values(imgs):
#     return np.median(np.abs(imgs), axis=(1, 2))


def get_sym_diffs(imgs, isym_imgs):
    sym_diffs = np.mean(np.abs(imgs - isym_imgs), axis=(1,))
    sym_diffs[np.isnan(sym_diffs)] = np.max(sym_diffs[~np.isnan(sym_diffs)])
    return sym_diffs


def get_min_img_value(img_values, bot_per=100 / 3):
    return np.percentile(img_values, bot_per)


def keep_top_imgs(imgs, isym_imgs, change_item=None, 
                  bot_per=100 / 3, reverse=False):
    change_item = imgs if change_item is None else change_item
    
    img_values = get_sym_diffs(imgs, isym_imgs)
    if not reverse:
        max_img_value = get_min_img_value(img_values, bot_per=bot_per)
        # print(max_img_value)
        kept_idx = np.where(img_values <= max_img_value)[0]
    else:
        min_img_value = get_min_img_value(img_values, bot_per=bot_per)
        kept_idx = np.where(img_values >= min_img_value)[0]
    kept_items = change_item[kept_idx, ...]
    return kept_items


###########
# Plot
def histplot_img_values(imgs, isym_imgs, bot_per=100 / 3):
    img_values = get_sym_diffs(imgs, isym_imgs)
    min_img_value = get_min_img_value(img_values, bot_per=bot_per)
    print(min_img_value)
    
    plt.figure(figsize=(2.5, 2.3))
    sns.histplot(x=img_values, color='lightgray')
    plt.axvline(x=min_img_value, color='maroon')
    plt.xlabel('Difference between\npooled and symmetry map')
    sns.despine(trim=True, offset=2)
    plt.show()
    plt.close()


def plot_kept_imgs(imgs, isym_imgs, n_plot=20):
    img_values = get_sym_diffs(imgs, isym_imgs)
    for i in np.random.choice(np.arange(imgs.shape[0]), n_plot):
        img = imgs[i, ...]
        isym_img = isym_imgs[i, ...]
        print(img_values[i])
        fig = plt.figure(figsize=(6, 2.6))
        ax = fig.add_subplot(121)
        sns.heatmap(isym_img, cmap='vlag', center=0, ax=ax, cbar=False)
        ax.axis('off')
        ax.set_aspect('equal')
        ax.set_title('Raw')
        ax = fig.add_subplot(122)
        sns.heatmap(img, cmap='vlag', center=0, ax=ax, cbar=False)
        ax.set_title('Symmetry')
        ax.set_aspect('equal')
        ax.axis('off')
        plt.show()
        plt.close()


if __name__ == '__main__':
    # save_path = '/data/home/cheyizhuo/project/centr_loc/data2/confor/img_diag'
    # loadmat_path = f"{save_path}/all_Data_X_diag.mat"
    
    save_path = '/data/home/cheyizhuo/project/centr_loc/data2/confor/img_undiag'
    loadmat_path = f"{save_path}/all_Data_X.mat"
    bot_per = 10

    all_data = loadmat(loadmat_path)
    all_names = all_data['theName']
    all_imgs = all_data['X']
    all_isym_imgs = all_data['X_isym']
    print(all_imgs.shape)
    
    histplot_img_values(all_imgs, all_isym_imgs, bot_per)
    
    kept_imgs = keep_top_imgs(all_imgs, all_isym_imgs, bot_per=bot_per)
    kept_isym_imgs = keep_top_imgs(all_imgs, all_isym_imgs, all_isym_imgs, bot_per=bot_per)
    plot_kept_imgs(kept_imgs, kept_isym_imgs)
    
    print('#########################')
    print('Below are eliminated images.')
    kept_imgs = keep_top_imgs(all_imgs, all_isym_imgs, reverse=True, bot_per=20)
    kept_isym_imgs = keep_top_imgs(all_imgs, all_isym_imgs, all_isym_imgs, reverse=True, bot_per=20)
    plot_kept_imgs(kept_imgs, kept_isym_imgs)
