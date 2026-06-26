import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import sys
from confor.utils import read_spe_confor, default_con_col, label_for_confor
from complex.utils import default_complex_col, Complex, norm_coms, label_for_local
from confor.utils import read_spe_confor, norm_confor_among_species, get_global_strength, Confor, keep_pos
from model.scatter.utils import scatter_set


def scatter_set(ax):
    ax.figure.set_figheight(2.6)
    ax.figure.set_figwidth(2.6)
    
    ax.figure.set_dpi(200)
    
    # ax.set_xlabel(label_for_local, color="#203864", fontweight='bold')
    # ax.set_ylabel(label_for_confor, color='#385723', fontweight='bold')
    ax.set_xlabel(label_for_local, fontweight='bold')
    ax.set_ylabel(label_for_confor, fontweight='bold')

    ax.set_xlim([2.25, 3.12])
    ax.set_ylim([-.2, 4.6])
    sns.despine(trim=True, ax=ax)


code_dir = os.path.abspath(__file__)
code_dir = '/'.join(code_dir.split('/')[:-1])
default_complex_file = f'{code_dir}/ave_complex.txt'

###########
# Read data
def read_spe_complex(input_cb, complex_file=default_complex_file):
    if isinstance(input_cb, str):
        cb_coms = Complex(input_cb)
        ave_cb_coms = cb_coms.get_mean_complex()
        cb_coms = pd.DataFrame({'file_name': 'Input',
                                default_complex_col: float(ave_cb_coms),
                                'input': True})
        cb_coms.index = ['Input']
    elif isinstance(input_cb, dict):
        cb_coms = {'file_name': [], 'input': [], default_complex_col: []}
        for file_name in input_cb:
            one_cb_coms = Complex(input_cb[file_name])
            ave_cb_coms = one_cb_coms.get_mean_complex()
            cb_coms['file_name'].append(f'{file_name}_input')
            cb_coms['input'].append(True)
            cb_coms[default_complex_col].append(float(ave_cb_coms))
        cb_coms = pd.DataFrame(cb_coms)
        cb_coms.index = cb_coms['file_name']
    else:
        raise ValueError('Unrecognized input.')
    
    coms = pd.read_table(complex_file)
    coms.index = coms['file_name']
    coms['input'] = False
    
    coms = pd.concat([cb_coms, coms])
    coms = norm_coms(coms)
    return coms


def read_global(input_gf, confor_dir=code_dir, intra=True):
    def _read_gf(gf_file, file_name="Input"):
        input_confors = Confor(gf_file)
        input_confors = input_confors.get_intra_sub(intra)
        input_confors = input_confors.values.groupby('type').agg({default_con_col: 'mean'})
        input_confors.reset_index(inplace=True)
        file_name = f'{file_name}_input' if file_name is not None else 'Input'
        input_confors['file_name'] = file_name
        input_confors['input'] = True
        return input_confors
    
    if isinstance(input_gf, str):
        input_confors = _read_gf(input_gf)
    elif isinstance(input_gf, dict):
        input_confors = []
        for file_name in input_gf:
            one_confors = _read_gf(input_gf[file_name], file_name)
            input_confors.append(one_confors)
        input_confors = pd.concat(input_confors, axis=0)
    else:
        raise ValueError('Unrecognized input.')
    
    confors = read_spe_confor(confor_dir, intra=intra, norm=False, only_keep_pos=False)
    confors['input'] = False
    
    confors = pd.concat([confors, input_confors], axis=0)
    confors = norm_confor_among_species(confors)
    confors = keep_pos(confors)
    return confors


def main_read(cb_file, gf_file,
              complex_file=default_complex_file,
              confor_dir=code_dir, intra=True):
    """
    Args:
        cb_file (str or dict): Either single file or files dictionary.
                               If dict, {species_name: file_path,
                                         species_name2: file_path2,}

        gf_file (str or dict): Same as cb_file
        intra (bool, optional): Intra global folding if True else False. Defaults to True.
    Output:
        cons DataFrame:
        # Species_name cb_score gf_score input
        # Homo sapiens  2.98    1.02    True / False

        confors DataFrame:
        # Species_name gf_score gf_type
        # Homo sapiens  1.02    center-end-axis
    """
    
    if isinstance(cb_file, dict) and isinstance(gf_file, dict):
        cb_names = list(cb_file.keys())
        cb_names.sort()
        gf_names = list(gf_file.keys())
        gf_names.sort()
        if cb_names != gf_names:
            warnings.warn(f'CB names {cb_names} != GF names {gf_names}')
    
    coms = read_spe_complex(cb_file, complex_file)
    confors = read_global(gf_file, confor_dir, intra)
    
    max_confors = get_global_strength(confors=confors)
    cons = pd.concat([max_confors, coms], axis=1, join='inner')
    
    cons.drop(columns=['file_name'], inplace=True)
    cons.rename(columns={default_con_col: 'gf_score',
                         default_complex_col: 'cb_score'},
                inplace=True)
    
    confors.drop(columns=['type_order'], inplace=True)
    confors.rename(columns={default_con_col: 'gf_score'},
                inplace=True)
    return cons, confors


###########
# Scatter plot
def scatter_plot(cb_file, gf_file, output=None, intra=True,
                 confor_dir=code_dir,
                 complex_file=default_complex_file):
    cons, _ = main_read(cb_file, gf_file, complex_file, confor_dir, intra)

    fig = plt.figure(figsize=(3, 2.6), dpi=200)
    ax = fig.add_subplot(111)
    sns.scatterplot(cons, x='cb_score',
                    y='gf_score',
                    color='lightgray', s=10,
                    ax=ax)
    input_cons = cons[cons['input']]
    sns.scatterplot(data=input_cons, x='cb_score',
                    y='gf_score',
                    color='black', s=20,
                    ax=ax)
    scatter_set(ax)
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == '__main__':
    from iutils.read_matrix import oe_dir
    
    cb_file = {'HN1': f'{oe_dir}/HN1/HN1.complexity.txt'}
    gf_file = {'HN1': f'{oe_dir}/HN1/HN1.confor.txt'}
    scatter_plot(cb_file, gf_file)
    
    # import argparse

    # parser = argparse.ArgumentParser(usage='Checkerboard.py -f sps_file -w index_file -o out_file',
    #                                  add_help=False, formatter_class=argparse.RawDescriptionHelpFormatter)

    # group = parser.add_argument_group("Required Parameters")
    # group.add_argument('-f', '--sps_file', type=str, required=True)
    # group.add_argument('-w', '--index_file', type=str, required=True)
    # group.add_argument('-o', '--out_file', type=str, required=True)

    # group1 = parser.add_argument_group("Optional Parameters")
    # group1.add_argument('-fo', '--fig_output', default=None)

    # args = vars(parser.parse_args())

    # scatter_plot(**args)

