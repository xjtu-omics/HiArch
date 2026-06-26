import glob
import numpy as np
import pandas as pd

import sys
from iutils.read_matrix import oe_dir


default_complex_col = 'accord'
class Complex:
    def __init__(self, input_complex,
                 plot_col=default_complex_col, intra=True):
        if isinstance(input_complex, str):
            self.values = pd.read_table(input_complex)
        elif isinstance(input_complex, pd.DataFrame):
            self.values = input_complex
        elif isinstance(input_complex, Complex):
            self.values = input_complex.values
        else:
            raise ValueError('Unrecognized input type.')
        self.value_names = [default_complex_col]
        
        self.plot_col = plot_col
        if self.plot_col not in self.value_names:
            raise ValueError('Cannot find input plot_col in value_names')
        
        self.values['intra'] = self.values['accord_chro'] == self.values['other_chro']
        if intra is not None:
            self.filt_complex('intra', intra, inplace=True)
    
    def filt_complex(self, key, value, inplace=False):
        new_values = self.values[self.values[key] == value].copy()
        if inplace:
            self.values = new_values
        else:
            return Complex(new_values)

    def get_mean_complex(self):
        return self.values[[self.plot_col]].mean(axis=0)
 
    def out_plot_series(self, rename_to='complex'):
        out_series = self.values[self.plot_col].copy()
        
        if rename_to is not None:
            out_series = self.values[self.plot_col].copy().rename(rename_to)
        return out_series


####################
# Read complexes
default_complex_dir = oe_dir
complex_suffix = '.complexity.txt'

label_for_local = 'Checkerboard score'

def read_comes(data_dir=default_complex_dir,
               suffix=complex_suffix, intra=True):
    value_df = []
    for file in glob.glob(f'{data_dir}/*/*{suffix}'):
        file_name = file.split('/')[-1].split(suffix)[0]
        
        complexes = Complex(file, intra=intra).values
        complexes['file_name'] = file_name
        
        value_df.append(complexes)
    return pd.concat(value_df, axis=0)


def norm_coms(values):
    value_arr = np.array(values[default_complex_col])
    min_value = np.percentile(value_arr, 2)
    print(min_value)
    values.loc[values[default_complex_col] < min_value, default_complex_col] = min_value
    return values


def get_spe_ave_comes(data_dir=default_complex_dir, intra=True, suffix=complex_suffix):
    values = read_comes(data_dir, intra=intra, suffix=suffix)
    values = values.groupby('file_name').agg({default_complex_col: 'mean'})
    values = values.reset_index()
    
    values = norm_coms(values)
    return values

