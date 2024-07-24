import os
import re
from tqdm import tqdm

import scipy.io
import numpy as np
import pandas as pd

from label_dict import label_dict

NUM_COLUMNS = 10

sum_columns = [f'sum_{i+1}' for i in range(NUM_COLUMNS)]
mean_columns = [f'mean_{i+1}' for i in range(NUM_COLUMNS)]
waveform_columns = [f'waveform_{i+1}' for i in range(NUM_COLUMNS)]
av_energy_columns = [f'av_energy_{i+1}' for i in range(NUM_COLUMNS)]
cls = ["class"]

column_names = sum_columns + mean_columns + waveform_columns + av_energy_columns + cls

# create an empty dataframe
final_database = pd.DataFrame(columns=column_names)

def extract_cols(file, ex):
    rep = file["rerepetition"].copy()
    emg = file["emg"].copy()
    lab = file["restimulus"].copy() 

    # rename the labels according to the label_dict
    new_lab = np.array([[label_dict[ex][lab[i][0]]] for i in range(lab.shape[0])])  
    return rep, emg, new_lab


def wavelength_form(x):
    return x.diff().abs().sum()


def av_signal_energy(x):
    return np.sum(x**2) / len(x)


def extract_features(dataframe):
    summed_data = dataframe.groupby([11, 0], as_index=False).sum()
    summed_data.columns = ['class', 'Group_0'] + sum_columns

    mean_data = dataframe.groupby([11, 0], as_index=False).mean()
    mean_data.columns = ['class', 'Group_0'] + mean_columns

    waveform_data = dataframe.groupby([11, 0], as_index=False).agg(wavelength_form)
    waveform_data.columns = ['class', 'Group_0'] + waveform_columns

    av_energy_data = dataframe.groupby([11, 0], as_index=False).agg(av_signal_energy)
    av_energy_data.columns = ['class', 'Group_0'] + av_energy_columns

    combined_data = pd.concat([summed_data.drop(columns=['Group_0']),
                               mean_data.drop(columns=['class', 'Group_0']),
                               waveform_data.drop(columns=['class', 'Group_0']),
                               av_energy_data.drop(columns=['class', 'Group_0'])], axis=1)

    return combined_data

directory = r'..\data\ninapro_DB1'
pattern = r'E(\d+)\.mat$'
files_limit = 0

for filename in tqdm(os.listdir(directory)):
    if filename.endswith('.mat'):
        files_limit += 1

        file_path = os.path.join(directory, filename)
        matlab_f = scipy.io.loadmat(file_path)

        exercise = int(re.search(pattern, filename).group(1))
        rep, emg, lab = extract_cols(matlab_f, exercise)
        
        df = pd.DataFrame(np.concatenate((rep, emg, lab), axis=1))
        df.drop(df[df[11] == 0.0].index, inplace=True)

        sub_df = extract_features(df)
        
        final_database = pd.concat([final_database, sub_df], ignore_index=True)
        

final_database.to_csv('processed_NinaDB1.csv', index=False) 
