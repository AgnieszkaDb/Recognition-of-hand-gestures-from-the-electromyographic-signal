import os
import re
from tqdm import tqdm
import scipy.io
import numpy as np
import pandas as pd
from label_dict import label_dict

class Data:
    NUM_COLUMNS = 10

    def __init__(self, directory='../../../data', pattern='E(\d+)/.mat$', files_limit=0):
        self.directory = directory
        self.pattern = pattern
        self.files_limit = files_limit

        self.sum_columns = [f'sum_{i+1}' for i in range(self.NUM_COLUMNS)]
        self.mean_columns = [f'mean_{i+1}' for i in range(self.NUM_COLUMNS)]
        self.waveform_columns = [f'waveform_{i+1}' for i in range(self.NUM_COLUMNS)]
        self.av_energy_columns = [f'av_energy_{i+1}' for i in range(self.NUM_COLUMNS)]
        self.cls = ["class"]

        self.column_names = self.sum_columns + self.mean_columns + self.waveform_columns + self.av_energy_columns + self.cls
        self.final_database = pd.DataFrame(columns=self.column_names)

    def extract_cols(self, file, ex):
        rep = file["rerepetition"].copy()
        emg = file["emg"].copy()
        lab = file["restimulus"].copy()
        new_lab = np.array([[label_dict[ex][lab[i][0]]] for i in range(lab.shape[0])])  
        return rep, emg, new_lab

    def wavelength_form(self, x):
        return x.diff().abs().sum()

    def av_signal_energy(self, x):
        return np.sum(x**2) / len(x)

    def extract_features(self, dataframe):
        summed_data = dataframe.groupby([11, 0], as_index=False).sum()
        summed_data.columns = ['class', 'Group_0'] + self.sum_columns

        mean_data = dataframe.groupby([11, 0], as_index=False).mean()
        mean_data.columns = ['class', 'Group_0'] + self.mean_columns

        waveform_data = dataframe.groupby([11, 0], as_index=False).agg(self.wavelength_form)
        waveform_data.columns = ['class', 'Group_0'] + self.waveform_columns

        av_energy_data = dataframe.groupby([11, 0], as_index=False).agg(self.av_signal_energy)
        av_energy_data.columns = ['class', 'Group_0'] + self.av_energy_columns

        combined_data = pd.concat([
            summed_data.drop(columns=['Group_0']),
            mean_data.drop(columns=['class', 'Group_0']),
            waveform_data.drop(columns=['class', 'Group_0']),
            av_energy_data.drop(columns=['class', 'Group_0'])
        ], axis=1)

        return combined_data

    def process_files(self):
        for subject in tqdm(os.listdir(self.directory)):
            self.files_limit += 1
            if self.files_limit == 3: break
            for filename in os.listdir(os.path.join(self.directory, subject)):
                if filename.endswith('.mat'):

                    file_path = os.path.join(self.directory, subject, filename)
                    matlab_f = scipy.io.loadmat(file_path)

                    exercise = int(filename[-5])
                    rep, emg, lab = self.extract_cols(matlab_f, exercise)

                    df = pd.DataFrame(np.concatenate((rep, emg, lab), axis=1))
                    df.drop(df[df[11] == 0.0].index, inplace=True)

                    sub_df = self.extract_features(df)

                    self.final_database = pd.concat([self.final_database, sub_df], ignore_index=True, sort=False)

    def save_to_csv(self, filename='processed_NinaDB1.csv'):
        self.final_database.to_csv(filename, index=False)

