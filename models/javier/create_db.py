import os
import re
from tqdm import tqdm

import scipy.io
import numpy as np
import pandas as pd

from label_dict import label_dict


def extract_cols(file, ex):
    rep = file["rerepetition"].copy()
    emg = file["emg"].copy()
    lab = file["restimulus"].copy() 

    # rename the labels according to the label_dict
    new_lab = np.array([[label_dict[ex][lab[i][0]]] for i in range(lab.shape[0])])  
    return rep, emg, new_lab

def extract_features(dataframe):
    summed_data = dataframe
    summed_data.columns = ['rep_num'] + electrode_cols + ['label']

    combined_data = summed_data.drop(columns=['rep_num'])

    return combined_data


if __name__ == "__main__":
    NUM_COLUMNS = 10

    electrode_cols = [f'Electrode {i+1}' for i in range(NUM_COLUMNS)]
    cls = ["label"]
    column_names = electrode_cols + cls
    final_database = pd.DataFrame(columns=column_names)

    directory = r'/home/agn/studia/magisterka/kopia_dockera/Datasets/__zip__/Ninapro-DB1/'
    pattern = r'E(\d+)\.mat$'

    for filename in tqdm(os.listdir(directory)):
        if filename.endswith('.mat'):
            file_path = os.path.join(directory, filename)
            matlab_f = scipy.io.loadmat(file_path)

            exercise = int(re.search(pattern, filename).group(1))
            rep, emg, lab = extract_cols(matlab_f, exercise)
            
            df = pd.DataFrame(np.concatenate((rep, emg, lab), axis=1))
            df.drop(df[df[11] == 0.0].index, inplace=True)

            sub_df = extract_features(df)
            
            final_database = pd.concat([final_database, sub_df], ignore_index=True)

    final_database.sort_values(by='label')
    print(final_database.shape)
    final_database.to_csv('processed_NinaDB1_try.csv', index=False) 
