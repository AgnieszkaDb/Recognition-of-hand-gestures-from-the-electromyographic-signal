import os
from tqdm import tqdm
import scipy.io
import numpy as np
import pandas as pd
from label_dict import label_dict

class Data:
    def __init__(self, directory, num_columns=10):
        self.directory = directory
        self.num_columns = num_columns
        self.electrode_cols = [f'Electrode {i+1}' for i in range(self.num_columns)]
        self.cls = ["label"]
        self.column_names = self.electrode_cols + self.cls
        self.final_database = pd.DataFrame(columns=self.column_names)

    def extract_cols(self, file, ex):
        rep = file["rerepetition"].copy()
        emg = file["emg"].copy()
        lab = file["restimulus"].copy()

        # Rename the labels according to the label_dict
        new_lab = np.array([[label_dict[ex][lab[i][0]]] for i in range(lab.shape[0])])  
        return rep, emg, new_lab

    def extract_features(self, dataframe):
        summed_data = dataframe
        summed_data.columns = ['rep_num'] + self.electrode_cols + ['label']
        combined_data = summed_data.drop(columns=['rep_num'])
        return combined_data

    def process_data(self, max_subjects=None):
        counter = 0

        for subject in tqdm(os.listdir(self.directory)):
            if max_subjects and counter == max_subjects:
                break
            counter += 1

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
        
        self.final_database.sort_values(by='label')
        return self.final_database

    def save_to_csv(self, filename='processed_data.csv'):
        self.final_database.to_csv(filename, index=False)
        print(f"Data saved to {filename}")


if __name__ == "__main__":
    data_processor = Data(directory='../../data')
    final_database = data_processor.process_data(max_subjects=2)
    print(final_database.shape)
    data_processor.save_to_csv('processed_NinaDB1_tests.csv')
