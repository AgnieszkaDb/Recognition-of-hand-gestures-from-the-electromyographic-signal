import os
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split

class Data:
    def __init__(self, root_dir, rows_padding=661, cols=11):
        self.root_dir = root_dir
        self.rows_padding = rows_padding
        self.cols = cols

    def load_and_aggregate_data(self):
        aggregated_data = []
        for patient in tqdm(os.listdir(self.root_dir)):
            patient_dir = os.path.join(self.root_dir, patient)
            if not os.path.isdir(patient_dir):
                continue

            for gesture in os.listdir(patient_dir):
                if gesture == "gesture-00":
                    continue

                gesture_dir = os.path.join(patient_dir, gesture)
                if not os.path.isdir(gesture_dir):
                    continue

                for rep_file in os.listdir(gesture_dir):
                    if rep_file.endswith('.mat'):
                        output_emg_array = np.zeros((self.rows_padding, self.cols))
                        rep_path = os.path.join(gesture_dir, rep_file)

                        data = loadmat(rep_path)
                        emg_data = data['emg']
                        label = data['stimulus']
                        combined_data = np.hstack((emg_data, label))
                        output_emg_array[:combined_data.shape[0], :] = combined_data

                        aggregated_data.append(output_emg_array)

        return np.array(aggregated_data)

    @staticmethod
    def save_mat(var_name, var_value):
        try: 
            var_value = np.transpose(var_value, (0, 2, 1))
        except:
            var_value = (np.transpose(var_value, (1, 0)))[0]

        file_name = f"{var_name}.mat"
        savemat(os.path.join(".", file_name), {var_name: var_value})
        print(f"Saved {var_name} to {file_name}")

    def split_and_save_data(self, aggregated_data):
        X = aggregated_data[:, :, :10]
        y = aggregated_data[:, :, 10]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        data_dict = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

        for var_name, var_value in data_dict.items():
            self.save_mat(var_name, var_value)

# Usage:
# data = Data(root_dir)
# aggregated_data = data.load_and_aggregate_data()
# data.split_and_save_data(aggregated_data)
