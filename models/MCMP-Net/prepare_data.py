import os
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split

root_dir = '/home/agn/studia/magisterka/kopia_dockera/Datasets/Datasets/Ninapro-DB1'
rows_padding = 661
cols = 11


def load_and_aggregate_data(root_dir):
    aggregated_data = []

    for patient in tqdm(os.listdir(root_dir)):
        patient_dir = os.path.join(root_dir, patient)
        if os.path.isdir(patient_dir):

            for gesture in os.listdir(patient_dir):
                gesture_dir = os.path.join(patient_dir, gesture)
                if gesture == "gesture-00": break
                if os.path.isdir(gesture_dir):
                    for rep_file in os.listdir(gesture_dir):
                        if rep_file.endswith('.mat'):
                            output_emg_array = np.zeros((rows_padding, cols))
                            rep_path = os.path.join(gesture_dir, rep_file)
                            
                            data = loadmat(rep_path)
                            emg_data = data['emg']
                            label = data['stimulus']
                            
                            combined_data = np.hstack((emg_data, label))
                            output_emg_array[:combined_data.shape[0], :] = combined_data

                        aggregated_data.append(output_emg_array)

    return aggregated_data


def save_mat(var_name, var_value):
    try: 
        var_value = np.transpose(var_value, (0, 2, 1))
    except:
        var_value = (np.transpose(var_value, (1, 0)))[0]

    print(var_name, var_value.shape)
    file_name = f"{var_name}.mat"
    savemat(os.path.join(".", file_name), {var_name: var_value})

    print(f"Saved {var_name} to {file_name} \n")


if __name__ == '__main__':
    aggregated_data = load_and_aggregate_data(root_dir)
    print("Shape of aggregated_data:", np.array(aggregated_data).shape, '\n')

    X = np.array(aggregated_data)[:, :, :10]
    y = np.array(aggregated_data)[:, :, 10]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    data_dict = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

    for var_name, var_value in data_dict.items():
        save_mat(var_name, var_value)