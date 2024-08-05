import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import scipy.io as scio
import numpy as np
import json

from generator import Data

# Usage:
root_dir = '../Hilbert/Datasets/Datasets/Ninapro-DB1'

data = Data(root_dir)
aggregated_data = data.load_and_aggregate_data()
data.split_and_save_data(aggregated_data)


# Load Data
def load_data():
    dataFile = 'X_train.mat'
    X_train = scio.loadmat(dataFile)['X_train']
    dataFile = 'X_test.mat'
    X_test = scio.loadmat(dataFile)['X_test']
    dataFile = 'y_train.mat'
    y_train = scio.loadmat(dataFile)['y_train'] - 1
    y_train = np.transpose(y_train)
    dataFile = 'y_test.mat'
    y_test = scio.loadmat(dataFile)['y_test'] - 1
    y_test = np.transpose(y_test)
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_data()

def minmaxscaler(data):
    min = np.amin(data)
    max = np.amax(data)
    return (data - min) / (max - min)

X_train = minmaxscaler(X_train)
X_test = minmaxscaler(X_test)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

batch_size = 32
train_dataset = train_dataset.batch(batch_size).shuffle(len(X_train))
test_dataset = test_dataset.batch(batch_size)

class PointNet(tf.keras.Model):
    def __init__(self):
        super(PointNet, self).__init__()
        self.conv1 = layers.Conv1D(256, 1, activation='relu')
        self.conv2 = layers.Conv1D(512, 1, activation='relu')
        self.conv3 = layers.Conv1D(1024, 1, activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()
        self.lin = layers.Dense(52)

    def call(self, x):
        x = tf.transpose(x, [0, 2, 1])  # Transpose to match the input dimensions expected by Conv1D
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.conv3(x1)
        x1 = self.bn3(x1)
        x1 = tf.reduce_max(x1, axis=1)  # Reduce max along the correct axis to match PyTorch behavior
        
        x2 = self.conv1(x)
        x2 = self.bn1(x2)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = self.conv3(x2)
        x2 = self.bn3(x2)
        x2 = tf.reduce_max(x2, axis=1)  # Reduce max along the correct axis

        x = tf.concat([x1, x2], axis=1)  # Concatenate along the feature dimension
        x = self.lin(x)
        return x

model = PointNet()

model.compile(optimizer=optimizers.Adam(learning_rate=0.001, beta_1=0.9,
                                        beta_2=0.999, epsilon=1e-08),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=2, validation_data=test_dataset)

model.save("../../logs/MCMP.keras")


def get_predictions_and_labels(model, test):
    y_pred_probs = model.predict(test)
    # print(f'y_pred_probs{ y_pred_probs}')
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_test = np.concatenate([y for x, y in test], axis=0)
    print(f'y_test: {y_test} \n y_pred: {y_pred}')
    return y_pred, y_test

y_pred, y_test = get_predictions_and_labels(model, test_dataset)
# y_test = np.argmax(y_test, axis=1)

history_data = { 
    'model_name': 'MCMP',
    'train_loss': history.history['loss'],
    'train_accuracy': history.history['accuracy'],
    'val_loss': history.history['val_loss'],
    'val_accuracy': history.history['val_accuracy'],
    'y_pred': y_pred.tolist(),
    'y_test': y_test.tolist()
}

with open('../../logs/history_MCMP.json', 'w') as f:
    json.dump(history_data, f)
