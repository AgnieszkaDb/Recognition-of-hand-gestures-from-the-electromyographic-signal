import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import scipy.io as scio
import numpy as np
import json

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

<<<<<<< Updated upstream
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

# Normalize the data
def minmaxscaler(data):
    min = np.amin(data)
    max = np.amax(data)
    return (data - min) / (max - min)

X_train = minmaxscaler(X_train)
X_test = minmaxscaler(X_test)

# Convert to TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# Batch the datasets
=======
>>>>>>> Stashed changes
batch_size = 32
train_dataset = train_dataset.batch(batch_size).shuffle(len(X_train))
test_dataset = test_dataset.batch(batch_size)

<<<<<<< Updated upstream
# Define the Model
=======
>>>>>>> Stashed changes
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
        
<<<<<<< Updated upstream
        # Processing a second pathway (same as first pathway for now)
=======
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
# Instantiate the model
model = PointNet()

# Compile the model
=======
model = PointNet()

>>>>>>> Stashed changes
model.compile(optimizer=optimizers.Adam(learning_rate=0.001, beta_1=0.9,
                                        beta_2=0.999, epsilon=1e-08),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

<<<<<<< Updated upstream
# Train the model
history = model.fit(train_dataset, epochs=2, validation_data=test_dataset)

# Save the model
model.save("model.keras")

# # Evaluate the model on the test data
# test_loss, test_acc = model.evaluate(test_dataset)
# print(f"Test accuracy: {test_acc:.4f}")


train_loss = history.history['loss']
train_accuracy = history.history['accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']

np.save('history/train_loss.npy', train_loss)
np.save('history/train_accuracy.npy', train_accuracy)
np.save('history/val_loss.npy', val_loss)
np.save('history/val_accuracy.npy', val_accuracy)

def get_predictions_and_labels(model, test):
    y_pred_probs = model.predict(test)
    print(f'y_pred_probs{ y_pred_probs}')
=======
history = model.fit(train_dataset, epochs=2, validation_data=test_dataset)

model.save("../../logs/MCMP.keras")


# train_loss = history.history['loss']
# train_accuracy = history.history['accuracy']
# val_loss = history.history['val_loss']
# val_accuracy = history.history['val_accuracy']

# np.save('history/train_loss.npy', train_loss)
# np.save('history/train_accuracy.npy', train_accuracy)
# np.save('history/val_loss.npy', val_loss)
# np.save('history/val_accuracy.npy', val_accuracy)

def get_predictions_and_labels(model, test):
    y_pred_probs = model.predict(test)
    # print(f'y_pred_probs{ y_pred_probs}')
>>>>>>> Stashed changes
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_test = np.concatenate([y for x, y in test], axis=0)
    print(f'y_test: {y_test} \n y_pred: {y_pred}')
    return y_pred, y_test

y_pred, y_test = get_predictions_and_labels(model, test_dataset)
# y_test = np.argmax(y_test, axis=1)

<<<<<<< Updated upstream

np.save('history/y_pred.npy', y_pred)
np.save('history/y_test.npy', y_test)

print("y_pred shape:", y_pred.shape)
print("y_test shape:", y_test.shape)
=======
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

# np.save('history/y_pred.npy', y_pred)
# np.save('history/y_test.npy', y_test)

# print("y_pred shape:", y_pred.shape)
# print("y_test shape:", y_test.shape)
>>>>>>> Stashed changes








