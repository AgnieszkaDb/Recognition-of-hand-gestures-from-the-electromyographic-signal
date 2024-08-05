
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

from generator import Data

import json
import pandas as pd
import numpy as np


data_processor = Data()
data_processor.process_files()
data_processor.save_to_csv('processed_NinaDB1.csv')

data_path = 'processed_NinaDB1.csv'  
data = pd.read_csv(data_path)

X = data.drop('class', axis=1).values  # Features
y = data['class'].values  # Labels
y = y - 1 # Relabel from 1-52 to 0-51

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test_s = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

y_train = to_categorical(y_train, num_classes=52)
y_val = to_categorical(y_val, num_classes=52)
y_test = to_categorical(y_test_s, num_classes=52)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(52, activation='softmax')  
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=2, batch_size=32, validation_data=(X_val, y_val))
# history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

def get_predictions_and_labels(model, test):
    y_pred_probs = model.predict(test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    return y_pred, y_test

y_pred, _ = get_predictions_and_labels(model, X_test)

history_data = {
    'model_name': 'Baseline',
    'train_loss': history.history['loss'],
    'train_accuracy': history.history['accuracy'],
    'val_loss': history.history['val_loss'],
    'val_accuracy': history.history['val_accuracy'],
    'y_pred': y_pred.tolist(),
    'y_test': np.argmax(y_test, axis=1).tolist()
}

with open('../../../logs/history_baseline.json', 'w') as f:
    json.dump(history_data, f)

model.save('../../../logs/baseline_model.keras')
