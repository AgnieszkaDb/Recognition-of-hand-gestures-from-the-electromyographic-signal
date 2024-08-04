import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

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

train_loss = history.history['loss']
train_accuracy = history.history['accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']

# Save the metrics to files
np.save('../history/train_loss.npy', train_loss)
np.save('../history/train_accuracy.npy', train_accuracy)
np.save('../history/val_loss.npy', val_loss)
np.save('../history/val_accuracy.npy', val_accuracy)

model.evaluate(X_val, y_val)
model.evaluate(X_test, y_test)

def get_predictions_and_labels(model, test):
    y_pred_probs = model.predict(test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    return y_pred, y_test

y_pred, _ = get_predictions_and_labels(model, X_test)

np.save('../history/y_pred.npy', y_pred)
np.save('../history/y_test.npy', y_test_s)

print("y_pred shape:", y_pred.shape)
print("y_test shape:", y_test_s.shape)



# test_loss, test_acc = model.evaluate(X_test, y_test)
# print(f'Test accuracy: {test_acc:.3f}')

model.save('baseline_model.h5')
