from os import system, listdir
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Lambda,
    Permute,
    Multiply,
)

from activations import Mish
from optimizers import Ranger
import losses as l
import callbacks as cb
from layers import Attention, LayerNormalization
from data import dataset, nina1_dataset, dataset_without_imu
from generator import generator

imu = False
if sys.argv and sys.argv[-1]=='imu':
    imu = True

# read in the data
data = dataset_without_imu("../../data")
# data = dataset("./ninaPro")

reps = np.unique(data.repetition)
print(reps)
val_reps = reps[3::2]
train_reps = reps[np.where(np.isin(reps, val_reps, invert=True))]
test_reps = val_reps[-1].copy()
val_reps = val_reps[:-1]

train = generator(data, list(train_reps), imu = imu)
validation = generator(data, list(val_reps), augment=False, imu = imu)
test = generator(data, [test_reps][0], augment=False, imu = imu)


# model parameters
timesteps = train[0][0].shape[1]
n_class = 53
n_features = train[0][0].shape[-1] # 16 channels

model_pars = {
    "timesteps": timesteps,
    "n_class": n_class,
    "n_features": n_features,
    "classifier_architecture": [500, 500, 2000],
    "dropout": [0.36, 0.36, 0.36],
}


# attention mechanism
def attention_simple(inputs, timesteps):
    input_dim = int(inputs.shape[-1])
    a = Permute((2, 1), name='transpose')(inputs)
    a = Dense(timesteps, activation='softmax',  name='attention_probs')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply(name='focused_attention')([inputs, a_probs])
    output_flat = Lambda(lambda x: K.sum(x, axis=1), name='temporal_average')(output_attention_mul)
    return output_flat, a_probs


def dense_model(timesteps, n_class, n_features, classifier_architecture, dropout):
    inputs = Input((timesteps, n_features))
    x = Dense(128, activation=Mish())(inputs)
    x = LayerNormalization()(x)
    x, a = attention_simple(x, timesteps)
    for d, dr in zip(classifier_architecture, dropout):
        x = Dropout(dr)(x)
        x = Dense(d, activation=Mish())(x)
        x = LayerNormalization()(x)
    outputs = Dense(n_class, activation="softmax")(x)
    model = Model(inputs, outputs)
    return model



model = dense_model(**model_pars)

cosine = cb.CosineAnnealingScheduler(
    T_max=50, eta_max=1e-3, eta_min=1e-5, verbose=1, epoch_start=5
)
loss = l.focal_loss(gamma=3., alpha=6.)
model.compile(Ranger(learning_rate=1e-3), loss=loss, metrics=["accuracy"])

print(model.summary())

history = model.fit(
    train,
    epochs=2,
    validation_data=validation,
    callbacks=[
        ModelCheckpoint(
            "main.h5",
            monitor="val_loss",
            keep_best_only=True,
            save_weights_only=False,
        ),
        cosine,
    ],
    shuffle = False,
)

def get_predictions_and_labels(model, test):
    y_pred_probs = model.predict(test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_test = np.concatenate([y for x, y in test], axis=0)
    return y_pred, y_test

y_pred, y_test = get_predictions_and_labels(model, test)
y_test = np.argmax(y_test, axis=1)

history_data = {
    'model_name': 'Simple model of attention',
    'train_loss': history.history['loss'],
    'train_accuracy': history.history['accuracy'],
    'val_loss': history.history['val_loss'],
    'val_accuracy': history.history['val_accuracy'],
    'y_pred': y_pred.tolist(),
    'y_test': y_test.tolist()
}

with open('../../logs/history_attention.json', 'w') as f:
    json.dump(history_data, f)

model.save('../../logs/attention.h5')



