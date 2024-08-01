from scipy import io
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf #2.13
import keras
from tensorflow import keras
from keras import optimizers
import numpy as np



from tensorflow.keras import optimizers
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU,SimpleRNN
from tensorflow.keras.layers import TimeDistributed
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Activation
from tensorflow import reshape
from keras import utils
from tensorflow.keras.layers import Conv1D,Conv2D, MaxPooling1D,AveragePooling1D
from tensorflow.keras.layers import Input, LocallyConnected1D
from tensorflow.keras.layers import SeparableConv1D, Bidirectional
from tensorflow.keras.layers import LocallyConnected2D
from tensorflow.keras.layers import ZeroPadding2D,ZeroPadding1D, MaxPooling2D, Bidirectional
from tensorflow.keras.regularizers import l2,l1
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import CSVLogger,LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint




FilePath1 = "processed_NinaDB1_tests.csv"
train_emg = pd.read_csv(FilePath1)


emg = train_emg.iloc[:, :10].values
label_train=train_emg.iloc[:,10].values
label_t=label_train-1
label=np.array(label_t).reshape(len(label_t),1)


from feature_utils import *  
import math
imageData=[]
imageLabel=[]
imageLength=500
featureData=[]
featureLabel = []
classes = 52
timeWindow = 20
strideWindow = 20

for i in range(classes):
    index = []
    for j in range(label.shape[0]):
        if(label[j,:]==i):
            index.append(j)
            
    iemg = emg[index,:]
    length = math.floor((iemg.shape[0]-imageLength)/imageLength)
    print("class ",i," number of sample: ",iemg.shape[0],length)

           
    for j in range(length):
        subImage = iemg[imageLength*j:imageLength*(j+1),:]
        
        N=25
        for k in range(25):
            rms = featureRMS(subImage[strideWindow*k:strideWindow*k+timeWindow,:])
            mav = featureMAV(subImage[strideWindow*k:strideWindow*k+timeWindow,:])
            wl  = featureWL( subImage[strideWindow*k:strideWindow*k+timeWindow,:])
            zc  = featureZC( subImage[strideWindow*k:strideWindow*k+timeWindow,:])
            ssc = featureSSC(subImage[strideWindow*k:strideWindow*k+timeWindow,:])
        
            featureStack = np.hstack((rms,mav,wl,zc,ssc))
            featureData.append(featureStack)
            featureLabel.append(i)
                                 
        imageData.append(subImage)
        imageLabel.append(i)
        
imageData = np.array(imageData)



featureData = np.array(featureData)
f_Label=np.array(featureLabel)
my_array_2d = f_Label[:, np.newaxis]

classes = 52
for i in range(classes):
    index = []
    for j in range(my_array_2d.shape[0]):
        if(my_array_2d[j,:]==i):
            index.append(j)

    f =featureData[index,:]
    

reshaped_featuredata = featureData.reshape(745, 25, 50)
# reshaped_featuredata = featureData.reshape(10443, 25, 50)

reshaped_featurelabels =f_Label.reshape(745, 25)
# reshaped_featurelabels =f_Label.reshape(10443, 25)

num_unique_featurevalues = np.unique(reshaped_featurelabels[:, 0]).shape[0]
selected_featurelabels = reshaped_featurelabels[:, 5]
feature_labels= selected_featurelabels[:, np.newaxis]

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

# 随机打乱数据和标签
N = reshaped_featuredata.shape[0]
index = np.random.permutation(N)
featuredata = reshaped_featuredata[index,:,:]
data  = imageData[index,:,:]
label = np.array(imageLabel)[index]
featurelabel = np.array(feature_labels)[index]

# 对数据升维,标签one-hot
featuredata = np.expand_dims(featuredata , axis=2)
featurelabel= convert_to_one_hot(featurelabel,52).T

data  = np.expand_dims(data, axis=2)
label = convert_to_one_hot(label,52).T

# 划分数据集
N = featuredata.shape[0]
num_train = round(N*0.8)
X_train_feature = featuredata[0:num_train,:,:]
Y_train_feature = featurelabel[0:num_train,:]
X_test_feature  =featuredata[num_train:N,:,:]
Y_test_feature  = featurelabel[num_train:N,:]
X_train = data[0:num_train,:,:]
Y_train = label[0:num_train,:]
X_test  = data[num_train:N,:,:]
Y_test  = label[num_train:N,:]


n_steps, n_length = 25, 20
n_depth=10
x_train = X_train.reshape(X_train.shape[0], n_steps, n_length,n_depth)
print('x_train shape: ', x_train.shape)
x_test = X_test.reshape(X_test.shape[0], n_steps, n_length,n_depth)
print('x_test shape: ', x_test.shape)
n_outputs = Y_train.shape[1]
LABELS=[]
for i in range(1,n_outputs+1,1):
  LABELS.append (i)
    

n_steps, n_length = 25, 5
n_depth=10
X_train_feature = X_train_feature.reshape(X_train.shape[0], n_steps, n_length,n_depth)
X_test_feature = X_test_feature.reshape(X_test.shape[0], n_steps, n_length,n_depth)
n_outputs = Y_train.shape[1]
LABELS=[]
for i in range(1,n_outputs+1,1):
  LABELS.append (i)
    

Y= np.array(imageLabel)
reshaped_array = Y.reshape(-1, 1)

are_corresponding = np.array_equal(Y_train_feature, Y_train)


def step_decay(epoch):
  initial_lrate = 1e-4
  drop = 0.1
  epochs_drop =70.0
  lrate = initial_lrate * tf.math.pow(drop,  
          tf.math.floor((1+epoch)/epochs_drop))
  return lrate
lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
verbose, epochs, batch_size = 0, 2, 32

# # from tensorflow.keras import optimizers
# from tensorflow.keras.layers import LSTM
# from tensorflow.keras.layers import GRU,SimpleRNN
# from tensorflow.keras.layers import TimeDistributed
# from sklearn import metrics
# from sklearn.metrics import classification_report
# from sklearn import preprocessing
# from tensorflow.keras.models import Sequential,Model
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape
# from tensorflow.keras.models import Sequential,Model
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape
# from tensorflow.keras.layers import Conv2D, MaxPooling2D,Activation
# from tensorflow import reshape
# from keras import utils
# from tensorflow.keras.layers import Conv1D,Conv2D, MaxPooling1D,AveragePooling1D
# from tensorflow.keras.layers import Input, LocallyConnected1D
# from tensorflow.keras.layers import SeparableConv1D, Bidirectional
# from tensorflow.keras.layers import LocallyConnected2D
# from tensorflow.keras.layers import ZeroPadding2D,ZeroPadding1D, MaxPooling2D, Bidirectional
# from tensorflow.keras.regularizers import l2,l1
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.callbacks import CSVLogger,LearningRateScheduler
# from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint




#写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input
from keras.regularizers import l2
learning_rate = 0.001


from tensorflow.keras.layers import Concatenate
learning_rate = 0.001
# input_feature = Input_feature(shape = (25, 5, 10))
feature_input = Input(shape=(25, 5, 10), name="feature_input")
raw_input = Input(shape=(25, 20, 10), name="raw_input")

feature_model = Sequential()

feature_model.add(TimeDistributed(Conv1D(filters=256,kernel_size=3, padding='same', kernel_initializer="he_normal",strides=2,kernel_regularizer=l1(1e-04)), \
                           input_shape=(25, 5, 10)))
feature_model.add(TimeDistributed(BatchNormalization(epsilon=1e-06, momentum=0.95, weights=None)))
# feature_model.add(TimeDistributed(MaxPooling1D(pool_size=1,strides=2)))
feature_model.add(TimeDistributed(Activation('relu')))
# feature_model.add(TimeDistributed(LSTM(50,return_sequences=True)))
# Raw_model.add(TimeDistributed(Dropout(0.3)))
feature_model.add(TimeDistributed(Conv1D(filters=256,kernel_size=3,padding="same",kernel_initializer="he_normal",strides=2,kernel_regularizer=l1(1e-04))))
feature_model.add(TimeDistributed(BatchNormalization(epsilon=1e-06, momentum=0.95, weights=None)))
feature_model.add(TimeDistributed(Activation('relu')))
feature_model.add(TimeDistributed(Dropout(0.2093)))

feature_model.add(TimeDistributed(Dropout(0.7)))
feature_model.add(TimeDistributed(Flatten()))
feature_output = feature_model(feature_input)

Raw_model = Sequential()
Raw_model.add(TimeDistributed(Conv1D(filters=256, kernel_size=3, padding='same', kernel_initializer="he_normal",strides=2,kernel_regularizer=l1(1e-04)), \
                           input_shape=(25, 20, 10)))
Raw_model.add(TimeDistributed(BatchNormalization(epsilon=1e-06, momentum=0.95, weights=None)))
# Raw_model.add(TimeDistributed(MaxPooling1D(pool_size=8,strides=2)))
Raw_model.add(TimeDistributed(Activation('relu')))
# Raw_model.add(TimeDistributed(Dropout(0.5)))
Raw_model.add(TimeDistributed(LSTM(200,return_sequences=True)))
Raw_model.add(TimeDistributed(Conv1D(filters=256,kernel_size=3,padding="same",kernel_initializer="he_normal",strides=2,kernel_regularizer=l1(1e-04))))
Raw_model.add(TimeDistributed(BatchNormalization(epsilon=1e-06, momentum=0.95, weights=None)))
Raw_model.add(TimeDistributed(Activation('relu')))
Raw_model.add(TimeDistributed(Dropout(0.2093)))

Raw_model.add(TimeDistributed(Flatten()))
Raw_output = Raw_model(raw_input)
concatenated_output = Concatenate(axis=2)([feature_output, Raw_output])
combined_model= Sequential()
# model.add(Flatten())
X=Bidirectional(LSTM(200,return_sequences=True))(concatenated_output )
X=Dropout(0.3)(X)
X=Bidirectional(LSTM(200,return_sequences=True))(X)
X=Dropout(0.3)(X)

X=Flatten()(X)
X=Dense(512, activation='relu',kernel_regularizer=l2(0.01))(X)#修改参数成tanh
X=Dropout(0.3)(X)
X=BatchNormalization(epsilon=1e-05, momentum=0.9, weights=None)(X)
# X=Dense(256, activation='tanh')(X)
X=Dense(52, activation='softmax')(X)
# combined_model = combined_model(inputs=[feature_input ,raw_input ],outputs=X)
combined_model = Model(inputs=[feature_input ,raw_input ], outputs=X)
print(combined_model.summary())


adam=optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999, amsgrad=False)
checkpoint_filepath = 'E:/jupyter notebook/jupyter notebook/me/vlog.12.5'
# model.load_weights(checkpoint_filepath) 
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,verbose=1, monitor='val_accuracy',save_weights_only=True,save_best_only=True)
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='auto', baseline=None, restore_best_weights=True)


combined_model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

#写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

checkpoint_callback = ModelCheckpoint('model.h5', save_best_only=True)
early = EarlyStopping(monitor='val_loss', patience=5)
def update_lr(epoch, lr):
    return lr * 0.9 if epoch > 10 else lr
lrate = LearningRateScheduler(update_lr)


from keras.callbacks import CSVLogger


csv_logger = CSVLogger('/home/agn/studia/magisterka/Recognition-of-hand-gestures-from-the-electromyographic-signal/models/javier/vlog.10.5', append=True, separator=';')

history = combined_model.fit(
    {'feature_input': X_train_feature, 'raw_input': x_train},
    Y_train,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[csv_logger, checkpoint_callback, lrate, early],
    validation_data=({'feature_input': X_test_feature, 'raw_input': x_test}, Y_test),
    verbose=1
)

print(history.history.keys())

best_index = history.history['val_accuracy'].index(max(history.history['val_accuracy']))
print('epoch_number',best_index+1)


import datetime
print('train accuracy and validation accuracy', history.history['accuracy'][best_index], history.history['val_accuracy'][best_index])
# model.load_weights(checkpoint_filepath) 

# ADDED
train_loss = history.history['loss']
train_accuracy = history.history['accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']

np.save('history/train_loss.npy', train_loss)
np.save('history/train_accuracy.npy', train_accuracy)
np.save('history/val_loss.npy', val_loss)
np.save('history/val_accuracy.npy', val_accuracy)

# ADDED
def get_predictions_and_labels(model, test):
    y_pred_probs = model.predict(test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    # y_test = np.concatenate([y for x, y in test], axis=0)
    return y_pred

y_pred = get_predictions_and_labels(combined_model, {'feature_input': X_test_feature, 'raw_input': x_test})

np.save('history/y_pred.npy', y_pred)
np.save('history/y_test.npy', Y_test)

print("y_pred shape:", y_pred.shape)
print("y_test shape:", Y_test.shape)


# print(datetime.datetime.now())
# # _, testaccuracy = combined_model.evaluate(x_test,Y_test, batch_size=batch_size, verbose=1)
# print(datetime.datetime.now())
# # print('val_accuracy',val_accuracy)
# # test_acc.append(testaccuracy)

# combined_model.save('E:/jupyter notebook/10.5')

# for i in history.history:
#     print("history",i)
# # facecolor='black'
# DB1a=plt.figure(figsize=(10, 8),edgecolor='black')
# # ax=figDB4.add_axes(("left", "bottom", "width", "height"))
# ax= DB1a.add_axes((1,1,1,1))
# ax.spines["top"].set_position(("outward", 0))
# ax.spines["bottom"].set_position(("data", 0))
# ax.spines["left"].set_position(("axes", 0))
# ax.spines["right"].set_position(("outward", 0))

# ax.spines["top"].set_color("black")
# ax.spines["left"].set_color("black")
# ax.spines["right"].set_color("black")
# ax.spines["bottom"].set_color("black")

# plt.plot(history.history['accuracy'], 'r', label='Accuracy of training data')
# plt.plot(history.history['val_accuracy'], 'b', label='Accuracy of validation data')
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Training Epoch')

# # plt2.ylim(0)
# # plt2.rcParams['figure.facecolor'] = 'white'

# # ax = fig.add_axes((left, bottom, width, height))
# ax.set(facecolor="white")
# # ax.patch.set_alpha(1.0)
# plt.ylim(0)
# plt.legend()
# plt.show()



# figDB1=plt.figure(figsize=(10, 8),edgecolor='black')
# font = {'family' : 'Aril',
#         'weight' : 'bold',
#         'size'   : 28}

# plt.rc('font', **font)

# # SMALL_SIZE = 12
# # MEDIUM_SIZE = 14
# # BIGGER_SIZE = 18

# ax.spines["top"].set_linewidth(3)
# ax.spines["top"].set_linestyle("-")
# ax.spines["bottom"].set_linewidth(3)
# ax.spines["bottom"].set_linestyle("-")
# ax.spines["left"].set_linewidth(3)
# ax.spines["left"].set_linestyle("-")
# ax.spines["right"].set_linewidth(3)
# ax.spines["right"].set_linestyle("-")
# # ax.spines["bottom"].set_capstyle("round")

# # plt2.rcParams['figure.facecolor'] = 'white'
# plt.plot(history.history['loss'], 'r-', linewidth=3, label='Loss on training data')
# plt.plot(history.history['val_loss'], 'b-', linewidth=3, label='Loss on validation data')
# # plt.title('Loss curve for NinaPro DB1',fontsize=18, weight = 'bold')

# plt.ylabel('Loss',fontsize=30, weight = 'bold')
# plt.xlabel('no. of training epochs',fontsize=30, weight = 'bold')
# # plt2.ylim(0)
# # plt2.rcParams['figure.facecolor'] = 'white'

# # ax = fig.add_axes((left, bottom, width, height))
# ax.set(facecolor="white")
# # ax.patch.set_alpha(1.0)
# plt.ylim(0)
# plt.legend()
# plt.show()


# from sklearn.preprocessing import LabelBinarizer
# from sklearn.metrics import classification_report, confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np

# # Function to show confusion matrix
# def show_confusion_matrix(validations, predictions, labels):
#     matrix = confusion_matrix(validations, predictions)
#     plt.figure(figsize=(20, 16))
#     sns.set(font_scale=0.8)
#     # Adjust the figure size and layout
#     plt.subplots_adjust(left=0.2, bottom=0.25)

#     sns.heatmap(matrix,
#                 cmap='coolwarm',
#                 linecolor='white',
#                 linewidths=1,
#                 xticklabels=labels,
#                 yticklabels=labels,
#                 annot=True,
#                 fmt='d')

#     # Rotate axis labels for better readability
#     num_rows = len(labels)
#     xtick_positions = np.arange(num_rows) + 0.7
#     plt.xticks(xtick_positions,labels,rotation=0, ha='right')
#     num_rows = len(labels)
#     ytick_positions = np.arange(num_rows) + 0.5
#     plt.yticks(ytick_positions, labels, rotation=0)

#     # Adjust font size of x-axis and y-axis labels
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)

    

#     plt.title('Confusion Matrix', fontsize=20)
#     plt.ylabel('True Label', fontsize=20)
#     plt.xlabel('Predicted Label', fontsize=20)

#     # Adjust layout to prevent overlapping
#     plt.tight_layout()

#     plt.show()

# # Assuming Y_train is your true labels for the training set

# # Transform true labels to binary indicator format
# label_binarizer = LabelBinarizer()
# Y_train_binary = label_binarizer.fit_transform(Y_train)

# # Make predictions on the training set
# y_pred_train = combined_model.predict({'feature_input': X_train_feature, 'raw_input': x_train})
# max_y_pred_train = np.argmax(y_pred_train, axis=1)

# # Display confusion matrix for the training set
# show_confusion_matrix(np.argmax(Y_train_binary, axis=1), max_y_pred_train, labels=[' 0', '1', '2', '3', '4', '5', '6','7', '8','9','10','11','12','13','14','15','16','17','18','19','20','21','22',
#                                                                                   '23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47',
#                                                                                   '48','49','50','51',])

# # Print classification report for the training set
# print("Classification Report (Training Set):")
# print(classification_report(np.argmax(Y_train_binary, axis=1), max_y_pred_train))


# from sklearn.preprocessing import LabelBinarizer
# from sklearn.metrics import classification_report, confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np

# # Function to show confusion matrix
# def show_confusion_matrix(validations, predictions, labels):
#     matrix = confusion_matrix(validations, predictions)
#     plt.figure(figsize=(20, 16))
#     sns.set(font_scale=0.8)
#     # Adjust the figure size and layout
#     plt.subplots_adjust(left=0.2, bottom=0.25)

#     sns.heatmap(matrix,
#                 cmap='coolwarm',
#                 linecolor='white',
#                 linewidths=1,
#                 xticklabels=labels,
#                 yticklabels=labels,
#                 annot=True,
#                 fmt='d')

#     # Rotate axis labels for better readability
#     num_rows = len(labels)
#     xtick_positions = np.arange(num_rows) + 0.7
#     plt.xticks(xtick_positions,labels,rotation=0, ha='right')
#     num_rows = len(labels)
#     ytick_positions = np.arange(num_rows) + 0.5
#     plt.yticks(ytick_positions, labels, rotation=0)

#     # Adjust font size of x-axis and y-axis labels
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)

    

#     plt.title('Confusion Matrix', fontsize=20)
#     plt.ylabel('True Label', fontsize=20)
#     plt.xlabel('Predicted Label', fontsize=20)

#     # Adjust layout to prevent overlapping
#     plt.tight_layout()

#     plt.show()

# # Assuming Y_train is your true labels for the training set

# # Transform true labels to binary indicator format
# label_binarizer = LabelBinarizer()
# Y_test_binary = label_binarizer.fit_transform(Y_test)

# # Make predictions on the training set
# y_pred_test = combined_model.predict({'feature_input': X_test_feature, 'raw_input': x_test})
# max_y_pred_test = np.argmax(y_pred_test, axis=1)

# # Display confusion matrix for the training set
# show_confusion_matrix(np.argmax(Y_test_binary, axis=1), max_y_pred_test, labels=[' 0', '1', '2', '3', '4', '5', '6','7', '8','9','10','11','12','13','14','15','16','17','18','19','20','21','22',
#                                                                                   '23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47',
#                                                                                   '48','49','50','51',])

# # Print classification report for the training set
# print("Classification Report (Vailding Set):")
# print(classification_report(np.argmax(Y_test_binary, axis=1), max_y_pred_test))
