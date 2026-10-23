# -*- coding: utf-8 -*-
"""
Traffic state prediction - Type 3 framing, treated as classification.

Predicts the next signal action (out of the encoded action space) from a
window of past state+action pairs, instead of regressing the raw next state.

Originally written/run in Google Colab - drive.mount() is only needed there.

Author: Mohammad Abbasishahkoo
Course project - Artificial Intelligence, Spring 2021
"""

# from google.colab import drive
# drive.mount('/gdrive')

import numpy as np
import pandas as pd
import os
import math
from random import randint

# keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM , Flatten
from keras.callbacks import ReduceLROnPlateau, EarlyStopping , ModelCheckpoint
from keras.models import load_model
from keras import optimizers

# sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

# plot
import matplotlib.pyplot as plt

# configs
DATASET = 1
DIR_PATH = os.path.join('data', 'DataSet-' + str(DATASET))
TYPE = 3    # problems type is fixed in this file
TEST_SIZE = 0.2
VAL_SIZE = 0.1  # validation size
LOOK_BACK = 10  # window size
BATCH_SIZE = 32
EPOCH = 50

# export files 
model_name = 'model_t'+str(TYPE)+'_ds'+str(DATASET)+'.h5'




def read_dataset(dir_path):
    files = os.listdir(dir_path)
    print(files)
    flag = 0
    for fn in files:
        if flag == 0:
            dataset = pd.read_csv(os.path.join(dir_path, fn), header=None)
            flag = 1
        else:
            df2 = pd.read_csv(os.path.join(dir_path, fn), header=None)
            dataset = pd.concat([dataset, df2], ignore_index=True, sort=False)
    return dataset


dataset = read_dataset(DIR_PATH)



data_value_0_8 = dataset.iloc[:, 0:9].values

target = dataset.iloc[:, 9:]
unique_target = dataset.iloc[:, 9:].drop_duplicates()
unique_target = unique_target.reset_index(drop=True)

dic = {}
for index, row in unique_target.iterrows():
    dic[(row[9], row[10], row[11], row[12])] = index
encode_data_value = []
for index, row in target.iterrows():
    i = dic.get((row[9], row[10], row[11], row[12]))
    encode_data_value.append([i])
data_value = np.concatenate((data_value_0_8, np.array(encode_data_value)), axis=1)

Xd = data_value[:,0:9]
Yd = data_value[:,9:]

Yd = np.reshape(Yd,(Yd.shape[0]))

# normalize the dataset trainX,TrainY, testX,testY
scaler = MinMaxScaler(feature_range=(0, 1))
Xd = scaler.fit_transform(Xd)

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y1 = encoder.fit_transform(Yd)

target = pd.get_dummies(y1).values

data_target = np.concatenate((Xd, target), axis=1)

print(data_target.shape)
print(target.shape)

# split into train and test sets
train, test = train_test_split(data_target, test_size=TEST_SIZE, shuffle=False)

def create_dataset_type3(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :9]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 9:])
    return np.array(dataX), np.array(dataY)

if TYPE == 3:
    trainX, trainY = create_dataset_type3(train, LOOK_BACK)
    testX, testY = create_dataset_type3(test, LOOK_BACK)

print("x_train", trainX.shape)
print("y_train", trainY.shape)
print()
print("x_test", testX.shape)
print("y_test", testY.shape)

from keras.optimizers import SGD,Adam
# create Model
num_features = trainX.shape[2]
model_output = trainY.shape[1]
model = Sequential()
model.add(LSTM(6, input_shape=(LOOK_BACK, num_features), return_sequences=True, activation='tanh'))
model.add(LSTM(6, activation='tanh'))
model.add(Dense(10,activation='tanh'))
model.add(Dense(10,activation='tanh'))
model.add(Dense(8,activation='tanh'))

model.add(Dense(model_output,activation='softmax'))

# adam = optimizers.Adam(lr=0.01, decay=1e-6)


model.compile(Adam(lr=0.04),'categorical_crossentropy',metrics=['accuracy'])
model.summary()

print("Model Input",model.input_shape,"Model Output",model.output_shape)

# callback for reduce learning rate and early stopping
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001)

# callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
#               EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]

callback = [es, mc]


history = model.fit(trainX, trainY,
                    epochs=10,
                    batch_size=BATCH_SIZE,
                    verbose=1,
                    validation_split=VAL_SIZE,
                    shuffle=False)


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

y_train_class = np.argmax(trainY,axis=1)
y_train_pred_class = np.argmax(trainPredict,axis=1)

print(classification_report(y_train_class,y_train_pred_class))

y_test_class = np.argmax(testY,axis=1)
y_pred_class = np.argmax(testPredict,axis=1)

print(classification_report(y_test_class,y_pred_class))

accuracy_score(y_test_class, y_pred_class)

