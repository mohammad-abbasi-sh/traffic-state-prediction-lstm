# -*- coding: utf-8 -*-
"""
Traffic state prediction - Type 1 & Type 2 framing (next-state regression).

Originally written/run in Google Colab. The drive.mount() call below is only
needed if you're running this in Colab with data stored on Google Drive -
otherwise just point DIR_PATH at a local folder and skip it.

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

# plot
import matplotlib.pyplot as plt

# configs
DATASET = 3
DIR_PATH = os.path.join('data', 'DataSet-' + str(DATASET))
TYPE = 1    # problems type
TEST_SIZE = 0.2
VAL_SIZE = 0.1  # validation size
LOOK_BACK = 10  # window size
BATCH_SIZE = 32
EPOCH = 300

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



if TYPE == 1 :
    # this is for type 1 : state(0) to state(n) --> state(n+1)
    data_value = dataset.iloc[:, 0:8].values

if TYPE == 2:
    data_value_0_7 = dataset.iloc[:, 0:8].values
    # data_value_10_14 = dataset.iloc[:, 9:].values

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
    data_value = np.concatenate((data_value_0_7, encode_data_value), axis=1)
    '''
    lst = [0] * len(unique_target)
    encode_target = []
    for index, row in target.iterrows():
        i = dic.get((row[9], row[10], row[11], row[12]))
        lst[i] = 1
        encode_target.append(lst)
        lst[i] = 0
    '''
if TYPE == 3:
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
    data_value = np.concatenate((data_value_0_8, encode_data_value), axis=1)



# normalize the dataset trainX,TrainY, testX,testY
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data_value)

                             

# split into train and test sets
train, test = train_test_split(data, test_size=TEST_SIZE, shuffle=False)



# convert an array of values into a dataset matrix
def create_dataset_type1(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    return np.array(dataX), np.array(dataY)


def create_dataset_type2(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :8])
    return np.array(dataX), np.array(dataY)


def create_dataset_type3(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = np.concatenate((dataset[i:(i + look_back), :8], dataset[i:(i + look_back), 9:]), axis=1)
        dataX.append(a)
        dataY.append(dataset[i + look_back, 8:])
    return np.array(dataX), np.array(dataY)


# reshape into X=t and Y=t+1
if TYPE == 1:
    trainX, trainY = create_dataset_type1(train, LOOK_BACK)
    testX, testY = create_dataset_type1(test, LOOK_BACK)

elif TYPE == 2:
    trainX, trainY = create_dataset_type2(train, LOOK_BACK)
    testX, testY = create_dataset_type2(test, LOOK_BACK)

elif TYPE == 3:
    trainX, trainY = create_dataset_type3(train, LOOK_BACK)
    testX, testY = create_dataset_type3(test, LOOK_BACK)



# reshape input to be [samples, time steps, features]
num_features = trainX.shape[2]
# time_steps = 1  # same look_back
trainX = np.reshape(trainX, (trainX.shape[0], LOOK_BACK, trainX.shape[2]))
testX = np.reshape(testX, (testX.shape[0], LOOK_BACK, testX.shape[2]))



print("x_train", trainX.shape)
print("y_train", trainY.shape)
print()
print("x_test", testX.shape)
print("y_test", testY.shape)

# create Model

model_output = trainY.shape[1]
model = Sequential()
model.add(LSTM(6, input_shape=(LOOK_BACK, num_features), return_sequences=True, activation='sigmoid'))
model.add(LSTM(6, activation='sigmoid'))
# model.add(Flatten())
model.add(Dense(model_output))

# adam = optimizers.Adam(lr=0.01, decay=1e-6)


model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.summary()

print("Model Input",model.input_shape,"Model Output",model.output_shape)

# callback for reduce learning rate and early stopping
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001)

# callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
#               EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]

callbacks = [es, mc]


history = model.fit(trainX, trainY,
                    epochs=EPOCH,
                    batch_size=BATCH_SIZE,
                    verbose=1,
                    validation_split=VAL_SIZE,
                    shuffle=False,
                    callbacks=callbacks)


title =' type_'+str(TYPE)+' ds'+str(DATASET)+' window = '+str(LOOK_BACK)


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
 
epochs = range(len(acc))
 
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy'+title)
plt.legend()
 
plt.figure()
 
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss'+title)
plt.legend()
 
plt.show()

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
print("item = 0")
print(testY[0])
print(testPredict[0])

count = 5
items = []
size_test = len(testPredict)
for i in range(count):
  items.append(randint(0, size_test))

for i in items:
  print("item =",i)
  print(testY[i])
  print(testPredict[i])

print("rmse with scalling")
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

if TYPE == 1:
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY)
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testY)
    
elif TYPE == 2:
    z = data_value[:trainY.shape[0],8:]     # last column
    trainY = np.concatenate((trainY,z),axis=1)
    trainY = scaler.inverse_transform(trainY)[:,:8]
    
    z = data_value[:trainPredict.shape[0],8:]    # last column
    trainPredict = np.concatenate((trainPredict,z),axis=1)
    trainPredict = scaler.inverse_transform(trainPredict)[:,:8]
    
    z = data_value[:testPredict.shape[0],8:]    # last column
    testPredict = np.concatenate((testPredict,z),axis=1)
    testPredict = scaler.inverse_transform(testPredict)[:,:8]
    
    z = data_value[:testY.shape[0],8:] # last column
    testY = np.concatenate((testY,z),axis=1)
    testY = scaler.inverse_transform(testY)[:,:8]
    
elif TYPE == 3:
    z = data_value[:trainY.shape[0],:8]
    trainY = np.concatenate((z,trainY),axis=1)
    trainY = scaler.inverse_transform(trainY)[:,8:]
    
    z = data_value[:trainPredict.shape[0],:8]
    trainPredict = np.concatenate((z,trainPredict),axis=1)
    trainPredict = scaler.inverse_transform(trainPredict)[:,8:]
    
    z = data_value[:testPredict.shape[0],:8]
    testPredict = np.concatenate((z,testPredict),axis=1)
    testPredict = scaler.inverse_transform(testPredict)[:,8:]
    
    z = data_value[:testY.shape[0],:8]
    testY = np.concatenate((z,testY),axis=1)
    testY = scaler.inverse_transform(testY)[:,8:]

print("rmse without scalling")
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

for i in items:
  print("item =",i)
  print(testY[i])
  print(testPredict[i])

print("item = 0")
print(testY[0])
print(testPredict[0])

