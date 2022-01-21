# reduce.py
import sys
import os

import ui.reduce_ui

reduce = ui.reduce_ui.Ui()

print(reduce.dataset_path)
print(reduce.queryset_path)
print(reduce.output_dataset_file)
print(reduce.output_query_file)

dataset = reduce.dataset_path # path to the dataset
queryset = reduce.queryset_path # path to the queryset
output_dataset_file = reduce.output_dataset_file # path to the output_dataset_file
output_queryset_file = reduce.output_query_file # path to the output_queryset_file

dir = 'models/' # the directory to which the model will be save or from which it will be loaded
model_name = 'compressor.h5'

from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, LSTM, RepeatVector
from keras.models import Model
from keras.models import model_from_json
from keras import regularizers
import keras
import datetime
import time
import requests as req
import json
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# %pylab inline

df = pd.read_csv(dataset, delimiter='\t', header=None)
df2 = pd.read_csv(queryset, delimiter='\t', header=None)

df.rename(columns = {0:'id'}, inplace=True) #rename the first column to "id"
df2.rename(columns = {0:'id'}, inplace=True) #rename the first column to "id"

import random
import tensorflow

def reproducibleResults(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tensorflow.random.set_seed(seed)
    np.random.seed(seed)

reproducibleResults(12345)

df.drop("id", axis=1, inplace=True)
df.sample(frac=1, random_state=1).reset_index(drop=True)

test = df

df2.drop("id", axis=1, inplace=True)
df2.sample(frac=1, random_state=1).reset_index(drop=True)

test2 = df2

test = test.transpose()
test2 = test2.transpose()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

for i in range(test.shape[1]):
    curr_ts = test[i].to_numpy()
    curr_ts = curr_ts.reshape(-1,1)
    test[i] = scaler.fit_transform(curr_ts)

for i in range(test2.shape[1]):
    curr_ts = test2[i].to_numpy()
    curr_ts = curr_ts.reshape(-1,1)
    test2[i] = scaler.fit_transform(curr_ts)

def create_dataset(X, time_steps=1, lag=1):
    Xs, ys = [], []
    for j in range(len(X.columns)):
      for i in range(0, len(X) - time_steps, lag):
          v = X[j].iloc[i:(i + time_steps)].values
          Xs.append(v)
          ys.append(X[j].iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

WINDOW = 50
LAG = 10

# reshape to (samples, window, n_features)

X_test, y_test = create_dataset(test, WINDOW, LAG)

X_test = X_test[:, :, np.newaxis]

X_test2, y_test2 = create_dataset(test2, WINDOW, LAG)

X_test2 = X_test2[:, :, np.newaxis]

full_path = dir + model_name
autoencoder = keras.models.load_model(full_path)

decoded_stocks = autoencoder.predict(X_test)
decoded_stocks2 = autoencoder.predict(X_test2)

print(decoded_stocks.shape)

subseries_per_series = int(decoded_stocks.shape[0] / test.shape[1])
print(subseries_per_series)

outcome = []

for i in range(test.shape[1]):
    index = i*subseries_per_series
    curr_ts = decoded_stocks[index]
    for j in range(1, subseries_per_series):
        curr_ts = np.concatenate([curr_ts, decoded_stocks[index+j]])
    # print(len(curr_ts))
    outcome.append(curr_ts)
print(len(outcome))