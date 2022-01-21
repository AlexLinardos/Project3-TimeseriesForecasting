# reduce.py
import tensorflow
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
import pandas as pd
import json
import requests as req
import time
import datetime
import keras
from keras import regularizers
from keras.models import model_from_json
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, LSTM, RepeatVector
import sys
import os

import ui.reduce_ui
import utils

reduce = ui.reduce_ui.Ui()

print(reduce.dataset_path)
print(reduce.queryset_path)
print(reduce.output_dataset_file)
print(reduce.output_query_file)

dataset = reduce.dataset_path  # path to the dataset
queryset = reduce.queryset_path  # path to the queryset
# path to the output_dataset_file
output_dataset_file = reduce.output_dataset_file
# path to the output_queryset_file
output_queryset_file = reduce.output_query_file

# the directory to which the model will be saved or from which it will be loaded
dir = 'models/'
model_name = 'compressor.h5'

# DATASET LOADING AND EDITING
df = pd.read_csv(dataset, delimiter='\t', header=None)
df2 = pd.read_csv(queryset, delimiter='\t', header=None)

df.rename(columns={0: 'id'}, inplace=True)  # rename the first column to "id"
df2.rename(columns={0: 'id'}, inplace=True)  # rename the first column to "id"

ids = df['id'].values
ids2 = df2['id'].values

seed = 12345
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
tensorflow.random.set_seed(seed)
np.random.seed(seed)


df.drop("id", axis=1, inplace=True)

test = df

df2.drop("id", axis=1, inplace=True)

test2 = df2

test = test.transpose()
test2 = test2.transpose()

# SCALING
scaler = MinMaxScaler(feature_range=(0, 1))

for i in range(test.shape[1]):
    curr_ts = test[i].to_numpy()
    curr_ts = curr_ts.reshape(-1, 1)
    test[i] = scaler.fit_transform(curr_ts)

for i in range(test2.shape[1]):
    curr_ts = test2[i].to_numpy()
    curr_ts = curr_ts.reshape(-1, 1)
    test2[i] = scaler.fit_transform(curr_ts)


# MAKING PREDICTIONS (aka reducing dimensionality of data)
WINDOW = 50
LAG = 10

# reshape to (samples, window, n_features)

X_test, y_test = utils.create_dataset(test, WINDOW, LAG)

X_test = X_test[:, :, np.newaxis]

X_test2, y_test2 = utils.create_dataset(test2, WINDOW, LAG)

X_test2 = X_test2[:, :, np.newaxis]

full_path = dir + model_name
autoencoder = keras.models.load_model(full_path)

decoded_stocks = autoencoder.predict(X_test)
decoded_stocks2 = autoencoder.predict(X_test2)


subseries_per_series = int(decoded_stocks.shape[0] / test.shape[1])


concatenated_ts = []
for i in range(test.shape[1]):
    index = i*subseries_per_series
    curr_ts = decoded_stocks[index]
    for j in range(1, subseries_per_series):
        curr_ts = np.concatenate([curr_ts, decoded_stocks[index+j]])
    concatenated_ts.append(curr_ts)

subseries_per_series2 = int(decoded_stocks2.shape[0] / test2.shape[1])
print(subseries_per_series2)

concatenated_ts2 = []
for i in range(test2.shape[1]):
    index = i*subseries_per_series2
    curr_ts = decoded_stocks2[index]
    for j in range(1, subseries_per_series2):
        curr_ts = np.concatenate([curr_ts, decoded_stocks2[index+j]])
    concatenated_ts2.append(curr_ts)


ids_df = pd.DataFrame(ids)
concatenated_ts_df = pd.DataFrame(np.array(concatenated_ts).squeeze())
output_df = pd.concat([ids_df, concatenated_ts_df], axis=1)
output_df = output_df.transpose()
output_df.reset_index(drop=True, inplace=True)
output_df = output_df.transpose()
output_csv = output_df.to_csv(
    output_dataset_file, index=False, header=False, sep='\t', line_terminator='\n')

ids_df2 = pd.DataFrame(ids2)
concatenated_ts_df2 = pd.DataFrame(np.array(concatenated_ts2).squeeze())
output_df2 = pd.concat([ids_df2, concatenated_ts_df2], axis=1)
output_df2 = output_df2.transpose()
output_df2.reset_index(drop=True, inplace=True)
output_df2 = output_df2.transpose()
output_csv2 = output_df2.to_csv(
    output_queryset_file, index=False, header=False, sep='\t', line_terminator='\n')
