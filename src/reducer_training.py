from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, LSTM, RepeatVector
from keras.models import Model
from keras import regularizers
import time
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import tensorflow
import utils

# path to the dataset
dataset = '/content/drive/My Drive/Colab Notebooks/Project3/nasdaq2007_17.csv'
# the directory to which the model will be saved
dir = 'models/'
model_name = 'new_compressor.h5'
test_samples = 2000


def plot_examples(stock_input, stock_decoded):
    n = 10
    plt.figure(figsize=(20, 4))
    for i, idx in enumerate(list(np.arange(0, test_samples, 200))):
        # display original
        ax = plt.subplot(2, n, i + 1)
        if i == 0:
            ax.set_ylabel("Input", fontweight=600)
        else:
            ax.get_yaxis().set_visible(False)
        plt.plot(stock_input[idx])
        ax.get_xaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        if i == 0:
            ax.set_ylabel("Output", fontweight=600)
        else:
            ax.get_yaxis().set_visible(False)
        plt.plot(stock_decoded[idx])
        ax.get_xaxis().set_visible(False)


df = pd.read_csv(dataset, delimiter='\t', header=None)

df.rename(columns={0: 'id'}, inplace=True)  # rename the first column to "id"

seed = 12345
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
tensorflow.random.set_seed(seed)
np.random.seed(seed)

# DATASET EDITING
df.drop("id", axis=1, inplace=True)
df.sample(frac=1, random_state=1).reset_index(drop=True)

split_index = int(df.shape[0]*0.9)

train = df.head(split_index)
test = df.tail(df.shape[0]-split_index).reset_index(drop=True)

train = train.transpose()
test = test.transpose()


# SCALING
scaler = MinMaxScaler(feature_range=(0, 1))

for i in range(train.shape[1]):
    curr_ts = train[i].to_numpy()
    curr_ts = curr_ts.reshape(-1, 1)
    train[i] = scaler.fit_transform(curr_ts)

for i in range(test.shape[1]):
    curr_ts = test[i].to_numpy()
    curr_ts = curr_ts.reshape(-1, 1)
    test[i] = scaler.transform(curr_ts)

# SPLIT TO X AND y
WINDOW = 50
LAG = 10

# reshape to (samples, window, n_features)

X_train, y_train = utils.create_dataset(train, WINDOW, LAG)
X_test, y_test = utils.create_dataset(test, WINDOW)

X_train = X_train[:, :, np.newaxis]
X_test = X_test[:, :, np.newaxis]

input_window = Input(shape=(WINDOW, 1))
x = Conv1D(16, 3, activation="relu", padding="same")(input_window)  # 10 dims
#x = BatchNormalization()(x)
x = MaxPooling1D(2, padding="same")(x)  # 5 dims
x = Conv1D(1, 3, activation="relu", padding="same")(x)  # 5 dims
#x = BatchNormalization()(x)
encoded = MaxPooling1D(2, padding="same")(x)  # 3 dims

encoder = Model(input_window, encoded)

# 3 dimensions in the encoded layer

x = Conv1D(1, 3, activation="relu", padding="same")(encoded)  # 3 dims
#x = BatchNormalization()(x)
x = UpSampling1D(2)(x)  # 6 dims
x = Conv1D(16, 2, activation='relu')(x)  # 5 dims
#x = BatchNormalization()(x)
x = UpSampling1D(2)(x)  # 10 dims
decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x)  # 10 dims
autoencoder = Model(input_window, decoded)
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
history = autoencoder.fit(X_train, X_train,
                          epochs=10,
                          batch_size=32,
                          shuffle=True,
                          validation_data=(X_test, X_test))

decoded_stocks = autoencoder.predict(X_test)

full_path = dir + model_name
model.save(full_path)

x_test_deep = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
plot_examples(x_test_deep, decoded_stocks)
