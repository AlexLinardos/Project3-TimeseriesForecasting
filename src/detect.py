# detect.py
import ui.detect_ui
import utils
import pandas as pd
import numpy as np
import os
import random
import tensorflow
import keras
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

detect = ui.detect_ui.Ui()

seed = 12345
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
tensorflow.random.set_seed(seed)
np.random.seed(seed)

# DATASET LOADING AND EDITING
df = pd.read_csv(detect.dataset_path, delimiter='\t', header=None)
df.rename(columns = {0:'id'}, inplace=True) #rename the first column to "id"

df.sample(frac=1, random_state=1).reset_index(drop=True) # shuffle the dataframe

chosen_series = df.sample(n=3, random_state=1)
print(chosen_series)

# get only the numeric values
column_names = chosen_series.columns.values.tolist()
values = chosen_series.iloc[:, column_names[1:]].values

# MODEL CREATION AND TRAINING

# dataset preparation and split
df.drop("id", axis=1, inplace=True)
df.sample(frac=1, random_state=1).reset_index(drop=True)

split_index = int(df.shape[0]*0.9)

train = df.head(split_index)
test = df.tail(df.shape[0]-split_index).reset_index(drop=True)

train = train.transpose() # quality of life change
test = test.transpose() # quality of life change

scaler = MinMaxScaler(feature_range=(0, 1))

for i in range(train.shape[1]):
  curr_ts = train[i].to_numpy()
  curr_ts = curr_ts.reshape(-1,1)
  train[i] = scaler.fit_transform(curr_ts)

for i in range(test.shape[1]):
  curr_ts = test[i].to_numpy()
  curr_ts = curr_ts.reshape(-1,1)
  test[i] = scaler.transform(curr_ts)


WINDOW = 50
LAG = 10

X_train, y_train = utils.create_dataset(train, WINDOW, LAG)
X_test, y_test = utils.create_dataset(test, WINDOW)

X_train = X_train[:, :, np.newaxis]
X_test = X_test[:, :, np.newaxis]

if detect.retrain == True:
  model = keras.models.Sequential([
  keras.layers.LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])),
  keras.layers.Dropout(0.2),
  keras.layers.RepeatVector(n = X_train.shape[1]),
  keras.layers.LSTM(100, return_sequences=True),
  keras.layers.Dropout(0.2),
  keras.layers.TimeDistributed(keras.layers.Dense(units = X_train.shape[2]))
  ])

  model.compile(optimizer = 'adam', loss = 'mae')

  history = model.fit(X_train, y_train, epochs = 10, batch_size=2048, validation_split=0.1)

model_name = 'autoencoder2.h5' # file name of saved model (file must be of type .h5)

if detect.retrain == True:
    full_path = './models/' + model_name
    model.save(full_path)

if detect.retrain == False:
    full_path = './models/' + model_name
    model = keras.models.load_model(full_path)
    model.summary()


# MAKING PREDICTIONS AND FINDING ANOMALIES

X_train_pred = model.predict(X_train)
train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

X_test_pred = model.predict(X_test)
test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

num_of_series = test.shape[1]

# choose n series randomly to plot
if num_of_series < detect.ts_number:
    print("Test set does not have ", detect.ts_number," time series. Defaulting to 3...")
    detect.ts_number = 3

plot_numbers = []
for i in range(detect.ts_number):
    plot_numbers.append(random.randint(0, num_of_series-1))

# generate the data to be plotted
num_values = (test.shape[0]-WINDOW)
series_preds = []
test_score_dfs = []
anomalies_dfs = []
for index in plot_numbers:
    series_preds.append(X_train_pred[index*num_values : index*num_values+num_values])
    score_df = pd.DataFrame()
    score_df['loss'] = test_mae_loss[index*num_values:index*num_values+num_values].squeeze()
    score_df['threshold'] = detect.mae
    score_df['anomaly'] = score_df.loss > score_df.threshold
    score_df['price'] = y_test[index*num_values:index*num_values+num_values]
    test_score_dfs.append(score_df)
    anomalies_dfs.append(score_df[score_df.anomaly==True])

# plot
figure, axis = plt.subplots(detect.ts_number, 1, figsize=(14,5))
for i in range(detect.ts_number):
    to_plot = scaler.inverse_transform(np.array(test_score_dfs[i]['price'])[:, np.newaxis])
    scores_np = test_score_dfs[i]['price'].values
    scores_np = scores_np.reshape(-1, 1)
    axis[i].plot(
    scaler.inverse_transform(scores_np),
    label='Time series'
    )
    anomalies_np = anomalies_dfs[i]['price'].values
    anomalies_np = anomalies_np.reshape(-1, 1)
    if anomalies_np.shape[0]>0:
        sns.scatterplot(
            ax = axis[i],
            x=anomalies_dfs[i].index,
            y=scaler.inverse_transform(anomalies_np).squeeze(),
            color=sns.color_palette()[3],
            s=50,
            label='Anomaly'
        )
    axis[i].xticks(rotation=25)
    axis[i].legend()

plt.show()