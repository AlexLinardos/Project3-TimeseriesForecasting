# forecast.py
import ui.forecast_ui
import pandas as pd
import numpy as np
import os
import random
import tensorflow
import keras
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

forecast = ui.forecast_ui.Ui()

seed = 12345
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
tensorflow.random.set_seed(seed)
np.random.seed(seed)

# DATASET LOADING AND EDITING
df = pd.read_csv(forecast.dataset_path, delimiter='\t', header=None)
df.rename(columns = {0:'id'}, inplace=True) #rename the first column to "id"

df.sample(frac=1, random_state=1).reset_index(drop=True) # shuffle the dataframe

chosen_series = df.sample(n=3, random_state=1)
print(chosen_series)

# get only the numeric values
column_names = chosen_series.columns.values.tolist()
values = chosen_series.iloc[:, column_names[1:]].values

# SINGLE TIME SERIES MODELS

## Model creation and training
df.sample(frac=1, random_state=1).reset_index(drop=True) # shuffle the dataframe

chosen_series = df.sample(n=forecast.ts_number, random_state=1)  # choose n time series

# get only the numeric values
column_names = chosen_series.columns.values.tolist()
values = chosen_series.iloc[:, column_names[1:]].values

window = 50
slice_index = int(len(values[0])*0.8)

X_train = []
y_train = []
X_valid = []
y_valid = []
scalers = []

for series in values:
    curr_scaler = MinMaxScaler(feature_range = (0, 1))
    scalers.append(curr_scaler)

    Xt_values = []
    yt_values = []
    known_series = series[:slice_index]
    known_series = known_series.reshape(-1, 1)
    known_series = curr_scaler.fit_transform(known_series)
    for i in range(window, slice_index):
        Xt_values.append(known_series[i-window : i])
        yt_values.append(known_series[i])
    X_train.append(Xt_values)
    y_train.append(yt_values)

    Xv_values = []
    yv_values = []
    inputs = series[slice_index-window:]
    inputs = inputs.reshape(-1, 1)
    inputs = curr_scaler.transform(inputs)
    for i in range(window, len(inputs)):
        Xv_values.append(inputs[i-window: i])
        yv_values.append(inputs[i])
    X_valid.append(Xv_values)
    y_valid.append(yv_values)

X_train, y_train = np.array(X_train), np.array(y_train)
X_valid, y_valid = np.array(X_valid), np.array(y_valid)

trained_models = []

for i in range(forecast.ts_number):
    model = keras.models.Sequential([
    keras.layers.LSTM(100, return_sequences=True, input_shape=(window, 1)),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(100),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1)
    ])

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    history = model.fit(X_train[i], y_train[i], epochs = 10, batch_size=128, validation_data=(X_valid[i], y_valid[i]))
    trained_models.append((model, history))

### Making and plotting predictions
predictions = []
realities = []
for i in range(len(trained_models)):
    predicted_stock = trained_models[i][0].predict(X_valid[i])
    predicted_stock = scalers[i].inverse_transform(predicted_stock)
    predicted_stock = np.squeeze(predicted_stock)
    real_stock = values[i][slice_index:]
    predictions.append(predicted_stock)
    realities.append(real_stock)

figure, axis = plt.subplots(forecast.ts_number, 1, figsize=(14,5))

for plot_n in range(forecast.ts_number):
    axis[plot_n, 0].plot(realities[plot_n], label="Real values", linewidth=2)
    axis[plot_n, 0].plot(predictions[plot_n], label="Predicted values", linewidth=2)

plt.show()

input("Press Enter to continue to the multi-time series model...")

# MULTI-TIME SERIES MODEL
model_name = 'final_model.h5' # file name of saved model (file must be of type .h5)

# get only the numeric values
column_names = df.columns.values.tolist()
cols_to_scale = column_names[1:]
values = df.iloc[:, column_names[1:]].values

## Model creation and training
window = 50
lag = 4
slice_index = int(len(values[0])*0.8)

X_train = []
y_train = []
X_valid = []
y_valid = []

scaler = MinMaxScaler(feature_range = (0, 1))
for series in values:
  known_series = series[:slice_index]
  known_series = known_series.reshape(-1, 1)
  known_series = scaler.fit_transform(known_series)
  for i in range(window, slice_index, lag):
    X_train.append(known_series[i-window : i])
    y_train.append(known_series[i])
    
  inputs = series[slice_index-window:]
  inputs = inputs.reshape(-1, 1)
  inputs = scaler.transform(inputs)
  for i in range(window, len(inputs)):
    X_valid.append(inputs[i-window: i])
    y_valid.append(inputs[i])

X_train, y_train = np.array(X_train), np.array(y_train)
X_valid, y_valid = np.array(X_valid), np.array(y_valid)

if forecast.retrain == True:
    model = keras.models.Sequential([
    keras.layers.LSTM(50, return_sequences=True, input_shape=(window, 1)),
    keras.layers.Dropout(0.3),
    keras.layers.LSTM(50),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1)
    ])

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    history = model.fit(X_train, y_train, epochs = 20, batch_size=2048, validation_data=(X_valid, y_valid))

if forecast.retrain == True:
    full_path = './models/' + model_name
    model.save(full_path)

if forecast.retrain == False:
    full_path = './models/' + model_name
    model = keras.models.load_model(full_path)
    model.summary()

# making predictions for all time series
predictions = model.predict(X_valid)
pred_length = values.shape[1] - slice_index

predicted_series = []
for i in range(0, len(predictions), pred_length):
    predicted_series.append(predictions[i:i+pred_length])
real_series = []
for i in range(0, len(y_valid), pred_length):
    real_series.append(y_valid[i:i+pred_length])

# get n random time series to plot
plot_numbers = []
for i in range(forecast.ts_number):
    plot_numbers.append(random.randint(0, len(df.index)))

figure, axis = plt.subplots(forecast.ts_number, 1, figsize=(14,5))

for plot_n in plot_numbers:
    axis[plot_n, 0].plot(realities[plot_n], label="Real values", linewidth=2)
    axis[plot_n, 0].plot(predictions[plot_n], label="Predicted values", linewidth=2)

plt.show()