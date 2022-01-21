# Project3 - Timeseries forecasting, anomaly detections and dimensionality reduction

### In this project we are making use of Neural Networks in order to implement models that can predict the future values of a time series, detect if there are any anomalies, or reduce its dimensionality. The project is written in Python and the dataset used for experimentation contains stock values.

## Team members:
* Λινάρδος Αλέξανδρος - sdi1600093
* Αντώνης Μήλιος - sdi1600100

Github repository: https://github.com/AlexLinardos/Project3-TimeseriesForecasting

## Table of contents:
* [Repository organisation](#repository-organisation)
* [Dependencies](#dependencies)
* [Forecasting](#forecasting)
* [Anomaly detection](#anomaly-detection)
* [Dimensionality reduction](#dimensionality-reduction)
    * [Performance comparisons](##performance-comparisons)

# Repository organisation
### __ATTENTION!__ Even though the code is also provided in .py files, we strongly recomend that you use Jupyter Notebooks (.ipynb) for studying and running the models instead, as the Notebooks provide narrated code and more flexibility during the plotting steps.

## Files by directory:
* __datasets/__
    * nasdaq2007_17.csv -> The dataset we used for our experiments
    * in_dataset.csv -> The dataset we used as input for the dimensionality reducer
    * in_queryset.csv -> The queryset we used as input for the dimensionality reducer
* __models/__ -> Contains the pre-trained models for our applications in .h5 format
    * autoencoder2.h5 -> Model for anomaly detection
    * final_model.h5 -> Model for forecasting
    * compressor.h5 -> Model for dimensionality reduction
* __notebooks/__
    * detect.ipynb -> Anomaly detection
    * forecast.ipynb -> Forecasting
    * reduce.ipynb -> Shows the results of the compression and decompression
    * reducer_training.ipynb -> Trains the autoencoder model
* __outputs/__ -> Contains the outputs of our dimensionality reducer
* __src/__
    * __ui/__ -> Contains modules used to implement the command line interfaces
        * detect_ui.py
        * forecast_ui.py
        * reduce_ui.py
    * detect.py -> Anomaly detection
    * forecast.py -> Forecasting
    * reduce.py -> Performs dimensionality reduction
    * reducer_training.py -> Trains the model for dimensionality reduction
    * utils.py -> Module that contains utility functions for general use

# Dependencies
To run our programs you will need Python 3.6 (or later) and the following modules:
* pandas
* numpy
* tensorflow
* sklearn
* keras
* matplotlib
* seaborn

# Forecasting

## Running the program
__As a .py__ : To run forecast.py execute the program from the command line while using the following format: `$python src/forecast.py -d <dataset> -n <number of time series selected> --retrain <optional>`

__WARNING!__ The --retrain parameter will cause the program to start the training of the model that fits on the entirety of the dataset from scratch (instead of just loading it) which may take *a lot* of time.

_Example:_ `python3 src/forecast.py -d datasets/nasdaq2007_17.csv -n 3`

__As a .ipynb__ : To be able to re-run the forecast.ipynb notebook make sure to first download the final_model.h5 file from the "models" directory on your computer or on Google Drive (if you are using Colab). You will also need a dataset in .csv format. You can download the one in the "datasets" directory or use your own.

## How it works
For this task we have created several LSTM neural networks that predicts the future values of a time series. Each model is trained on only the time series for which its will predict the future values. Before feeding the time series to our models we split them into several smaller series by taking the first N values, then sliding to the right by one value and repeating the process such as each sub-series consists of the same values as the previous, except for one (the last value). The number of values of each sub-series (aka the "window") and the number of values we slide over may vary. 

We 've also created and trained a model that uses the same techniques, although it is trained using the entirety of the dataset instead of just one time series.

## Combating overfitting
We avoid overfitting our models by using the following techniques:
* Dataset shuffling: We shuffle our data in order to ensure that both the train and the validation/test sets are representative of the whole dataset.
* Data scaling : We scale our data using a MinMaxScaler in the (0, 1) range. We have also tried using the StandardScaler but the MinMaxScaler yielded way better results in terms of loss.
* Dropout layers: Dropout layers assigns neurons a probability of being temporarily ignored for a training step. We set this probability to 0.2.
* Hyperparameter tuning: In each model, we experimented to find the right number of epochs, layers, neurons and batch size in order to prevent overfitting while minimizing the loss at the same time.

To prove that our models are not overfitting we plotted (in the forecast.ipynb notebook) the training and validation MSE history. The results gave us the confirmation that our models indeed do not overfit. In some cases, the validation loss appears to be lower than the training loss. This phenomenon is probably caused by the fact that the dropout is only active during training or (less likely) the validation set is "easy". In any case, it is not something worrying.


# Anomaly detection
## Running the program
__As a .py__ : To run detect.py execute the program from the command line while using the following format: `$python src/detect.py -d <dataset> -n <number of time series selected> -mae <error values as double> --retrain <optional>`

_Example:_ `python3 src/detect.py -d datasets/nasdaq2007_17.csv -n 3 -mae 0.32`

__WARNING!__ The --retrain parameter will cause the program to start the training of the model from scratch (instead of just loading it) which may take *a lot* of time.

For an indication about the values you could use for the -mae parameter you can take a look at the MAE graphs in the corresponding notebook.

__As a .ipynb__ : To be able to re-run the detect.ipynb notebook make sure to first download the autoencoder2.h5 file from the "models" directory on your computer or on Google Drive (if you are using Colab). You will also need a dataset in .csv format. You can download the one in the "datasets" directory or use your own.

## How it works
For this task we have created an LSTM autoencoder. The model is trained on the entirety of the dataset. By plotting the MAE of the training set (as seen in the detect.ipynb notebook), we see that most of the loss is bellow 0.3. That observation gives as a good idea about where we should set the threshold, after which a value is considered an anomaly. After making our predictions on the test set and monitoring the MAE, we compare it to the threshold value and if it exceeds it we mark the corresponding point as an anomaly.

## Combating overfitting
We avoid overfitting our models by using the following techniques:
* Dataset shuffling: We shuffle our data in order to ensure that both the train and the validation/test sets are representative of the whole dataset.
* Data scaling : We scale our data using a MinMaxScaler in the (0, 1) range. We have also tried using the StandardScaler but the MinMaxScaler yielded way better results in terms of loss.
* Dropout layers: Dropout layers assigns neurons a probability of being temporarily ignored for a training step. We set this probability to 0.2.
* Hyperparameter tuning: In each model, we experimented to find the right number of epochs, layers, neurons and batch size in order to prevent overfitting while minimizing the loss at the same time.

To prove that our models are not overfitting we plotted (in the detect.ipynb notebook) the training and validation MSE history. The results gave us the confirmation that our models indeed do not overfit.

# Dimensionality reduction
## Running the program
__As a .py__: To run reduce.py execute the program from the command line while using the following format: 
`$python reduce.py -d <dataset> -q <queryset> -od <output_dataset_file> -oq <output_query_file>`

_Example:_ `python3 src/reduce.py -d datasets/in_dataset.csv -q datasets/in_queryset.csv -od reduced_dataset.csv -oq reduced_queryset.csv`

If you would like to re-train the model or train a new model execute the "reducer_training.py" program (as such: `python3 src/reducer_training.py`). If you decide to do this you could also change the first 4 variables of the code of "reducer_training.py" according to your liking.

__As a .ipynb__ : To be able to re-run the reduce.ipynb notebook make sure to first download the autoencoder2.h5 file from the "models" directory on your computer or on Google Drive (if you are using Colab). You will also need a dataset, as well as a queryset, in .csv format. You can download the ones in the "datasets" directory or use your own. If you would like to see our graphs about this model or re-train it, please refer to the "reducer_training.ipynb" notebook.

## How it works
For this task we have created another autoencoder, althouth this time its a Convolutional autoencoder with the intention of taking time series as input and reducing their dimensions while trying to maintain low loss of data. Thus, we use the same dataset for both X and y sets while fitting the model. Basically, it compresses time series. We used a window of 50 for splitting the time series and kept the latent dimension to 3. 

## Combating overfitting
We avoid overfitting our models by using the following techniques:
* Data scaling : We scale our data using a MinMaxScaler in the (0, 1) range. We have also tried using the StandardScaler but the MinMaxScaler yielded way better results in terms of loss.
* Hyperparameter tuning: In each model, we experimented to find the right number of epochs, layers, neurons and batch size in order to prevent overfitting while minimizing the loss at the same time.

To prove that our models are not overfitting we plotted (in the reducer_training.ipynb notebook) the training and validation binary cross-entropy loss history. The results gave us the confirmation that our models indeed do not overfit.

## Performance comparisons
__(PENDING...)__

In order to examine how good our dimensionality reduction model trully did, we used its output to run the code of our previous project, which performs Nearest Neighbour searching and Clustering specifically for time series, using the specialized Frechet distance metric (for information consider visiting the repo at https://github.com/AlexLinardos/Project2-TimeSeries_Hashing_and_Searching). The results of our experiments are included in the "outputs" directory of the repository.