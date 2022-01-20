import numpy as np

def create_dataset(X, time_steps=1, lag=1):
    Xs, ys = [], []
    for j in range(len(X.columns)):
      for i in range(0, len(X) - time_steps, lag):
          v = X[j].iloc[i:(i + time_steps)].values
          Xs.append(v)
          ys.append(X[j].iloc[i + time_steps])
    return np.array(Xs), np.array(ys)