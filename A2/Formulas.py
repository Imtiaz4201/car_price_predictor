import numpy as np


# Eval Metrics
def mse(y_pred, y_true):
    s = np.size(y_true)
    z = ((y_pred - y_true) ** 2).sum() / s
    return z


def rmse(y_pred, y_true):
    mse = mse(y_pred, y_true)
    z = np.sqrt(mse)
    return z


def r2(y_pred, y_true):
    SST = np.sum((y_true - np.mean(y_true)) ** 2)
    SSE = np.sum((y_true - y_pred) ** 2)
    if SST == 0:
        return 0.0
    r2 = 1 - (SSE / SST)
    return r2
