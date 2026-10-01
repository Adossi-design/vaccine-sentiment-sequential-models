"""This module provides the RMSE and MAE metrics used to score the models."""
import numpy as np


def rmse(y_true, y_pred):
    """Return the root mean squared error, the main metric of the challenge."""
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    """Return the mean absolute error, the average distance between prediction and true label."""
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    return float(np.mean(np.abs(y_true - y_pred)))
