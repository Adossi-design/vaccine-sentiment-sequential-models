"""This module provides the RMSE and MAE metrics used to score the models."""
import numpy as np

from .common import rmse  # one RMSE for the whole project, with its shape check

__all__ = ["mae", "rmse"]


def mae(y_true, y_pred):
    """Return the mean absolute error, the average distance between prediction and true label."""
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}.")
    return float(np.mean(np.abs(y_true - y_pred)))
