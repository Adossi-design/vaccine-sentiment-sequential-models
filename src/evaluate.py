"""This module provides the RMSE and MAE metrics, the prediction files and the experiment log shared by all five models."""
import json
from datetime import datetime

import numpy as np
import pandas as pd

from .config import EXPERIMENT_LOG, ID_COL, RESULTS_DIR


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


def save_validation_predictions(model_name, ids, y_true, y_pred):
    """Save the predictions to results/<model_name>_validation_predictions.csv (for example model_name="ridge")."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{model_name}_validation_predictions.csv"
    pd.DataFrame({ID_COL: list(ids),
                  "true_label": np.asarray(y_true, float).ravel(),
                  "predicted_label": np.asarray(y_pred, float).ravel()}).to_csv(path, index=False)
    return path


# The columns follow the Minimum Experiment Record of the group plan with the id, member, MAE and training time added.
LOG_COLUMNS = ["experiment_id", "model", "member", "input_representation", "data_split",
               "preprocessing", "hyperparameters", "experiment_change", "validation_rmse",
               "validation_mae", "train_seconds", "observation", "timestamp"]


def log_experiment(record):
    """Add one experiment as a row in results/experiment_results.csv replacing any row with the same experiment_id."""
    row = {column: record.get(column, "") for column in LOG_COLUMNS}
    if isinstance(row["hyperparameters"], dict):
        row["hyperparameters"] = json.dumps(row["hyperparameters"], sort_keys=True)
    row["timestamp"] = datetime.now().isoformat(timespec="seconds")
    if EXPERIMENT_LOG.exists():
        log = pd.read_csv(EXPERIMENT_LOG, dtype=str)
        log = log[log["experiment_id"] != row["experiment_id"]]
    else:
        log = pd.DataFrame(columns=LOG_COLUMNS)
    log = pd.concat([log, pd.DataFrame([row]).astype(str)], ignore_index=True)
    EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(EXPERIMENT_LOG, index=False)
    return log
