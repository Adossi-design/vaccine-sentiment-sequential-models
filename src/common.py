"""Shared helpers for every notebook in this project.

They encode the group's rules so that all five models are trained and
evaluated the same way:

* one shared train/validation split (``splits/``), never re-split;
* one prediction-file format (``tweet_id, true_label, predicted_label``);
* one experiment record per run, with the group's eight required fields.

Text cleaning is deliberately not done here. It is decided and documented in
``01_data_and_baselines.ipynb``; model notebooks get clean rows by selecting
the shared split IDs.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import shutil
import subprocess
import sys
from datetime import date
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SPLITS_DIR = ROOT / "splits"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
MODELS_DIR = ROOT / "models"

IN_COLAB = "google.colab" in sys.modules
SEED = 42
# SHA-256 of the official Zindi Train.csv (1,258,959 bytes), so every member
# can confirm they train on the same file.
TRAIN_SHA256 = "89e73c8f013a972bb33751403e1a2505eb3926aa4d86ae315208c78835e63ed6"
MODEL_NAMES = ("ridge", "svr", "rnn", "lstm", "gru")
VALID_LABELS = (-1, 0, 1)
PREDICTION_COLUMNS = ["tweet_id", "true_label", "predicted_label"]
EXPERIMENT_COLUMNS = [
    "run_id",
    "date",
    "owner",
    "model",
    "approach",
    "input_representation",
    "data_split",
    "preprocessing",
    "hyperparameters",
    "change_from_previous",
    "val_rmse",
    "observation",
    "environment",
]


# ---------------------------------------------------------------- data


def fetch_train_csv(drive_path: str | Path | None = None) -> Path:
    """Make sure ``data/Train.csv`` exists and is the official Zindi file.

    Zindi's rules forbid publishing the challenge data, so it is not in this
    repository. Locally, copy ``Train.csv`` into ``data/``. On Colab, pass the
    path of your copy in Google Drive, or leave ``drive_path`` empty to be
    asked to upload the file.
    """
    target = DATA_DIR / "Train.csv"
    if not target.exists():
        if not IN_COLAB:
            raise FileNotFoundError(
                f"{target} not found. Download Train.csv from Zindi and put it "
                "in data/ (see docs/data.md)."
            )
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if drive_path:
            from google.colab import drive

            drive.mount("/content/drive")
            shutil.copyfile(drive_path, target)
        else:
            from google.colab import files

            print("Choose Train.csv in the upload dialog.")
            uploaded = files.upload()
            if len(uploaded) != 1:
                raise ValueError("Upload exactly one file: Train.csv.")
            target.write_bytes(next(iter(uploaded.values())))
    if hashlib.sha256(target.read_bytes()).hexdigest() != TRAIN_SHA256:
        raise ValueError(
            f"{target} is not the official Zindi Train.csv, so results would "
            "not be comparable with the rest of the group. Delete it and "
            "download the file again from Zindi."
        )
    return target


def load_train(path: str | Path | None = None) -> pd.DataFrame:
    """Read the raw Zindi ``Train.csv`` exactly as pandas parses it.

    The raw file contains a malformed record, so do not train on this frame
    directly: pass it to :func:`load_train_validation` (or apply the
    notebook 01 cleaning first).
    """
    path = Path(path) if path else fetch_train_csv()
    return pd.read_csv(path, dtype={"tweet_id": str})


def load_split_ids() -> tuple[pd.Series, pd.Series]:
    """Return the shared ``(train_ids, validation_ids)`` created by notebook 01."""
    paths = {
        "train": SPLITS_DIR / "train_ids.csv",
        "validation": SPLITS_DIR / "validation_ids.csv",
    }
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in splits/. They are created once by "
            "01_data_and_baselines.ipynb; do not make a new split."
        )
    ids = {}
    for name, path in paths.items():
        ids[name] = pd.read_csv(path, dtype=str, keep_default_na=False)["tweet_id"]
        if ids[name].duplicated().any():
            raise ValueError(f"{path.name} contains duplicate tweet_ids.")
    overlap = set(ids["train"]) & set(ids["validation"])
    if overlap:
        raise ValueError(f"{len(overlap)} tweet_ids are in both splits.")
    return ids["train"], ids["validation"]


def select_rows(df: pd.DataFrame, ids: pd.Series) -> pd.DataFrame:
    """Rows of ``df`` whose ``tweet_id`` is in ``ids``, in the order of ``ids``.

    Fails loudly if an ID is missing or a selected row has an invalid label,
    which means the notebook 01 cleaning has not been applied to ``df``.
    """
    if df["tweet_id"].duplicated().any():
        raise ValueError("The data contains duplicate tweet_ids.")
    indexed = df.set_index("tweet_id")
    missing = ids[~ids.isin(indexed.index)]
    if len(missing):
        raise KeyError(f"{len(missing)} split IDs not found, e.g. {missing.iloc[0]!r}.")
    rows = indexed.loc[ids.to_numpy()].reset_index()
    if not rows["label"].isin(VALID_LABELS).all():
        raise ValueError(
            "Selected rows contain labels outside {-1, 0, 1}. Apply the "
            "notebook 01 cleaning before selecting rows."
        )
    return rows


def load_train_validation(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(train_df, validation_df)`` for the shared split.

    ``df`` defaults to the raw ``Train.csv``; pass a cleaned frame if
    notebook 01 repairs rows instead of dropping them.
    """
    df = load_train() if df is None else df
    train_ids, val_ids = load_split_ids()
    return select_rows(df, train_ids), select_rows(df, val_ids)


# ---------------------------------------------------------------- evaluation


def rmse(y_true, y_pred) -> float:
    """Root mean squared error, the challenge metric."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}.")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def save_predictions(model: str, val_df: pd.DataFrame, y_pred) -> Path:
    """Write ``results/<model>_validation_predictions.csv`` in the shared format.

    ``val_df`` is the validation frame the predictions were made on, in the
    same row order; ``tweet_id`` and ``true_label`` both come from it, so they
    cannot drift apart. Checks that the file covers exactly the shared
    validation IDs, then prints the RMSE of what was written.
    """
    if model not in MODEL_NAMES:
        raise ValueError(f"model must be one of {MODEL_NAMES}, got {model!r}.")
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if len(y_pred) != len(val_df):
        raise ValueError(f"{len(y_pred)} predictions for {len(val_df)} validation rows.")
    out = pd.DataFrame(
        {
            "tweet_id": val_df["tweet_id"].astype(str).to_numpy(),
            "true_label": val_df["label"].astype(float).to_numpy(),
            "predicted_label": y_pred,
        }
    )
    if not np.isfinite(out[["true_label", "predicted_label"]].to_numpy()).all():
        raise ValueError("Labels or predictions contain NaN or infinite values.")
    _, val_ids = load_split_ids()
    if len(out) != len(val_ids) or set(out["tweet_id"]) != set(val_ids):
        raise ValueError("Predictions must cover exactly the shared validation IDs.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{model}_validation_predictions.csv"
    out.to_csv(path, index=False)
    score = rmse(out["true_label"], out["predicted_label"])
    print(f"Saved {path.relative_to(ROOT)} (validation RMSE {score:.4f})")
    return path


# ---------------------------------------------------------------- experiment log


def _read_log(path: Path) -> pd.DataFrame:
    # Read every field as text so run IDs like "01" and entries like "None"
    # or "N/A" survive a round trip; only val_rmse is numeric.
    log = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "val_rmse" in log:
        log["val_rmse"] = pd.to_numeric(log["val_rmse"])
    return log


def _json_default(value):
    return value.item() if isinstance(value, np.generic) else str(value)


def log_experiment(
    *,
    run_id: str,
    owner: str,
    model: str,
    approach: str,
    input_representation: str,
    preprocessing: str,
    hyperparameters: dict,
    change_from_previous: str,
    val_rmse: float,
    observation: str,
    data_split: str = "shared splits/train_ids.csv + splits/validation_ids.csv",
) -> pd.DataFrame:
    """Add or update one run in ``results/experiments/<model>_experiments.csv``.

    Each model has its own log so teammates never edit the same file. Logging
    the same ``run_id`` again replaces that row, so re-running a notebook
    does not duplicate runs. The ``environment`` column (library versions and
    GPU) is filled in automatically. Returns the model's full log.
    """
    if model not in MODEL_NAMES:
        raise ValueError(f"model must be one of {MODEL_NAMES}, got {model!r}.")
    run_id = str(run_id)
    row = {
        "run_id": run_id,
        "date": date.today().isoformat(),
        "owner": owner,
        "model": model,
        "approach": approach,
        "input_representation": input_representation,
        "data_split": data_split,
        "preprocessing": preprocessing,
        "hyperparameters": json.dumps(hyperparameters, sort_keys=True, default=_json_default),
        "change_from_previous": change_from_previous,
        "val_rmse": round(float(np.asarray(val_rmse).item()), 6),
        "observation": observation,
        "environment": environment_summary(),
    }
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPERIMENTS_DIR / f"{model}_experiments.csv"
    old = _read_log(path) if path.exists() else pd.DataFrame(columns=EXPERIMENT_COLUMNS)
    records = old.to_dict("records")
    run_ids = [r["run_id"] for r in records]
    if run_id in run_ids:
        records[run_ids.index(run_id)] = row
    else:
        records.append(row)
    extra = [c for c in old.columns if c not in EXPERIMENT_COLUMNS]
    log = pd.DataFrame(records, columns=EXPERIMENT_COLUMNS + extra)
    log.to_csv(path, index=False)
    return log


def merge_experiment_logs() -> pd.DataFrame:
    """Combine every model's log into ``results/experiment_results.csv``.

    Notebook 01 calls this after the baseline runs, and notebook 05 calls it
    again once all five models are logged.
    """
    logs = [_read_log(p) for p in sorted(EXPERIMENTS_DIR.glob("*_experiments.csv"))]
    if not logs:
        raise FileNotFoundError("No experiment logs in results/experiments/ yet.")
    merged = pd.concat(logs, ignore_index=True).reindex(columns=EXPERIMENT_COLUMNS)
    merged.to_csv(RESULTS_DIR / "experiment_results.csv", index=False)
    return merged


# ---------------------------------------------------------------- reproducibility


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and (if installed) Keras/TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import keras
    except ImportError:
        return
    keras.utils.set_random_seed(seed)


def _gpu_name() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    names = out.stdout.strip().splitlines()
    return names[0] if out.returncode == 0 and names else None


def environment_summary() -> str:
    """Python, key library versions and GPU, e.g. for the experiment log.

    GRU/LSTM results can differ slightly between library versions and GPU
    types, so every run records where it ran.
    """
    parts = [f"Python {platform.python_version()}"]
    for package in ("numpy", "pandas", "scikit-learn", "tensorflow", "keras"):
        try:
            parts.append(f"{package} {metadata.version(package)}")
        except metadata.PackageNotFoundError:
            pass
    parts.append(f"GPU {_gpu_name() or 'none'}")
    return "; ".join(parts)


def print_environment() -> None:
    """Print the environment summary, one item per line."""
    print(environment_summary().replace("; ", "\n"))
