"""This module defines the file paths, column names and random seed used throughout the project."""
from pathlib import Path

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = PROJECT_ROOT / "Train.csv"
SPLITS_DIR = PROJECT_ROOT / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENT_LOG = RESULTS_DIR / "experiment_results.csv"

ID_COL = "tweet_id"
TEXT_COL = "safe_text"
TARGET_COL = "label"
AGREEMENT_COL = "agreement"

LABELS = (-1.0, 0.0, 1.0)
