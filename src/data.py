"""This module loads Train.csv, repairs its broken record and creates the shared train/validation split."""
import re

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import AGREEMENT_COL, ID_COL, SEED, SPLITS_DIR, TARGET_COL, TEXT_COL


def ensure_train_csv():
    """Return the path to Train.csv after checking it is the official file."""
    from .common import fetch_train_csv  # The import is placed here because common.py imports this module.
    return fetch_train_csv()


def repair_broken_records(df):
    """Merge a tweet that an unquoted line break split across two rows and return the repaired table with the repaired ids."""
    df = df.copy()
    repaired, to_drop = [], []
    for i in df.index[df[TARGET_COL].isna()]:
        if i + 1 not in df.index:
            continue
        # The next row holds the rest of the text in tweet_id, the label in safe_text and the agreement in label.
        next_row = df.loc[i + 1]
        shifted_label = str(next_row[TEXT_COL]).strip()
        if pd.isna(next_row[AGREEMENT_COL]) and shifted_label in {"-1", "0", "1", "-1.0", "0.0", "1.0"}:
            df.loc[i, TEXT_COL] = str(df.loc[i, TEXT_COL]).strip() + "\n" + str(next_row[ID_COL])
            df.loc[i, TARGET_COL] = float(shifted_label)
            df.loc[i, AGREEMENT_COL] = float(next_row[TARGET_COL])
            to_drop.append(i + 1)
            repaired.append(df.loc[i, ID_COL])
    return df.drop(index=to_drop).reset_index(drop=True), repaired


def load_train(path=None, verbose=True):
    """Load Train.csv and return it with the broken record repaired."""
    path = path or ensure_train_csv()
    raw = pd.read_csv(path, dtype={ID_COL: str, TEXT_COL: str})
    df, repaired = repair_broken_records(raw)
    df[TARGET_COL] = df[TARGET_COL].astype(float)
    df[AGREEMENT_COL] = df[AGREEMENT_COL].astype(float)
    if verbose:
        print(f"Read {len(raw):,} rows -> {len(df):,} records after repairing {repaired}")
    assert df[[ID_COL, TEXT_COL, TARGET_COL, AGREEMENT_COL]].notna().all().all()
    assert df[ID_COL].is_unique
    return df


def duplicate_group_key(text):
    """Normalise a tweet by lowercasing it and removing placeholders, retweet markers and punctuation so that copies share one key."""
    key = text.lower()
    key = re.sub(r"<url>|<user>|\brt\b", " ", key)
    key = re.sub(r"[^a-z0-9#]+", " ", key)
    return " ".join(key.split())


def make_split(df, n_folds=5, seed=SEED):
    """Create the shared 80/20 train/validation split keeping duplicate tweets together and stratifying by label and agreement."""
    groups = df[TEXT_COL].map(duplicate_group_key)
    # Each stratum combines label and agreement, for example "-1_0.33".
    strata = df[TARGET_COL].astype(int).astype(str) + "_" + df[AGREEMENT_COL].round(2).astype(str)
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    # The first of the five folds (20% of the tweets) becomes the validation set.
    _, val_idx = next(splitter.split(df, strata, groups))
    split = pd.Series("train", index=df.index, name="split")
    split.iloc[val_idx] = "validation"
    return split


def save_split(df, split):
    """Save the tweet ids of the training and validation parts to splits/train_ids.csv and splits/validation_ids.csv."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("train", "validation"):
        ids = df.loc[split == name, [ID_COL]].sort_values(ID_COL)
        ids.to_csv(SPLITS_DIR / f"{name}_ids.csv", index=False)


def load_split(df):
    """Load the saved split files and return the training and validation data frames."""
    parts = []
    for name in ("train", "validation"):
        ids = pd.read_csv(SPLITS_DIR / f"{name}_ids.csv", dtype={ID_COL: str})[ID_COL]
        parts.append(df[df[ID_COL].isin(ids)].reset_index(drop=True))
    assert len(parts[0]) + len(parts[1]) == len(df), "split files do not cover the dataset"
    assert not set(parts[0][ID_COL]) & set(parts[1][ID_COL]), "train and validation overlap"
    return parts[0], parts[1]
