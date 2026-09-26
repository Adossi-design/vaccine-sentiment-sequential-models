# Shared train/validation split

Created once in `notebooks/01_data_and_baselines.ipynb`. All five models use exactly these rows, so their validation RMSEs are comparable. Nobody makes a new split.

| File | Format | Used for |
|---|---|---|
| `train_ids.csv` | one column, `tweet_id` | fitting tokenizers, vectorizers and model parameters |
| `validation_ids.csv` | one column, `tweet_id` | evaluation only |

Load the rows with:

```python
train_df, val_df = common.load_train_validation()
```

This checks that neither file has duplicates, that the two files do not overlap, and that every selected row has a label in {-1, 0, 1}.

Validation labels are never used to fit a tokenizer, vectorizer or model parameters.
