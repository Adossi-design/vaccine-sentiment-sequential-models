# Shared helpers (`src/common.py`)

| Function | Purpose |
|---|---|
| `fetch_train_csv(drive_path=None)` | makes sure `data/Train.csv` exists (uploads it or copies it from Drive on Colab) and checks it against the official file |
| `load_train()` | `Train.csv` as a DataFrame, with notebook 01's repair of the malformed record (`src/data.py`) |
| `load_split_ids()` | the shared train and validation IDs, checked for duplicates and overlap |
| `load_train_validation(df=None)` | train and validation rows for the shared split (`df` defaults to `load_train()`) |
| `rmse(y_true, y_pred)` | the challenge metric |
| `save_predictions(model, val_df, y_pred)` | writes `results/<model>_validation_predictions.csv` and prints its RMSE |
| `log_experiment(...)` | adds or updates one run in `results/experiments/<model>_experiments.csv` |
| `merge_experiment_logs()` | combines all logs into `results/experiment_results.csv` |
| `set_seed(seed=42)` | seeds Python, NumPy and Keras |
| `environment_summary()` / `print_environment()` | Python and library versions plus GPU name, for the record |

Paths are available as `common.ROOT`, `DATA_DIR`, `SPLITS_DIR`, `RESULTS_DIR`, `FIGURES_DIR`, `EXPERIMENTS_DIR` and `MODELS_DIR`.

## Typical use in a model notebook

```python
train_df, val_df = common.load_train_validation()

# ... fit on train_df only, predict val_df ...

common.save_predictions("gru", val_df, y_pred)
common.log_experiment(
    run_id="gru-01",
    owner="Serein Byiringiro Shima",
    model="gru",
    approach="Embedding -> GRU -> Dense(1)",
    input_representation="token IDs from TextVectorization",
    preprocessing="agreed notebook 01 text preparation",
    hyperparameters={"embedding_dim": 64, "units": 64, "learning_rate": 1e-3},
    change_from_previous="first run",
    val_rmse=common.rmse(val_df["label"], y_pred),
    observation="what the result suggests and what to try next",
)
```
