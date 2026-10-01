# Group conventions

Rules 1 to 4, 6 and 7 are the group's shared rules; 5 and 8 keep runs reproducible and file names consistent. `src/common.py` checks rules 1 and 3 and provides helpers for 4, 5 and 8 (see [helpers.md](helpers.md)).

1. **One split.** Load data with `common.load_train_validation()`. Never call `train_test_split` or make another split.
2. **No validation leakage.** Fit tokenizers, vectorizers and model parameters on the training rows only.
3. **One prediction format.** Save validation predictions with `common.save_predictions(model, val_df, y_pred)`, where `val_df` is the validation frame the predictions were made on.
4. **Log every run.** Call `common.log_experiment(...)` after each experiment with the measured RMSE and what the run changed and why. Keep experiment notes and literature sources in a final *Notes and sources* section of each model notebook.
5. **Reproducible runs.** Call `common.set_seed()` (seed 42) before building a model. `log_experiment` records library versions and the GPU, because recurrent models can give slightly different results on another GPU type or TensorFlow version. For bitwise-identical reruns on the same GPU, also call `tf.config.experimental.enable_op_determinism()` (slower).
6. **Keep hard cases, not copies of the data.** Note large-error examples for the error analysis in notebook 05, but never write tweet text to a file outside `data/`: join predictions to `Train.csv` on `tweet_id` at run time. A saved `.ipynb` is a file too, so saved outputs show `tweet_id`, labels and predictions, never `safe_text`; read the text locally while you work, and in markdown describe or paraphrase a tweet and cite its `tweet_id` instead of quoting it.
7. **Justified preprocessing only.** Keep the raw `safe_text`; apply only the cleaning decided in notebook 01 from the EDA or literature, which is `prepare_series()` in `src/preprocess.py`.
8. **Consistent names.** Figures go to `results/figures/<model or eda>_<what>.png` (e.g. `gru_learning_curve.png`); models to `models/simple_rnn.keras`, `models/lstm.keras` and `models/gru.keras`.

## Repository layout

```
├── README.md
├── requirements.txt
├── docs/              # these documents
├── data/              # Train.csv goes here; never committed (see data.md)
├── notebooks/         # 01 to 05
├── src/
│   ├── common.py      # shared loading, evaluation and experiment-logging helpers
│   ├── config.py      # notebook 01: paths, column names, seed
│   ├── data.py        # notebook 01: record repair and split creation
│   ├── preprocess.py  # notebook 01: the agreed text cleaning
│   └── evaluate.py    # notebook 01: RMSE and MAE
├── splits/            # train_ids.csv, validation_ids.csv (created by 01)
├── results/
│   ├── <model>_validation_predictions.csv
│   ├── experiments/   # one experiment log per model
│   ├── experiment_results.csv
│   └── figures/
├── models/            # saved .keras models
└── report/
```

## Shared split

Created once in `notebooks/01_data_and_baselines.ipynb`. All five models use exactly these rows, so their validation RMSEs are comparable.

| File | Format | Used for |
|---|---|---|
| `splits/train_ids.csv` | one column, `tweet_id` | fitting tokenizers, vectorizers and model parameters |
| `splits/validation_ids.csv` | one column, `tweet_id` | evaluation only |

`common.load_train_validation()` checks that neither file has duplicates, that the two files do not overlap, and that every selected row has a label in {-1, 0, 1}.

## Result files

`<model>` is one of `ridge`, `svr`, `rnn`, `lstm`, `gru`.

| Path | Written by | Contents |
|---|---|---|
| `results/<model>_validation_predictions.csv` | each model notebook, with `common.save_predictions` | columns `tweet_id, true_label, predicted_label`, one row per validation tweet |
| `results/experiments/<model>_experiments.csv` | each model notebook, with `common.log_experiment` | one row per run |
| `results/experiment_results.csv` | `common.merge_experiment_logs()`, called at the end of notebook 01 and again in notebook 05 | every run from every model; never edit it by hand |
| `results/figures/` | all notebooks | PNG figures |

Each model keeps its own experiment log so that teammates never edit the same file. Logging an existing `run_id` again replaces that row, so re-running a notebook does not duplicate runs.

## Experiment record

Every run records the group's eight required fields, plus `run_id`, `date`, `owner` and `environment` (Python, library versions and GPU, filled in automatically).

| Required field | Column |
|---|---|
| Model / approach | `model`, `approach` |
| Input representation | `input_representation` |
| Data split | `data_split` |
| Preprocessing | `preprocessing` |
| Hyperparameters | `hyperparameters` (JSON) |
| Experiment change | `change_from_previous` (what changed and why) |
| Validation RMSE | `val_rmse` (pass the value returned by `common.rmse` on the validation predictions; never type it by hand) |
| Observation | `observation` (what the result suggests and what to try next) |
