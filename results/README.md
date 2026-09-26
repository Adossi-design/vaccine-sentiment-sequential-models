# Results

`<model>` is one of `ridge`, `svr`, `rnn`, `lstm`, `gru`.

| Path | Written by | Contents |
|---|---|---|
| `<model>_validation_predictions.csv` | each model notebook, with `common.save_predictions` | columns `tweet_id, true_label, predicted_label`, one row per validation tweet |
| `experiments/<model>_experiments.csv` | each model notebook, with `common.log_experiment` | one row per run |
| `experiment_results.csv` | `common.merge_experiment_logs()`, called at the end of notebook 01 and again in notebook 05 | every run from every model; never edit it by hand |
| `figures/` | all notebooks | PNG figures named `<model or eda>_<what>.png`, e.g. `gru_learning_curve.png` |

Each model keeps its own experiment log so that teammates never edit the same file. Logging an existing `run_id` again replaces that row, so re-running a notebook does not duplicate runs.

## Experiment record fields

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
