# Vaccine Sentiment Analysis Using Sequential NLP Models

Group project for Formative Assignment 2, *Research-Informed Sequential Models for NLP and Language Technologies* (Machine Learning Techniques I, ALU, September 2026 term).

We predict the sentiment of tweets towards vaccination and compare five modelling approaches: two TF-IDF baselines and three recurrent neural networks that read each tweet as a sequence of tokens. The assignment's research question asks how effectively sequential models can address a real-world language problem, and what evidence supports their strengths and limitations.

## Task setup

| Item | Choice |
|---|---|
| Data | `Train.csv` from the Zindi challenge *To Vaccinate or Not to Vaccinate* (tweets labelled through Crowdbreaks); see [data/README.md](data/README.md) |
| Input | `safe_text`, the tweet text |
| Target | `label`: -1 negative, 0 neutral, 1 positive, treated as a regression target as in the official starter notebook |
| Metric | RMSE on the shared validation split (the challenge metric) |
| Not a model input | `agreement` (annotator agreement) |
| Not used | Zindi `Test.csv` (no competition submission) |

## Models and notebooks

| # | Approach | Type | Owner | Notebook |
|---|---|---|---|---|
| 1 | TF-IDF + Ridge regression | baseline | Adossi Fred William | `01_data_and_baselines.ipynb` |
| 2 | TF-IDF + Support Vector Regression | baseline | Adossi Fred William | `01_data_and_baselines.ipynb` |
| 3 | Embedding → Simple RNN → Dense(1) | neural sequential | Honourgod Kilechukwu Levison | `02_simple_rnn.ipynb` |
| 4 | Embedding → LSTM → Dense(1) | neural sequential | Parfait Christian Henry UHIRIVE | `03_lstm.ipynb` |
| 5 | Embedding → GRU → Dense(1) | neural sequential | Serein Byiringiro Shima | `04_gru.ipynb` |
| | Comparison and error analysis | | whole group | `05_model_comparison_and_error_analysis.ipynb` |

`01_data_and_baselines.ipynb` (Part 1) also holds the exploratory data analysis and creates the shared split. Run it first, then 02 to 04 in any order, then 05.

## Repository layout

```
├── README.md
├── requirements.txt
├── data/              # Train.csv goes here; never committed (see data/README.md)
├── notebooks/         # 01 to 05, one per part of the group plan
├── src/common.py      # shared loading, evaluation and experiment-logging helpers
├── splits/            # train_ids.csv, validation_ids.csv (created by 01)
├── results/
│   ├── <model>_validation_predictions.csv
│   ├── experiments/   # one experiment log per model
│   ├── experiment_results.csv
│   └── figures/
├── models/            # saved .keras models (optional)
└── report/
```

## Running on Google Colab

1. Open a notebook directly from GitHub, for example
   `https://colab.research.google.com/github/Adossi-design/vaccine-sentiment-sequential-models/blob/main/notebooks/04_gru.ipynb`
   (or in Colab: *File → Open notebook → GitHub*).
2. Choose a CPU or GPU runtime (*Runtime → Change runtime type*). A T4 GPU speeds up the recurrent models.
3. Run the setup cell below. It is the first code cell of every notebook.
4. When asked, upload `Train.csv`, or load it from Drive (see [data/README.md](data/README.md)).

```python
# Setup: works on Google Colab and locally.
import subprocess
import sys
from pathlib import Path

REPO = "vaccine-sentiment-sequential-models"
if "google.colab" in sys.modules:
    ROOT = Path("/content") / REPO
    if not ROOT.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        f"https://github.com/Adossi-design/{REPO}.git", str(ROOT)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                    str(ROOT / "requirements.txt")], check=True)
else:
    ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "src" / "common.py").exists())
sys.path.insert(0, str(ROOT))

from src import common

common.print_environment()
common.set_seed()
```

Files a notebook writes on Colab (split IDs, predictions, experiment logs, figures, models) disappear when the session ends. Download them and commit them from a local clone. Notebooks 02 to 05 need `splits/` on `main` before they can run.

## Running locally

Use Python 3.10 to 3.13 (Colab uses 3.13). TensorFlow 2.21 has no wheels for Python 3.14.

```bash
git clone https://github.com/Adossi-design/vaccine-sentiment-sequential-models.git
cd vaccine-sentiment-sequential-models
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt jupyterlab
jupyter lab
```

Put `Train.csv` in `data/` first.

## Group conventions

Rules 1 to 4, 6 and 7 are the group's shared rules; 5 and 8 keep runs reproducible and file names consistent. `src/common.py` checks rules 1 and 3 and provides helpers for 4, 5 and 8.

1. **One split.** Load data with `common.load_train_validation()`. Never call `train_test_split` or make another split.
2. **No validation leakage.** Fit tokenizers, vectorizers and model parameters on the training rows only.
3. **One prediction format.** Save validation predictions with `common.save_predictions(model, val_df, y_pred)`, where `val_df` is the validation frame the predictions were made on. It writes `results/<model>_validation_predictions.csv` with the columns `tweet_id, true_label, predicted_label`.
4. **Log every run.** Call `common.log_experiment(...)` after each experiment with the measured RMSE and what the run changed and why. Keep experiment notes and literature sources in a final *Notes and sources* section of each model notebook.
5. **Reproducible runs.** Call `common.set_seed()` (seed 42) before building a model. `log_experiment` records library versions and the GPU, because recurrent models can give slightly different results on another GPU type or TensorFlow version. For bitwise-identical reruns on the same GPU, also call `tf.config.experimental.enable_op_determinism()` (slower).
6. **Keep hard cases, not copies of the data.** Note large-error examples for the error analysis in notebook 05, but never write tweet text to a file outside `data/`: join predictions to `Train.csv` on `tweet_id` at run time, and print only small samples (for example the 10 to 20 largest errors per model).
7. **Justified preprocessing only.** Keep the raw `safe_text`; apply only the cleaning decided in notebook 01 from the EDA or literature.
8. **Figures.** Save to `common.FIGURES_DIR` as `<model or eda>_<what>.png`, e.g. `gru_learning_curve.png`.

## Shared helpers (`src/common.py`)

| Function | Purpose |
|---|---|
| `fetch_train_csv(drive_path=None)` | makes sure `data/Train.csv` exists (uploads it or copies it from Drive on Colab) and checks it against the official file |
| `load_train()` | the raw `Train.csv` as a DataFrame |
| `load_split_ids()` | the shared train and validation IDs, checked for duplicates and overlap |
| `load_train_validation(df=None)` | train and validation rows for the shared split |
| `rmse(y_true, y_pred)` | the challenge metric |
| `save_predictions(model, val_df, y_pred)` | writes a prediction file in the shared format and prints its RMSE |
| `log_experiment(...)` | adds or updates one run in `results/experiments/<model>_experiments.csv` |
| `merge_experiment_logs()` | combines all logs into `results/experiment_results.csv` |
| `set_seed(seed=42)` | seeds Python, NumPy and Keras |
| `environment_summary()` / `print_environment()` | Python and library versions plus GPU name, for the record |

## Notes for the recurrent-model notebooks (Keras 3)

Colab runs Keras 3, which changes a few APIs that older tutorials use.

- `keras.preprocessing.text.Tokenizer` no longer exists. Use `keras.layers.TextVectorization(max_tokens=..., output_sequence_length=max_len)` and call `.adapt()` on the **training** texts only.
- `TextVectorization` lowercases and strips punctuation by default, so `<user>` becomes `user` and `#vaccine` becomes `vaccine`. Keep that only if notebook 01 justifies it; otherwise pass `standardize="lower"` or a custom function.
- Its vocabulary already reserves index 0 for padding and 1 for unknown words, so use `Embedding(input_dim=vectorizer.vocabulary_size(), ...)` with no extra +1.
- It keeps the first `max_len` tokens and pads at the end. `keras.utils.pad_sequences` pads and truncates at the start by default; with `mask_zero=True` the fast cuDNN GRU/LSTM kernel requires end padding (`padding="post"`).
- Save models as `models/simple_rnn.keras`, `models/lstm.keras` and `models/gru.keras`. If `TextVectorization` sits outside the model, also save its vocabulary and `max_len`.
- scikit-learn 1.6 on Colab has no `mean_squared_error(squared=False)`. Use `common.rmse`.

## References

- Müller, M. M., & Salathé, M. (2019). Crowdbreaks: Tracking health trends using public social media data and crowdsourcing. *Frontiers in Public Health, 7*, Article 81. https://doi.org/10.3389/fpubh.2019.00081
- Zindi. (2020). *To vaccinate or not to vaccinate: It's not a question* [Data set]. https://zindi.africa/competitions/to-vaccinate-or-not-to-vaccinate
