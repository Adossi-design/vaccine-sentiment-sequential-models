# Vaccine Sentiment Analysis Using Sequential NLP Models

Group project for Formative Assignment 2, *Research-Informed Sequential Models for NLP and Language Technologies* (Machine Learning Techniques I, ALU, September 2026 term).

We predict the sentiment of tweets towards vaccination (-1 negative, 0 neutral, 1 positive, treated as regression and scored by RMSE) and compare two TF-IDF baselines with three recurrent neural networks that read each tweet as a sequence of tokens.

| Notebook | Approach | Owner |
|---|---|---|
| `01_data_and_baselines.ipynb` | EDA, shared split, TF-IDF + Ridge, TF-IDF + SVR | Adossi Fred William |
| `02_simple_rnn.ipynb` | Simple RNN | Honourgod Kilechukwu Levison |
| `03_lstm.ipynb` | LSTM | Parfait Christian Henry UHIRIVE |
| `04_gru.ipynb` | GRU | Serein Byiringiro Shima |
| `05_model_comparison_and_error_analysis.ipynb` | comparison and error analysis | whole group |

Run 01 first (it creates the shared split), then 02 to 04 in any order, then 05.

## Quick start

Open a notebook in Colab at `https://colab.research.google.com/github/Adossi-design/vaccine-sentiment-sequential-models/blob/main/notebooks/<notebook>.ipynb`, run the first cell, and upload `Train.csv` when asked. The data is not in this repository; [docs/data.md](docs/data.md) explains how to get it.

## Documentation

| Document | Contents |
|---|---|
| [docs/setup.md](docs/setup.md) | running on Colab and locally, the setup cell |
| [docs/data.md](docs/data.md) | getting `Train.csv`, data rules, columns, source |
| [docs/conventions.md](docs/conventions.md) | group rules, repository layout, file formats, experiment record |
| [docs/helpers.md](docs/helpers.md) | `src/common.py` reference |
| [docs/keras3-notes.md](docs/keras3-notes.md) | Keras 3 notes for the recurrent models |
