# Vaccine Sentiment Analysis Using Sequential NLP Models

Group project for Formative Assignment 2, *Research-Informed Sequential Models for NLP and Language Technologies* (Machine Learning Techniques I, ALU, September 2026 term).

We predict the sentiment of tweets towards vaccination (-1 negative, 0 neutral, 1 positive, treated as regression and scored by RMSE) and compare two TF-IDF baselines with three recurrent neural networks that read each tweet as a sequence of tokens.

The brief asks for five approaches, at least three of them neural sequential models (sections 2 and 3), but it also refers to "the three approaches" (sections 2 and 3), "the three selected approaches" (report Methodology) and "the three models selected" (demo video). Our five (two TF-IDF baselines and three recurrent networks) satisfy both readings: the three recurrent networks are the sequential approaches, and the baselines are the comparison the brief allows.

| Notebook | Approach | Owner |
|---|---|---|
| `01_data_and_baselines.ipynb` | EDA, shared split, TF-IDF + Ridge, TF-IDF + SVR | Adossi Fred William |
| `02_simple_rnn.ipynb` | Simple RNN | Honourgod Kilechukwu Levison |
| `03_lstm.ipynb` | LSTM | Parfait Christian Henry UHIRIWE |
| `04_gru.ipynb` | GRU | Serein Byiringiro Shima |
| `05_model_comparison_and_error_analysis.ipynb` | comparison and error analysis | whole group |

Run the notebooks in order, 01 to 05. Notebook 01 creates the shared split, and notebook 03 reads two files that notebook 02 writes: `models/simple_rnn_vectorizer.json` (to reuse its vocabulary size and sequence length) and `results/rnn_validation_predictions.csv` (for its comparison with the Simple RNN). Notebooks 02 and 04 also compare against the other models' prediction files when they exist; these are committed, so their comparison tables assume all five files are present.

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
