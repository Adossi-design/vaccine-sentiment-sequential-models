# Vaccine Sentiment Analysis Using Sequential NLP Models

Group project for Formative Assignment 2, *Research-Informed Sequential Models for NLP and Language Technologies* (Machine Learning Techniques I, ALU, September 2026 term).

We predict the sentiment of tweets towards vaccination (-1 negative, 0 neutral, 1 positive, treated as regression and scored by RMSE) and compare two TF-IDF baselines with three recurrent neural networks that read each tweet as a sequence of tokens.

The brief asks for five approaches, at least three of them neural sequential models (sections 2 and 3), but it also refers to "the three approaches" (sections 2 and 3), "the three selected approaches" (report Methodology) and "the three models selected" (demo video). Our five (two TF-IDF baselines and three recurrent networks) satisfy both readings: the three recurrent networks are the sequential approaches, and the baselines are the comparison the brief allows.

## Deliverables

- Report: [report/Group18_report.pdf](report/Group18_report.pdf)
- Code: this repository, <https://github.com/Adossi-design/vaccine-sentiment-sequential-models>
- Demo video: the link is added here before submission.
- Contribution tracker: the link is added here before submission.

## Results

Validation results on the same 1,999 tweets of the shared split, copied from [results/model_comparison.csv](results/model_comparison.csv) (notebook 05). The interval is a paired-bootstrap 95% interval for the RMSE, macro F1 rounds each prediction to the nearest label, and the AUC measures how well a model ranks anti-vaccine tweets below the rest.

| Approach | Validation RMSE | 95% interval | MAE | Macro F1 | AUC, anti-vaccine vs rest |
|---|---|---|---|---|---|
| TF-IDF + Ridge | 0.5628 | 0.542 to 0.583 | 0.4213 | 0.5016 | 0.6411 |
| TF-IDF + SVR | 0.5740 | 0.555 to 0.593 | 0.4626 | 0.5127 | 0.7158 |
| GRU | 0.5769 | 0.557 to 0.596 | 0.4436 | 0.4756 | 0.6137 |
| LSTM | 0.5780 | 0.556 to 0.599 | 0.4337 | 0.4731 | 0.5789 |
| Simple RNN | 0.5855 | 0.564 to 0.606 | 0.4458 | 0.4445 | 0.5496 |
| Always predicting the training mean (0.3013) | 0.6459 | | 0.5656 | | |

TF-IDF + Ridge has the lowest RMSE and TF-IDF + SVR ranks anti-vaccine tweets best; all three recurrent models beat always predicting the mean, but none beats Ridge. Starting the GRU's embedding from fine-tuned GloVe Twitter vectors (notebook 04, round 4) brought its three-seed mean RMSE from 0.5762 to 0.5700, still behind Ridge. That gain is below the notebook's noise threshold, so the GRU in the table is the saved gru-09, whose embedding is learned from scratch.

## Notebooks

Each notebook name opens it in Colab.

| Notebook | Approach | Owner |
|---|---|---|
| [`01_data_and_baselines.ipynb`](https://colab.research.google.com/github/Adossi-design/vaccine-sentiment-sequential-models/blob/main/notebooks/01_data_and_baselines.ipynb) | EDA, shared split, TF-IDF + Ridge, TF-IDF + SVR | Adossi Fred William |
| [`02_simple_rnn.ipynb`](https://colab.research.google.com/github/Adossi-design/vaccine-sentiment-sequential-models/blob/main/notebooks/02_simple_rnn.ipynb) | Simple RNN | Honourgod Kilechukwu Levison |
| [`03_lstm.ipynb`](https://colab.research.google.com/github/Adossi-design/vaccine-sentiment-sequential-models/blob/main/notebooks/03_lstm.ipynb) | LSTM | Parfait Christian Henry UHIRIWE |
| [`04_gru.ipynb`](https://colab.research.google.com/github/Adossi-design/vaccine-sentiment-sequential-models/blob/main/notebooks/04_gru.ipynb) | GRU, with a fourth round that starts its embedding from GloVe Twitter vectors | Serein Byiringiro Shima |
| [`05_model_comparison_and_error_analysis.ipynb`](https://colab.research.google.com/github/Adossi-design/vaccine-sentiment-sequential-models/blob/main/notebooks/05_model_comparison_and_error_analysis.ipynb) | comparison and error analysis, plus follow-up checks: learning-curve retrains, the three recurrent cells on identical settings, and a cleaning ablation | whole group |

Run the notebooks in order, 01 to 05. Notebook 01 creates the shared split, and notebook 03 reads two files that notebook 02 writes: `models/simple_rnn_vectorizer.json` (to reuse its vocabulary size and sequence length) and `results/rnn_validation_predictions.csv` (for its comparison with the Simple RNN). Notebooks 02 and 04 also compare against the other models' prediction files when they exist; these are committed, so their comparison tables assume all five files are present.

## Quick start

Open a notebook in Colab from the table above, run the first cell, and upload `Train.csv` when asked. The data is not in this repository; [docs/data.md](docs/data.md) explains how to get it.

## Reproducing the numbers

Notebooks 02, 04 and 05 were run on CPU with TensorFlow 2.21, Keras 3.15 and scikit-learn 1.9.1 (the `environment` column of the experiment logs), and they reproduce their logged numbers when rerun on a CPU with these versions. Notebook 01's TF-IDF models use iterative solvers and were run on Windows, so a rerun elsewhere can differ in the sixth decimal. Notebook 03 (LSTM) was run on a Colab T4 GPU with TensorFlow 2.20; notebook 05 reproduced its saved run (lstm-final) exactly on CPU with TensorFlow 2.21, but the other LSTM runs were not rechecked. Other hardware or versions can change the last digits. The follow-up checks in section 7 of notebook 05 train small Keras models and take several minutes on CPU; they compare each retrain with its logged result and only warn, rather than stop, when the runtime differs from the logged one. Notebook 04's round 4 reads the GloVe rows for the GRU's vocabulary from the committed `models/gru_glove_100d.npz` and downloads the 1.5 GB GloVe archive only if that file is missing or no longer matches the vocabulary.

## Documentation

| Document | Contents |
|---|---|
| [docs/setup.md](docs/setup.md) | running on Colab and locally, the setup cell |
| [docs/data.md](docs/data.md) | getting `Train.csv`, data rules, columns, source |
| [docs/conventions.md](docs/conventions.md) | group rules, repository layout, file formats, experiment record |
| [docs/helpers.md](docs/helpers.md) | `src/common.py` reference |
| [docs/keras3-notes.md](docs/keras3-notes.md) | Keras 3 notes for the recurrent models |
