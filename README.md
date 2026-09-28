# Vaccine Sentiment Analysis with Sequential NLP Models

This repository contains our group project for Formative Assignment 2, *Research-Informed Sequential Models for NLP and Language Technologies*. Tweets about vaccines show how people feel about vaccination so being able to tell automatically whether a tweet is against vaccines, neutral or in favour of them could help people who follow public opinion on this topic. Our aim is to find out how well sequential models can do this task and to use the evidence from our own experiments to explain where each approach works well and where it struggles.

## The data

We use the dataset from the Zindi challenge [Sentiment Analysis: To Vaccinate or Not to Vaccinate](https://zindi.africa/competitions/sentiment-analysis-to-vaccinate-or-not-to-vaccinate-its-not-a-question/data). Only `Train.csv` comes with labels so we use it for both training and validation, and each tweet in it has a `label` of -1 (negative, anti-vaccine), 0 (neutral) or 1 (positive, pro-vaccine). Because these labels have a natural order and the official starter notebook treats them as numbers, we follow the same approach and treat the task as regression with RMSE as the main score.

## Models and who worked on them

We compare five approaches so that the sequential models can be judged against simpler methods rather than on their own. The two TF-IDF baselines ignore word order and show how far we can get with word counts alone while the Simple RNN, LSTM and GRU read each tweet as a sequence, which lets us test whether word order and memory actually improve the results on this data.

| # | Approach | Group member | Notebook |
|---|---|---|---|
| 1 | TF-IDF + Ridge regression (baseline) | Adossi Fred William | `01_data_and_baselines.ipynb` |
| 2 | TF-IDF + Support Vector Regression (baseline) | Adossi Fred William | `01_data_and_baselines.ipynb` |
| 3 | Simple RNN | Honourgod Kilechukwu Levison | `02_simple_rnn.ipynb` |
| 4 | LSTM | Parfait Christian Henry UHIRIVE | `03_lstm.ipynb` |
| 5 | GRU | Serein Byiringiro Shima | `04_gru.ipynb` |

Once all five models are trained the last notebook, `05_model_comparison_and_error_analysis.ipynb`, brings their results together so that we can compare them side by side and study the tweets they get wrong.

## What is in the repository

The notebooks contain the analysis and the experiments while the `src/` folder holds the code that all of them share so that every model loads, cleans, splits and scores the data in exactly the same way.

```
README.md
requirements.txt
Train.csv                      not in the repository, download it from Zindi and place it here
notebooks/                     notebooks 01 to 05 (see the table above)
src/                           code shared by all the notebooks
  config.py                    file paths, column names and the random seed
  data.py                      loads Train.csv, fixes the broken record, makes the split
  preprocess.py                the text cleaning that every model uses
  evaluate.py                  RMSE and MAE, prediction files and the experiment log
splits/                        train_ids.csv and validation_ids.csv, used by every model
results/                       prediction files and experiment_results.csv
results/figures/               the figures we use in the report
models/                        saved models (optional)
report/                        the final report
```

## How to run the notebooks

Because the Zindi rules do not allow the challenge data to be hosted online, `Train.csv` is not included in this repository and each person has to download it from the challenge page linked above before running anything. The notebooks are written to run on Google Colab with very little setup so opening a notebook from this repository and running all the cells is enough: the first cell clones the repository and, if `Train.csv` is missing, asks you to upload it. To run them on your own computer instead install the packages with `pip install -r requirements.txt` and put `Train.csv` in the main folder. In both cases notebook `01` has to be run first because it creates the shared split that all the other notebooks load.

## Rules we all follow

Comparing five models only makes sense if the differences in their results come from the models themselves and not from the way each person prepared the data. For this reason every model loads the data with `load_train()` and the split with `load_split(df)` from `src/data.py` and cleans the text with `prepare_series()` from `src/preprocess.py` using the default settings. To avoid leaking information from the validation set, tokenizers, vocabularies and TF-IDF are always fitted on the training split only. Every model is then evaluated in the same way, reporting RMSE as the main score together with MAE on the validation split, saving its predictions with `save_validation_predictions()` and recording each experiment with `log_experiment()` from `src/evaluate.py` so that all the results can later be compared in one place.
