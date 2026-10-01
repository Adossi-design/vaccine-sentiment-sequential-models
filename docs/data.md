# Data

`Train.csv` is **not** in this repository. The challenge rules say the data belongs to Zindi and the competition host, and they forbid uploading it to public sites such as GitHub. `.gitignore` blocks the challenge files (and any CSV other than the project's own outputs) to help prevent committing them by accident.

## Getting Train.csv

Every group member joins the competition on Zindi before downloading or receiving the data, because the rules only allow sharing it with participants.

1. Sign in to Zindi, open [To Vaccinate or Not to Vaccinate: It's not a Question](https://zindi.africa/competitions/to-vaccinate-or-not-to-vaccinate), accept the rules and download `Train.csv` from the Data tab.
2. **Locally:** save it as `data/Train.csv`.
3. **On Colab:** either upload it when the notebook asks, or keep a copy in your Google Drive and pass its path:

   ```python
   common.fetch_train_csv(drive_path="/content/drive/MyDrive/vaccine-sentiment/Train.csv")
   ```

   A Colab session's files are private to you, which the rules allow.

`common.fetch_train_csv()` compares the file's SHA-256 with the official file and stops with an error if they differ, so everyone trains on identical data.

We use only `Train.csv`. `Test.csv` has no labels, and the group is not submitting to the competition.

## What this repository does publish

- `splits/`: tweet IDs only.
- `results/<model>_validation_predictions.csv`: tweet ID, true label and prediction for each validation tweet, as required for comparing the five models.
- Notebook outputs: tweet IDs, labels and predictions, never tweet text. A committed notebook publishes its saved outputs, so display examples by `tweet_id` (drop `safe_text` before displaying) and paraphrase them in markdown. Before pushing, check that no saved output contains `<user>` or `<url>`.

## Columns

| Column | Meaning |
|---|---|
| `tweet_id` | unique tweet identifier |
| `safe_text` | tweet text; usernames and URLs are replaced by `<user>` and `<url>` |
| `label` | sentiment towards vaccination: -1 negative, 0 neutral, 1 positive (treated as a regression target) |
| `agreement` | share of the three annotators who agreed on the label; analysed in the EDA, not used as a model input |

The raw file contains a malformed record (one tweet split across two lines). `01_data_and_baselines.ipynb` explains why it is repaired rather than dropped; the repair is `repair_broken_records()` in `src/data.py`, and `common.load_train()` applies it, so every model sees the same 10,000 tweets.

## Source

The tweets were collected and labelled through Crowdbreaks (Müller & Salathé, 2019) and published by Zindi as a knowledge challenge.

- Müller, M. M., & Salathé, M. (2019). Crowdbreaks: Tracking health trends using public social media data and crowdsourcing. *Frontiers in Public Health, 7*, Article 81. https://doi.org/10.3389/fpubh.2019.00081
- Zindi. (2020). *To vaccinate or not to vaccinate: It's not a question* [Data set]. https://zindi.africa/competitions/to-vaccinate-or-not-to-vaccinate
