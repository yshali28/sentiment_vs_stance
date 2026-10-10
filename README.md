# A Target-Aware Framework for Disentangling Stance from Sentiment in Social Media Text

Speech and Language Processing course project.

Sentiment analysis tells you the tone of a tweet (positive, negative, neutral). Stance detection tells you the author's actual position on a specific target (in favour, against, or neither). The two are often treated as the same thing, but they often point in different directions. A tweet can be angry in tone while supporting a cause, or cheerful while opposing it. This project measures how often that happens in a real labelled dataset. It then tests whether a target-aware transformer model can classify these divergent cases that a simple bag-of-words baseline gets wrong.

## Dataset

- **SemEval-2016 Task 6** (stance detection in tweets), using the 2017 re-release that adds sentiment labels to the same tweets. Each tweet has both a stance label and a sentiment label.
- Stance labels: `FAVOR`, `AGAINST`, `NONE`. Sentiment labels: `pos`, `neg`, `other`.
- 2,914 training tweets and 1,249 test tweets across the 5 official Task A targets: Atheism, Climate Change is a Real Concern, Feminist Movement, Hillary Clinton, Legalization of Abortion.
- Source: https://www.saifmohammad.com/WebPages/StanceDataset.htm
- Data mirror used: https://github.com/nchand/tweet-stance-classification

The files the code reads are `data/train.csv` and `data/test.csv`. `data/train_raw.csv` and `data/test_raw.csv` are the original downloads and are not used by the pipeline.

## Approach

### Baseline (TF-IDF + logistic regression)

Tweets are cleaned. URLs, mentions and the `#SemST` tag are removed, CamelCase hashtags are split into words, and negation words are kept. The text is turned into TF-IDF features over unigrams and bigrams. Two separate logistic regression classifiers are trained on the same split, one for stance and one for sentiment. The model has no knowledge of the target.

- Official stance macro-F1 (averaged over FAVOR and AGAINST only, as in the SemEval evaluation): **0.590**
- Overall stance accuracy: **0.585**

### Target-aware BERTweet

`vinai/bertweet-base`, fine-tuned on a Kaggle GPU. The target is given to the model together with the tweet, using the tokenizer's sentence-pair encoding (target as the first segment, tweet as the second). This lets the model use the target when it predicts the stance.

- Official stance macro-F1: **0.681**, which is **+0.092** over the baseline
- Overall stance accuracy: **0.685**
- Kaggle notebook (version 2): https://www.kaggle.com/code/yshali/slp-proj

The notebook is not part of this repository.

## Key finding: the divergent subset

A tweet is counted as **divergent** when its gold stance and gold sentiment point in opposite directions: `FAVOR` with negative sentiment, or `AGAINST` with positive sentiment. In these cases, using sentiment in place of stance is always wrong.

- **29.5%** of test tweets (369 of 1,249) are divergent. In the training set it is 29.1%.
- On this subset, stance accuracy goes from **0.572** (baseline) to **0.672** (BERTweet).
- BERTweet got **84** divergent tweets right that the baseline got wrong (recovered).
- **74** divergent tweets were wrong for both models.
- BERTweet got **47** divergent tweets wrong that the baseline had right (broken).
- The net gain is 37 tweets. BERTweet is not better on every hard case.

Per-target results (test set, from `outputs/per_target_breakdown.csv`):

| Target | Divergence rate | Overall acc. baseline / BERTweet | Divergent-subset acc. baseline / BERTweet |
|---|---|---|---|
| Atheism | 65.9% | 0.577 / 0.727 | 0.607 / 0.807 |
| Climate Change is a Real Concern | 35.5% | 0.574 / 0.822 | 0.550 / 0.817 |
| Legalization of Abortion | 28.6% | 0.643 / 0.654 | 0.688 / 0.512 |
| Feminist Movement | 16.8% | 0.491 / 0.589 | 0.521 / 0.521 |
| Hillary Clinton | 12.2% | 0.634 / 0.695 | 0.278 / 0.444 |

The clearest gains are on the two targets with the highest divergence, Atheism and Climate Change. BERTweet improves both overall accuracy and divergent-subset accuracy there. On the other three targets, overall accuracy improves a little. Divergent-subset accuracy drops for Legalization of Abortion, stays the same for Feminist Movement, and improves only for Hillary Clinton. The per-target divergent subsets are small (36 to 145 tweets), so these numbers are noisier than the totals.

## Demo

There is a small Streamlit app in `app.py`.

1. Install the requirements:

   ```
   pip install -r requirements.txt
   ```

2. Run the Kaggle notebook (https://www.kaggle.com/code/yshali/slp-proj) and download `models.zip` from its output.
3. Unzip it in the project root so that `models/stance_model` and `models/sentiment_model` sit next to `app.py`.
4. Start the app:

   ```
   streamlit run app.py
   ```

The app has two tabs.

- **Try a tweet**: pick one of the example divergent test tweets, or type your own tweet and choose a target. Click Predict. It shows the stance and sentiment predicted by the baseline and by BERTweet side by side, and tells you if the two point in opposite directions.
- **Browse test results**: a table of all 1,249 test tweets with the gold labels and both models' stance predictions. You can filter by target, show only divergent tweets, and filter by outcome (BERTweet fixed it, BERTweet broke it, both right, both wrong). This tab reads `outputs/phase1_vs_phase2_merged.csv`.

If `models/` is missing, the BERTweet half of the "Try a tweet" tab shows a "models not found" message. The baseline half and the browse tab still work.

`models/` is not in the repository because of its size (about 1 GB). BERTweet results can vary slightly between runs because of GPU non-determinism, so a new training run may not give exactly the numbers above.

## Repository structure

```
.
├── app.py                             Streamlit demo
├── requirements.txt                   Python packages for the code and the demo
├── data/
│   ├── train.csv, test.csv            dual-labelled SemEval data read by the pipeline
│   └── train_raw.csv, test_raw.csv    original downloads (not used by the code)
├── src/                               baseline pipeline (run from inside src/)
│   ├── data_loader.py                 loads the CSVs, keeps the 5 Task A targets, renames columns
│   ├── preprocess.py                  tweet cleaning (URLs, mentions, #SemST, hashtag splitting; keeps negations)
│   ├── baselines.py                   TF-IDF + logistic regression (or linear SVM) training and evaluation
│   ├── disentangle.py                 divergence rate, Cramer's V, sentiment-as-proxy test, per-target divergence, plots
│   ├── main.py                        runs the full baseline pipeline and the divergence analysis
│   └── export_predictions.py          saves baseline row-level test predictions for the comparison step
├── compare_phase1_phase2.py           merges baseline and BERTweet predictions, computes head-to-head metrics
├── confusion_and_per_target.py        confusion matrices and per-target accuracy breakdown
├── phase2_predictions.csv             BERTweet test predictions downloaded from the Kaggle notebook
├── outputs/                           generated results and figures
└── models/                            BERTweet weights (not in the repo, see Demo)
```

The scripts `compare_phase1_phase2.py` and `phase2_predictions.csv`, and the output files with `phase1` / `phase2` in their names, keep their old names. The code was not changed.

### Outputs

| File | Produced by |
|---|---|
| `results_summary.json` | `src/main.py` (baseline F1, divergence rates, Cramer's V, proxy test, per-target divergence on train) |
| `stance_vs_sentiment_heatmap.png`, `stance_within_sentiment.png` | `src/main.py` |
| `phase1_predictions.csv` | `src/export_predictions.py` |
| `phase1_vs_phase2_merged.csv`, `phase1_vs_phase2.png` | `compare_phase1_phase2.py` |
| `confusion_matrices.png`, `per_target_accuracy.png`, `per_target_breakdown.csv` | `confusion_and_per_target.py` |

## How to reproduce

Requirements (Python 3.11+): install with `pip install -r requirements.txt`. The baseline scripts only need `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib` and `seaborn`.

1. **(Optional) Run the baseline analysis.** Prints classification reports and the divergence analysis, and writes `outputs/results_summary.json` plus two figures.

   ```
   cd src
   python main.py            # or: python main.py --model svm
   ```

2. **Export the baseline predictions.** Writes `outputs/phase1_predictions.csv`. Run from inside `src/`.

   ```
   cd src
   python export_predictions.py
   cd ..
   ```

3. **Train BERTweet on Kaggle.** Open https://www.kaggle.com/code/yshali/slp-proj, turn on a GPU, and run all cells. The notebook writes `phase2_predictions.csv` with columns `text, target, stance, sentiment, pred_stance, pred_sentiment`. It also writes `models.zip`.

4. **Download `phase2_predictions.csv`** from the notebook's output and put it in the project root (next to `compare_phase1_phase2.py`, not inside `outputs/`).

5. **Compare the two models.** Run from the project root. This merges on `(text, target)`, prints macro-F1, overall and divergent-subset accuracy, and the recovered / both-wrong / broken counts. It writes `outputs/phase1_vs_phase2_merged.csv` and `outputs/phase1_vs_phase2.png`.

   ```
   python compare_phase1_phase2.py
   ```

6. **Confusion matrices and per-target breakdown.** Run from the project root after step 5. It reads the merged CSV.

   ```
   python confusion_and_per_target.py
   ```

The baseline is deterministic, so steps 1 and 2 reproduce the baseline numbers exactly. BERTweet results may vary slightly between Kaggle runs because of GPU non-determinism in fine-tuning.
