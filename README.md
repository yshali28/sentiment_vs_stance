# A Target-Aware Framework for Disentangling Stance from Sentiment in Social Media Text

Speech and Language Processing course project.

Sentiment analysis tells you the tone of a tweet (positive, negative, neutral). Stance detection tells you the author's actual position on a specific target (in favour, against, or neither). The two are often treated as interchangeable, but they frequently point in different directions: a tweet can be angry in tone while supporting a cause, or cheerful while opposing it. This project measures how often that happens in a real labelled dataset, and then tests whether a target-aware transformer model can correctly classify these hard, divergent cases that a simple bag-of-words baseline gets wrong.

## Dataset

- **SemEval-2016 Task 6** (stance detection in tweets), using the 2017 re-release that adds sentiment labels to the same tweets, so each tweet has both a stance label and a sentiment label.
- Stance labels: `FAVOR`, `AGAINST`, `NONE`. Sentiment labels: `pos`, `neg`, `other`.
- 2,914 training tweets and 1,249 test tweets across the 5 official Task A targets: Atheism, Climate Change is a Real Concern, Feminist Movement, Hillary Clinton, Legalization of Abortion.
- Source: https://www.saifmohammad.com/WebPages/StanceDataset.htm
- Data mirror used: https://github.com/nchand/tweet-stance-classification

The files the code reads are `data/train.csv` and `data/test.csv`. `data/train_raw.csv` and `data/test_raw.csv` are the original downloads and are not used by the pipeline.

## Approach

### Phase 1: TF-IDF + Logistic Regression baseline

Tweets are cleaned (URLs, mentions and the `#SemST` tag removed, CamelCase hashtags split into words, negation words kept) and vectorised with TF-IDF over unigrams and bigrams. Two separate Logistic Regression classifiers are trained on the same split, one for stance and one for sentiment. The model has no explicit knowledge of the target.

- Official stance macro-F1 (averaged over FAVOR and AGAINST only, as in the SemEval evaluation): **0.590** (locally reproduced: 0.5895)

### Phase 2: Target-aware BERTweet

`vinai/bertweet-base`, fine-tuned on a Kaggle GPU. The target is passed to the model alongside the tweet using the tokenizer's sentence-pair encoding (target as the first segment, tweet as the second), so the model can condition its stance prediction on what the tweet is about.

- Official stance macro-F1: **0.681**, an improvement of **+0.091** over Phase 1
- Training notebook: https://www.kaggle.com/code/yshali/slp-proj

## Key finding: the divergent subset

A tweet is counted as **divergent** when its gold stance and gold sentiment point in opposite directions: `FAVOR` with negative sentiment, or `AGAINST` with positive sentiment. These are the cases where using sentiment as a stand-in for stance fails by definition.

- **29.5%** of test tweets (369 of 1,249) are divergent (29.1% of training tweets).
- On this subset, stance accuracy rose from **0.572** (Phase 1) to **0.672** (Phase 2).
- Phase 2 correctly classified **86** divergent tweets that Phase 1 got wrong, but also misclassified **49** that Phase 1 had right. The net gain is positive, but Phase 2 is not strictly better on every hard case.

Per-target results (test set, from `outputs/per_target_breakdown.csv`):

| Target | Divergence rate | Overall acc. P1 / P2 | Divergent-subset acc. P1 / P2 |
|---|---|---|---|
| Atheism | 65.9% | 0.577 / 0.750 | 0.607 / 0.793 |
| Climate Change is a Real Concern | 35.5% | 0.574 / 0.840 | 0.550 / 0.917 |
| Legalization of Abortion | 28.6% | 0.643 / 0.657 | 0.688 / 0.488 |
| Feminist Movement | 16.8% | 0.491 / 0.593 | 0.521 / 0.458 |
| Hillary Clinton | 12.2% | 0.634 / 0.685 | 0.278 / 0.472 |

The clearest gains are on the two highest-divergence targets, Atheism and Climate Change, where Phase 2 improves both overall and on the divergent subset. On the lower-divergence targets the picture is mixed: overall accuracy still improves for all three, but divergent-subset accuracy drops for Legalization of Abortion and Feminist Movement, and only Hillary Clinton improves. Note that the per-target divergent subsets are small (36 to 145 tweets), so these per-target numbers are noisier than the aggregate.

## Repository structure

```
.
├── data/
│   ├── train.csv, test.csv            dual-labelled SemEval data read by the pipeline
│   └── train_raw.csv, test_raw.csv    original downloads (not used by the code)
├── src/                               Phase 1 pipeline (run from inside src/)
│   ├── data_loader.py                 loads the CSVs, keeps the 5 Task A targets, renames columns
│   ├── preprocess.py                  tweet cleaning (URLs, mentions, #SemST, hashtag splitting; keeps negations)
│   ├── baselines.py                   TF-IDF + Logistic Regression (or linear SVM) training and evaluation
│   ├── disentangle.py                 divergence rate, Cramer's V, sentiment-as-proxy test, per-target divergence, plots
│   ├── main.py                        runs the full Phase 1 pipeline and the divergence analysis
│   └── export_predictions.py          saves Phase 1 row-level test predictions for the comparison step
├── compare_phase1_phase2.py           merges Phase 1 and Phase 2 predictions, computes head-to-head metrics
├── confusion_and_per_target.py        confusion matrices and per-target accuracy breakdown
├── phase2_predictions.csv             Phase 2 test predictions downloaded from the Kaggle notebook
└── outputs/                           generated results and figures
```

Phase 2 training is not in this repository; it lives in the Kaggle notebook linked above.

### Outputs

| File | Produced by |
|---|---|
| `results_summary.json` | `src/main.py` (Phase 1 F1, divergence rates, Cramer's V, proxy test, per-target divergence on train) |
| `stance_vs_sentiment_heatmap.png`, `stance_within_sentiment.png` | `src/main.py` |
| `phase1_predictions.csv` | `src/export_predictions.py` |
| `phase1_vs_phase2_merged.csv`, `phase1_vs_phase2.png` | `compare_phase1_phase2.py` |
| `confusion_matrices.png`, `per_target_accuracy.png`, `per_target_breakdown.csv` | `confusion_and_per_target.py` |

## How to reproduce

Requirements (Python 3.11+): `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`.

```
pip install pandas numpy scikit-learn scipy matplotlib seaborn
```

1. **(Optional) Run the Phase 1 analysis.** Prints classification reports and the divergence analysis, and writes `outputs/results_summary.json` plus two figures.

   ```
   cd src
   python main.py            # or: python main.py --model svm
   ```

2. **Export Phase 1 predictions.** Writes `outputs/phase1_predictions.csv`. Run from inside `src/`.

   ```
   cd src
   python export_predictions.py
   cd ..
   ```

3. **Train Phase 2 on Kaggle.** Open https://www.kaggle.com/code/yshali/slp-proj, enable a GPU, and run all cells. The notebook writes `phase2_predictions.csv` with columns `text, target, stance, sentiment, pred_stance, pred_sentiment`.

4. **Download `phase2_predictions.csv`** from the notebook's output and place it in the project root (next to `compare_phase1_phase2.py`, not inside `outputs/`).

5. **Compare the two phases.** Run from the project root. Merges on `(text, target)`, prints macro-F1, overall and divergent-subset accuracy, and the recovered / broken counts, and writes `outputs/phase1_vs_phase2_merged.csv` and `outputs/phase1_vs_phase2.png`.

   ```
   python compare_phase1_phase2.py
   ```

6. **Confusion matrices and per-target breakdown.** Run from the project root after step 5; it reads the merged CSV.

   ```
   python confusion_and_per_target.py
   ```

Phase 1 is deterministic, so steps 1 and 2 reproduce the numbers above exactly. Phase 2 results may vary slightly between Kaggle runs due to GPU non-determinism in fine-tuning.
