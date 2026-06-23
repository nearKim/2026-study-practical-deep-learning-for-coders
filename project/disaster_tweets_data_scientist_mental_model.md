# Disaster Tweets: Expert Data Scientist Mental Model

## Objective And Metric

The data scientist receives one objective: classify tweets as real disaster (`1`) or not real disaster (`0`).

The requested target metric is accuracy, so the experiment selector optimizes accuracy. Accuracy is appropriate here because the user asked for "highest possible accuracy" and the labels are binary.

F1 is still tracked as a guardrail. The training data has more non-disaster tweets than disaster tweets, so a model can sometimes improve accuracy while becoming worse at finding the positive disaster class.

```text
target=1 rate: 42.97%
target=0 rate: 57.03%
```

## Iteration 1: Honest Transformer Baseline

### 1. Root Cause Analysis

The first useful model is not the first random split score. Random row splits allow duplicate or near-duplicate tweets to appear in both train and validation. That can inflate validation accuracy.

The stricter baseline uses a grouped split so normalized duplicate tweets stay in one split. That makes the metric harder but more honest.

### 2. Enhancement Strategy And Justification

Use `distilroberta-base` with keyword context:

```text
keyword: <keyword> tweet: <tweet text>
```

Justification:

- `keyword` is real metadata in the dataset.
- A Transformer receives one text sequence, so prepending metadata is the simplest architecture-preserving way to use it.
- The split is fixed and saved, so later experiments compare against the same validation and holdout rows.
- The decision threshold is selected on validation only, then transferred unchanged to holdout.

### 3. How The PyTorch Code Is Built

The code path is:

```text
CSV -> grouped split -> duplicate curation on train only -> model_text
-> Hugging Face tokenizer -> PyTorch Transformer classifier
-> validation probability threshold sweep -> holdout audit
```

`train_model()` builds a PyTorch-backed Hugging Face `Trainer` around `AutoModelForSequenceClassification`. The model outputs logits, and the code converts those logits to class-1 probabilities before threshold tuning.

### 4. Result And Analysis

```text
run: 20260623_003253_distilroberta_lr2e-05_seed42_4bedf51f
validation accuracy: 0.8341
validation F1:       0.7915
holdout accuracy:    0.8110
holdout F1:          0.7662
threshold:           0.605
```

Analysis:

- This is an honest grouped-split baseline.
- It is below 90%.
- The model is useful, but there is still lexical signal that a dense Transformer may not use perfectly on a small dataset.

## Iteration 2: Seed Robustness

### 1. Root Cause Analysis

The first baseline may be limited by optimization variance. Fine-tuning small NLP datasets can change materially with a different random seed because the classification head, data order, and dropout masks all change.

### 2. Enhancement Strategy And Justification

Train the same `distilroberta-base` setup with seed `13`.

Justification:

- Same architecture isolates seed variance from model-family changes.
- If the second seed wins, the problem was partly optimization noise.
- If averaging seeds wins, the two models make complementary errors.

### 3. How The PyTorch Code Is Built

The PyTorch training code stays the same. Only the experiment config changes:

```text
model_name: distilroberta-base
learning_rate: 2e-5
seed: 13
text_variant: keyword_tweet
```

The ensemble selector then tests seed42 + seed13 probabilities on validation. It chooses weights using validation accuracy, not holdout.

### 4. Result And Analysis

```text
run: 20260623_015914_distilroberta_lr2e-05_seed13_229915b9
validation accuracy: 0.8358
validation F1:       0.8000
holdout accuracy:    0.8215
holdout F1:          0.7923
threshold:           0.500
```

Seed-only ensemble result:

```text
selected weights: [0.00, 1.00]
validation accuracy: 0.8358
holdout accuracy:    0.8215
```

Analysis:

- Seed 13 is better than seed 42.
- Averaging both seeds does not help; validation selection keeps seed 13 alone.
- The root cause is seed variance, not complementary seed errors.

## Iteration 3: Hybrid Sparse + Transformer Ensemble

### 1. Root Cause Analysis

The Transformer still misses some sparse lexical cues: exact disaster phrases, hashtags, compact spellings, and metadata tokens. A TF-IDF logistic model is weaker alone, but it can add complementary signal because it sees word and character n-grams directly.

### 2. Enhancement Strategy And Justification

Add one sparse model:

```text
TF-IDF word n-grams + TF-IDF character n-grams -> LogisticRegression
```

Then ensemble its class-1 probabilities with the saved PyTorch Transformer probabilities.

Justification:

- TF-IDF is fast and strong on short text.
- Character n-grams help with misspellings, hashtags, and social-media fragments.
- Logistic regression is the right simple classifier for high-dimensional sparse text.
- The sparse model writes the same artifact shape as a Transformer run, so the existing ensemble selector can score it fairly.
- I did not rewrite logistic regression in PyTorch because it would add code without improving the model; PyTorch remains the right tool for the Transformer component.

### 3. How The Code Is Built Using PyTorch

The hybrid has two parts:

```text
PyTorch path:
CSV -> tokenizer -> DistilRoBERTa -> logits -> class-1 probabilities

Sparse path:
CSV -> keyword/location/text string -> TF-IDF -> logistic regression -> class-1 probabilities

Final path:
validation-selected weighted probability average -> threshold -> predictions
```

The final classifier is still built around PyTorch Transformer probabilities, but it adds a non-neural sparse companion because the evidence showed complementary signal.

### 4. Result And Analysis

TF-IDF alone:

```text
run: 20260623_020958_tfidf_keyword_location_logreg_bb4d1c3a
validation accuracy: 0.8012
holdout accuracy:    0.8005
threshold:           0.435
```

Best transformer + TF-IDF ensemble:

```text
runs:
- 20260623_003253_distilroberta_lr2e-05_seed42_4bedf51f
- 20260623_015914_distilroberta_lr2e-05_seed13_229915b9
- 20260623_020958_tfidf_keyword_location_logreg_bb4d1c3a

weights:             [0.29, 0.22, 0.49]
threshold:           0.540
validation accuracy: 0.8394
validation F1:       0.7996
holdout accuracy:    0.8259
holdout F1:          0.7881
artifact:            project/experiments/ensembles/20260623_021457_weighted_ensemble
```

Analysis:

- This is the best validation-selected result from the three-loop local run.
- It improves validation accuracy over both standalone Transformer seeds.
- It improves holdout accuracy over the best standalone Transformer seed.
- It still does not support a 90% claim.

## Decision Log

The code kept in `project/disaster_tweets_complex.py` is the hybrid path:

- PyTorch Transformer candidate training remains available.
- TF-IDF logistic regression now writes normal run artifacts.
- The ensemble selector chooses weights and threshold on validation only.
- Holdout is used as an audit, not as a tuning target.

The current honest best local result is:

```text
validation accuracy: 83.94%
holdout accuracy:    82.59%
```

To reach 90% on the current validation set of `1127` rows, the model would need at least `1015` correct predictions. The best current validation result has `946` correct predictions, so it needs about `69` more correct validation examples.

That gap is too large to claim solved from threshold tuning or seed changes. The next defensible path is stronger pretrained encoders, cross-validation, and external-data or competition-specific cleaning, but those require more training/download time and must still be judged on grouped holdout or cross-validation.

## Root Cause: Why This Setup Did Not Reach 90%

Two independent subagent reviews reached the same conclusion: the current result is not blocked by one bad threshold or one missing weight setting. It is the local ceiling of the tried setup: two `distilroberta-base` seeds plus one TF-IDF logistic-regression companion on one grouped split.

### 1. The Gap Is Too Large For Tuning

Current best ensemble:

```text
validation: 946 / 1127 correct = 83.94%
holdout:    944 / 1143 correct = 82.59%
```

To reach 90%:

```text
validation needs: 1015 / 1127 correct
validation gap:  69 more correct predictions

holdout needs: 1029 / 1143 correct
holdout gap:  85 more correct predictions
```

A threshold change cannot plausibly create that many extra correct predictions. The validation-selected threshold is `0.54`; even tuning the threshold on holdout, which is not a valid workflow, would only reach about `83.64%` holdout accuracy.

### 2. False Negatives Dominate

The best ensemble under-predicts real disasters:

```text
validation false positives:  53
validation false negatives: 128

holdout false positives:     70
holdout false negatives:    129
```

This means the main error mode is recall: many real disaster tweets are classified as non-disaster. Raising accuracy above 90% would require recovering many of those missed disaster examples without creating many new false positives.

### 3. Label Noise And Ambiguity Set A Real Limit

The dataset contains contradictory duplicate labels:

```text
exact duplicate text groups:             69
exact duplicate rows:                   179
exact duplicate conflict groups:         18
exact duplicate conflict rows:           55

normalized duplicate groups:            305
normalized duplicate rows:              975
normalized duplicate conflict groups:    69
normalized duplicate conflict rows:     246
```

The grouped split prevents normalized duplicate leakage across train, validation, and holdout. That is good experiment hygiene, but it also removes the easy shortcut that can inflate a random row split.

Held-out conflict burden:

```text
validation rows in normalized conflict groups: 28
validation errors in those groups:              9

holdout rows in normalized conflict groups:    38
holdout errors in those groups:                15
```

These conflicts do not explain the full 90% gap by themselves, but they explain why the dataset has a real ceiling: some rows are not cleanly learnable from text alone.

### 4. The Current Models Are Not Diverse Enough

The component models are:

```text
1. distilroberta-base, seed 42
2. distilroberta-base, seed 13
3. TF-IDF word/character n-grams + logistic regression
```

The second Transformer seed helped, but the two Transformer seeds are highly correlated. The seed-only ensemble selected seed 13 alone:

```text
seed-only weights: [0.00, 1.00]
```

TF-IDF adds different lexical signal, but it is weaker alone:

```text
TF-IDF validation accuracy: 80.12%
TF-IDF holdout accuracy:    80.05%
```

The final hybrid ensemble improved the best seed by only:

```text
validation: +4 correct rows over seed 13
holdout:    +5 correct rows over seed 13
```

Even an oracle that picked the correct answer whenever any of the three component models was correct would reach only:

```text
validation oracle over current base models: 89.80%
holdout oracle over current base models:    88.01%
```

So the current model pool does not contain enough complementary signal to honestly reach 90%, even with perfect row-by-row selection.

### 5. Stronger Models Were Planned But Not Run

`project/disaster_tweets_complex.py` plans stronger candidates:

```text
cardiffnlp/twitter-roberta-base
microsoft/deberta-v3-base
vinai/bertweet-base
```

But the completed artifact set contains only:

```text
two distilroberta-base runs
one TF-IDF logistic-regression run
four weighted ensemble artifacts
```

Therefore, `82-84%` is the honest ceiling for the tried local setup, not a theoretical ceiling for the problem. A 90% attempt would need a broader validated search: stronger pretrained encoders, more learning-rate/seed coverage, cross-validation, and dataset-specific label cleaning.
