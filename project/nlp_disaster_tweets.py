from __future__ import annotations

import html
import math
import re
from typing import Any
from urllib.parse import unquote

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ==============================================================================
# NLP with Disaster Tweets - Fine-Tuning distilroberta-base
# ==============================================================================
# Instructions:
# - Download the dataset from Kaggle: https://www.kaggle.com/competitions/nlp-getting-started/data
#   Place `train.csv` and `test.csv` in the `project/data/` folder.
# - Fill in the missing code where indicated by `# TODO` comments.

MODEL_NAME = "distilroberta-base"
MODEL_TEXT_COLUMN = "model_text"
DEFAULT_TEXT_VARIANT = "keyword_tweet"

# ==============================================================================
# STEP 1: Data Exploration & Preprocessing
# ==============================================================================
def normalize_tweet_text(text: str) -> str:
    """
    Normalize tweet-specific noise without deleting predictive signal.
    """
    # URLs and mentions are not just noise in this dataset: their presence has
    # label signal. We replace them with tokens instead of removing them.
    normalized = html.unescape(str(text))
    normalized = normalized.replace("\x89Ûª", "'")
    normalized = re.sub(r"https?://\S+|www\.\S+", " url ", normalized)
    normalized = re.sub(r"<.*?>", " ", normalized)
    normalized = re.sub(r"@\w+", " user ", normalized)
    normalized = re.sub(r"#(\w+)", r"hashtag \1", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def load_and_clean_data(csv_path: str, clean_text: bool = False) -> pd.DataFrame:
    """
    Load the CSV data and optionally clean the tweets.
    """
    df = pd.read_csv(csv_path)

    if clean_text:
        # Step 1a: Keep the original table shape and normalize only the text.
        # This keeps the learning problem honest: no labels or metadata are
        # changed while we test whether tweet normalization helps.
        df = df.copy()
        df["text"] = df["text"].map(normalize_tweet_text)

    return df


def resolve_duplicate_training_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse exact duplicate training tweets to one majority-label example.
    """
    # Only apply this to training folds. Validation rows should stay untouched,
    # because editing validation labels would inflate the metric.
    rows: list[pd.Series] = []

    for _, group in df.groupby("text", sort=False):
        if len(group) == 1 or "target" not in group:
            rows.append(group.iloc[0])
            continue

        counts = group["target"].value_counts()
        if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
            continue

        row = group.iloc[0].copy()
        row["target"] = int(counts.idxmax())
        rows.append(row)

    if not rows:
        return df.iloc[0:0].copy()

    return pd.DataFrame(rows).reset_index(drop=True)


def add_model_text_features(
    df: pd.DataFrame,
    text_variant: str = DEFAULT_TEXT_VARIANT,
) -> pd.DataFrame:
    """
    Build the text field that the Transformer will actually read.
    """
    # Step 1b: The original dataset has useful metadata in `keyword`.
    # A Transformer receives one string per example, so we prepend the keyword
    # to the tweet text instead of changing the model architecture.
    featured_df = df.copy()
    keywords = (
        featured_df["keyword"]
        .fillna("none")
        .astype(str)
        .map(lambda keyword: unquote(keyword).strip() or "none")
    )
    tweets = featured_df["text"].astype(str).str.strip()

    if text_variant == "keyword_tweet":
        # Step 1c: The labels "keyword:" and "tweet:" make the structure
        # explicit. The model can learn that the first phrase is metadata and
        # the second phrase is the user-written tweet.
        featured_df[MODEL_TEXT_COLUMN] = "keyword: " + keywords + " tweet: " + tweets
    elif text_variant == "topic_sentence":
        # This variant reads more like natural language. It is an experiment,
        # not a guaranteed improvement; the runner lets us compare it fairly.
        featured_df[MODEL_TEXT_COLUMN] = "topic: " + keywords + ". " + tweets
    else:
        raise ValueError(f"Unknown text_variant: {text_variant}")

    return featured_df

# ==============================================================================
# STEP 2: Tokenization & Datasets
# ==============================================================================
def dataframe_to_dataset(df: pd.DataFrame) -> Dataset:
    """
    Convert a pandas dataframe into the Dataset format expected by Trainer.
    """
    dataset = Dataset.from_pandas(df, preserve_index=False)
    if "target" in dataset.column_names:
        dataset = dataset.rename_column("target", "labels")
    return dataset


def prepare_datasets(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    curate_training_duplicates: bool = False,
    text_variant: str = DEFAULT_TEXT_VARIANT,
) -> tuple[Dataset, Dataset]:
    """
    Split the dataframe into train and validation sets, 
    and convert them into Hugging Face `Dataset` objects.
    """
    # Step 2a: Split labeled examples into training and validation rows.
    # `stratify` preserves the 0/1 class balance in both splits, which makes
    # validation metrics less noisy for this slightly imbalanced dataset.
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["target"],
    )

    if curate_training_duplicates:
        train_df = resolve_duplicate_training_labels(train_df)

    train_df = add_model_text_features(train_df, text_variant=text_variant)
    val_df = add_model_text_features(val_df, text_variant=text_variant)

    # Step 2b: Hugging Face models expect the supervised target column to be
    # called "labels"; with "target", Trainer would not pass labels to the model.
    train_dataset = dataframe_to_dataset(train_df)
    val_dataset = dataframe_to_dataset(val_df)

    return train_dataset, val_dataset


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: Any,
    max_length: int = 128,
    text_column: str = MODEL_TEXT_COLUMN,
    padding: bool | str = "max_length",
) -> Dataset:
    """
    Apply the tokenizer to one Dataset.
    """
    def tokenize_function(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # Step 2d: Neural networks do not read Python strings directly.
        # The tokenizer converts each tweet into integer token IDs and an
        # attention mask that tells the model which positions are real tokens.
        return tokenizer(
            examples[text_column],
            padding=padding,
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(tokenize_function, batched=True)


def tokenize_datasets(
    train_dataset: Dataset,
    val_dataset: Dataset,
    tokenizer: Any,
    max_length: int = 128,
    text_column: str = MODEL_TEXT_COLUMN,
    padding: bool | str = "max_length",
) -> tuple[Dataset, Dataset]:
    """
    Apply the tokenizer to the text column in the datasets.
    """
    # Step 2e: `batched=True` inside tokenize_dataset sends many tweets to the
    # tokenizer at once, which is faster than tokenizing one row per call.
    tokenized_train = tokenize_dataset(
        train_dataset,
        tokenizer,
        max_length=max_length,
        text_column=text_column,
        padding=padding,
    )
    tokenized_val = tokenize_dataset(
        val_dataset,
        tokenizer,
        max_length=max_length,
        text_column=text_column,
        padding=padding,
    )

    return tokenized_train, tokenized_val

# ==============================================================================
# STEP 3 & 4: Model Setup & Training Loop
# ==============================================================================
def compute_metrics(pred: Any) -> dict[str, float]:
    """
    Compute accuracy and F1 score for the validation set.
    """
    # Step 5a: The model outputs logits: raw scores for class 0 and class 1.
    # `argmax(-1)` chooses the class with the larger score for each tweet.
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Step 5b: Accuracy is easy to interpret; F1 is useful here because the
    # dataset has more non-disaster tweets than disaster tweets.
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)

    return {"accuracy": acc, "f1": f1}

def positive_class_probabilities(logits: np.ndarray) -> np.ndarray:
    """
    Convert two-class logits into class-1 probabilities.
    """
    # Step 5c: Logits are unnormalized model scores. Softmax turns the two
    # scores into probabilities that add to 1. We only need the probability of
    # class 1: "real disaster".
    shifted_logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return probabilities[:, 1]


def threshold_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """
    Score class-1 probabilities after applying one decision threshold.
    """
    preds = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }


def threshold_sweep(
    probabilities: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """
    Score a list of thresholds so the cutoff choice is inspectable.
    """
    rows = [
        threshold_metrics(probabilities, labels, float(threshold))
        for threshold in thresholds
    ]
    return pd.DataFrame(rows)


def find_best_accuracy_threshold(
    logits: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Find the validation threshold that maximizes accuracy.
    """
    # Step 5d: `argmax` is a fixed decision rule. Threshold tuning asks a more
    # direct question: "how confident must the model be before we predict 1?"
    # We choose that cutoff on validation data only, never on the hidden test set.
    if thresholds is None:
        thresholds = np.round(np.arange(0.1, 0.91, 0.01), 2)

    probabilities = positive_class_probabilities(logits)
    sweep = threshold_sweep(probabilities, labels, thresholds)

    if sweep.empty:
        raise ValueError("At least one threshold is required.")

    best_row = sweep.sort_values(
        ["accuracy", "f1"],
        ascending=False,
    ).iloc[0]
    return best_row.to_dict()

def train_model(
    tokenized_train: Dataset,
    tokenized_val: Dataset,
    output_dir: str = "project/results",
    learning_rate: float = 2e-5,
    num_train_epochs: float = 3,
    train_batch_size: int = 16,
    eval_batch_size: int = 16,
    weight_decay: float = 0.01,
    run_training: bool = True,
    model_name: str = MODEL_NAME,
    seed: int = 42,
    warmup_ratio: float = 0.0,
    warmup_steps: int | None = None,
    eval_strategy: str = "epoch",
    save_strategy: str = "epoch",
    eval_steps: int | None = None,
    save_steps: int | None = None,
    save_total_limit: int | None = None,
    save_only_model: bool = True,
    gradient_accumulation_steps: int = 1,
    tokenizer: Any | None = None,
    use_dynamic_padding: bool = False,
) -> Trainer:
    """
    Initialize the model and train it using Hugging Face's Trainer API.
    """
    set_seed(seed)

    # Step 3a: Load the pretrained Transformer encoder and attach a fresh
    # two-class classification head for labels 0 and 1.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )

    # Step 4a: TrainingArguments controls optimization and evaluation cadence.
    # Since this experiment is trying to increase accuracy, the checkpoint
    # selector should optimize accuracy instead of F1.
    if warmup_steps is None:
        steps_per_epoch = math.ceil(len(tokenized_train) / train_batch_size)
        update_steps_per_epoch = math.ceil(
            steps_per_epoch / gradient_accumulation_steps
        )
        total_training_steps = math.ceil(update_steps_per_epoch * num_train_epochs)
        warmup_steps = round(total_training_steps * warmup_ratio)
        if warmup_ratio > 0 and warmup_steps == 0:
            warmup_steps = 1

    training_kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "num_train_epochs": num_train_epochs,
        "weight_decay": weight_decay,
        "eval_strategy": eval_strategy,
        "save_strategy": save_strategy,
        "save_only_model": save_only_model,
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "greater_is_better": True,
        "remove_unused_columns": True,
        "report_to": "none",
        "seed": seed,
        "data_seed": seed,
        "warmup_steps": warmup_steps,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    if eval_steps is not None:
        training_kwargs["eval_steps"] = eval_steps
    if save_steps is not None:
        training_kwargs["save_steps"] = save_steps
    if save_total_limit is not None:
        training_kwargs["save_total_limit"] = save_total_limit

    training_args = TrainingArguments(**training_kwargs)

    if use_dynamic_padding and tokenizer is None:
        raise ValueError("tokenizer is required when use_dynamic_padding=True")

    data_collator = (
        DataCollatorWithPadding(tokenizer=tokenizer)
        if use_dynamic_padding and tokenizer is not None
        else None
    )

    # Step 4b: Trainer wires together the model, tokenized data, training
    # settings, and metric function. It owns the PyTorch training loop.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    if run_training:
        print("Starting training...")
        trainer.train()

    return trainer

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
if __name__ == "__main__":
    # 1. Load Data (assuming you have downloaded it to project/data/train.csv)
    df = load_and_clean_data("project/data/train.csv", clean_text=False)

    # 2. Prepare Splits
    train_ds, val_ds = prepare_datasets(df)

    # 3. Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 4. Tokenize
    tokenized_train, tokenized_val = tokenize_datasets(train_ds, val_ds, tokenizer)

    # 5. Train
    trainer = train_model(tokenized_train, tokenized_val)

    # 6. Evaluate
    metrics = trainer.evaluate()
    print("Validation Metrics:", metrics)

    # 7. Tune the final decision threshold on validation predictions.
    predictions = trainer.predict(tokenized_val)
    threshold_metrics = find_best_accuracy_threshold(
        predictions.predictions,
        predictions.label_ids,
    )
    print("Best Validation Threshold:", threshold_metrics)
