from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from transformers import Trainer

import project.nlp_disaster_tweets as app


class FakeTokenizer:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(
        self,
        texts: list[str],
        padding: str,
        truncation: bool,
        max_length: int,
    ) -> dict[str, list[list[int]]]:
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
        self.texts.extend(texts)
        return {
            "input_ids": [[idx + 1, idx + 2, 0] for idx, _ in enumerate(texts)],
            "attention_mask": [[1, 1, 0] for _ in texts],
        }


class DisasterTweetsTest(unittest.TestCase):
    def test_normalize_tweet_text_preserves_signal_tokens(self) -> None:
        text = app.normalize_tweet_text(
            "@responder Check https://example.com #Wildfire &amp; smoke"
        )

        self.assertEqual(text, "user Check url hashtag Wildfire & smoke")

    def test_load_and_clean_data_normalizes_common_tweet_noise(self) -> None:
        df = app.load_and_clean_data("project/data/train.csv", clean_text=True)

        self.assertEqual(len(df), 7613)
        self.assertFalse(df["text"].str.contains(r"http\S+", regex=True).any())
        self.assertFalse(df["text"].str.contains(r"@\w+", regex=True).any())
        self.assertTrue(df["text"].str.contains(r"\burl\b", regex=True).any())

    def test_add_model_text_features_prepends_keyword_context(self) -> None:
        df = pd.DataFrame(
            {
                "keyword": ["body%20bags", np.nan],
                "text": ["example tweet", "missing keyword tweet"],
            }
        )

        featured = app.add_model_text_features(df)

        self.assertEqual(
            featured["model_text"].tolist(),
            [
                "keyword: body bags tweet: example tweet",
                "keyword: none tweet: missing keyword tweet",
            ],
        )

    def test_add_model_text_features_supports_natural_template(self) -> None:
        df = pd.DataFrame({"keyword": ["forest%20fire"], "text": ["near town"]})

        featured = app.add_model_text_features(df, text_variant="topic_sentence")

        self.assertEqual(featured["model_text"].tolist(), ["topic: forest fire. near town"])

    def test_resolve_duplicate_training_labels_keeps_majority_label(self) -> None:
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5, 6],
                "text": ["same", "same", "same", "tie", "tie", "unique"],
                "target": [1, 1, 0, 0, 1, 0],
            }
        )

        curated = app.resolve_duplicate_training_labels(df)

        self.assertEqual(curated["text"].tolist(), ["same", "unique"])
        self.assertEqual(curated["target"].tolist(), [1, 0])

    def test_dataframe_to_dataset_renames_target_when_present(self) -> None:
        df = pd.DataFrame(
            {
                "id": [1],
                "keyword": ["fire"],
                "location": [""],
                "text": ["example"],
                "target": [1],
                "model_text": ["keyword: fire tweet: example"],
            }
        )

        dataset = app.dataframe_to_dataset(df)

        self.assertIn("labels", dataset.column_names)
        self.assertNotIn("target", dataset.column_names)

    def test_prepare_datasets_splits_and_renames_labels(self) -> None:
        df = pd.DataFrame(
            {
                "id": range(10),
                "keyword": ["fire"] * 10,
                "location": [""] * 10,
                "text": [f"tweet {idx}" for idx in range(10)],
                "target": [0, 1] * 5,
            }
        )

        train_dataset, val_dataset = app.prepare_datasets(
            df,
            test_size=0.2,
            random_state=1,
        )

        self.assertEqual(len(train_dataset), 8)
        self.assertEqual(len(val_dataset), 2)
        self.assertIn("labels", train_dataset.column_names)
        self.assertNotIn("target", train_dataset.column_names)
        self.assertIn("model_text", train_dataset.column_names)

    def test_tokenize_datasets_adds_model_inputs(self) -> None:
        df = pd.DataFrame(
            {
                "id": range(10),
                "keyword": ["fire"] * 10,
                "location": [""] * 10,
                "text": [f"tweet {idx}" for idx in range(10)],
                "target": [0, 1] * 5,
            }
        )
        train_dataset, val_dataset = app.prepare_datasets(
            df,
            test_size=0.2,
            random_state=1,
        )
        tokenizer = FakeTokenizer()

        tokenized_train, tokenized_val = app.tokenize_datasets(
            train_dataset,
            val_dataset,
            tokenizer,
            max_length=3,
        )

        self.assertIn("input_ids", tokenized_train.column_names)
        self.assertIn("attention_mask", tokenized_val.column_names)
        self.assertEqual(tokenizer.padding, "max_length")
        self.assertTrue(tokenizer.truncation)
        self.assertTrue(
            all(text.startswith("keyword: fire tweet:") for text in tokenizer.texts)
        )

    def test_compute_metrics_returns_accuracy_and_f1(self) -> None:
        pred = type(
            "Prediction",
            (),
            {
                "label_ids": np.array([0, 1, 1, 0]),
                "predictions": np.array(
                    [
                        [0.9, 0.1],
                        [0.2, 0.8],
                        [0.7, 0.3],
                        [0.6, 0.4],
                    ]
                ),
            },
        )()

        metrics = app.compute_metrics(pred)

        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)

    def test_find_best_accuracy_threshold_uses_probabilities(self) -> None:
        probabilities = np.array([0.2, 0.4, 0.6, 0.8])
        logits = np.log(np.column_stack([1 - probabilities, probabilities]))
        labels = np.array([0, 1, 1, 1])

        threshold_result = app.find_best_accuracy_threshold(
            logits,
            labels,
            thresholds=np.array([0.3, 0.5]),
        )

        self.assertEqual(threshold_result["threshold"], 0.3)
        self.assertEqual(threshold_result["accuracy"], 1.0)

    def test_threshold_sweep_returns_one_row_per_threshold(self) -> None:
        probabilities = np.array([0.2, 0.4, 0.6, 0.8])
        labels = np.array([0, 1, 1, 1])

        sweep = app.threshold_sweep(probabilities, labels, np.array([0.3, 0.5]))

        self.assertEqual(sweep["threshold"].tolist(), [0.3, 0.5])
        self.assertEqual(sweep.loc[0, "accuracy"], 1.0)

    def test_train_model_builds_trainer_without_training_when_disabled(self) -> None:
        df = pd.DataFrame(
            {
                "id": range(10),
                "keyword": ["fire"] * 10,
                "location": [""] * 10,
                "text": [f"tweet {idx}" for idx in range(10)],
                "target": [0, 1] * 5,
            }
        )
        train_dataset, val_dataset = app.prepare_datasets(df, test_size=0.2)
        tokenized_train, tokenized_val = app.tokenize_datasets(
            train_dataset,
            val_dataset,
            FakeTokenizer(),
            max_length=3,
        )
        model = torch.nn.Linear(3, 2)

        with patch.object(
            app.AutoModelForSequenceClassification,
            "from_pretrained",
            return_value=model,
        ):
            trainer = app.train_model(
                tokenized_train,
                tokenized_val,
                run_training=False,
                output_dir="/tmp/disaster-tweets-test-results",
                model_name="unit-test-model",
                seed=7,
                warmup_ratio=0.06,
                eval_strategy="steps",
                save_strategy="steps",
                eval_steps=100,
                save_steps=100,
            )

        self.assertIsInstance(trainer, Trainer)
        self.assertIs(trainer.model, model)
        self.assertEqual(trainer.args.metric_for_best_model, "accuracy")
        self.assertEqual(trainer.args.seed, 7)
        self.assertEqual(trainer.args.warmup_steps, 1)
        self.assertEqual(trainer.args.eval_strategy.value, "steps")
        self.assertEqual(trainer.args.eval_steps, 100)
        self.assertTrue(trainer.args.save_only_model)


if __name__ == "__main__":
    unittest.main()
