from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import project.experiment_runner as runner


class ExperimentRunnerTest(unittest.TestCase):
    def sample_dataframe(self) -> pd.DataFrame:
        rows = []
        for group_idx in range(20):
            for copy_idx in range(2):
                rows.append(
                    {
                        "id": group_idx * 10 + copy_idx,
                        "keyword": "fire" if group_idx % 2 else "storm",
                        "location": "",
                        "text": f"duplicate tweet group {group_idx}",
                        "target": group_idx % 2,
                    }
                )
        return pd.DataFrame(rows)

    def test_create_or_load_split_is_deterministic_and_grouped(self) -> None:
        df = self.sample_dataframe()

        with tempfile.TemporaryDirectory() as tmp_dir:
            first_path = Path(tmp_dir) / "split_a.csv"
            second_path = Path(tmp_dir) / "split_b.csv"

            first = runner.create_or_load_split(df, first_path, seed=11)
            second = runner.create_or_load_split(df, second_path, seed=11)

        pd.testing.assert_frame_equal(first.sort_values("id"), second.sort_values("id"))
        self.assertEqual(set(first["split"]), {"train", "val", "holdout"})
        self.assertTrue((first.groupby("text_group")["split"].nunique() == 1).all())

    def test_evaluate_threshold_transfer_applies_validation_threshold_to_holdout(self) -> None:
        result = runner.evaluate_threshold_transfer(
            val_probabilities=np.array([0.2, 0.4, 0.6, 0.8]),
            val_labels=np.array([0, 1, 1, 1]),
            holdout_probabilities=np.array([0.2, 0.7]),
            holdout_labels=np.array([0, 1]),
            thresholds=np.array([0.3, 0.5]),
        )

        self.assertEqual(result["threshold"], 0.3)
        self.assertEqual(result["val"]["accuracy"], 1.0)
        self.assertEqual(result["holdout"]["threshold"], 0.3)
        self.assertEqual(result["holdout"]["accuracy"], 1.0)

    def test_save_predictions_writes_required_columns(self) -> None:
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "keyword": ["fire", "storm"],
                "text": ["smoke downtown", "cloudy day"],
                "target": [1, 0],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "predictions.csv"
            runner.save_predictions(path, df, np.array([0.7, 0.4]), threshold=0.6)
            saved = pd.read_csv(path)

        self.assertEqual(
            saved.columns.tolist(),
            [
                "id",
                "keyword",
                "text",
                "target",
                "prob_1",
                "pred_0_50",
                "pred_tuned",
                "correct",
            ],
        )
        self.assertEqual(saved["correct"].tolist(), [True, True])

    def test_append_leaderboard_preserves_existing_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "leaderboard.csv"
            runner.append_leaderboard(path, {"run_id": "one", "accuracy": 0.8})
            runner.append_leaderboard(path, {"run_id": "two", "accuracy": 0.9})
            saved = pd.read_csv(path)

        self.assertEqual(saved["run_id"].tolist(), ["one", "two"])
        self.assertEqual(saved["accuracy"].tolist(), [0.8, 0.9])

    def test_config_hash_changes_when_config_changes(self) -> None:
        baseline = runner.ExperimentConfig(name="baseline", learning_rate=2e-5)
        changed = runner.ExperimentConfig(name="baseline", learning_rate=3e-5)

        self.assertNotEqual(runner.config_hash(baseline), runner.config_hash(changed))


if __name__ == "__main__":
    unittest.main()
