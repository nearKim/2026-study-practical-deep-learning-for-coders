from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import project.disaster_tweets_complex as complex_app


class DisasterTweetsComplexTest(unittest.TestCase):
    def make_run_dir(
        self,
        root: Path,
        name: str,
        val_probabilities: list[float],
        holdout_probabilities: list[float],
        test_probabilities: list[float] | None = None,
    ) -> Path:
        run_dir = root / name
        run_dir.mkdir()
        val_targets = [0, 1, 1, 0]
        holdout_targets = [0, 1]

        pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "target": val_targets,
                "prob_1": val_probabilities,
            }
        ).to_csv(run_dir / "val_predictions.csv", index=False)
        pd.DataFrame(
            {
                "id": [5, 6],
                "target": holdout_targets,
                "prob_1": holdout_probabilities,
            }
        ).to_csv(run_dir / "holdout_predictions.csv", index=False)

        if test_probabilities is not None:
            pd.DataFrame(
                {
                    "id": [7, 8],
                    "prob_1": test_probabilities,
                }
            ).to_csv(run_dir / "test_predictions.csv", index=False)

        return run_dir

    def test_quick_candidate_configs_include_stronger_models(self) -> None:
        configs = complex_app.build_candidate_configs(quick=True)

        self.assertEqual(
            [config.model_name for config in configs],
            [
                "distilroberta-base",
                "cardiffnlp/twitter-roberta-base",
                "microsoft/deberta-v3-base",
            ],
        )

    def test_simplex_weight_grid_sums_to_one(self) -> None:
        weights = complex_app.simplex_weight_grid(n_models=3, step=0.5)

        self.assertTrue(all(np.isclose(weight.sum(), 1.0) for weight in weights))
        self.assertIn([0.0, 0.5, 0.5], [weight.tolist() for weight in weights])

    def test_select_best_ensemble_uses_validation_not_holdout_to_choose_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            strong_val = self.make_run_dir(
                root,
                "strong_val",
                val_probabilities=[0.1, 0.9, 0.8, 0.2],
                holdout_probabilities=[0.9, 0.1],
            )
            weak_val = self.make_run_dir(
                root,
                "weak_val",
                val_probabilities=[0.9, 0.1, 0.2, 0.8],
                holdout_probabilities=[0.1, 0.9],
            )

            selection = complex_app.select_best_ensemble(
                [strong_val, weak_val],
                weight_step=0.5,
                threshold_step=0.1,
            )

        self.assertEqual(selection.weights, [1.0, 0.0])
        self.assertEqual(selection.val["accuracy"], 1.0)
        self.assertEqual(selection.holdout["accuracy"], 0.0)

    def test_write_ensemble_artifacts_outputs_submission_when_test_predictions_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = self.make_run_dir(
                root,
                "single",
                val_probabilities=[0.1, 0.9, 0.8, 0.2],
                holdout_probabilities=[0.2, 0.7],
                test_probabilities=[0.2, 0.8],
            )
            selection = complex_app.select_best_ensemble(
                [run_dir],
                weight_step=1.0,
                threshold_step=0.1,
            )
            output_dir = complex_app.write_ensemble_artifacts(
                selection,
                output_dir=root / "ensemble",
            )

            submission = pd.read_csv(output_dir / "submission.csv")

        self.assertEqual(submission.columns.tolist(), ["id", "target"])
        self.assertEqual(submission["target"].tolist(), [0, 1])

    def test_tfidf_run_writes_ensemble_compatible_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_dir = root / "data"
            experiments_dir = root / "experiments"
            data_dir.mkdir()
            (experiments_dir / "splits").mkdir(parents=True)
            train_rows = pd.DataFrame(
                {
                    "id": [1, 2, 3, 4, 5, 6, 7, 8],
                    "keyword": ["fire", "fire", "music", "music", "fire", "music", "fire", "music"],
                    "location": ["NY", "", "LA", "", "NY", "", "LA", ""],
                    "text": [
                        "Fire damage near homes",
                        "Wild fire damage reported",
                        "Music concert tonight",
                        "New music concert announced",
                        "Fire damage warning",
                        "Music concert tickets",
                        "Fire damage spreads",
                        "Music concert review",
                    ],
                    "target": [1, 1, 0, 0, 1, 0, 1, 0],
                }
            )
            test_rows = train_rows.drop(columns=["target"]).assign(id=lambda df: df["id"] + 100)
            train_rows.to_csv(data_dir / "train.csv", index=False)
            test_rows.to_csv(data_dir / "test.csv", index=False)
            pd.DataFrame(
                {
                    "id": [1, 2, 3, 4, 5, 6, 7, 8],
                    "split": ["train", "train", "train", "train", "val", "val", "holdout", "holdout"],
                    "text_group": [f"group-{index}" for index in range(8)],
                }
            ).to_csv(experiments_dir / "splits" / "split_v1.csv", index=False)

            run_dir = complex_app.run_tfidf_experiment(
                data_dir=data_dir,
                experiments_dir=experiments_dir,
            )

            val_predictions = pd.read_csv(run_dir / "val_predictions.csv")
            holdout_predictions = pd.read_csv(run_dir / "holdout_predictions.csv")
            test_predictions = pd.read_csv(run_dir / "test_predictions.csv")
            metrics_exists = (run_dir / "metrics.json").exists()
            submission_exists = (run_dir / "submission.csv").exists()

            self.assertEqual(val_predictions["id"].tolist(), [5, 6])
            self.assertEqual(holdout_predictions["id"].tolist(), [7, 8])
            self.assertTrue(val_predictions["prob_1"].between(0, 1).all())
            self.assertTrue(test_predictions["prob_1"].between(0, 1).all())
            self.assertTrue(metrics_exists)
            self.assertTrue(submission_exists)

    def test_tfidf_probability_selects_positive_class_by_label(self) -> None:
        class FakeModel:
            named_steps = {"classifier": SimpleNamespace(classes_=np.array([1, 0]))}

            def predict_proba(self, text: pd.Series) -> np.ndarray:
                return np.array([[0.7, 0.3] for _ in text])

        df = pd.DataFrame(
            {
                "keyword": ["fire", "music"],
                "location": ["NY", ""],
                "text": ["fire damage", "music concert"],
            }
        )

        probabilities = complex_app._predict_tfidf_probabilities(
            FakeModel(),
            df,
            complex_app.BEST_TFIDF_SPEC,
        )

        np.testing.assert_array_equal(probabilities, np.array([0.7, 0.7]))

    def test_select_best_ensemble_rejects_mismatched_prediction_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            first = self.make_run_dir(
                root,
                "first",
                val_probabilities=[0.1, 0.9, 0.8, 0.2],
                holdout_probabilities=[0.2, 0.7],
            )
            second = self.make_run_dir(
                root,
                "second",
                val_probabilities=[0.1, 0.9, 0.8, 0.2],
                holdout_probabilities=[0.2, 0.7],
            )
            val_predictions = pd.read_csv(second / "val_predictions.csv")
            val_predictions.loc[0, "id"] = 99
            val_predictions.to_csv(second / "val_predictions.csv", index=False)

            with self.assertRaisesRegex(ValueError, "same id and target rows"):
                complex_app.select_best_ensemble([first, second])


if __name__ == "__main__":
    unittest.main()
