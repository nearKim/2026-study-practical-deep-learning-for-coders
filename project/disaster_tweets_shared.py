from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from project.experiment_runner import ExperimentConfig
from project.nlp_disaster_tweets import (
    DEFAULT_TEXT_VARIANT,
    MODEL_NAME,
    threshold_metrics,
    threshold_sweep,
)


DATA_DIR = Path("project/data")
EXPERIMENTS_DIR = Path("project/experiments")


def baseline_config(
    *,
    name: str = "baseline_keyword_distilroberta",
    model_name: str = MODEL_NAME,
    learning_rate: float = 2e-5,
    seed: int = 42,
    split_version: str = "v1",
) -> ExperimentConfig:
    """
    Build the default disaster-tweet experiment config used by simple/complex runs.
    """
    return ExperimentConfig(
        name=name,
        model_name=model_name,
        text_variant=DEFAULT_TEXT_VARIANT,
        clean_text=False,
        curate_training_duplicates=True,
        max_length=128,
        learning_rate=learning_rate,
        num_train_epochs=3,
        train_batch_size=16,
        eval_batch_size=16,
        weight_decay=0.01,
        seed=seed,
        split_version=split_version,
        warmup_ratio=0.06,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        gradient_accumulation_steps=1,
        use_dynamic_padding=True,
    )


def write_config(path: Path, config: ExperimentConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(asdict(config), file, indent=2, sort_keys=True)


def read_run_metrics(run_dir: Path) -> dict[str, Any]:
    with (run_dir / "metrics.json").open() as file:
        return json.load(file)


def prediction_column(index: int) -> str:
    return f"prob_1_model_{index}"


def align_labeled_prediction_files(
    run_dirs: list[Path],
    file_name: str,
) -> pd.DataFrame:
    """
    Align labeled prediction files by `id` and `target` before ensembling.
    """
    if not run_dirs:
        raise ValueError("At least one run directory is required.")

    merged: pd.DataFrame | None = None
    for index, run_dir in enumerate(run_dirs):
        frame = pd.read_csv(run_dir / file_name)
        required_columns = {"id", "target", "prob_1"}
        missing_columns = required_columns.difference(frame.columns)
        if missing_columns:
            raise ValueError(f"{run_dir / file_name} missing {sorted(missing_columns)}")

        model_frame = frame[["id", "target", "prob_1"]].rename(
            columns={"prob_1": prediction_column(index)}
        )
        if merged is None:
            merged = model_frame
        else:
            expected_rows = len(merged)
            merged = merged.merge(model_frame, on=["id", "target"], validate="one_to_one")
            if len(merged) != expected_rows or len(model_frame) != expected_rows:
                raise ValueError("Prediction files must contain the same id and target rows.")

    if merged is None:
        raise ValueError("No prediction files were loaded.")
    return merged


def align_unlabeled_prediction_files(
    run_dirs: list[Path],
    file_name: str = "test_predictions.csv",
) -> pd.DataFrame:
    """
    Align test prediction files by `id`; test has no label column.
    """
    if not run_dirs:
        raise ValueError("At least one run directory is required.")

    merged: pd.DataFrame | None = None
    for index, run_dir in enumerate(run_dirs):
        frame = pd.read_csv(run_dir / file_name)
        required_columns = {"id", "prob_1"}
        missing_columns = required_columns.difference(frame.columns)
        if missing_columns:
            raise ValueError(f"{run_dir / file_name} missing {sorted(missing_columns)}")

        model_frame = frame[["id", "prob_1"]].rename(
            columns={"prob_1": prediction_column(index)}
        )
        if merged is None:
            merged = model_frame
        else:
            expected_rows = len(merged)
            merged = merged.merge(model_frame, on="id", validate="one_to_one")
            if len(merged) != expected_rows or len(model_frame) != expected_rows:
                raise ValueError("Prediction files must contain the same ids.")

    if merged is None:
        raise ValueError("No prediction files were loaded.")
    return merged


def weighted_probabilities(aligned_predictions: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    probability_columns = [prediction_column(index) for index in range(len(weights))]
    missing_columns = set(probability_columns).difference(aligned_predictions.columns)
    if missing_columns:
        raise ValueError(f"Missing probability columns: {sorted(missing_columns)}")
    return aligned_predictions[probability_columns].to_numpy() @ weights


def best_threshold_for_accuracy(
    probabilities: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, float]:
    """
    Select threshold on validation only; holdout must use this exact threshold.
    """
    sweep = threshold_sweep(probabilities, labels, thresholds)
    if sweep.empty:
        raise ValueError("At least one threshold is required.")
    return sweep.sort_values(["accuracy", "f1"], ascending=False).iloc[0].to_dict()


def labeled_prediction_output(
    aligned_predictions: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    output = aligned_predictions[["id", "target"]].copy()
    output["prob_1"] = probabilities
    output["pred_tuned"] = (probabilities >= threshold).astype(int)
    output["correct"] = output["pred_tuned"].eq(output["target"])
    return output


def unlabeled_submission_output(
    aligned_predictions: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": aligned_predictions["id"],
            "target": (probabilities >= threshold).astype(int),
        }
    )


def score_with_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    return threshold_metrics(probabilities, labels, threshold)
