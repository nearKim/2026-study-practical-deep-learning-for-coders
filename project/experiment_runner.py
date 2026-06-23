from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from project.nlp_disaster_tweets import (
    DEFAULT_TEXT_VARIANT,
    MODEL_NAME,
    add_model_text_features,
    dataframe_to_dataset,
    load_and_clean_data,
    normalize_tweet_text,
    positive_class_probabilities,
    resolve_duplicate_training_labels,
    threshold_metrics,
    threshold_sweep,
    tokenize_dataset,
    train_model,
)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "experiment"
    model_name: str = MODEL_NAME
    text_variant: str = DEFAULT_TEXT_VARIANT
    clean_text: bool = False
    curate_training_duplicates: bool = False
    max_length: int = 128
    learning_rate: float = 2e-5
    num_train_epochs: float = 3
    train_batch_size: int = 16
    eval_batch_size: int = 16
    weight_decay: float = 0.01
    seed: int = 42
    split_version: str = "v1"
    warmup_ratio: float = 0.06
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    eval_steps: int | None = 100
    save_steps: int | None = 100
    save_total_limit: int | None = 2
    save_only_model: bool = True
    gradient_accumulation_steps: int = 1
    use_dynamic_padding: bool = True
    train_csv: str = "train.csv"
    test_csv: str = "test.csv"


@dataclass(frozen=True)
class RunResult:
    run_id: str
    run_dir: Path
    metrics: dict[str, Any]


def load_config(path: Path) -> ExperimentConfig:
    with path.open() as file:
        config = json.load(file)
    return ExperimentConfig(**config)


def config_hash(config: ExperimentConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return sha1(payload.encode("utf-8")).hexdigest()[:8]


def build_run_id(config: ExperimentConfig, started_at: datetime | None = None) -> str:
    timestamp = started_at or datetime.now(timezone.utc)
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "-", config.name).strip("-")
    return f"{timestamp:%Y%m%d_%H%M%S}_{safe_name}_{config_hash(config)}"


def make_text_group(text: str) -> str:
    # Grouping by normalized text prevents the same duplicated tweet from
    # appearing in both train and validation, which would leak signal.
    return normalize_tweet_text(text).lower()


def create_or_load_split(df: pd.DataFrame, path: Path, seed: int) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)

    required_columns = {"id", "text", "target"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing split columns: {sorted(missing_columns)}")

    split_df = df[["id", "text", "target"]].copy()
    split_df["text_group"] = split_df["text"].map(make_text_group)
    group_targets = (
        split_df.groupby("text_group", as_index=False)["target"]
        .agg(lambda labels: int(labels.mode().iloc[0]))
        .rename(columns={"target": "group_target"})
    )

    train_groups, temp_groups = _train_test_split_groups(
        group_targets,
        test_size=0.30,
        seed=seed,
    )
    val_groups, holdout_groups = _train_test_split_groups(
        temp_groups,
        test_size=0.50,
        seed=seed,
    )

    split_lookup = {
        **{group: "train" for group in train_groups["text_group"]},
        **{group: "val" for group in val_groups["text_group"]},
        **{group: "holdout" for group in holdout_groups["text_group"]},
    }
    output = split_df[["id", "text_group"]].copy()
    output["split"] = output["text_group"].map(split_lookup)
    output = output[["id", "split", "text_group"]].sort_values("id").reset_index(drop=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    return output


def _train_test_split_groups(
    group_df: pd.DataFrame,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = _safe_stratify(group_df["group_target"])
    train_groups, test_groups = train_test_split(
        group_df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    return train_groups.reset_index(drop=True), test_groups.reset_index(drop=True)


def _safe_stratify(labels: pd.Series) -> pd.Series | None:
    counts = labels.value_counts()
    return labels if len(counts) > 1 and counts.min() >= 2 else None


def evaluate_threshold_transfer(
    val_probabilities: np.ndarray,
    val_labels: np.ndarray,
    holdout_probabilities: np.ndarray,
    holdout_labels: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    if thresholds is None:
        thresholds = np.round(np.arange(0.40, 0.805, 0.005), 3)

    sweep = threshold_sweep(val_probabilities, val_labels, thresholds)
    if sweep.empty:
        raise ValueError("At least one threshold is required.")

    best_val = sweep.sort_values(["accuracy", "f1"], ascending=False).iloc[0].to_dict()
    threshold = float(best_val["threshold"])
    holdout = threshold_metrics(holdout_probabilities, holdout_labels, threshold)
    return {"threshold": threshold, "val": best_val, "holdout": holdout}


def save_predictions(
    path: Path,
    df: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> None:
    if len(df) != len(probabilities):
        raise ValueError("df and probabilities must have the same length.")

    columns = ["id", "keyword", "text"]
    if "target" in df.columns:
        columns.append("target")

    output = df[columns].copy()
    output["prob_1"] = probabilities
    output["pred_0_50"] = (probabilities >= 0.50).astype(int)
    output["pred_tuned"] = (probabilities >= threshold).astype(int)
    if "target" in output.columns:
        output["correct"] = output["pred_tuned"].eq(output["target"])

    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)


def append_leaderboard(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([row])
    if path.exists():
        leaderboard = pd.concat([pd.read_csv(path), new_row], ignore_index=True)
    else:
        leaderboard = new_row
    leaderboard.to_csv(path, index=False)


def run_experiment(
    config: ExperimentConfig,
    data_dir: Path = Path("project/data"),
    experiments_dir: Path = Path("project/experiments"),
    run_training: bool = True,
) -> RunResult:
    run_id = build_run_id(config)
    run_dir = experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    train_df = load_and_clean_data(str(data_dir / config.train_csv), config.clean_text)
    split_path = experiments_dir / "splits" / f"split_{config.split_version}.csv"
    split_df = create_or_load_split(train_df, split_path, config.seed)
    split_train_df, split_val_df, split_holdout_df = _apply_split(train_df, split_df)

    if config.curate_training_duplicates:
        split_train_df = resolve_duplicate_training_labels(split_train_df)

    split_train_df = add_model_text_features(split_train_df, config.text_variant)
    split_val_df = add_model_text_features(split_val_df, config.text_variant)
    split_holdout_df = add_model_text_features(split_holdout_df, config.text_variant)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    padding = False if config.use_dynamic_padding else "max_length"
    tokenized_train = tokenize_dataset(
        dataframe_to_dataset(split_train_df),
        tokenizer,
        max_length=config.max_length,
        padding=padding,
    )
    tokenized_val = tokenize_dataset(
        dataframe_to_dataset(split_val_df),
        tokenizer,
        max_length=config.max_length,
        padding=padding,
    )
    tokenized_holdout = tokenize_dataset(
        dataframe_to_dataset(split_holdout_df),
        tokenizer,
        max_length=config.max_length,
        padding=padding,
    )

    trainer = train_model(
        tokenized_train,
        tokenized_val,
        output_dir=str(run_dir / "checkpoints"),
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        weight_decay=config.weight_decay,
        run_training=run_training,
        model_name=config.model_name,
        seed=config.seed,
        warmup_ratio=config.warmup_ratio,
        eval_strategy=config.eval_strategy,
        save_strategy=config.save_strategy,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        save_only_model=config.save_only_model,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        tokenizer=tokenizer,
        use_dynamic_padding=config.use_dynamic_padding,
    )

    val_predictions = trainer.predict(tokenized_val)
    holdout_predictions = trainer.predict(tokenized_holdout)
    val_probabilities = positive_class_probabilities(val_predictions.predictions)
    holdout_probabilities = positive_class_probabilities(holdout_predictions.predictions)
    threshold_result = evaluate_threshold_transfer(
        val_probabilities,
        val_predictions.label_ids,
        holdout_probabilities,
        holdout_predictions.label_ids,
    )

    threshold = threshold_result["threshold"]
    threshold_sweep(
        val_probabilities,
        val_predictions.label_ids,
        np.round(np.arange(0.40, 0.805, 0.005), 3),
    ).to_csv(run_dir / "thresholds.csv", index=False)
    save_predictions(run_dir / "val_predictions.csv", split_val_df, val_probabilities, threshold)
    save_predictions(
        run_dir / "holdout_predictions.csv",
        split_holdout_df,
        holdout_probabilities,
        threshold,
    )
    _save_test_predictions(config, data_dir, run_dir, trainer, tokenizer, padding, threshold)

    metrics = {
        "run_id": run_id,
        "config": asdict(config),
        "threshold": threshold,
        "val": threshold_result["val"],
        "holdout": threshold_result["holdout"],
        "row_counts": {
            "train": len(split_train_df),
            "val": len(split_val_df),
            "holdout": len(split_holdout_df),
        },
    }
    _write_json(run_dir / "config.json", asdict(config))
    _write_json(run_dir / "metrics.json", metrics)
    pd.DataFrame(trainer.state.log_history).to_json(
        run_dir / "trainer_log_history.json",
        orient="records",
        indent=2,
    )

    leaderboard_row = {
        "run_id": run_id,
        "config_name": config.name,
        "split_version": config.split_version,
        "seed": config.seed,
        "model_name": config.model_name,
        "text_variant": config.text_variant,
        "val_acc_tuned": threshold_result["val"]["accuracy"],
        "holdout_acc_tuned": threshold_result["holdout"]["accuracy"],
        "best_threshold": threshold,
        "notes": "",
    }
    append_leaderboard(experiments_dir / "leaderboard.csv", leaderboard_row)
    return RunResult(run_id=run_id, run_dir=run_dir, metrics=metrics)


def compare_runs(leaderboard_path: Path = Path("project/experiments/leaderboard.csv")) -> pd.DataFrame:
    leaderboard = pd.read_csv(leaderboard_path)
    sort_columns = [column for column in ["holdout_acc_tuned", "val_acc_tuned"] if column in leaderboard]
    return leaderboard.sort_values(sort_columns, ascending=False) if sort_columns else leaderboard


def _apply_split(
    df: pd.DataFrame,
    split_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = df.merge(split_df[["id", "split"]], on="id", how="inner")
    train_df = merged.loc[merged["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
    val_df = merged.loc[merged["split"] == "val"].drop(columns=["split"]).reset_index(drop=True)
    holdout_df = merged.loc[merged["split"] == "holdout"].drop(columns=["split"]).reset_index(drop=True)
    return train_df, val_df, holdout_df


def _save_test_predictions(
    config: ExperimentConfig,
    data_dir: Path,
    run_dir: Path,
    trainer: Any,
    tokenizer: Any,
    padding: bool | str,
    threshold: float,
) -> None:
    test_path = data_dir / config.test_csv
    if not test_path.exists():
        return

    test_df = load_and_clean_data(str(test_path), config.clean_text)
    test_df = add_model_text_features(test_df, config.text_variant)
    tokenized_test = tokenize_dataset(
        dataframe_to_dataset(test_df),
        tokenizer,
        max_length=config.max_length,
        padding=padding,
    )
    test_predictions = trainer.predict(tokenized_test)
    probabilities = positive_class_probabilities(test_predictions.predictions)
    save_predictions(run_dir / "test_predictions.csv", test_df, probabilities, threshold)

    submission = pd.DataFrame(
        {
            "id": test_df["id"],
            "target": (probabilities >= threshold).astype(int),
        }
    )
    submission.to_csv(run_dir / "submission.csv", index=False)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run disaster-tweet experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--data-dir", type=Path, default=Path("project/data"))
    run_parser.add_argument("--experiments-dir", type=Path, default=Path("project/experiments"))
    run_parser.add_argument("--no-train", action="store_true")

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument(
        "--leaderboard",
        type=Path,
        default=Path("project/experiments/leaderboard.csv"),
    )

    args = parser.parse_args()
    if args.command == "run":
        result = run_experiment(
            load_config(args.config),
            data_dir=args.data_dir,
            experiments_dir=args.experiments_dir,
            run_training=not args.no_train,
        )
        print(result.run_dir)
    elif args.command == "compare":
        print(compare_runs(args.leaderboard).to_string(index=False))


if __name__ == "__main__":
    main()
