from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from project.disaster_tweets_shared import (
    DATA_DIR,
    EXPERIMENTS_DIR,
    align_labeled_prediction_files,
    align_unlabeled_prediction_files,
    baseline_config,
    labeled_prediction_output,
    score_with_threshold,
    unlabeled_submission_output,
    weighted_probabilities,
    write_config,
)
from project.experiment_runner import (
    ExperimentConfig,
    append_leaderboard,
    create_or_load_split,
    evaluate_threshold_transfer,
    run_experiment,
)
from project.nlp_disaster_tweets import (
    load_and_clean_data,
    normalize_tweet_text,
    resolve_duplicate_training_labels,
    threshold_sweep,
)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    model_name: str
    learning_rates: tuple[float, ...]
    seeds: tuple[int, ...]
    train_batch_size: int = 16
    eval_batch_size: int = 16
    clean_text: bool = False


@dataclass(frozen=True)
class EnsembleSelection:
    run_dirs: list[Path]
    weights: list[float]
    threshold: float
    val: dict[str, float]
    holdout: dict[str, float]


@dataclass(frozen=True)
class TfidfSpec:
    name: str = "tfidf_keyword_location_logreg"
    text_variant: str = "keyword_location_flag"
    c: float = 1.0
    class_weight: str | None = None
    seed: int = 42
    split_version: str = "v1"
    train_csv: str = "train.csv"
    test_csv: str = "test.csv"
    curate_training_duplicates: bool = True


BEST_TFIDF_SPEC = TfidfSpec()


def candidate_specs(
    *,
    include_bertweet: bool = False,
    quick: bool = False,
) -> list[CandidateSpec]:
    """
    Return the model search space ordered by expected return on training time.
    """
    # The complex approach is not "train longer." It changes model family,
    # learning rate, seed, and final probability averaging.
    seeds = (42,) if quick else (13, 42, 777)
    specs = [
        CandidateSpec(
            name="distilroberta",
            model_name="distilroberta-base",
            learning_rates=(2e-5,) if quick else (1e-5, 2e-5, 3e-5),
            seeds=seeds,
        ),
        CandidateSpec(
            name="twitter-roberta",
            model_name="cardiffnlp/twitter-roberta-base",
            learning_rates=(2e-5,) if quick else (1e-5, 2e-5, 3e-5),
            seeds=seeds,
        ),
        CandidateSpec(
            name="deberta-v3",
            model_name="microsoft/deberta-v3-base",
            learning_rates=(1.5e-5,) if quick else (1e-5, 1.5e-5, 2e-5),
            seeds=seeds,
            train_batch_size=8,
            eval_batch_size=8,
        ),
    ]

    if include_bertweet:
        specs.append(
            CandidateSpec(
                name="bertweet",
                model_name="vinai/bertweet-base",
                learning_rates=(2e-5,) if quick else (1e-5, 2e-5, 3e-5),
                seeds=seeds,
            )
        )

    return specs


def build_candidate_configs(
    *,
    include_bertweet: bool = False,
    quick: bool = False,
    split_version: str = "v1",
) -> list[ExperimentConfig]:
    configs = []
    for spec in candidate_specs(include_bertweet=include_bertweet, quick=quick):
        for learning_rate in spec.learning_rates:
            for seed in spec.seeds:
                config = replace(
                    baseline_config(
                        name=f"{spec.name}_lr{learning_rate:g}_seed{seed}",
                        model_name=spec.model_name,
                        learning_rate=learning_rate,
                        seed=seed,
                        split_version=split_version,
                    ),
                    clean_text=spec.clean_text,
                    train_batch_size=spec.train_batch_size,
                    eval_batch_size=spec.eval_batch_size,
                )
                configs.append(config)
    return configs


def write_candidate_configs(
    configs: list[ExperimentConfig],
    output_dir: Path,
) -> list[Path]:
    paths = []
    for config in configs:
        path = output_dir / f"{config.name}.json"
        write_config(path, config)
        paths.append(path)
    return paths


def run_candidate_search(
    configs: list[ExperimentConfig],
    *,
    data_dir: Path = DATA_DIR,
    experiments_dir: Path = EXPERIMENTS_DIR,
    max_runs: int | None = None,
) -> list[Path]:
    """
    Train candidates sequentially and return their artifact directories.
    """
    selected_configs = configs[:max_runs] if max_runs is not None else configs
    run_dirs = []
    for config in selected_configs:
        result = run_experiment(
            config,
            data_dir=data_dir,
            experiments_dir=experiments_dir,
            run_training=True,
        )
        run_dirs.append(result.run_dir)
    return run_dirs


def build_tfidf_model_text(
    df: pd.DataFrame,
    text_variant: str = BEST_TFIDF_SPEC.text_variant,
) -> pd.Series:
    """
    Build the sparse-model text used by the best observed hybrid candidate.
    """
    if text_variant != "keyword_location_flag":
        raise ValueError(f"Unknown TF-IDF text_variant: {text_variant}")

    # The sparse model benefits from explicit lexical fields. The Transformer
    # can infer context from token order; TF-IDF cannot, so we expose the
    # keyword and whether a user supplied location as plain tokens.
    keywords = (
        df["keyword"]
        .fillna("none")
        .astype(str)
        .map(lambda keyword: unquote(keyword).strip() or "none")
    )
    location_present = df["location"].fillna("").astype(str).str.strip().ne("")
    location_flags = pd.Series(
        np.where(location_present, "location_present", "location_missing"),
        index=df.index,
    )
    tweets = df["text"].astype(str).map(normalize_tweet_text)
    return "keyword: " + keywords + " " + location_flags + " tweet: " + tweets


def build_tfidf_pipeline(spec: TfidfSpec = BEST_TFIDF_SPEC) -> Pipeline:
    """
    Return the best sparse lexical model from the local validation loop.
    """
    # Word n-grams catch disaster phrases such as "forest fire"; character
    # n-grams catch misspellings, hashtags, and compact social-media wording.
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
        ]
    )
    classifier = LogisticRegression(
        C=spec.c,
        class_weight=spec.class_weight,
        max_iter=3000,
        random_state=spec.seed,
        solver="liblinear",
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def run_tfidf_experiment(
    spec: TfidfSpec = BEST_TFIDF_SPEC,
    *,
    data_dir: Path = DATA_DIR,
    experiments_dir: Path = EXPERIMENTS_DIR,
) -> Path:
    """
    Train the best sparse model and write Transformer-compatible artifacts.
    """
    run_id = _build_tfidf_run_id(spec)
    run_dir = experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    train_df = load_and_clean_data(str(data_dir / spec.train_csv), clean_text=False)
    split_path = experiments_dir / "splits" / f"split_{spec.split_version}.csv"
    split_df = create_or_load_split(train_df, split_path, spec.seed)
    split_train_df, split_val_df, split_holdout_df = _apply_split(train_df, split_df)

    if spec.curate_training_duplicates:
        split_train_df = resolve_duplicate_training_labels(split_train_df)

    model = build_tfidf_pipeline(spec)
    train_text = build_tfidf_model_text(split_train_df, spec.text_variant)
    model.fit(train_text, split_train_df["target"].astype(int))

    val_probabilities = _predict_tfidf_probabilities(model, split_val_df, spec)
    holdout_probabilities = _predict_tfidf_probabilities(model, split_holdout_df, spec)
    threshold_result = evaluate_threshold_transfer(
        val_probabilities,
        split_val_df["target"].to_numpy(),
        holdout_probabilities,
        split_holdout_df["target"].to_numpy(),
    )
    threshold = threshold_result["threshold"]

    thresholds = np.round(np.arange(0.40, 0.805, 0.005), 3)
    threshold_sweep(
        val_probabilities,
        split_val_df["target"].to_numpy(),
        thresholds,
    ).to_csv(run_dir / "thresholds.csv", index=False)
    save_tfidf_predictions(run_dir / "val_predictions.csv", split_val_df, val_probabilities, threshold)
    save_tfidf_predictions(
        run_dir / "holdout_predictions.csv",
        split_holdout_df,
        holdout_probabilities,
        threshold,
    )
    _save_tfidf_test_predictions(spec, data_dir, run_dir, model, threshold)

    metrics = {
        "run_id": run_id,
        "model_name": "tfidf_logistic_regression",
        "config": asdict(spec),
        "threshold": threshold,
        "val": threshold_result["val"],
        "holdout": threshold_result["holdout"],
        "row_counts": {
            "train": len(split_train_df),
            "val": len(split_val_df),
            "holdout": len(split_holdout_df),
        },
    }
    _write_json(run_dir / "config.json", asdict(spec))
    _write_json(run_dir / "metrics.json", metrics)
    append_leaderboard(
        experiments_dir / "leaderboard.csv",
        {
            "run_id": run_id,
            "config_name": spec.name,
            "split_version": spec.split_version,
            "seed": spec.seed,
            "model_name": "tfidf_logistic_regression",
            "text_variant": spec.text_variant,
            "val_acc_tuned": threshold_result["val"]["accuracy"],
            "holdout_acc_tuned": threshold_result["holdout"]["accuracy"],
            "best_threshold": threshold,
            "notes": "best sparse lexical companion for transformer ensemble",
        },
    )
    return run_dir


def save_tfidf_predictions(
    path: Path,
    df: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> None:
    if len(df) != len(probabilities):
        raise ValueError("df and probabilities must have the same length.")

    columns = [column for column in ["id", "keyword", "location", "text", "target"] if column in df]
    output = df[columns].copy()
    output["prob_1"] = probabilities
    output["pred_0_50"] = (probabilities >= 0.50).astype(int)
    output["pred_tuned"] = (probabilities >= threshold).astype(int)
    if "target" in output.columns:
        output["correct"] = output["pred_tuned"].eq(output["target"])
    output.to_csv(path, index=False)


def simplex_weight_grid(n_models: int, step: float = 0.25) -> list[np.ndarray]:
    """
    Generate non-negative ensemble weights that sum to 1.
    """
    if n_models <= 0:
        raise ValueError("n_models must be positive.")

    units_float = 1 / step
    units = round(units_float)
    if not np.isclose(units, units_float):
        raise ValueError("step must divide 1.0 exactly, for example 0.25 or 0.1.")

    grids: list[np.ndarray] = []

    def build(prefix: list[int], remaining: int, slots_left: int) -> None:
        if slots_left == 1:
            grids.append(np.array([*prefix, remaining], dtype=float) / units)
            return
        for value in range(remaining + 1):
            build([*prefix, value], remaining - value, slots_left - 1)

    build([], units, n_models)
    return grids


def select_best_ensemble(
    run_dirs: list[Path],
    *,
    weight_step: float = 0.25,
    threshold_step: float = 0.005,
) -> EnsembleSelection:
    """
    Choose ensemble weights and threshold using validation data only.
    """
    val_predictions = align_labeled_prediction_files(run_dirs, "val_predictions.csv")
    holdout_predictions = align_labeled_prediction_files(run_dirs, "holdout_predictions.csv")
    thresholds = np.round(np.arange(0.40, 0.805, threshold_step), 3)

    best_selection: EnsembleSelection | None = None
    best_key: tuple[float, float] | None = None

    for weights in simplex_weight_grid(len(run_dirs), step=weight_step):
        val_probabilities = weighted_probabilities(val_predictions, weights)
        val_metrics = _best_threshold_for_accuracy_fast(
            val_probabilities,
            val_predictions["target"].to_numpy(),
            thresholds,
        )
        key = (val_metrics["accuracy"], val_metrics["f1"])

        if best_key is None or key > best_key:
            threshold = val_metrics["threshold"]
            holdout_probabilities = weighted_probabilities(holdout_predictions, weights)
            holdout_metrics = score_with_threshold(
                holdout_probabilities,
                holdout_predictions["target"].to_numpy(),
                threshold,
            )
            best_selection = EnsembleSelection(
                run_dirs=run_dirs,
                weights=weights.tolist(),
                threshold=threshold,
                val=val_metrics,
                holdout=holdout_metrics,
            )
            best_key = key

    if best_selection is None:
        raise ValueError("No ensemble candidate was evaluated.")
    return best_selection


def _best_threshold_for_accuracy_fast(
    probabilities: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, float]:
    if len(thresholds) == 0:
        raise ValueError("At least one threshold is required.")

    label_positive = labels == 1
    predictions = probabilities[:, np.newaxis] >= thresholds[np.newaxis, :]
    true_positive = np.logical_and(predictions, label_positive[:, np.newaxis]).sum(axis=0)
    false_positive = np.logical_and(predictions, ~label_positive[:, np.newaxis]).sum(axis=0)
    false_negative = np.logical_and(~predictions, label_positive[:, np.newaxis]).sum(axis=0)

    accuracy = (predictions == label_positive[:, np.newaxis]).mean(axis=0)
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2 * true_positive, 2 * true_positive + false_positive + false_negative)
    best_index = max(range(len(thresholds)), key=lambda index: (accuracy[index], f1[index]))

    return {
        "threshold": float(thresholds[best_index]),
        "accuracy": float(accuracy[best_index]),
        "f1": float(f1[best_index]),
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
    }


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


def write_ensemble_artifacts(
    selection: EnsembleSelection,
    output_dir: Path | None = None,
) -> Path:
    """
    Write ensemble metrics, validation predictions, holdout predictions, and submission.
    """
    run_dirs = selection.run_dirs
    if output_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = EXPERIMENTS_DIR / "ensembles" / f"{timestamp}_weighted_ensemble"
    output_dir.mkdir(parents=True, exist_ok=False)

    weights = np.array(selection.weights)
    val_predictions = align_labeled_prediction_files(run_dirs, "val_predictions.csv")
    holdout_predictions = align_labeled_prediction_files(run_dirs, "holdout_predictions.csv")
    val_probabilities = weighted_probabilities(val_predictions, weights)
    holdout_probabilities = weighted_probabilities(holdout_predictions, weights)

    labeled_prediction_output(
        val_predictions,
        val_probabilities,
        selection.threshold,
    ).to_csv(output_dir / "val_predictions.csv", index=False)
    labeled_prediction_output(
        holdout_predictions,
        holdout_probabilities,
        selection.threshold,
    ).to_csv(output_dir / "holdout_predictions.csv", index=False)

    if all((run_dir / "test_predictions.csv").exists() for run_dir in run_dirs):
        test_predictions = align_unlabeled_prediction_files(run_dirs)
        test_probabilities = weighted_probabilities(test_predictions, weights)
        test_predictions.assign(prob_1=test_probabilities).to_csv(
            output_dir / "test_predictions.csv",
            index=False,
        )
        unlabeled_submission_output(
            test_predictions,
            test_probabilities,
            selection.threshold,
        ).to_csv(output_dir / "submission.csv", index=False)

    metrics = {
        "run_dirs": [str(run_dir) for run_dir in run_dirs],
        "weights": selection.weights,
        "threshold": selection.threshold,
        "val": selection.val,
        "holdout": selection.holdout,
    }
    with (output_dir / "metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2, sort_keys=True)
    return output_dir


def plan_dataframe(configs: list[ExperimentConfig]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": config.name,
                "model_name": config.model_name,
                "learning_rate": config.learning_rate,
                "seed": config.seed,
                "batch_size": config.train_batch_size,
            }
            for config in configs
        ]
    )


def _apply_split(
    df: pd.DataFrame,
    split_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = df.merge(split_df[["id", "split"]], on="id", how="inner")
    train_df = merged.loc[merged["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
    val_df = merged.loc[merged["split"] == "val"].drop(columns=["split"]).reset_index(drop=True)
    holdout_df = merged.loc[merged["split"] == "holdout"].drop(columns=["split"]).reset_index(drop=True)
    return train_df, val_df, holdout_df


def _predict_tfidf_probabilities(
    model: Pipeline,
    df: pd.DataFrame,
    spec: TfidfSpec,
) -> np.ndarray:
    text = build_tfidf_model_text(df, spec.text_variant)
    classifier = model.named_steps["classifier"]
    class_matches = np.flatnonzero(classifier.classes_ == 1)
    if len(class_matches) != 1:
        raise ValueError("TF-IDF classifier must expose exactly one positive class.")
    return model.predict_proba(text)[:, int(class_matches[0])]


def _save_tfidf_test_predictions(
    spec: TfidfSpec,
    data_dir: Path,
    run_dir: Path,
    model: Pipeline,
    threshold: float,
) -> None:
    test_path = data_dir / spec.test_csv
    if not test_path.exists():
        return

    test_df = load_and_clean_data(str(test_path), clean_text=False)
    probabilities = _predict_tfidf_probabilities(model, test_df, spec)
    save_tfidf_predictions(run_dir / "test_predictions.csv", test_df, probabilities, threshold)
    unlabeled_submission_output(test_df[["id"]], probabilities, threshold).to_csv(
        run_dir / "submission.csv",
        index=False,
    )


def _build_tfidf_run_id(spec: TfidfSpec) -> str:
    payload = json.dumps(asdict(spec), sort_keys=True, separators=(",", ":"))
    config_hash = sha1(payload.encode("utf-8")).hexdigest()[:8]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{spec.name}_{config_hash}"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggressive disaster-tweet search.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--quick", action="store_true")
    plan_parser.add_argument("--include-bertweet", action="store_true")

    write_parser = subparsers.add_parser("write-configs")
    write_parser.add_argument("--quick", action="store_true")
    write_parser.add_argument("--include-bertweet", action="store_true")
    write_parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENTS_DIR / "configs" / "complex",
    )

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--quick", action="store_true")
    run_parser.add_argument("--include-bertweet", action="store_true")
    run_parser.add_argument("--max-runs", type=int)
    run_parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    run_parser.add_argument("--experiments-dir", type=Path, default=EXPERIMENTS_DIR)

    ensemble_parser = subparsers.add_parser("ensemble")
    ensemble_parser.add_argument("--run-dir", type=Path, action="append", required=True)
    ensemble_parser.add_argument("--output-dir", type=Path)
    ensemble_parser.add_argument("--weight-step", type=float, default=0.25)
    ensemble_parser.add_argument("--threshold-step", type=float, default=0.005)

    tfidf_parser = subparsers.add_parser("tfidf")
    tfidf_parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    tfidf_parser.add_argument("--experiments-dir", type=Path, default=EXPERIMENTS_DIR)
    tfidf_parser.add_argument("--split-version", default=BEST_TFIDF_SPEC.split_version)
    tfidf_parser.add_argument("--seed", type=int, default=BEST_TFIDF_SPEC.seed)

    args = parser.parse_args()
    if args.command == "plan":
        configs = build_candidate_configs(
            include_bertweet=args.include_bertweet,
            quick=args.quick,
        )
        print(plan_dataframe(configs).to_string(index=False))
    elif args.command == "write-configs":
        configs = build_candidate_configs(
            include_bertweet=args.include_bertweet,
            quick=args.quick,
        )
        for path in write_candidate_configs(configs, args.output_dir):
            print(path)
    elif args.command == "run":
        configs = build_candidate_configs(
            include_bertweet=args.include_bertweet,
            quick=args.quick,
        )
        for run_dir in run_candidate_search(
            configs,
            data_dir=args.data_dir,
            experiments_dir=args.experiments_dir,
            max_runs=args.max_runs,
        ):
            print(run_dir)
    elif args.command == "ensemble":
        selection = select_best_ensemble(
            args.run_dir,
            weight_step=args.weight_step,
            threshold_step=args.threshold_step,
        )
        output_dir = write_ensemble_artifacts(selection, args.output_dir)
        print(json.dumps(asdict(selection), indent=2, sort_keys=True, default=str))
        print(output_dir)
    elif args.command == "tfidf":
        spec = replace(
            BEST_TFIDF_SPEC,
            split_version=args.split_version,
            seed=args.seed,
        )
        print(
            run_tfidf_experiment(
                spec,
                data_dir=args.data_dir,
                experiments_dir=args.experiments_dir,
            )
        )


if __name__ == "__main__":
    main()
