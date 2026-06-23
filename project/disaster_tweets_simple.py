from __future__ import annotations

import argparse
from pathlib import Path

from project.disaster_tweets_shared import DATA_DIR, EXPERIMENTS_DIR, baseline_config
from project.experiment_runner import ExperimentConfig, run_experiment


def build_simple_config() -> ExperimentConfig:
    """
    Return the readable baseline: one model, one split, one threshold.
    """
    # This is the teaching version. It uses the proven keyword-augmented
    # DistilRoBERTa setup before the complex file tries model sweeps/ensembles.
    return baseline_config()


def run_simple_experiment(
    *,
    data_dir: Path = DATA_DIR,
    experiments_dir: Path = EXPERIMENTS_DIR,
    run_training: bool = True,
) -> Path:
    result = run_experiment(
        build_simple_config(),
        data_dir=data_dir,
        experiments_dir=experiments_dir,
        run_training=run_training,
    )
    return result.run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the simple disaster-tweet baseline.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--experiments-dir", type=Path, default=EXPERIMENTS_DIR)
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Build the Trainer and artifacts path without training.",
    )
    args = parser.parse_args()

    run_dir = run_simple_experiment(
        data_dir=args.data_dir,
        experiments_dir=args.experiments_dir,
        run_training=not args.no_train,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
