"""Tests for the model training module."""

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from src.models.train import run_training


def create_sample_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25, 32, 40, 45, 28, 50, 37, 29],
            "income": [30000, 45000, 52000, 61000, 34000, 72000, 48000, 36000],
            "employment_type": ["salaried", "self", "salaried", "self", "salaried", "self", "salaried", "self"],
            "default_status": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


def test_run_training_creates_artifacts(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    dataset = create_sample_dataset()

    train = dataset.iloc[:6]
    validation = dataset.iloc[6:]
    test = dataset.iloc[6:]

    train.to_parquet(processed_dir / "train.parquet", index=False)
    validation.to_parquet(processed_dir / "validation.parquet", index=False)
    test.to_parquet(processed_dir / "test.parquet", index=False)

    artifacts_dir = tmp_path / "artifacts"

    config_content = {
        "data": {
            "raw_dir": "data",
            "interim_dir": "data/interim",
            "processed_dir": str(processed_dir),
            "application_filename": "application_record.csv",
            "credit_filename": "credit_record.csv",
            "output_dataset": "processed_credit.parquet",
        },
        "preprocessing": {
            "id_column": "ID",
            "target_column": "default_status",
            "test_size": 0.2,
            "validation_size": 0.1,
            "random_state": 42,
            "balance_method": "none",
        },
        "modeling": {
            "algorithms": ["logistic_regression"],
            "scoring": "roc_auc",
        },
        "artifacts_dir": str(artifacts_dir),
    }

    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(config_content), encoding="utf-8")

    run_training(config_path=config_path)

    model_path = artifacts_dir / "logistic_regression_pipeline.joblib"
    metrics_path = artifacts_dir / "logistic_regression_metrics.json"

    assert model_path.exists()
    assert metrics_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "accuracy" in metrics
