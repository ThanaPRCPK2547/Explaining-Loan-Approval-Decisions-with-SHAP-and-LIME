"""Tests for the explainability module."""

from pathlib import Path

import joblib
import matplotlib
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.explain import run_explanations


matplotlib.use("Agg")
pytest.importorskip("shap")
pytest.importorskip("lime")


def build_pipeline() -> Pipeline:
    numeric_features = ["age", "income"]
    categorical_features = ["employment_type"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    estimator = LogisticRegression(max_iter=100)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def test_run_explanations_creates_outputs(tmp_path: Path, monkeypatch) -> None:
    processed_dir = tmp_path / "processed"
    artifacts_dir = tmp_path / "artifacts"
    output_dir = tmp_path / "reports"
    processed_dir.mkdir()
    artifacts_dir.mkdir()

    data = pd.DataFrame(
        {
            "age": [25, 32, 40, 45],
            "income": [30000, 45000, 52000, 61000],
            "employment_type": ["salaried", "self", "salaried", "self"],
            "default_status": [0, 1, 0, 1],
        }
    )
    validation = data.copy()
    validation.to_parquet(processed_dir / "validation.parquet", index=False)

    pipeline = build_pipeline()
    pipeline.fit(data.drop(columns=["default_status"]), data["default_status"])

    model_path = artifacts_dir / "logistic_regression_pipeline.joblib"
    joblib.dump(pipeline, model_path)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
data:
  raw_dir: data
  interim_dir: data/interim
  processed_dir: "{processed_dir}"
  application_filename: application_record.csv
  credit_filename: credit_record.csv
  output_dataset: processed_credit.parquet

preprocessing:
  id_column: ID
  target_column: default_status
  test_size: 0.2
  validation_size: 0.1
  random_state: 0

modeling:
  algorithms:
    - logistic_regression
  scoring: roc_auc

artifacts_dir: "{artifacts_dir}"

explainability:
  shap:
    sample_size: 2
  lime:
    sample_size: 1
    num_features: 2
  output_dir: "{output_dir}"
""",
        encoding="utf-8",
    )

    run_explanations(config_path=config_path, model_names=["logistic_regression"])

    shap_output = output_dir / "logistic_regression_shap_summary.png"
    lime_dir = output_dir / "logistic_regression_lime"

    assert shap_output.exists()
    assert lime_dir.exists()
    assert any(lime_dir.iterdir())
