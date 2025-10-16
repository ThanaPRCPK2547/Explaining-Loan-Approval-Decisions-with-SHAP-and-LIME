"""Model training utilities for the loan approval project."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class TrainingConfig:
    processed_dir: Path
    target_column: str
    algorithms: List[str]
    scoring: str
    random_state: int
    balance_method: str
    artifacts_dir: Path


def load_training_config(config_path: Path) -> TrainingConfig:
    with config_path.open("r", encoding="utf-8") as config_file:
        config: Dict[str, Any] = yaml.safe_load(config_file)

    data_cfg = config["data"]
    prep_cfg = config["preprocessing"]
    model_cfg = config["modeling"]

    processed_dir = Path(data_cfg["processed_dir"])
    artifacts_dir = Path(config.get("artifacts_dir", "artifacts"))

    return TrainingConfig(
        processed_dir=processed_dir,
        target_column=prep_cfg["target_column"],
        algorithms=model_cfg["algorithms"],
        scoring=model_cfg.get("scoring", "roc_auc"),
        random_state=prep_cfg.get("random_state", 42),
        balance_method=prep_cfg.get("balance_method", "none"),
        artifacts_dir=artifacts_dir,
    )


def load_datasets(processed_dir: Path, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = processed_dir / "train.parquet"
    val_path = processed_dir / "validation.parquet"
    test_path = processed_dir / "test.parquet"

    if not train_path.exists():
        raise FileNotFoundError(f"Processed train dataset not found at {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Processed validation dataset not found at {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Processed test dataset not found at {test_path}")

    train = pd.read_parquet(train_path)
    validation = pd.read_parquet(val_path)
    test = pd.read_parquet(test_path)

    for dataset in (train, validation, test):
        if target_column not in dataset.columns:
            raise ValueError(f"Target column '{target_column}' missing from dataset.")

    return train, validation, test


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical = features.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = features.select_dtypes(exclude=["object", "category"]).columns.tolist()

    transformers = []

    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )

    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )

    if not transformers:
        raise ValueError("No valid features found for preprocessing.")

    return ColumnTransformer(transformers=transformers)


def create_estimator(
    algorithm: str,
    random_state: int,
) -> Any:
    if algorithm == "logistic_regression":
        return LogisticRegression(
            max_iter=200,
            class_weight="balanced",
            random_state=random_state,
        )
    if algorithm == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            class_weight="balanced",
        )
    raise ValueError(f"Unsupported algorithm '{algorithm}'")


def build_pipeline(
    estimator: Any,
    preprocessor: ColumnTransformer,
    balance_method: str,
    random_state: int,
) -> ImbPipeline:
    steps: List[Tuple[str, Any]] = [("preprocess", preprocessor)]

    if balance_method.lower() == "smote":
        steps.append(
            (
                "sampler",
                SMOTE(random_state=random_state),
            )
        )

    steps.append(("model", estimator))
    return ImbPipeline(steps=steps)


def evaluate_model(pipeline: ImbPipeline, features: pd.DataFrame, target: pd.Series) -> Dict[str, Any]:
    predictions = pipeline.predict(features)
    probs = None
    try:
        probs = pipeline.predict_proba(features)[:, 1]
    except AttributeError:
        pass

    metrics: Dict[str, Any] = {
        "accuracy": accuracy_score(target, predictions),
        "classification_report": classification_report(target, predictions, output_dict=True),
    }
    if probs is not None:
        metrics["roc_auc"] = roc_auc_score(target, probs)

    return metrics


def persist_artifacts(
    artifacts_dir: Path,
    algorithm_name: str,
    pipeline: ImbPipeline,
    metrics: Dict[str, Any],
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / f"{algorithm_name}_pipeline.joblib"
    metrics_path = artifacts_dir / f"{algorithm_name}_metrics.json"

    joblib.dump(pipeline, model_path)
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)


def run_training(config_path: Path) -> None:
    config = load_training_config(config_path)
    project_root = Path(__file__).resolve().parents[2]

    processed_dir = (project_root / config.processed_dir).resolve()
    artifacts_dir = (project_root / config.artifacts_dir).resolve()

    train_df, val_df, test_df = load_datasets(processed_dir, config.target_column)

    x_train = train_df.drop(columns=[config.target_column])
    y_train = train_df[config.target_column]

    x_val = val_df.drop(columns=[config.target_column])
    y_val = val_df[config.target_column]

    combined_features = pd.concat([x_train, x_val], axis=0).reset_index(drop=True)

    preprocessor = build_preprocessor(combined_features)

    for algorithm in config.algorithms:
        estimator = create_estimator(algorithm, config.random_state)
        pipeline = build_pipeline(
            estimator=estimator,
            preprocessor=preprocessor,
            balance_method=config.balance_method,
            random_state=config.random_state,
        )

        pipeline.fit(x_train, y_train)

        metrics = evaluate_model(pipeline, x_val, y_val)
        combined_features = pd.concat([x_train, x_val], axis=0).reset_index(drop=True)
        combined_target = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

        pipeline.fit(combined_features, combined_target)
        persist_artifacts(artifacts_dir, algorithm, pipeline, metrics)
