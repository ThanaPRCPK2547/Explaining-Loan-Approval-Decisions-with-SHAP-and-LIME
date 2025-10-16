"""Generate SHAP and LIME explanations for trained models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml
from lime.lime_tabular import LimeTabularExplainer
from scipy.sparse import issparse


@dataclass
class ExplainConfig:
    processed_dir: Path
    target_column: str
    artifacts_dir: Path
    shap_sample_size: int
    lime_sample_size: int
    lime_num_features: int
    lime_num_samples: int
    output_dir: Path
    random_state: int


def load_explain_config(config_path: Path) -> ExplainConfig:
    with config_path.open("r", encoding="utf-8") as file:
        raw_config: Dict[str, Any] = yaml.safe_load(file)

    data_cfg = raw_config["data"]
    prep_cfg = raw_config["preprocessing"]
    explain_cfg = raw_config.get("explainability", {})

    processed_dir = Path(data_cfg["processed_dir"])
    artifacts_dir = Path(raw_config.get("artifacts_dir", "artifacts/models"))

    shap_cfg = explain_cfg.get("shap", {})
    lime_cfg = explain_cfg.get("lime", {})

    return ExplainConfig(
        processed_dir=processed_dir,
        target_column=prep_cfg["target_column"],
        artifacts_dir=artifacts_dir,
        shap_sample_size=int(shap_cfg.get("sample_size", 200)),
        lime_sample_size=int(lime_cfg.get("sample_size", 5)),
        lime_num_features=int(lime_cfg.get("num_features", 10)),
        lime_num_samples=int(lime_cfg.get("num_samples", 1000)),
        output_dir=Path(lime_cfg.get("output_dir", explain_cfg.get("output_dir", "reports/explanations"))),
        random_state=int(prep_cfg.get("random_state", 42)),
    )


def load_validation_data(processed_dir: Path, target_column: str) -> pd.DataFrame:
    validation_path = processed_dir / "validation.parquet"
    if not validation_path.exists():
        raise FileNotFoundError(f"Validation dataset not found at {validation_path}")
    validation = pd.read_parquet(validation_path)
    if target_column not in validation:
        raise ValueError(f"Target column '{target_column}' missing from validation dataset.")
    return validation


def select_model_files(artifacts_dir: Path, model_names: Optional[Sequence[str]] = None) -> List[Path]:
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory not found at {artifacts_dir}")

    model_files = sorted(artifacts_dir.glob("*_pipeline.joblib"))
    if not model_files:
        raise FileNotFoundError(f"No model artifacts found in {artifacts_dir}")

    if model_names:
        normalized = {name.lower() for name in model_names}
        model_files = [
            path for path in model_files if path.stem.replace("_pipeline", "") in normalized
        ]

    if not model_files:
        raise ValueError("No matching model artifacts found for the specified names.")

    return model_files


def get_feature_names(preprocessor: Any, sample: pd.DataFrame) -> List[str]:
    if hasattr(preprocessor, "get_feature_names_out"):
        return list(preprocessor.get_feature_names_out())
    transformed = preprocessor.transform(sample)
    if issparse(transformed):
        transformed = transformed.toarray()
    return [f"feature_{i}" for i in range(transformed.shape[1])]


def compute_shap_values(
    pipeline: Any,
    X_sample: pd.DataFrame,
    feature_names: Sequence[str],
) -> Dict[str, Any]:
    estimator = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]
    X_processed = preprocessor.transform(X_sample)
    if issparse(X_processed):
        X_processed = X_processed.toarray()

    if estimator.__class__.__name__.lower().startswith("randomforest"):
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_processed)
        values = shap_values[1] if isinstance(shap_values, list) else shap_values
        base = (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, tuple, np.ndarray))
            else explainer.expected_value
        )
        return {"values": values, "data": X_processed, "base_values": base}

    if estimator.__class__.__name__.lower().startswith("logisticregression"):
        explainer = shap.LinearExplainer(estimator, X_processed)
        shap_values = explainer.shap_values(X_processed)
        return {"values": shap_values, "data": X_processed, "base_values": explainer.expected_value}

    # Fallback to model-agnostic explainer
    explainer = shap.KernelExplainer(
        lambda data: pipeline.predict_proba(pd.DataFrame(data, columns=X_sample.columns))[:, 1],
        shap.sample(X_sample, min(50, len(X_sample))),
    )
    shap_values = explainer.shap_values(X_sample, nsamples="auto")
    values = shap_values[1] if isinstance(shap_values, list) else shap_values
    return {"values": values, "data": X_sample.values, "base_values": explainer.expected_value}


def save_shap_summary(
    shap_payload: Dict[str, Any],
    feature_names: Sequence[str],
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_payload["values"], shap_payload["data"], feature_names=feature_names, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def generate_lime_explanations(
    pipeline: Any,
    X_val: pd.DataFrame,
    sample_indices: Iterable[int],
    config: ExplainConfig,
    output_dir: Path,
    class_names: Sequence[str],
) -> None:
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    transformed = preprocessor.transform(X_val)
    if issparse(transformed):
        transformed = transformed.toarray()
    feature_names = get_feature_names(preprocessor, X_val)

    explainer = LimeTabularExplainer(
        training_data=transformed,
        feature_names=feature_names,
        class_names=class_names,
        discretize_continuous=True,
        random_state=config.random_state,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for idx in sample_indices:
        transformed_instance = np.asarray(transformed[idx]).ravel()

        def predict_fn(data: np.ndarray) -> np.ndarray:
            return model.predict_proba(data)

        explanation = explainer.explain_instance(
            data_row=transformed_instance,
            predict_fn=predict_fn,
            num_features=config.lime_num_features,
            num_samples=config.lime_num_samples,
        )
        html_path = output_dir / f"lime_instance_{idx}.html"
        html_path.write_text(explanation.as_html(), encoding="utf-8")


def run_explanations(
    config_path: Path,
    model_names: Optional[Sequence[str]] = None,
) -> None:
    config = load_explain_config(config_path)
    project_root = Path(__file__).resolve().parents[2]

    processed_dir = (project_root / config.processed_dir).resolve()
    artifacts_dir = (project_root / config.artifacts_dir).resolve()
    output_dir = (project_root / config.output_dir).resolve()

    validation = load_validation_data(processed_dir, config.target_column)
    X_val = validation.drop(columns=[config.target_column])

    if X_val.empty:
        raise ValueError("Validation dataset has no feature columns to explain.")

    model_files = select_model_files(artifacts_dir, model_names)
    sample_size = min(len(X_val), config.shap_sample_size)
    lime_sample_size = min(len(X_val), config.lime_sample_size)

    if sample_size == 0:
        raise ValueError("No samples available for SHAP explanations.")

    sample_indices = np.random.RandomState(config.random_state).choice(
        len(X_val), size=sample_size, replace=False
    )
    shap_sample = X_val.iloc[sample_indices]

    lime_indices = np.random.RandomState(config.random_state).choice(
        len(X_val), size=lime_sample_size, replace=False
    )

    for model_path in model_files:
        model_name = model_path.stem.replace("_pipeline", "")
        pipeline = joblib.load(model_path)
        preprocessor = pipeline.named_steps["preprocess"]

        if shap_sample.isnull().any().any():
            shap_sample_filled = shap_sample.fillna(shap_sample.median(numeric_only=True))
        else:
            shap_sample_filled = shap_sample

        feature_names = get_feature_names(preprocessor, shap_sample_filled)
        print(f"[INFO] Computing SHAP values for {model_name} on {len(shap_sample_filled)} samples...")
        shap_values = compute_shap_values(pipeline, shap_sample_filled, feature_names)

        shap_output_path = output_dir / f"{model_name}_shap_summary.png"
        print(f"[INFO] Saving SHAP summary to {shap_output_path}")
        save_shap_summary(
            shap_values,
            feature_names=feature_names,
            output_path=shap_output_path,
            title=f"SHAP Summary - {model_name}",
        )

        lime_output_dir = output_dir / f"{model_name}_lime"
        print(f"[INFO] Generating LIME explanations for {model_name} on {len(lime_indices)} samples...")
        generate_lime_explanations(
            pipeline=pipeline,
            X_val=X_val,
            sample_indices=lime_indices,
            config=config,
            output_dir=lime_output_dir,
            class_names=["no_default", "default"],
        )

        summary_path = output_dir / f"{model_name}_summary.json"
        summary = {
            "model": model_name,
            "shap_summary": str(shap_output_path.relative_to(output_dir.parent if output_dir.parent.exists() else output_dir)),
            "lime_examples": [f"{lime_output_dir.name}/lime_instance_{idx}.html" for idx in lime_indices],
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[INFO] Explanation artifacts written for {model_name}")
