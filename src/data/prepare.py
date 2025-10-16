"""Data loading and preparation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

DEFAULT_STATUS_CODES = {"2", "3", "4", "5"}
STATUS_MAPPING = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "C": 0,
    "X": 0,
}


@dataclass
class PipelineConfig:
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    application_filename: str
    credit_filename: str
    output_dataset: str
    id_column: str
    target_column: str
    test_size: float
    validation_size: float
    random_state: int


def load_pipeline_config(config_path: Path) -> PipelineConfig:
    with config_path.open("r", encoding="utf-8") as file:
        config: Dict[str, Any] = yaml.safe_load(file)

    data_cfg = config["data"]
    prep_cfg = config["preprocessing"]

    return PipelineConfig(
        raw_dir=Path(data_cfg["raw_dir"]),
        interim_dir=Path(data_cfg["interim_dir"]),
        processed_dir=Path(data_cfg["processed_dir"]),
        application_filename=data_cfg["application_filename"],
        credit_filename=data_cfg["credit_filename"],
        output_dataset=data_cfg["output_dataset"],
        id_column=prep_cfg["id_column"],
        target_column=prep_cfg["target_column"],
        test_size=prep_cfg["test_size"],
        validation_size=prep_cfg["validation_size"],
        random_state=prep_cfg["random_state"],
    )


def ensure_directories(*directories: Path) -> None:
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def clean_application_data(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    df = df.drop_duplicates(subset=[id_column]).copy()
    df.columns = [col.lower() for col in df.columns]

    day_columns = [col for col in df.columns if col.startswith("days_")]
    for col in day_columns:
        df[col] = df[col].abs()

    binary_columns = ["flag_own_car", "flag_own_realty"]
    for col in binary_columns:
        if col in df.columns:
            df[col] = df[col].map({"Y": 1, "N": 0}).fillna(df[col])

    return df


def map_status_to_int(status: str) -> int:
    return STATUS_MAPPING.get(status, 0)


def build_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    credit = df.copy()
    credit["STATUS"] = credit["STATUS"].astype(str)
    credit["status_num"] = credit["STATUS"].map(map_status_to_int).astype(int)

    grouped = credit.groupby("ID")

    features = grouped.agg(
        num_credit_records=("status_num", "size"),
        max_status_delay=("status_num", "max"),
        avg_status_delay=("status_num", "mean"),
        months_on_book=("MONTHS_BALANCE", lambda x: int(abs(x.max() - x.min())) + 1),
    )

    recent_status = (
        credit.sort_values(["ID", "MONTHS_BALANCE"], ascending=[True, False])
        .groupby("ID")
        .first()["status_num"]
        .rename("recent_status")
    )

    default_target = (
        grouped["STATUS"]
        .apply(lambda s: int(any(code in DEFAULT_STATUS_CODES for code in s)))
        .rename("default_status")
    )

    features = pd.concat([features, recent_status, default_target], axis=1)
    features = features.reset_index()
    features.columns = [col.lower() for col in features.columns]

    return features


def merge_datasets(
    applications: pd.DataFrame,
    credit_features: pd.DataFrame,
    id_column: str,
) -> pd.DataFrame:
    applications = applications.rename(columns={id_column.lower(): "id"})
    dataset = applications.merge(credit_features, on="id", how="inner")
    dataset = dataset.dropna(subset=["default_status"])
    return dataset


def split_dataset(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    validation_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = df.drop(columns=[target_column])
    target = df[target_column]

    stratify = _safe_stratify(target)
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    validation_ratio = validation_size / (1 - test_size)
    stratify_train = _safe_stratify(y_train_val)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=validation_ratio,
        random_state=random_state,
        stratify=stratify_train,
    )

    train = x_train.assign(**{target_column: y_train})
    validation = x_val.assign(**{target_column: y_val})
    test = x_test.assign(**{target_column: y_test})

    return train, validation, test


def _safe_stratify(target: pd.Series | pd.DataFrame) -> pd.Series | None:
    if target.nunique() <= 1:
        return None

    if target.value_counts().min() < 2:
        return None

    return target


def run_data_preparation(config_path: Path) -> None:
    config = load_pipeline_config(config_path)
    project_root = Path(__file__).resolve().parents[2]

    raw_dir = (project_root / config.raw_dir).resolve()
    interim_dir = (project_root / config.interim_dir).resolve()
    processed_dir = (project_root / config.processed_dir).resolve()

    ensure_directories(interim_dir, processed_dir)

    application_path = raw_dir / config.application_filename
    credit_path = raw_dir / config.credit_filename

    applications = pd.read_csv(application_path)
    credit_records = pd.read_csv(credit_path)

    applications_clean = clean_application_data(applications, config.id_column)
    credit_features = build_credit_features(credit_records)

    dataset = merge_datasets(applications_clean, credit_features, config.id_column)
    dataset = dataset.rename(columns={"default_status": config.target_column})

    output_path = processed_dir / config.output_dataset
    dataset.to_parquet(output_path, index=False)

    train, validation, test = split_dataset(
        dataset,
        target_column=config.target_column,
        test_size=config.test_size,
        validation_size=config.validation_size,
        random_state=config.random_state,
    )

    train.to_parquet(processed_dir / "train.parquet", index=False)
    validation.to_parquet(processed_dir / "validation.parquet", index=False)
    test.to_parquet(processed_dir / "test.parquet", index=False)
