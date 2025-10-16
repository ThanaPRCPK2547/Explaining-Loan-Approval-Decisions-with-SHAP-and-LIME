"""Unit tests for data preparation utilities."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.prepare import (
    build_credit_features,
    clean_application_data,
    merge_datasets,
    split_dataset,
)


def test_clean_application_data_handles_duplicates_and_flags():
    df = pd.DataFrame(
        {
            "ID": [1, 1, 2],
            "DAYS_BIRTH": [-10000, -10001, -12000],
            "FLAG_OWN_CAR": ["Y", "Y", "N"],
        }
    )
    result = clean_application_data(df, id_column="ID")

    assert len(result) == 2
    assert result.loc[result["id"] == 1, "days_birth"].item() == 10000
    assert set(result["flag_own_car"].unique()) <= {0, 1}


def test_build_credit_features_creates_target_and_features():
    credit = pd.DataFrame(
        {
            "ID": [1, 1, 2, 2],
            "MONTHS_BALANCE": [0, -1, 0, -1],
            "STATUS": ["0", "2", "0", "0"],
        }
    )
    features = build_credit_features(credit)

    default_map = dict(zip(features["id"], features["default_status"]))
    assert default_map[1] == 1
    assert default_map[2] == 0
    assert "num_credit_records" in features.columns


def test_merge_and_split_dataset(tmp_path: Path):
    applications = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "feature": [0.1, 0.2, 0.3],
        }
    )
    credit_features = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "default_status": [1, 0, 1],
        }
    )

    merged = merge_datasets(applications, credit_features, id_column="ID")

    train, val, test = split_dataset(
        merged,
        target_column="default_status",
        test_size=0.2,
        validation_size=0.2,
        random_state=42,
    )

    assert len(train) + len(val) + len(test) == len(merged)
    assert {"default_status"}.issubset(train.columns)
