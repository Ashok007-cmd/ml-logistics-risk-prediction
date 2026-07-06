"""Unit tests for preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    build_preprocessor,
    chronological_split,
    clean_raw,
    frequency_encode,
    standardize_dtypes,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame(
        {
            "Late_delivery_risk": rng.integers(0, 2, size=n),
            "Shipping Mode": rng.choice(["Standard Class", "First Class"], size=n),
            "Days for shipment (scheduled)": rng.integers(1, 10, size=n),
            "Latitude": rng.uniform(20, 50, n),
            "Longitude": rng.uniform(-130, -70, n),
            "Delivery Status": rng.choice(["Late delivery", "On time"], size=n),
            "Customer Email": ["test@test.com"] * n,
            "order date (DateOrders)": pd.date_range("2024-01-01", periods=n, freq="D"),
        }
    )


def test_clean_raw_drops_leak_columns(sample_df):
    result = clean_raw(sample_df)
    assert "Delivery Status" not in result.columns
    assert "Customer Email" not in result.columns


def test_clean_raw_preserves_target(sample_df):
    result = clean_raw(sample_df)
    assert "Late_delivery_risk" in result.columns


def test_clean_raw_target_is_int(sample_df):
    result = clean_raw(sample_df)
    assert result["Late_delivery_risk"].dtype == int


def test_standardize_dtypes_handles_dates(sample_df):
    result = standardize_dtypes(sample_df)
    assert pd.api.types.is_datetime64_any_dtype(result["order date (DateOrders)"])


def test_chronological_split_maintains_order(sample_df):
    train, val, test = chronological_split(sample_df)
    total = len(train) + len(val) + len(test)
    assert total == len(sample_df)
    assert len(train) > len(val) >= len(test)
    assert train["order date (DateOrders)"].max() <= val["order date (DateOrders)"].min()


def test_frequency_encode_creates_column(sample_df):
    result, mapping = frequency_encode(sample_df, ["Shipping Mode"])
    assert "Shipping Mode_freq" in result.columns
    assert "Shipping Mode" in mapping


def test_build_preprocessor_returns_transformer():
    transformer = build_preprocessor()
    assert hasattr(transformer, "fit_transform")
    assert hasattr(transformer, "transform")


def test_clean_raw_removes_all_leaks(sample_df):
    sample_df["Days for shipping (real)"] = 5
    sample_df["Benefit per order"] = 100.0
    result = clean_raw(sample_df)
    for col in [
        "Delivery Status",
        "Days for shipping (real)",
        "Benefit per order",
        "Customer Email",
    ]:
        assert col not in result.columns


def test_chronological_split_no_date_column(sample_df):
    df_no_date = sample_df.drop(columns=["order date (DateOrders)"])
    train, val, test = chronological_split(df_no_date)
    total = len(train) + len(val) + len(test)
    assert total == len(sample_df)
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0


def test_frequency_encode_nan_handling():
    df = pd.DataFrame({"col": ["A", "B", "A", None, "B", "A"]})
    result, mapping = frequency_encode(df, ["col"])
    assert "col_freq" in result.columns
    # Nulls should map to 0 or have a non-NaN value in mapping/freq column
    assert not result["col_freq"].isna().any()
