"""Unit tests for feature engineering module."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    add_geo_features,
    add_shipping_interactions,
    compute_lag_features,
    extract_temporal,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame(
        {
            "order date (DateOrders)": pd.date_range("2024-01-01", periods=n, freq="D"),
            "shipping date (DateOrders)": pd.date_range("2024-01-03", periods=n, freq="D"),
            "Shipping Mode": rng.choice(["Standard Class", "First Class"], size=n),
            "Latitude": rng.uniform(20, 50, n),
            "Longitude": rng.uniform(-130, -70, n),
            "Days for shipment (scheduled)": rng.integers(1, 10, size=n),
            "Late_delivery_risk": rng.integers(0, 2, size=n),
        }
    )


def test_extract_temporal_adds_month(sample_df):
    result = extract_temporal(sample_df)
    assert "order_month" in result.columns


def test_extract_temporal_adds_is_weekend(sample_df):
    result = extract_temporal(sample_df)
    assert "order_is_weekend" in result.columns


def test_extract_temporal_excludes_shipping_date(sample_df):
    """Shipping date is the actual (post-hoc) ship date — using it, or anything
    derived from it, would leak the target. Only order-date features are safe."""
    result = extract_temporal(sample_df)
    assert "processing_days" not in result.columns
    assert not any(c.startswith("ship_") for c in result.columns)


def test_add_geo_features_adds_region(sample_df):
    result = add_geo_features(sample_df)
    assert "geo_region" in result.columns


def test_add_shipping_interactions(sample_df):
    result = add_shipping_interactions(sample_df)
    assert "ship_mode_x_scheduled_days" in result.columns


def test_compute_lag_features(sample_df):
    result, rates = compute_lag_features(sample_df)
    assert "Shipping Mode_delay_rate" in result.columns
    assert 0 <= result["Shipping Mode_delay_rate"].iloc[0] <= 1
    assert isinstance(rates, dict)


def test_compute_lag_features_with_precomputed_rates(sample_df):
    result1, rates = compute_lag_features(sample_df)
    result2, _ = compute_lag_features(sample_df, rates=rates)
    pd.testing.assert_series_equal(
        result1["Shipping Mode_delay_rate"],
        result2["Shipping Mode_delay_rate"],
    )


def test_numericize_string_features_hashes_string_cols(sample_df):
    from src.features import _numericize_string_features

    df = add_geo_features(sample_df)
    df = add_shipping_interactions(df)
    assert pd.api.types.is_string_dtype(df["geo_region"]) or df["geo_region"].dtype == object
    df2 = _numericize_string_features(df.copy())
    assert df2["geo_region"].dtype == np.int64
    assert df2["ship_mode_x_scheduled_days"].dtype == np.int64


def test_compute_lag_features_missing_target_returns_unchanged(sample_df):
    no_target = sample_df.drop(columns=["Late_delivery_risk"])
    result, rates = compute_lag_features(no_target)
    assert "Shipping Mode_delay_rate" not in result.columns
    # with explicit rates still works
    result2, _ = compute_lag_features(
        no_target,
        rates={
            "Standard Class": 0.3,
            "First Class": 0.5,
        },
    )
    assert "Shipping Mode_delay_rate" in result2.columns
