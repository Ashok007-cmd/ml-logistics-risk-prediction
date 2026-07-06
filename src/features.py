"""
Feature engineering module.

Creates temporal, geographical, and interaction features from
the cleaned DataCo dataset.

All transforms must be safe to apply at inference time (no future
information leakage).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import TARGET_COL

# String-engineered columns that ColumnTransformer would silently drop.
# We hash them to int so they survive remainder="drop".
_STRING_FEATS = ["geo_region", "ship_mode_x_scheduled_days", "ship_mode_x_month"]


def _numericize_string_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic hash: string engineered cols → int64 so they survive ColumnTransformer."""
    for col in _STRING_FEATS:
        if col in df.columns:
            hashed = pd.util.hash_pandas_object(df[col], hash_key="lrp_hash_16bytes")
            df[col] = hashed.astype(np.int64)
    return df


def extract_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """Extract date-based features from the order date.

    Deliberately excludes "shipping date (DateOrders)": in this dataset that
    column is the *actual* ship-completion date, not a promised/scheduled
    date. It's set once the shipment is done, so it (and anything derived
    from it, like processing-days) is unavailable at real prediction time
    and leaks the target — date arithmetic on it reconstructs
    `Days for shipping (real)` (a LEAK_COLS member) closely enough to match
    `Late_delivery_risk` ~97% of the time on its own.
    """
    df = df.copy()

    col = "order date (DateOrders)"
    if col in df.columns:
        dt = pd.to_datetime(df[col], errors="coerce")
        df["order_year"] = dt.dt.year
        df["order_month"] = dt.dt.month
        df["order_dayofweek"] = dt.dt.dayofweek
        df["order_quarter"] = dt.dt.quarter
        df["order_is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)

    return df


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create geographical interaction features from lat/lon."""
    df = df.copy()

    lat_col = "Latitude"
    lon_col = "Longitude"

    if lat_col in df.columns and lon_col in df.columns:
        lat = pd.to_numeric(df[lat_col], errors="coerce")
        lon = pd.to_numeric(df[lon_col], errors="coerce")

        df["lat_rounded"] = (lat * 4).round() / 4
        df["lon_rounded"] = (lon * 4).round() / 4
        df["geo_region"] = df["lat_rounded"].astype(str) + "_" + df["lon_rounded"].astype(str)
    return df


def add_shipping_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features involving shipping mode."""
    df = df.copy()
    if "Shipping Mode" not in df.columns:
        return df

    sm = df["Shipping Mode"]
    if "Days for shipment (scheduled)" in df.columns:
        d = pd.to_numeric(df["Days for shipment (scheduled)"], errors="coerce")
        df["ship_mode_x_scheduled_days"] = sm.astype(str) + "_" + d.astype(str)

    if "order_month" in df.columns:
        df["ship_mode_x_month"] = sm.astype(str) + "_" + df["order_month"].astype(str)

    return df


def compute_lag_features(
    df: pd.DataFrame,
    group_col: str = "Shipping Mode",
    rates: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute rolling average delay rate per group as a proxy lag feature.

    When called during training (rates=None), computes rates from this
    dataframe's target.  When called during inference, pass the pre-computed
    rates from training to prevent target leakage.

    Always returns (df, rates_dict).
    """
    df = df.copy()
    if group_col not in df.columns:
        return df, rates or {}

    if rates is None:
        if TARGET_COL not in df.columns:
            return df, {}
        rates = df.groupby(group_col)[TARGET_COL].mean().to_dict()

    df[f"{group_col}_delay_rate"] = df[group_col].map(rates)
    return df, rates


def build_feature_pipeline(
    df: pd.DataFrame,
    lag_rates: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run all feature engineering steps in sequence.

    When lag_rates is None, computes rates from the data (fit mode).
    When lag_rates is provided, uses them (transform mode).

    Always returns (df, rates).  Callers in transform mode may ignore
    the second element.
    """
    df = extract_temporal(df)
    df = add_geo_features(df)
    df = add_shipping_interactions(df)
    df = _numericize_string_features(df)
    df, lag_rates = compute_lag_features(df, rates=lag_rates)
    return df, lag_rates
