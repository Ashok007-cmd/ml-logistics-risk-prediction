"""
Preprocessing module.

Cleans raw DataCo data, imputes missing values, encodes categoricals,
splits chronologically, and wraps transforms into a reusable pipeline.
"""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    DROP_COLS,
    HIGH_CARD_CATS,
    LEAK_COLS,
    LOW_CARD_CATS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
    VAL_SIZE,
)
from src.log_utils import get_logger

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logger = get_logger(__name__)


EXPECTED_COLS = (
    LOW_CARD_CATS + HIGH_CARD_CATS + NUMERIC_FEATURES + [TARGET_COL] + ["order date (DateOrders)"]
)


def prepare_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Construct Customer Full Name and rename Category Name if needed."""
    df = df.copy()
    if "Customer Full Name" not in df.columns:
        if "Customer Fname" in df.columns and "Customer Lname" in df.columns:
            df["Customer Full Name"] = (
                df["Customer Fname"].astype(str) + " " + df["Customer Lname"].astype(str)
            )
        elif "Customer Fname" in df.columns:
            df["Customer Full Name"] = df["Customer Fname"].astype(str)
        else:
            df["Customer Full Name"] = "Unknown Customer"

    if "Product Category Name" not in df.columns and "Category Name" in df.columns:
        df = df.rename(columns={"Category Name": "Product Category Name"})
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Assert that required columns exist with compatible types."""
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Got columns: {list(df.columns)}")

    if TARGET_COL in df.columns:
        unique_vals = df[TARGET_COL].dropna().unique()
        if not set(unique_vals).issubset(
            {0, 1, "0", "1", "Late delivery risk", "Late_delivery_risk", True, False}
        ):
            logger.warning("Unexpected target values: %s", unique_vals)


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns with no predictive value or that leak the target."""
    cols_to_drop = [c for c in DROP_COLS + LEAK_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    if TARGET_COL in df.columns:
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)

    return df


def standardize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert date columns to datetime, ensure numeric columns are numeric."""
    date_cols = [c for c in df.columns if "date" in c.lower()]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def chronological_split(
    df: pd.DataFrame,
    date_col: str = "order date (DateOrders)",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by time: oldest 70% train, next 15% val, newest 15% test."""
    if date_col not in df.columns:
        train, temp = train_test_split(
            df,
            test_size=TEST_SIZE + VAL_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True,
        )
        val, test = train_test_split(
            temp,
            test_size=TEST_SIZE / (TEST_SIZE + VAL_SIZE),
            random_state=RANDOM_STATE,
            shuffle=True,
        )
        return train, val, test

    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * (1 - TEST_SIZE - VAL_SIZE))
    val_end = int(n * (1 - TEST_SIZE))

    train = df_sorted.iloc[:train_end]
    val = df_sorted.iloc[train_end:val_end]
    test = df_sorted.iloc[val_end:]

    logger.info("Chronological split: train=%d, val=%d, test=%d", len(train), len(val), len(test))
    return train, val, test


def build_preprocessor(
    extra_numeric_cols: list[str] | None = None,
) -> ColumnTransformer:
    """Build a reusable ColumnTransformer.

    Parameters
    ----------
    extra_numeric_cols:
        Additional numeric columns (engineered features, freq-encoded cats)
        that should be passed through without transformation.
    """
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categ_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = [
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categ_pipe, LOW_CARD_CATS),
    ]

    if extra_numeric_cols:
        transformers.append(("extra", "passthrough", extra_numeric_cols))

    return ColumnTransformer(transformers, remainder="drop")


def frequency_encode(
    df: pd.DataFrame,
    columns: list[str],
    mappings: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply frequency encoding to high-cardinality columns.

    If mappings are provided, apply them (inference mode).
    Otherwise, learn mappings from the data (fit mode).
    """
    df = df.copy()
    result_mappings = mappings or {}

    for col in columns:
        if col not in df.columns:
            continue
        if mappings is None:
            mapping_dict = df[col].value_counts().to_dict()
            result_mappings[col] = mapping_dict
        else:
            mapping_dict = mappings[col]
        freq = df[col].map(mapping_dict).fillna(0)
        df[f"{col}_freq"] = freq

    return df, result_mappings
