"""
Central configuration for the ML pipeline.

All tunable parameters, paths, and model hyperparameters live here
so they can be changed without touching business logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Data ─────────────────────────────────────────────────────────────
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Overridable via env var so tests (and parallel training runs) can point
# at an isolated directory instead of colliding with production artifacts.
MODELS_DIR = Path(os.environ.get("MODELS_DIR", str(PROJECT_ROOT / "models")))

DATASET_URL = (
    "https://raw.githubusercontent.com/ashishpatel26/"
    "DataCo-SMART-SUPPLY-CHAIN-FOR-BIG-DATA-ANALYSIS/"
    "main/DataCoSupplyChainDataset.csv"
)
DATASET_FILENAME = "supply_chain.csv"

# ── Column names ─────────────────────────────────────────────────────
TARGET_COL = "Late_delivery_risk"

# Features that would leak the target (post-delivery knowledge)
LEAK_COLS = [
    "Delivery Status",
    "Days for shipping (real)",
    "Order Status",
    "Benefit per order",
    "Sales per customer",
]

# Low-cardinality categoricals (one-hot encode)
LOW_CARD_CATS = [
    "Shipping Mode",
    "Customer Segment",
    "Customer Country",
    "Market",
    "Type",
]

# High-cardinality categoricals (frequency encode)
HIGH_CARD_CATS = [
    "Product Name",
    "Customer Full Name",
    "Order City",
    "Customer City",
    "Customer State",
    "Order Region",
    "Product Category Name",
    "Department Name",
]

# Numeric features (excluding LEAK_COLS which are dropped by clean_raw)
NUMERIC_FEATURES = [
    "Days for shipment (scheduled)",
    "Latitude",
    "Longitude",
    "Order Item Discount",
    "Order Item Product Price",
    "Order Item Quantity",
    "Product Price",
    "Order Item Total",
    "Order Profit Per Order",
    "Product Card Id",
    "Customer Zipcode",
]

# Columns with no predictive value (PII, constants, empties)
DROP_COLS = [
    "Customer Email",
    "Customer Password",
    "Product Description",
    "Product Status",
]

# ── Preprocessing ────────────────────────────────────────────────────
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # from remaining after test split
RANDOM_STATE = 42

# ── Model (XGBoost) ──────────────────────────────────────────────────


@dataclass
class XGBParams:
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    min_child_weight: int = 3
    gamma: float = 0.0
    scale_pos_weight: float = 1.0  # ~55/45 split → near 1.0
    eval_metric: str = "aucpr"
    random_state: int = RANDOM_STATE
    verbosity: int = 1
    n_jobs: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and k != "early_stopping_rounds"
        }


# ── Threshold tuning ─────────────────────────────────────────────────
THRESHOLD_SEARCH_RANGE = np.arange(0.15, 0.60, 0.01)

# ── Paths ────────────────────────────────────────────────────────────
MODEL_UBJ_PATH = MODELS_DIR / "xgb_model.ubj"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
ENCODERS_PATH = MODELS_DIR / "encoders.pkl"
