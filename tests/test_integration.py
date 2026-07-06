"""
End-to-end integration test.

Creates synthetic data, runs the full pipeline (clean → feature → train → predict),
and validates that probabilities and predictions are produced without errors.
"""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest

from src.config import MODELS_DIR, RANDOM_STATE, TARGET_COL
from src.predictor import Predictor
from src.train import objective


@pytest.fixture(scope="module")
def synthetic_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    n = 200
    rows = []
    for i in range(n):
        rows.append(
            {
                "Type": rng.choice(["DEBIT", "CREDIT", "TRANSFER"]),
                "Shipping Mode": rng.choice(
                    ["Standard Class", "First Class", "Second Class", "Same Day"]
                ),
                "Customer Segment": rng.choice(["Consumer", "Corporate", "Home Office"]),
                "Market": rng.choice(["US", "LATAM", "Europe", "Pacific", "Africa"]),
                "Order Region": rng.choice(["North", "South", "East", "West", "Central"]),
                "Product Category Name": rng.choice(
                    [
                        "Office Machines",
                        "Chairs",
                        "Tables",
                        "Phones",
                        "Storage",
                        "Art",
                        "Binders",
                        "Paper",
                        "Appliances",
                    ]
                ),
                "Department Name": rng.choice(["Furniture", "Office Supplies", "Technology"]),
                "Customer City": rng.choice(["NYC", "LA", "Chicago", "Houston", "Miami"]),
                "Customer State": rng.choice(["NY", "CA", "IL", "TX", "FL"]),
                "Customer Country": "United States",
                "Customer Full Name": f"Customer_{i}",
                "Order City": f"City_{i}",
                "Product Name": f"Product_{i}",
                "Customer Zipcode": int(rng.integers(10000, 99999)),
                "Product Card Id": int(rng.integers(1, 1000)),
                "Latitude": round(rng.uniform(25, 48), 4),
                "Longitude": round(rng.uniform(-122, -72), 4),
                "Days for shipment (scheduled)": int(rng.integers(1, 8)),
                "Days for shipping (real)": int(rng.integers(1, 10)),
                "Order Item Discount": round(rng.uniform(0, 0.5), 2),
                "Order Item Quantity": int(rng.integers(1, 10)),
                "Sales per customer": round(rng.uniform(10, 500), 2),
                "Order Item Product Price": round(rng.uniform(10, 500), 2),
                "Product Price": round(rng.uniform(10, 500), 2),
                "Order Item Total": round(rng.uniform(10, 500), 2),
                "Order Profit Per Order": round(rng.uniform(-50, 200), 2),
                "order date (DateOrders)": f"2024-{int(rng.integers(1, 12)):02d}-"
                f"{int(rng.integers(1, 28)):02d}",
                "shipping date (DateOrders)": f"2024-{int(rng.integers(1, 12)):02d}-"
                f"{int(rng.integers(1, 28)):02d}",
                "Delivery Status": "Late delivery",
                TARGET_COL: int(rng.binomial(1, 0.3)),
            }
        )
    df = pd.DataFrame(rows)

    train = df.iloc[:140]
    val = df.iloc[140:170]
    test = df.iloc[170:200]
    return train, val, test


@pytest.fixture(scope="module")
def trained_pipeline(synthetic_data):
    train, val, _ = synthetic_data
    result = objective(train, val)
    yield result
    if MODELS_DIR.exists():
        shutil.rmtree(MODELS_DIR)


class TestEndToEnd:
    def test_training_completes(self, trained_pipeline):
        assert "best_params" in trained_pipeline
        assert "val_auc" in trained_pipeline
        assert isinstance(trained_pipeline["val_auc"], float)

    def test_model_files_exist(self, trained_pipeline):
        assert (MODELS_DIR / "xgb_model_final.ubj").exists()
        assert (MODELS_DIR / "preprocessor.pkl").exists()
        assert (MODELS_DIR / "encoders.pkl").exists()
        assert (MODELS_DIR / "lag_rates.pkl").exists()
        assert (MODELS_DIR / "best_threshold.json").exists()
        assert (MODELS_DIR / "experiments.csv").exists()

    def test_prediction(self, synthetic_data, trained_pipeline):
        p = Predictor()
        df = pd.DataFrame(
            [
                {
                    "Shipping Mode": "Standard Class",
                    "Customer Segment": "Consumer",
                    "Market": "US",
                    "Order Region": "North",
                    "Product Category Name": "Office Machines",
                    "Department Name": "Technology",
                    "Customer City": "NYC",
                    "Customer State": "NY",
                    "Customer Country": "United States",
                    "Type": "DEBIT",
                    "Product Name": "X",
                    "Customer Full Name": "Y",
                    "Order City": "Z",
                    "Days for shipment (scheduled)": 4,
                    "Latitude": 40.7,
                    "Longitude": -74.0,
                    "Order Item Discount": 0.0,
                    "Order Item Product Price": 150.0,
                    "Order Item Quantity": 2,
                    "Product Price": 150.0,
                    "Order Item Total": 300.0,
                    "Order Profit Per Order": 0.0,
                    "Sales per customer": 300.0,
                    "Product Card Id": 1,
                    "Customer Zipcode": 0,
                    "order date (DateOrders)": "2024-01-15",
                }
            ]
        )
        prob, risk = p.risk_score(df)
        assert 0 <= prob <= 1
        assert risk in ("HIGH RISK", "LOW RISK")

    def test_experiment_log_exists(self, trained_pipeline):
        df = pd.read_csv(MODELS_DIR / "experiments.csv")
        assert not df.empty
        assert "val_auc" in df.columns
        assert "best_f2" in df.columns

    def test_full_report_on_held_out_test_split(self, synthetic_data, trained_pipeline):
        """Regression test: full_report() must run on a split the model never trained
        or tuned the threshold on. This is the exact code path that silently broke
        (preprocessor/model feature-shape mismatch) with no test ever exercising it."""
        from src.evaluate import full_report

        _, _, test_df = synthetic_data
        metrics = full_report(test_df)
        assert "test_roc_auc" in metrics
        assert 0 <= metrics["test_roc_auc"] <= 1
        assert "leaked_columns" in metrics

    def test_api_predict(self, trained_pipeline):
        from fastapi.testclient import TestClient

        from app.api import app

        with TestClient(app) as client:
            payload = {
            "Shipping Mode": "Standard Class",
            "Customer Segment": "Consumer",
            "Customer City": "NYC",
            "Customer State": "NY",
            "Customer Country": "United States",
            "Market": "US",
            "Order Region": "North",
            "Product Category Name": "Office Machines",
            "Department Name": "Technology",
            "Type": "DEBIT",
            "Product Name": "X",
            "Customer Full Name": "Y",
            "Order City": "Z",
            "Days for shipment (scheduled)": 4,
            "Latitude": 40.7,
            "Longitude": -74.0,
            "Order Item Discount": 0.0,
            "Order Item Product Price": 150.0,
            "Order Item Quantity": 2,
            "Product Price": 150.0,
            "Order Item Total": 300.0,
            "Order Profit Per Order": 0.0,
            "Sales per customer": 300.0,
            "Product Card Id": 1,
            "Customer Zipcode": 0,
            "order date (DateOrders)": "2024-01-15",
        }

            response = client.post("/predict", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert "probability" in data
            assert "risk_label" in data
            assert "threshold_used" in data
            assert "contributions" in data
            assert len(data["contributions"]) > 0
