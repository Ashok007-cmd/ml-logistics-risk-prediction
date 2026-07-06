"""
Predictor class: single entry point for inference.

Encapsulates model + preprocessor + encoders + threshold so that
app, evaluation, and batch scripts share one code path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import MODELS_DIR
from src.features import build_feature_pipeline
from src.preprocess import frequency_encode, prepare_raw


class Predictor:
    """End-to-end prediction wrapper.

    Usage
    -----
    predictor = Predictor()
    probability = predictor.predict_proba(raw_df)[0, 1]
    """

    def __init__(self, model_dir: Path = MODELS_DIR) -> None:
        self.model_dir = model_dir
        self.model = self._load_model()
        self.preprocessor = joblib.load(model_dir / "preprocessor.pkl")
        self.encoders: dict[str, Any] = joblib.load(model_dir / "encoders.pkl")
        self.lag_rates: dict[str, Any] = joblib.load(model_dir / "lag_rates.pkl")
        with (model_dir / "best_threshold.json").open() as f:
            self.threshold: dict[str, Any] = json.load(f)
        self.feature_names: list[str] = list(self.preprocessor.feature_names_in_)

    # ── loading ─────────────────────────────────────────────────────

    def _load_model(self) -> xgb.XGBClassifier:
        model = xgb.XGBClassifier()
        final = self.model_dir / "xgb_model_final.ubj"
        fallback = self.model_dir / "xgb_model.ubj"
        path = final if final.exists() else fallback
        model.load_model(str(path))
        return model

    # ── feature pipeline (mirrors train-time) ───────────────────────

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full feature-engineering chain (temporal, geo, interactions, freq-encode)."""
        df = prepare_raw(df)
        df, _ = build_feature_pipeline(df, lag_rates=self.lag_rates)
        df, _ = frequency_encode(df, list(self.encoders.keys()), mappings=self.encoders)
        return df.reindex(columns=self.feature_names, fill_value=0)

    # ── prediction ──────────────────────────────────────────────────

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return shape (n, 2) — [prob_on_time, prob_late]."""
        df_feat = self.engineer_features(df)
        x_mat = self.preprocessor.transform(df_feat)
        return self.model.predict_proba(x_mat)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return binary predictions using the tuned threshold."""
        probs = self.predict_proba(df)[:, 1]
        return np.asarray((probs >= self.threshold["threshold"]).astype(int))

    def risk_score(self, df: pd.DataFrame) -> tuple[float, str]:
        """Return (late_probability, risk_label) for a single row."""
        prob = self.predict_proba(df)[0, 1]
        label = "HIGH RISK" if prob >= self.threshold["threshold"] else "LOW RISK"
        return float(prob), label

    # ── interpretability ────────────────────────────────────────────

    def feature_contributions(
        self,
        df: pd.DataFrame,
        top_k: int = 8,
    ) -> pd.DataFrame:
        """Return SHAP-style per-feature contributions for a single row."""
        df_feat = self.engineer_features(df)
        x_mat = self.preprocessor.transform(df_feat)
        booster = self.model.get_booster()
        contrib = booster.predict(xgb.DMatrix(x_mat), pred_contribs=True)
        feature_names = self.preprocessor.get_feature_names_out()
        cols = list(feature_names) + ["bias"]
        contrib_df = pd.DataFrame(contrib, columns=cols).drop(columns=["bias"])
        melted = contrib_df.T.reset_index()
        melted.columns = ["feature", "contribution"]
        melted = melted.sort_values("contribution", key=abs, ascending=False)
        return melted.head(top_k)
