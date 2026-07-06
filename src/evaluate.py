"""
Evaluation module.

Provides metrics, feature importance analysis, leakage audit,
plots, and reports for model performance assessment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    fbeta_score,
    roc_auc_score,
)

from src.config import DROP_COLS, LEAK_COLS, MODELS_DIR, TARGET_COL
from src.log_utils import get_logger
from src.predictor import Predictor
from src.preprocess import clean_raw, prepare_raw, standardize_dtypes

logger = get_logger(__name__)


def load_artifacts(model_dir: Path = MODELS_DIR) -> Predictor:
    return Predictor(model_dir)


def evaluate(
    model: xgb.XGBClassifier,
    x_test: pd.DataFrame,
    y_true: pd.Series,
    threshold: float = 0.5,
    prefix: str = "",
) -> dict[str, float]:
    """Run standard classification metrics."""
    y_prob = model.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    f2 = fbeta_score(y_true, y_pred, beta=2)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    report = classification_report(y_true, y_pred, output_dict=True)

    metrics = {
        f"{prefix}roc_auc": float(auc),
        f"{prefix}f2_score": float(f2),
        f"{prefix}precision": float(report["1"]["precision"]),
        f"{prefix}recall": float(report["1"]["recall"]),
        f"{prefix}f1_score": float(report["1"]["f1-score"]),
        f"{prefix}accuracy": float(report["accuracy"]),
        f"{prefix}true_negatives": int(tn),
        f"{prefix}false_positives": int(fp),
        f"{prefix}false_negatives": int(fn),
        f"{prefix}true_positives": int(tp),
    }

    logger.info(
        "%s — AUC=%.4f, F2=%.4f, Recall=%.4f, Precision=%.4f",
        prefix.strip("_"),
        auc,
        f2,
        report["1"]["recall"],
        report["1"]["precision"],
    )
    return metrics


def feature_importance(model: xgb.XGBClassifier, feature_names: list[str]) -> pd.DataFrame:
    """Return a DataFrame with gain, cover, and frequency importance."""
    imp_types = {
        "gain": model.get_booster().get_score(importance_type="gain"),
        "cover": model.get_booster().get_score(importance_type="cover"),
        "weight": model.get_booster().get_score(importance_type="weight"),
    }
    fkey_map = {fname: f"f{i}" for i, fname in enumerate(feature_names)}
    rows = []
    for fname in feature_names:
        fkey = fkey_map[fname]
        rows.append(
            {
                "feature": fname,
                "gain": imp_types["gain"].get(fkey, 0),
                "cover": imp_types["cover"].get(fkey, 0),
                "weight": imp_types["weight"].get(fkey, 0),
            }
        )
    df = pd.DataFrame(rows).sort_values("gain", ascending=False)
    logger.info("Top-5 features by gain:\n%s", df.head(5).to_string(index=False))
    return df


def leakage_audit(imp_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Check that known leak columns have zero importance, and identify zero-importance features.

    Returns a tuple of:
    - List of any leaked columns that have non-zero importance.
    - List of legitimate features that have zero importance (zero gain).
    """
    leaked = []
    for col in LEAK_COLS + DROP_COLS:
        match = imp_df[imp_df["feature"].str.contains(col.replace(" ", "_"), case=False, na=False)]
        if not match.empty and match["gain"].sum() > 0:
            leaked.append(col)

    if leaked:
        logger.warning("Leaked columns with non-zero importance: %s", leaked)
    else:
        logger.info("OK — no leaked columns found in feature set")

    # Check for zero importance features (features present in the model but with gain == 0)
    zero_imp = imp_df[imp_df["gain"] == 0]["feature"].tolist()
    if zero_imp:
        logger.warning("Features with zero importance (gain=0): %s", zero_imp)
    else:
        logger.info("OK — all features have non-zero importance")

    return leaked, zero_imp


def full_report(
    test_df: pd.DataFrame,
    model_dir: Path = MODELS_DIR,
) -> dict[str, Any]:
    """Run full evaluation on the test set, including feature importance + audit."""
    p = load_artifacts(model_dir)

    df = prepare_raw(test_df)
    df = clean_raw(df)
    df = standardize_dtypes(df)
    df_feat = p.engineer_features(df)
    x_processed = p.preprocessor.transform(df_feat)

    y = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)

    metrics: dict[str, Any] = evaluate(
        p.model, x_processed, y, threshold=p.threshold["threshold"], prefix="test_"
    )
    metrics["threshold_used"] = p.threshold["threshold"]

    logger.info(
        "Confusion Matrix: TN=%d FP=%d FN=%d TP=%d",
        metrics["test_true_negatives"],
        metrics["test_false_positives"],
        metrics["test_false_negatives"],
        metrics["test_true_positives"],
    )

    feature_names = p.preprocessor.get_feature_names_out()
    imp_df = feature_importance(p.model, feature_names)
    leaked, zero_imp = leakage_audit(imp_df)

    metrics["leaked_columns"] = leaked
    metrics["zero_importance_columns"] = zero_imp
    return metrics
