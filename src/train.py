"""
Training pipeline.

Orchestrates preprocessing → feature engineering → model training →
threshold tuning → serialization + experiment logging.
"""

from __future__ import annotations

import csv
import json
import warnings
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.config import (
    ENCODERS_PATH,
    HIGH_CARD_CATS,
    MODEL_UBJ_PATH,
    MODELS_DIR,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    TARGET_COL,
    THRESHOLD_SEARCH_RANGE,
    XGBParams,
)
from src.features import build_feature_pipeline
from src.log_utils import get_logger
from src.preprocess import (
    build_preprocessor,
    clean_raw,
    frequency_encode,
    prepare_raw,
    standardize_dtypes,
    validate_schema,
)

warnings.filterwarnings("ignore", category=UserWarning)

logger = get_logger(__name__)

EXPERIMENT_LOG = MODELS_DIR / "experiments.csv"


def _log_experiment(params: dict[str, Any], metrics: dict[str, Any]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    row = {**params, **metrics, "timestamp": datetime.now().isoformat()}
    file_exists = EXPERIMENT_LOG.exists()
    with EXPERIMENT_LOG.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(row)
    logger.info("Experiment logged to %s", EXPERIMENT_LOG)


def _extract_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x_mat = df.drop(columns=[TARGET_COL], errors="ignore")
    y = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)
    return x_mat, y


def objective(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> dict[str, Any]:
    """Train XGBoost with hyperparameter tuning and return best model + metrics."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Prepare raw inputs ───────────────────────────────────────────
    train_df = prepare_raw(train_df)
    val_df = prepare_raw(val_df)

    # ── Validate ─────────────────────────────────────────────────────
    validate_schema(train_df)
    validate_schema(val_df)

    # ── Clean & standardize ──────────────────────────────────────────
    train_df = clean_raw(train_df)
    val_df = clean_raw(val_df)

    train_df = standardize_dtypes(train_df)
    val_df = standardize_dtypes(val_df)

    # ── Capture original columns before feature engineering ──────────
    original_cols = set(train_df.columns)

    # ── Feature engineering (fit on train, transform val) ────────────
    train_df, lag_rates = build_feature_pipeline(train_df)
    val_df, _ = build_feature_pipeline(val_df, lag_rates=lag_rates)

    # ── Frequency encoding ───────────────────────────────────────────
    train_df, enc_mappings = frequency_encode(train_df, HIGH_CARD_CATS)
    val_df, _ = frequency_encode(val_df, HIGH_CARD_CATS, mappings=enc_mappings)

    joblib.dump(enc_mappings, ENCODERS_PATH)
    joblib.dump(lag_rates, MODELS_DIR / "lag_rates.pkl")

    # ── Separate X / y ───────────────────────────────────────────────
    x_train, y_train = _extract_xy(train_df)
    x_val, y_val = _extract_xy(val_df)

    # ── Identify engineered numeric columns ──────────────────────────
    numeric_cols = x_train.select_dtypes(include=np.number).columns
    eng_cols = numeric_cols.difference(list(original_cols)).tolist()

    # ── NaN check ────────────────────────────────────────────────────
    nan_cols = x_train.columns[x_train.isna().any()].tolist()
    if nan_cols:
        logger.warning("%d columns contain NaN: %s", len(nan_cols), nan_cols)
    logger.info("Train shape=%s | Val shape=%s", x_train.shape, x_val.shape)

    # ── Build preprocessor ───────────────────────────────────────────
    preprocessor = build_preprocessor(extra_numeric_cols=eng_cols)
    x_train_processed = preprocessor.fit_transform(x_train)
    x_val_processed = preprocessor.transform(x_val)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    # ── Hyperparameter search ────────────────────────────────────────
    base_params = XGBParams()
    xgb_clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric=base_params.eval_metric,
        random_state=RANDOM_STATE,
        n_jobs=base_params.n_jobs,
        verbosity=0,
    )

    search_params = {
        "n_estimators": [200, 400, 600],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5],
        "reg_lambda": [0.1, 1.0, 5.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=search_params,
        n_iter=30,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1,
    )
    search.fit(x_train_processed, y_train)

    best_model = search.best_estimator_
    assert isinstance(best_model, xgb.XGBClassifier), "best_estimator_ must be an XGBClassifier"

    # ── Evaluate on validation set ───────────────────────────────────
    y_val_prob = best_model.predict_proba(x_val_processed)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_prob)
    logger.info("Val ROC-AUC: %.4f", val_auc)

    # ── Threshold tuning (maximize F2) ───────────────────────────────
    results = []
    y_val_arr = np.asarray(y_val)
    y_val_prob_arr = np.asarray(y_val_prob)
    for t in THRESHOLD_SEARCH_RANGE:
        y_pred = (y_val_prob_arr >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_val_arr == 1))
        fp = np.sum((y_pred == 1) & (y_val_arr == 0))
        fn = np.sum((y_pred == 0) & (y_val_arr == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f2 = (
            (5 * precision * recall) / (4 * precision + recall)
            if (4 * precision + recall) > 0
            else 0.0
        )

        results.append(
            {
                "threshold": float(t),
                "precision": float(precision),
                "recall": float(recall),
                "f2": float(f2),
            }
        )

    best_t = max(results, key=lambda r: r["f2"])
    logger.info(
        "Best threshold=%.2f, recall=%.4f, precision=%.4f, F2=%.4f",
        best_t["threshold"],
        best_t["recall"],
        best_t["precision"],
        best_t["f2"],
    )

    # ── Save artifacts ───────────────────────────────────────────────
    best_model.save_model(str(MODEL_UBJ_PATH))
    logger.info("Model saved to %s", MODEL_UBJ_PATH)

    with (MODELS_DIR / "best_threshold.json").open("w") as f:
        json.dump(best_t, f)

    # ── Retrain with early stopping on held-out validation ────────────
    final_params: dict[str, Any] = dict(base_params.to_dict())
    final_params.update(search.best_params_)
    final_params["n_jobs"] = base_params.n_jobs
    final_model = xgb.XGBClassifier(
        objective="binary:logistic",
        early_stopping_rounds=50,
        **final_params,
    )
    final_model.fit(
        x_train_processed,
        y_train,
        eval_set=[(x_val_processed, y_val)],
        verbose=False,
    )
    final_model.save_model(str(MODELS_DIR / "xgb_model_final.ubj"))

    # ── Log experiment ───────────────────────────────────────────────
    metrics = {
        "val_auc": round(val_auc, 4),
        "best_threshold": round(best_t["threshold"], 2),
        "best_recall": round(best_t["recall"], 4),
        "best_precision": round(best_t["precision"], 4),
        "best_f2": round(best_t["f2"], 4),
    }
    _log_experiment(search.best_params_, metrics)

    return {
        "best_params": search.best_params_,
        "val_auc": val_auc,
        "best_threshold": best_t,
    }


if __name__ == "__main__":
    from src.config import DATASET_FILENAME, DATASET_URL
    from src.evaluate import full_report
    from src.ingest import download_raw_csv, load_raw, save_processed
    from src.preprocess import chronological_split

    logger.info("Starting logistics risk prediction training pipeline")

    # 1. Download and Ingest
    csv_path = download_raw_csv(DATASET_URL, DATASET_FILENAME)
    raw_df = load_raw(csv_path)

    # 2. Chronological Split & Preprocessing
    train_df, val_df, test_df = chronological_split(raw_df)
    save_processed(train_df, val_df, test_df)

    # 3. Train & Tune (Runs the objective function)
    logger.info("Training XGBoost model and tuning threshold...")
    pipeline_results = objective(train_df, val_df)

    # 4. Evaluate on test set
    logger.info("Running evaluation on test set...")
    test_metrics = full_report(test_df)
    logger.info("Pipeline training complete. Test Set Results:")
    for k, v in test_metrics.items():
        if isinstance(v, (int, float)):
            logger.info("  %s: %.4f", k, v)
        elif isinstance(v, list):
            logger.info("  %s: %s", k, v)
