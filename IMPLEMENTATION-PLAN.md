# Implementation Plan: Logistics Risk Prediction Pipeline

**Status:** Phases 1–7 complete. Model trained end-to-end on the full DataCo dataset;
FastAPI + Streamlit deployment, tests, and CI are in place. See REVIEW.md for the latest
code-review pass and its resolution status.

---

## Architecture Overview

```
┌──────────┐    ┌──────────────┐    ┌───────────┐    ┌─────────────┐
│  Raw CSV  │───▶│  Ingest &    │───▶│ Feature   │───▶│  Train &    │
│ (Kaggle)  │    │  Preprocess  │    │ Engineer  │    │  Evaluate   │
└──────────┘    └──────────────┘    └───────────┘    └─────────────┘
                                                           │
                                                           ▼
                                                    ┌──────────────┐
                                                    │ Serialized    │
                                                    │ Model (.ubj)  │
                                                    └──────────────┘
                                                           │
                                                           ▼
                                              ┌──────────────────────┐
                                              │ FastAPI (app/api.py)  │
                                              │ loads model once,     │
                                              │ serves POST /predict  │
                                              └──────────────────────┘
                                                           │  HTTP
                                                           ▼
                                              ┌──────────────────────┐
                                              │ Streamlit frontend    │
                                              │ (app/streamlit_app.py)│
                                              └──────────────────────┘
```

## Phase Breakdown

### Phase 1: Foundation & Project Scaffold (1–2 hrs)
- Create directory structure
- Craft `requirements.txt` with pinned deps
- Write `src/__init__.py` and stub modules
- Seed-setting, logging config

### Phase 2: Data Ingestion & Preprocessing (2–3 hrs)
- Implement `src/ingest.py` — download, cache, parse CSV
- Implement `src/preprocess.py` — column rename, drop leaks, impute, encode
- Chronological train/val/test split (70/15/15)
- Schema validation via `pandera` or manual assertions

### Phase 3: Feature Engineering (2–3 hrs)
- Temporal features: order_month, dayofweek, delay_lag_* (per shipping mode)
- Geo features: distance_km, region clusters
- Encoder: frequency encoding for high-cardinality cats
- All transforms in a single `sklearn.Transformer` or `Pipeline` step

### Phase 4: Model Training & Threshold Tuning (3–4 hrs)
- XGBoost with `RandomizedSearchCV` + `StratifiedKFold(5)`
- Probability calibration via `predict_proba`
- Threshold optimization for recall@0.30, F2-score
- Native `save_model('model.ubj')` serialization
- Experiment logging to CSV/MLflow

### Phase 5: Evaluation & Validation (1–2 hrs)
- ROC-AUC, Precision-Recall curves, confusion matrix
- Temporal holdout validation
- Feature importance (gain, cover, SHAP)
- Leakage audit: verify excluded columns have zero importance

### Phase 6: Streamlit Deployment App (2–3 hrs)
- Form inputs mapping DataCo features
- Load serialized model + preprocessing pipeline
- Real-time risk assessment with probability display
- Explanation section (SHAP waterfall or feature contributions)

### Phase 7: Testing & Documentation (1–2 hrs)
- `pytest` unit tests for preprocessing, features, metrics
- Integration test: CSV → prediction end-to-end
- README update with usage instructions, architecture diagram

---

## Dependency Map

```
Phase 1 ──▶ Phase 2 ──▶ Phase 3 ──▶ Phase 4 ──▶ Phase 5
                              │                      │
                              │                      ▼
                              │                 Phase 6
                              │                      │
                              ▼                      ▼
                         Phase 7 ◀────────────────────┘
```

---

## Key Decisions (from advisor-researcher analysis)

| Decision | Chosen | Rationale |
|---|---|---|
| Model | XGBoost | Best AUC on DataCo, pickle-compatible, native UBJSON serialization |
| Serialization | `model.save_model('model.ubj')` | Avoids pickle RCE risk + cross-version breakage |
| Deployment | Streamlit | Ops staff need forms, not APIs |
| Threshold | ~0.30 (tune during Phase 4) | Maximize recall; FN >> FP cost |
| CV Strategy | StratifiedKFold(5) + TimeSeriesSplit validation | Handles class balance + temporal structure |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Pickle version mismatch | Medium | High | Use XGBoost native format |
| Temporal data leakage | Medium | Critical | Chronological split; fold-aware lags |
| Class imbalance worse than expected | Low | Medium | scale_pos_weight fallback in Phase 4 |
| High-cardinality encoding | Medium | Medium | Frequency encoding; target encoding with CV |
| Streamlit single-user bottleneck | Low (ops team size) | Medium | Docker + nginx reverse proxy if needed |
