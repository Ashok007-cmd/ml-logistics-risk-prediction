# Code Review Report: ML Logistics Risk Prediction

**Reviewed:** 2026-06-30
**Depth:** deep
**Files Reviewed:** 11
**Status:** issues_found

> **Update (2026-07-06): all findings below have been resolved.** The three
> critical bugs (CR-01/02/03), the four high-severity issues, and the logging
> gap have all been fixed in the current codebase. This report is kept as a
> historical record of the review process. A more serious issue was found
> afterwards and fixed separately: `src/features.py` was deriving features
> from the actual (post-shipment) `shipping date (DateOrders)` column, which
> leaked the target and produced invalid validation metrics (AUC 0.99, recall
> 1.0) — see `src/features.py::extract_temporal` for the fix and rationale.

## Summary

An end-to-end ML pipeline for predicting late-delivery risk using XGBoost. The project is well-structured with clear separation of concerns across modules, a configuration dataclass, and test coverage. However, **three critical bugs** exist: (1) an off-by-one error in threshold tuning that selects the wrong precision/recall values, (2) a column contradiction in config that will crash the preprocessor at runtime, and (3) incorrect column names passed to frequency encoding in the evaluation inference path. Additionally, a centralized logging module is defined but never used, and several edge cases (NaN handling, version pinning, test markers) are unaddressed.

| Severity | Count |
|----------|-------|
| CRITICAL | 3 |
| HIGH     | 4 |
| MEDIUM   | 6 |
| LOW      | 4 |
| **Total** | **17** |

---

## Critical Issues

### CR-01: Off-by-one error in threshold selection — precision/recall indexing is shifted

**File:** `src/train.py:145-159`

**Issue:**
`precision_recall_curve` returns `precision` and `recall` arrays of length `n+1` but `thresholds` of length `n`. The first element `precision[0]` / `recall[0]` corresponds to the "all-positive" case (threshold ≈ 0) with no matching threshold entry.

The code appends `1.0` to `thresholds` to make lengths equal (line 146), but this **shifts the mapping**: after appending, `thresholds[i]` maps to `precision[i+1]` / `recall[i+1]` in the original arrays — but the code uses `precision[idx]` / `recall[idx]` directly, which is incorrect.

**Concrete example with 8 thresholds:**

| Index | thresholds (after append) | precision | correct mapping |
|-------|--------------------------|-----------|-----------------|
| 0     | 0.9                      | p@all-positive (≈threshold 0) | **should be skipped** |
| 1     | 0.8                      | p@threshold 0.9                | actually p@0.9 |
| 4     | 0.5                      | p@threshold 0.6                | should be p@0.5 |
| 7     | 0.2                      | p@threshold 0.3                | should be p@0.2 |
| 8     | 1.0 (appended)           | p@threshold 0.2                | should be p@last |

For `t=0.5`, `np.argmin(np.abs(thresholds - 0.5))` → index 4 → `precision[4]` reads the precision for threshold 0.6 (off by one). The optimal threshold selection is systematically wrong.

**Impact:** The model will use a suboptimal decision threshold, directly degrading late-delivery risk classification quality.

**Fix:**
```python
# Instead of appending, use the original thresholds and offset by 1:
precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_prob)
# precisions[0] and recalls[0] are the "all positive" case — skip them
# thresholds[i] corresponds to precisions[i+1], recalls[i+1]

results = []
for t in THRESHOLD_SEARCH_RANGE:
    idx = np.argmin(np.abs(thresholds - t))
    results.append({
        "threshold": float(t),
        "precision": float(precisions[idx + 1]),   # ← offset by 1
        "recall": float(recalls[idx + 1]),          # ← offset by 1
        "f2": float(
            (5 * precisions[idx + 1] * recalls[idx + 1])
            / (4 * precisions[idx + 1] + recalls[idx + 1] + 1e-10)
        ),
    })
```

---

### CR-02: Column contradiction — `"Sales per customer"` in both `LEAK_COLS` and `NUMERIC_FEATURES`

**File:** `src/config.py:37,67`

**Issue:**
The string `"Sales per customer"` appears in **both** `LEAK_COLS` (line 37) and `NUMERIC_FEATURES` (line 67). During training, `clean_raw()` drops `LEAK_COLS` columns first. When `build_preprocessor()` later references `NUMERIC_FEATURES`, the column no longer exists in the DataFrame, causing `ColumnTransformer.fit_transform()` to raise a `ValueError` / `KeyError` for the missing column.

This means the pipeline **cannot complete training on any dataset that has the column** (which is every dataset, since it's in the raw schema).

**Why existing tests don't catch it:** The integration test in `test_integration.py` generates synthetic data WITH the column, but `clean_raw()` drops it and the preprocessor then crashes before reaching the AUC assertion. If the test has been run, it would have failed — the empty `models/` directory suggests training was never completed successfully.

**Fix:**
```python
# Option A: Remove "Sales per customer" from NUMERIC_FEATURES if truly a leak
NUMERIC_FEATURES = [
    "Days for shipment (scheduled)",
    "Latitude",
    "Longitude",
    "Order Item Discount",
    # "Sales per customer",  # removed — column is dropped as leak
    "Order Item Product Price",
    "Order Item Quantity",
    "Product Price",
    "Order Item Total",
    "Order Profit Per Order",
    "Product Card Id",
    "Customer Zipcode",
]

# Option B: If NOT a leak, remove it from LEAK_COLS
LEAK_COLS = [
    "Delivery Status",
    "Days for shipping (real)",
    "Order Status",
    "Benefit per order",
    # "Sales per customer",  # removed if not a leak
]
```

---

### CR-03: Wrong column names passed to `frequency_encode` in evaluation inference path

**File:** `src/evaluate.py:162`

**Issue:**
`full_report()` calls `frequency_encode()` with column names derived from `freq_cols` — which are ALL columns ending in `_freq` (e.g., `"Product Name_freq"`). But `frequency_encode` expects the **original** column names (e.g., `"Product Name"`).

Since `frequency_encode` checks `if col not in df.columns: continue`, and columns like `"Product Name_freq"` already exist in the DataFrame (created by `build_feature_pipeline`), the function silently does nothing. The frequency encoding is **skipped entirely** during evaluation inference.

```python
freq_cols = [c for c in df.columns if c.endswith("_freq")]                      # ← ["Product Name_freq", ...]
df, _ = frequency_encode(df, freq_cols, mappings=artifacts["encoders"])          # ← skips all — wrong names
```

**Impact:** The evaluation/test set is processed without frequency encoding. The preprocessor and model were trained WITH frequency-encoded features, but the evaluation path produces different (wrong) feature columns. When the preprocessor transforms this data, it may produce mismatched output columns or silently use wrong values, invalidating all test metrics.

**Fix:**
```python
# Use the original high-cardinality columns (the encoder dict keys ARE those names)
original_columns = list(artifacts["encoders"].keys())  # e.g. ["Product Name", "Customer Full Name", "Order City"]
df, _ = frequency_encode(df, original_columns, mappings=artifacts["encoders"])
```

---

## High Issues

### HI-01: Centralized logging module defined but never used

**File:** `src/log_utils.py` (entire file) + all other source files

**Issue:**
`log_utils.py` provides a `get_logger()` function with structured formatting (timestamps, levels, module names). It is **never imported or used** by any other module. All modules use `print()` instead, which lacks severity levels, timestamps, output routing, and is not configurable.

This is a systematic maintainability gap that also blocks observability (no way to adjust log levels, no file output, no structured logging).

**Fix:**
```python
# In each module, replace:
print(f"[train] Val ROC-AUC: {val_auc:.4f}")
# With:
logger = get_logger(__name__)
logger.info("Val ROC-AUC: %.4f", val_auc)
```

This applies to: `ingest.py:28,30,33,44,54`, `preprocess.py:53,62,105`, `train.py:56,141,142,162,163,168`, `evaluate.py:89,90,113,137,139`

---

### HI-02: Dead code — `warned` list populated but never returned or logged

**File:** `src/evaluate.py:130-134`

**Issue:**
In `leakage_audit()`, the second loop collects categorical columns with non-zero gain into a `warned` list. This list is never returned, printed, or used — it is pure dead computation. Meanwhile, potentially important information about legitimate features having zero importance is silently discarded.

```python
warned = []                     # ← populated but never surfaced
for col in LOW_CARD_CATS + HIGH_CARD_CATS:
    ...
    if not match.empty and match["gain"].sum() > 0:
        warned.append(col)      # ← dead store
```

**Fix:**
```python
def leakage_audit(imp_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Returns (leaked_cols, zero_importance_features)."""
    ...
    return leaked, warned       # or print warned if appropriate
```

---

### HI-03: Synthetic integration test has overwritten column — dead initialization

**File:** `tests/test_integration.py:66-67`

**Issue:**
Line 66 sets `"Late_delivery_risk": 1`, but line 67 immediately overwrites this with `TARGET_COL: int(rng.binomial(1, 0.3))`. Since `TARGET_COL == "Late_delivery_risk"`, the first assignment is dead code that goes nowhere.

```python
"Late_delivery_risk": 1,                                # ← overwritten by line 67
TARGET_COL: int(rng.binomial(1, 0.3)),                  # ← TARGET_COL == "Late_delivery_risk"
```

**Fix:**
Remove the redundant line 66.

---

### HI-04: Missing pytest marker registration for `@pytest.mark.streamlit`

**File:** `tests/test_integration.py:100`

**Issue:**
The `@pytest.mark.streamlit` decorator on `test_prediction` is not registered in any `pytest.ini`, `pyproject.toml`, or `conftest.py`. When running `pytest -m streamlit` or even during normal test collection, pytest will emit `PytestUnknownMarkWarning`.

While this is a warning by default, unregistered markers can cause silent skips with `-W error::pytest.PytestUnknownMarkWarning` and confuse developers trying to selectively run tests.

**Fix:**
Create or add to a `pytest.ini`:
```ini
[pytest]
markers =
    streamlit: tests requiring streamlit extras
```

---

## Medium Issues

### ME-01: O(n²) feature name resolution in `feature_importance` via `list.index()`

**File:** `src/evaluate.py:105`

**Issue:**
For each feature name, `feature_names.index(fname)` performs a linear scan of the entire list. With hundreds of features (especially after OHE expansion), this is O(n²) — a simple enumeration would be O(n).

```python
for fname in feature_names:
    fkey = f"f{feature_names.index(fname)}"     # ← O(n²)
```

**Fix:**
```python
for idx, fname in enumerate(feature_names):
    fkey = f"f{idx}"
```

---

### ME-02: Inconsistent random number generator usage across the project

**File:** `tests/test_features.py:17` vs `tests/test_integration.py:23`

**Issue:**
`test_features.py` uses legacy `np.random.seed(42)` while `test_integration.py` uses modern `np.random.default_rng(RANDOM_STATE)`. The legacy API is deprecated and produces different streams than the new `Generator` API. This inconsistency means:
1. Tests are not reproducible across numpy versions
2. Visual test failures when switching to numpy ≥ 2.0
3. Confusing for developers maintaining the test suite

**Fix:**
```python
# test_features.py
@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)  # use Generator API
    n = 50
    return pd.DataFrame({
        "Latitude": rng.uniform(20, 50, n),
        ...
    })
```

---

### ME-03: Hardcoded dummy values in Streamlit app inference may produce unreliable predictions

**File:** `app/streamlit_app.py:119-148`

**Issue:**
The inference form hardcodes several feature values that differ from the training distribution:
- `"Customer Zipcode": 0` — invalid zipcode, far from training data distribution
- `"Product Card Id": 1` — arbitrary ID
- `"Customer City": ""` — empty string
- `"Customer State": ""` — empty string
- `"Order Profit Per Order": 0.0` — always zero profit
- `"order date (DateOrders)": "2024-01-15"` — static date

The preprocessor's StandardScaler will center and scale these values based on training statistics. For extreme values like `Zipcode=0`, the scaled value will fall far outside the training distribution, potentially causing unreliable predictions.

**Fix:**
```python
# Replace hardcoded defaults with reasonable fallbacks or make them user inputs
"Customer Zipcode": 10001,       # reasonable US zipcode
"Product Card Id": 0,            # sentinel (model learns it as "unknown")
"Customer City": "Unknown",
"Customer State": "NY",
"Order Profit Per Order": 0.0,   # document as "assumes zero profit"
```

---

### ME-04: `unsafe_allow_html=True` in Streamlit app creates XSS surface (low risk, bad pattern)

**File:** `app/streamlit_app.py:175`

**Issue:**
`st.markdown(..., unsafe_allow_html=True)` renders raw HTML. While the interpolated variables (`risk_color`, `risk`) are not user-controllable in the current code, this pattern sets a dangerous precedent. If a future change passes user input (e.g., customer name) into this markdown call, it becomes a stored XSS vector.

**Fix:**
```python
st.markdown(f"### Risk Assessment: **{risk}**")
```
Use native Streamlit components without raw HTML — the styling can be achieved with `st.success()` / `st.error()` instead.

---

### ME-05: Non-deterministic set ordering in `EXPECTED_COLS`

**File:** `src/preprocess.py:36`

**Issue:**
`EXPECTED_COLS` includes `list({"order date (DateOrders)", "shipping date (DateOrders)"})`. Set literal ordering is an implementation detail of CPython and is not guaranteed across Python versions or interpreters. The tuple's element order is non-deterministic.

While this doesn't affect correctness (validation iterates all elements), it makes the error message order inconsistent across runs, which can confuse debugging.

```python
EXPECTED_COLS = (
    LOW_CARD_CATS + HIGH_CARD_CATS + NUMERIC_FEATURES
    + [TARGET_COL] + list({"order date (DateOrders)", "shipping date (DateOrders)"})
                               # ^ set ordering unpredictable
)
```

**Fix:**
```python
from src.config import (
    ...
)

DATE_COLS = ["order date (DateOrders)", "shipping date (DateOrders)"]

EXPECTED_COLS = (
    LOW_CARD_CATS + HIGH_CARD_CATS + NUMERIC_FEATURES
    + [TARGET_COL] + DATE_COLS
)
```

---

### ME-06: `early_stopping_rounds` passed to final model without `eval_set`

**File:** `src/train.py:177-186`

**Issue:**
`XGBParams()` includes `early_stopping_rounds: int = 50`. The dictionary from `to_dict()` is passed to `final_model = xgb.XGBClassifier(...)`. However, `final_model.fit(X_full, y_full)` does not provide `eval_set`. XGBoost silently ignores `early_stopping_rounds` when no evaluation set is provided, but the parameter definition is misleading.

**Fix:**
```python
# Option A: Remove early_stopping_rounds from final model params
final_params = dict(base_params.to_dict())
final_params.update(search.best_params_)
final_params.pop("early_stopping_rounds", None)  # not used without eval_set

# Option B: Or provide eval_set if validation is desired during final training
final_model.fit(X_full, y_full, eval_set=[(X_full, y_full)])
```

---

## Low Issues

### LO-01: Unused imports

| File | Line | Import | Reason |
|------|------|--------|--------|
| `src/config.py` | 12 | `import numpy as np` | Only used for `np.arange` on line 117 — could be `from numpy import arange` |
| `src/train.py` | 20 | `from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold` | Both used — OK, but `StratifiedKFold` could be imported alone |

### LO-02: Inconsistent file open style

| File | Line | Pattern | Preferred |
|------|------|---------|-----------|
| `src/train.py` | 51 | `open(EXPERIMENT_LOG, "a", ...)` | `Path.open()` for consistency |
| `src/evaluate.py` | 46 | `open(model_dir / "best_threshold.json")` | `Path.open()` |

### LO-03: Imports inside function body (style)

**File:** `src/evaluate.py:32-33`

`import joblib` and `import xgboost as xgb` inside `load_artifacts()`. These should be at module level for discoverability and to surface import errors at load time rather than call time.

### LO-04: `print()` used throughout instead of structured logging

**Files:** `src/ingest.py:28-33,44-45,53-54`, `src/preprocess.py:53,62,105`, `src/train.py:56,141-142,162-163,168`, `src/evaluate.py:89-93,113,137-139` — all use `print()` where `logging` would provide level control, timestamps, and output routing.

While the `log_utils.py` module exists (HI-01), the entire codebase defaults to `print()`.

---

## Appendix: Test Quality Assessment

| Test File | Lines | What It Covers | Gaps |
|-----------|-------|----------------|------|
| `test_features.py` | 87 | `extract_temporal`, `add_geo_features`, `add_shipping_interactions`, `compute_lag_features` | No test for `build_feature_pipeline` orchestration; no edge case for missing columns |
| `test_preprocess.py` | 82 | `clean_raw`, `standardize_dtypes`, `chronological_split`, `frequency_encode`, `build_preprocessor` | No test for NaN handling in frequency encode; no test for chronological split without date column |
| `test_integration.py` | 161 | Full pipeline from `objective()` through prediction and model file existence | **Cannot pass** due to CR-02 (column contradiction crash); `test_prediction` requires `streamlit` extra with no skip-if-missing guard |

### Key test gaps:
1. **No data leakage tests** — no fixture that verifies leak columns don't appear in final features
2. **No NaN/edge-case coverage** — missing values in categoricals, null coordinates, empty DataFrames
3. **No "no date column" fallback test** for `chronological_split` — the fallback path is untested
4. **Integration test is broken** by CR-02 — `test_training_completes` would crash before assertion
5. **`test_prediction` subprocess** spawns a Python process rather than using module imports directly — fragile and harder to debug

---

_Reviewed: 2026-06-30_
_Reviewer: gsd-code-reviewer (deep analysis)_
_Depth: deep_
