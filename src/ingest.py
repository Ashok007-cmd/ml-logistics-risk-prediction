"""
Data ingestion module.

Downloads the DataCo Smart Supply Chain dataset, caches it locally,
and provides a clean interface for loading raw data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.log_utils import get_logger

logger = get_logger(__name__)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_raw_csv(url: str, filename: str = "supply_chain.csv") -> Path:
    ensure_dirs()
    dest = RAW_DIR / filename
    if dest.exists():
        logger.info("Cached raw data found at %s", dest)
        return dest
    logger.info("Downloading from %s ...", url)
    df = pd.read_csv(url, encoding="ISO-8859-1")
    df.to_csv(dest, index=False, encoding="utf-8")
    logger.info("Saved to %s (shape=%s)", dest, df.shape)
    return dest


def load_raw(path: Path | None = None) -> pd.DataFrame:
    if path is None:
        path = RAW_DIR / "supply_chain.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run ingest.download_raw_csv() first."
        )
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def save_processed(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    ensure_dirs()
    train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    val.to_parquet(PROCESSED_DIR / "val.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    logger.info("Processed splits saved to %s", PROCESSED_DIR)


def load_processed() -> dict[str, pd.DataFrame]:
    return {
        split: pd.read_parquet(PROCESSED_DIR / f"{split}.parquet")
        for split in ("train", "val", "test")
    }
