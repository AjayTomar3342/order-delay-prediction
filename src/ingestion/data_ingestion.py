from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd
from src.utils.logger import get_logger


class DataIngestion:
    """
    Handles ingestion of raw CSV data with encoding safety, validation, and logging.
    """

    def __init__(
        self,
        file_path: str,
        required_columns: List[str],
        log_level: int = logging.INFO,
    ) -> None:
        self.file_path = Path(file_path)
        self.required_columns = required_columns
        # Only change: add file_prefix for proper logging
        self.logger = get_logger(
            self.__class__.__name__, log_level, log_dir="logs", file_prefix="run_data_ingestion"
        )

    def load_data(self) -> pd.DataFrame:
        self.logger.info(f"Starting data ingestion from {self.file_path}")

        if not self.file_path.exists():
            self.logger.error(f"File not found: {self.file_path}")
            raise FileNotFoundError(f"File not found: {self.file_path}")

        try:
            df = pd.read_csv(self.file_path, encoding="utf-8")
            self.logger.info("Loaded CSV using UTF-8 encoding")
        except UnicodeDecodeError:
            self.logger.warning("UTF-8 decoding failed, trying latin-1 encoding")
            df = pd.read_csv(self.file_path, encoding="latin-1")
            self.logger.info("Loaded CSV using latin-1 encoding")
        except Exception as e:
            self.logger.error("Failed to read CSV file", exc_info=True)
            raise e

        # Log shape of loaded data
        self.logger.info(f"Data ingestion completed | Shape: {df.shape}")

        return df
