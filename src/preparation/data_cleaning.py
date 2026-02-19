from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Union

import pandas as pd
from src.utils.logger import get_logger


class DataCleaning:
    """
    Handles basic data cleaning steps:
    - Column dropping
    - Validation of required columns
    - NA filling (config-driven)
    - Type conversions
    - Date parsing
    - Duplicate removal
    """

    def __init__(
        self,
        drop_columns: List[str],
        required_columns: List[str],
        na_fill_values: Dict[str, Union[int, float, str]] = {},
        type_conversion: Dict[str, str] = {},
        date_columns: List[str] = [],
        drop_duplicates: bool = True,
        log_level: int = logging.INFO,
    ) -> None:
        self.drop_columns = drop_columns
        self.required_columns = required_columns
        self.na_fill_values = na_fill_values
        self.type_conversion = type_conversion
        self.date_columns = date_columns
        self.drop_duplicates = drop_duplicates
        self.logger = get_logger(self.__class__.__name__, log_level, file_prefix="run_data_cleaning")

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(
            f"Data Cleaning Logs"
        )
        self.logger.info(f"Starting data cleaning | Initial shape: {df.shape}")

        # -----------------------------
        # Drop columns
        # -----------------------------
        cols_to_drop = [c for c in self.drop_columns if c in df.columns]
        df_clean = df.drop(columns=cols_to_drop)
        self.logger.info(f"Dropped {len(cols_to_drop)} columns. Remaining: {df_clean.shape[1]}")

        # -----------------------------
        # Fill NA values
        # -----------------------------
        total_na_filled = 0
        for col, fill_value in self.na_fill_values.items():
            if col in df_clean.columns:
                na_count = df_clean[col].isna().sum()
                df_clean[col] = df_clean[col].fillna(fill_value)
                total_na_filled += na_count
                self.logger.info(f"Filled {na_count} NAs in column '{col}' with '{fill_value}'")
        if total_na_filled == 0:
            self.logger.info("No missing values were filled in this step (all specified columns already complete)")
        else:
            self.logger.info(f"Total NAs filled: {total_na_filled}")

        # -----------------------------
        # Type conversion
        # -----------------------------
        conversions_done = 0
        for col, dtype in self.type_conversion.items():
            if col in df_clean.columns:
                before_non_null = df_clean[col].notna().sum()
                try:
                    if dtype.lower() == "datetime":
                        df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce")
                    else:
                        df_clean[col] = df_clean[col].astype(dtype)
                    after_non_null = df_clean[col].notna().sum()
                    conversions_done += 1
                    self.logger.info(f"Converted column '{col}' to {dtype}")
                    if before_non_null == after_non_null:
                        self.logger.info(f" - No data loss in '{col}' after conversion")
                    else:
                        self.logger.info(f" - {before_non_null - after_non_null} values became NaT/NaN after conversion")
                except Exception as e:
                    self.logger.error(f"Failed to convert column '{col}' to {dtype}: {e}")
        if conversions_done == 0:
            self.logger.info("No type conversions were applied (columns not found)")

        # ---------------------------------
        # Date parsing (for extra columns)
        # ---------------------------------
        date_parsed_count = 0
        for col in self.date_columns:
            if col in df_clean.columns:
                before_non_null = df_clean[col].notna().sum()
                df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce")
                after_non_null = df_clean[col].notna().sum()
                date_parsed_count += 1
                self.logger.info(f"Parsed column '{col}' as datetime")
                if before_non_null == after_non_null:
                    self.logger.info(f" - No data loss in '{col}' after date parsing")
                else:
                    self.logger.info(f" - {before_non_null - after_non_null} values became NaT after date parsing")
        if date_parsed_count == 0:
            self.logger.info("No date columns were parsed")

        # -----------------------------
        # Remove duplicates
        # -----------------------------
        duplicates_removed = 0
        if self.drop_duplicates:
            before = df_clean.shape[0]
            df_clean = df_clean.drop_duplicates()
            duplicates_removed = before - df_clean.shape[0]
            if duplicates_removed == 0:
                self.logger.info("No duplicates found to remove")
            else:
                self.logger.info(f"Removed {duplicates_removed} duplicate rows")

        # -----------------------------
        # Validate required columns exist
        # -----------------------------
        missing = set(self.required_columns) - set(df_clean.columns)
        if missing:
            self.logger.error(f"Required columns missing after cleaning: {missing}")
            raise ValueError(f"Required columns missing after cleaning: {missing}")

        self.logger.info(
            f"Data cleaning completed successfully | Final shape: {df_clean.shape} | "
            f"Impact: NAs filled={total_na_filled}, Type conversions={conversions_done}, "
            f"Date columns parsed={date_parsed_count}, Duplicates removed={duplicates_removed}"
        )

        # -----------------------------
        # Data Type Change
        # -----------------------------
        df_clean["order date (DateOrders)"] = pd.to_datetime(df_clean["order date (DateOrders)"])

        return df_clean
