from __future__ import annotations

import logging
import pandas as pd
import numpy as np

from src.utils.logger import get_logger


class FeatureEngineer:
    """
    Handles feature engineering for Order Delay Prediction project.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        log_level: int = logging.INFO
    ) -> None:
        self.df = df.copy()
        self.logger = get_logger(self.__class__.__name__, log_level, file_prefix="run_feature_engineering")

        self.logger.info(
            f"Feature Engineering Logs"
        )

        self.logger.info(
            f"FeatureEngineer initialized | Input shape: {self.df.shape}"
        )

    def convert_datatypes(self) -> None:
        self.logger.info("Converting data types")

        before_non_null = self.df["order date (DateOrders)"].notna().sum()
        self.df["order date (DateOrders)"] = pd.to_datetime(
            self.df["order date (DateOrders)"], errors="coerce"
        )
        after_non_null = self.df["order date (DateOrders)"].notna().sum()

        if before_non_null == after_non_null:
            self.logger.info("Datetime conversion completed with no data loss")
        else:
            self.logger.warning(
                f"{before_non_null - after_non_null} values became NaT during datetime conversion"
            )

    def engineer_datetime_features(self) -> None:
        self.logger.info("Engineering datetime features")

        self.df["order_hour"] = self.df["order date (DateOrders)"].dt.hour
        self.df["order_day_of_week"] = self.df["order date (DateOrders)"].dt.dayofweek
        self.df["order_month"] = self.df["order date (DateOrders)"].dt.month
        self.df["is_weekend"] = (
            self.df["order_day_of_week"].isin([5, 6]).astype(int)
        )

        self.df.drop(columns=["order date (DateOrders)"], inplace=True)

        self.logger.info(
            "Datetime features created: order_hour, order_day_of_week, "
            "order_month, is_weekend | Original datetime column dropped"
        )

    def engineer_business_features(self) -> None:
        self.logger.info("Engineering business features")

        self.df["discount_per_item"] = (
            self.df["Order Item Discount"]
            / self.df["Order Item Quantity"].replace(0, 1)
        )

        self.logger.info("Feature created: discount_per_item")

    def engineer_geographical_features(self) -> None:
        self.logger.info("Engineering geographical features")

        self.df["cross_country_flag"] = (
            self.df["Customer Country"] != self.df["Order Country"]
        ).astype(int)

        self.logger.info("Feature created: cross_country_flag")

    def clean_categorical_features(self) -> None:
        categorical_cols = [
            "Type",
            "Category Name",
            "Customer City",
            "Customer Country",
            "Customer Segment",
            "Customer State",
            "Department Name",
            "Market",
            "Order City",
            "Order Country",
            "Order Region",
            "Order State",
            "Shipping Mode",
        ]

        self.logger.info(
            f"Standardizing categorical columns: {categorical_cols}"
        )

        for col in categorical_cols:
            if col in self.df.columns:
                self.df[col] = (
                    self.df[col]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                )

        self.logger.info("Categorical feature cleaning completed")

    def correlation_feature_selection(
        self,
        target_col: str,
        corr_threshold: float = 0.01
    ) -> None:
        """
        Perform initial correlation-based feature selection.
        """
        self.logger.info(
            f"Starting correlation-based feature selection | "
            f"Target='{target_col}', Threshold={corr_threshold}"
        )

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if target_col not in numeric_cols:
            self.logger.error(
                f"Target column '{target_col}' not found in numeric columns"
            )
            raise ValueError(
                f"Target column '{target_col}' not found in numeric columns"
            )

        numeric_cols.remove(target_col)

        # Drop zero-variance columns
        zero_var_cols = [
            col for col in numeric_cols if self.df[col].nunique() <= 1
        ]

        if zero_var_cols:
            self.df.drop(columns=zero_var_cols, inplace=True)
            self.logger.info(
                f"Dropped {len(zero_var_cols)} zero-variance columns: {zero_var_cols}"
            )
        else:
            self.logger.info("No zero-variance numeric columns found")

        # Recompute numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove(target_col)

        # Correlation filtering
        corr_matrix = self.df[numeric_cols + [target_col]].corr()
        target_corr = corr_matrix[target_col].abs()

        low_corr_cols = [
            col for col in target_corr.index
            if col != target_col and target_corr[col] < corr_threshold
        ]

        if low_corr_cols:
            self.df.drop(columns=low_corr_cols, inplace=True)
            self.logger.info(
                f"Dropped {len(low_corr_cols)} low-correlation features: {low_corr_cols}"
            )
        else:
            self.logger.info("No low-correlation features dropped")

        self.logger.info(
            f"Correlation-based feature selection completed | "
            f"Remaining shape: {self.df.shape}"
        )

    def run_feature_engineering(self) -> pd.DataFrame:
        self.logger.info("Starting feature engineering pipeline")

        self.convert_datatypes()
        self.engineer_datetime_features()
        self.engineer_business_features()
        self.engineer_geographical_features()
        self.clean_categorical_features()

        self.logger.info(
            f"Feature engineering completed successfully | Output shape: {self.df.shape}"
        )

        return self.df
