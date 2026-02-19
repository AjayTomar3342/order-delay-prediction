from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import snowflake.connector
from src.utils.logger import get_logger


class SnowflakeFeatureStore:
    """
    Handles creation and retrieval of feature tables in Snowflake.
    """

    def __init__(
        self,
        user: str,
        password: str,
        account: str,
        warehouse: str,
        database: str,
        schema: str,
        table_name: str,
        log_level: int = logging.INFO,
    ) -> None:
        self.user = user
        self.password = password
        self.account = account
        self.warehouse = warehouse
        self.database = database
        self.schema = schema
        self.table_name = table_name

        self.logger = get_logger(self.__class__.__name__, log_level)
        self.conn: Optional[snowflake.connector.SnowflakeConnection] = None

    # -----------------------------
    # Connection
    # -----------------------------
    def connect(self) -> None:
        try:
            self.conn = snowflake.connector.connect(
                user=self.user,
                password=self.password,
                account=self.account,
                warehouse=self.warehouse,
                database=self.database,
                schema=self.schema,
            )
            self.logger.info("Connected to Snowflake successfully")
        except Exception as e:
            self.logger.error(f"Failed to connect to Snowflake: {e}")
            raise

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.logger.info("Snowflake connection closed")

    # -----------------------------
    # Upload feature table
    # -----------------------------
    def create_or_replace_feature_table(self, df: pd.DataFrame) -> None:
        if self.conn is None:
            raise RuntimeError("Snowflake connection not established")

        cursor = self.conn.cursor()

        try:
            self.logger.info(
                f"Creating/Replacing feature table '{self.table_name}' "
                f"| Rows: {df.shape[0]} | Columns: {df.shape[1]}"
            )

            # Drop existing table
            cursor.execute(
                f"DROP TABLE IF EXISTS {self.database}.{self.schema}.{self.table_name}"
            )

            # Create table schema dynamically
            columns_sql = []
            for col, dtype in df.dtypes.items():
                if "int" in str(dtype):
                    snow_type = "INT"
                elif "float" in str(dtype):
                    snow_type = "FLOAT"
                else:
                    snow_type = "STRING"
                columns_sql.append(f'"{col}" {snow_type}')

            create_sql = f"""
            CREATE TABLE {self.database}.{self.schema}.{self.table_name} (
                {", ".join(columns_sql)}
            )
            """

            cursor.execute(create_sql)

            # Insert data
            insert_sql = f"""
            INSERT INTO {self.database}.{self.schema}.{self.table_name}
            VALUES ({", ".join(["%s"] * len(df.columns))})
            """

            cursor.executemany(insert_sql, df.values.tolist())

            self.conn.commit()

            self.logger.info("Feature table uploaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to create/upload feature table: {e}")
            raise
        finally:
            cursor.close()

    # -----------------------------
    # Download feature table
    # -----------------------------
    def download_feature_table(self) -> pd.DataFrame:
        if self.conn is None:
            raise RuntimeError("Snowflake connection not established")

        self.logger.info(
            f"Downloading feature table '{self.table_name}' from Snowflake"
        )

        try:
            query = f"""
            SELECT *
            FROM {self.database}.{self.schema}.{self.table_name}
            """

            df = pd.read_sql(query, self.conn)
            self.logger.info(
                f"Downloaded feature table | Shape: {df.shape}"
            )
            return df

        except Exception as e:
            self.logger.error(f"Failed to download feature table: {e}")
            raise
