from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple
import shutil
import joblib

import pandas as pd
import yaml
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.logger import get_logger


class ModelTrainer:
    """
    Handles preprocessing, model training, evaluation,
    MLflow tracking, and saving the best model for deployment.
    """

    def __init__(
        self,
        config_path: str,
        log_level: int = logging.INFO,
    ) -> None:
        self.config_path = Path(config_path)
        self.logger = get_logger(
            self.__class__.__name__,
            log_level,
            log_dir="logs",
            file_prefix="run_modeling",
        )

        self.config = self._load_config()
        self.preprocessor = self.build_preprocessor()

        self._setup_mlflow()

    # ---------------------------------------------------
    # MLflow Setup
    # ---------------------------------------------------
    def _setup_mlflow(self) -> None:
        project_root = self.config_path.parent.parent
        self.mlruns_path = project_root / "mlruns"

        self.mlruns_path.mkdir(exist_ok=True)
        self.mlruns_path.chmod(0o755)

        # Optional cleanup (keeps only latest run)
        for item in self.mlruns_path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)

        mlflow.set_tracking_uri(f"file://{self.mlruns_path.resolve()}")
        mlflow.set_experiment("order_delay_prediction")

        print("\nMLflow tracking enabled\n")

    # ---------------------------------------------------
    def _load_config(self) -> Dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    # ---------------------------------------------------
    def split_features_target(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        target_col = self.config["modeling"]["target_column"]
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return X, y

    # ---------------------------------------------------
    def train_test_split_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        split_cfg = self.config["modeling"]["train_test_split"]
        return train_test_split(
            X,
            y,
            test_size=split_cfg["test_size"],
            random_state=split_cfg["random_state"],
            stratify=y if split_cfg.get("stratify", False) else None,
        )

    # ---------------------------------------------------
    def build_preprocessor(self) -> ColumnTransformer:
        num_cols = self.config["modeling"]["numerical_columns"]
        cat_cols = self.config["modeling"]["categorical_columns"]

        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    cat_cols,
                ),
            ]
        )

    # ---------------------------------------------------
    def _get_models(self) -> Dict[str, object]:
        cfg = self.config["modeling"]["models"]
        return {
            "logistic_regression": LogisticRegression(**cfg["logistic_regression"]),
            "random_forest": RandomForestClassifier(**cfg["random_forest"], n_jobs=-1),
            "hist_gradient_boosting": HistGradientBoostingClassifier(
                max_iter=cfg["gradient_boosting"]["n_estimators"],
                max_depth=cfg["gradient_boosting"]["max_depth"],
                learning_rate=cfg["gradient_boosting"]["learning_rate"],
                random_state=cfg["gradient_boosting"]["random_state"],
            ),
        }

    # ---------------------------------------------------
    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Dict[str, Dict[str, float]]:

        models = self._get_models()
        metrics_cfg = self.config["modeling"]["evaluation_metrics"]

        sample_frac = self.config["sample_frac"]
        X_train_sample = X_train.sample(frac=sample_frac, random_state=42)
        y_train_sample = y_train.loc[X_train_sample.index]

        results: Dict[str, Dict[str, float]] = {}
        best_model_name = None
        best_score = -1
        best_pipeline = None

        for model_name, model in models.items():
            print(f"\nTraining model: {model_name}")

            full_pipeline = Pipeline(
                [
                    ("preprocessor", self.preprocessor),
                    ("model", model),
                ]
            )

            with mlflow.start_run(run_name=model_name):
                full_pipeline.fit(X_train_sample, y_train_sample)

                y_pred = full_pipeline.predict(X_test)
                y_prob = (
                    full_pipeline.predict_proba(X_test)[:, 1]
                    if hasattr(full_pipeline, "predict_proba")
                    else None
                )

                model_results: Dict[str, float] = {}

                if "accuracy" in metrics_cfg:
                    model_results["accuracy"] = accuracy_score(y_test, y_pred)
                if "precision" in metrics_cfg:
                    model_results["precision"] = precision_score(y_test, y_pred)
                if "recall" in metrics_cfg:
                    model_results["recall"] = recall_score(y_test, y_pred)
                if "f1_score" in metrics_cfg:
                    model_results["f1_score"] = f1_score(y_test, y_pred)
                if "roc_auc" in metrics_cfg and y_prob is not None:
                    model_results["roc_auc"] = roc_auc_score(y_test, y_prob)

                mlflow.log_params(model.get_params())
                mlflow.log_metrics(model_results)
                mlflow.sklearn.log_model(full_pipeline, artifact_path="model")

                results[model_name] = model_results

                # Track best model
                score = model_results.get("roc_auc", 0)
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
                    best_pipeline = full_pipeline

        # ---------------------------------------------------
        # Save Best Model
        # ---------------------------------------------------
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)

        model_path = artifacts_dir / "best_model.pkl"
        joblib.dump(best_pipeline, model_path)

        print(f"\nBest model: {best_model_name}")
        print(f"Saved to: {model_path}\n")

        return results
