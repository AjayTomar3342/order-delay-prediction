from pathlib import Path
import pandas as pd

from src.utils.config_loader import load_config
from src.ingestion.data_ingestion import DataIngestion
from src.preparation.data_cleaning import DataCleaning
from src.preparation.feature_engineering import FeatureEngineer
from src.modeling.modeling import ModelTrainer


def main() -> None:
    config = load_config("config/config.yaml")

    # -----------------------------
    # Data Ingestion
    # -----------------------------
    ingestion = DataIngestion(
        file_path=config["paths"]["raw_data"],
        required_columns=config["columns"]["processed_required_columns"]
    )
    df_raw = ingestion.load_data()

    # -----------------------------
    # Data Cleaning
    # -----------------------------
    cleaning_config = config["columns"]["cleaning"]
    cleaning = DataCleaning(
        drop_columns=config["columns"]["drop_columns"],
        required_columns=config["columns"]["processed_required_columns"],
        na_fill_values=cleaning_config.get("na_fill_values", {}),
        type_conversion=cleaning_config.get("type_conversion", {}),
        date_columns=cleaning_config.get("date_columns", []),
        drop_duplicates=cleaning_config.get("drop_duplicates", True)
    )
    df_clean = cleaning.clean(df_raw)

    processed_path = Path(config["paths"]["processed_data"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(processed_path, index=False)

    print(f"Cleaned data saved to: {processed_path}")

    # -----------------------------
    # Feature Engineering
    # -----------------------------
    df_cleaned = pd.read_csv(processed_path)

    fe = FeatureEngineer(df_cleaned)
    df_features = fe.run_feature_engineering()

    fe.correlation_feature_selection(
        target_col=config["columns"]["target"],
        corr_threshold=0.01
    )

    df_features = fe.df

    feature_path = Path(config["paths"]["feature_engineered_file"])
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(feature_path, index=False)

    print(f"Feature engineering completed. Data saved to: {feature_path}")

    # -----------------------------
    # Model Training
    # -----------------------------
    print("Starting model training...")

    model_trainer = ModelTrainer(config_path="config/config.yaml")

    X, y = model_trainer.split_features_target(df_features)
    X_train, X_test, y_train, y_test = model_trainer.train_test_split_data(X, y)

    model_results = model_trainer.train_and_evaluate(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    print("Model evaluation results:")
    for model_name, metrics in model_results.items():
        print(f"{model_name}: {metrics}")

    print("\nProduction model saved in: artifacts/best_model.pkl\n")


if __name__ == "__main__":
    main()
# uvicorn src.api.app:app --reload



## Check running containers
# docker ps


## Stop the container
# docker stop <container_id_or_name>

## Restart the same container
# docker start <container_id_or_name>

## Remove a container
# docker rm <container_id_or_name>

## Run a new container from image
# docker run -d -p 8000:8000 --name order-delay-api order-delay-api


