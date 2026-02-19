from pathlib import Path
import joblib
import pandas as pd


class ModelPredictor:
    """
    Handles loading the trained model pipeline
    and performing predictions.
    """

    def __init__(self) -> None:
        model_path = Path("artifacts/best_model.pkl")

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                "Run training first."
            )

        self.model = joblib.load(model_path)

    def predict(self, input_data: dict) -> dict:
        """
        Accepts input dictionary,
        converts to correct column format,
        returns prediction and probability.
        """

        # Map API field names back to original training column names
        column_mapping = {
            "Days_for_shipment_scheduled": "Days for shipment (scheduled)",
            "Category_Name": "Category Name",
            "Customer_City": "Customer City",
            "Customer_Country": "Customer Country",
            "Customer_Segment": "Customer Segment",
            "Customer_State": "Customer State",
            "Department_Name": "Department Name",
            "Order_City": "Order City",
            "Order_Country": "Order Country",
            "Order_Region": "Order Region",
            "Order_State": "Order State",
            "Shipping_Mode": "Shipping Mode",
        }

        # Rename keys
        renamed_data = {
            column_mapping.get(k, k): v for k, v in input_data.items()
        }

        df = pd.DataFrame([renamed_data])

        prediction = self.model.predict(df)[0]

        if hasattr(self.model, "predict_proba"):
            probability = self.model.predict_proba(df)[0][1]
        else:
            probability = None

        return {
            "prediction": int(prediction),
            "probability": float(probability) if probability is not None else None,
        }
