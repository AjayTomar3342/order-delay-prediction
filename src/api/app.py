from fastapi import FastAPI
from src.api.schema import OrderInput
from src.api.predictor import ModelPredictor

app = FastAPI(
    title="Order Delay Prediction API",
    version="1.0.0"
)

# Load model once at startup
predictor = ModelPredictor()


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/predict")
def predict(order: OrderInput):
    """
    Predict order delay.
    """
    result = predictor.predict(order.dict())
    return result
