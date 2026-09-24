from fastapi import FastAPI

from pydantic import BaseModel

from src.artifacts import ARTIFACTS

from src.predict import predict_order

from prometheus_fastapi_instrumentator import Instrumentator

from prometheus_client import Counter


app = FastAPI(
    title="Olist Late Delivery Prediction API",
    version="1.0.0",
    description="API for predicting whether an Olist order will be delivered late.",
)

Instrumentator().instrument(app).expose(app)


prediction_errors_total = Counter(
    "prediction_errors_total",
    "Total number of prediction errors",
)


class OrderRequest(BaseModel):
    order_purchase_timestamp: str
    order_estimated_delivery_date: str
    customer_state: str
    seller_ids: list[str]
    n_items: int
    n_distinct_products: int
    total_price: float
    total_freight: float
    n_payments: int
    total_payment_value: float
    max_installments: int


class PredictionResponse(BaseModel):
    ok: bool
    is_late: int | None = None
    probability: float | None = None
    threshold: float | None = None
    model_name: str | None = None
    model_version: str | None = None
    error: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model")
def model_info():
    return {
        "model_name": ARTIFACTS.model_name,
        "model_version": ARTIFACTS.model_version,
        "threshold": ARTIFACTS.threshold,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(order: OrderRequest):
    result = predict_order(order.model_dump())

    if not result["ok"]:
        prediction_errors_total.inc()

    return result
