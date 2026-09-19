"""
The single public entry point of the inference pipeline: given one raw
order, return whether it is predicted late or on time, with a probability.

This is the function a future API route (Step 7) will call directly. It
does not know anything about HTTP, JSON payloads, or web frameworks -- it
only knows how to turn a Python dict into a prediction, using the loaded
model and the same feature-building logic as the notebooks.
"""

from src.artifacts import ARTIFACTS
from src.features import build_features, InvalidOrderError


def predict_order(order: dict) -> dict:
    """
    Parameters
    ----------
    order : dict
        A raw order, in the shape documented at the top of src/features.py.

    Returns
    -------
    dict with:
        is_late      : int (0 or 1) -- the final prediction, using the
                       threshold chosen in Notebook 6 (not the default 0.5)
        probability  : float -- the model's raw predicted probability of
                       being late
        threshold    : float -- the threshold that was applied
        model_name   : str
        model_version: str
    """
    features_df = build_features(order)

    # predict_proba returns [[P(class=0), P(class=1)]]; we want P(late).
    probability = float(ARTIFACTS.model.predict_proba(features_df)[0, 1])
    is_late = int(probability >= ARTIFACTS.threshold)

    return {
        "is_late": is_late,
        "probability": probability,
        "threshold": ARTIFACTS.threshold,
        "model_name": ARTIFACTS.model_name,
        "model_version": ARTIFACTS.model_version,
    }


if __name__ == "__main__":
    # Manual check: "python -m src.predict" from the project root (task3/)
    sample_order = {
        "order_purchase_timestamp": "2018-05-01T10:00:00",
        "order_estimated_delivery_date": "2018-05-20",
        "customer_state": "SP",
        "seller_ids": ["some_seller_id"],
        "n_items": 1,
        "n_distinct_products": 1,
        "total_price": 100.0,
        "total_freight": 15.0,
        "n_payments": 1,
        "total_payment_value": 115.0,
        "max_installments": 1,
    }

    try:
        result = predict_order(sample_order)
        print("Prediction result:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    except InvalidOrderError as e:
        print(f"Invalid order: {e}")
