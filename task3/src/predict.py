"""
The single public entry point of the inference pipeline: given one raw
order, return whether it is predicted late or on time, with a probability.
 
This is the function a future API route (Step 7) will call directly. It
does not know anything about HTTP, JSON payloads, or web frameworks -- it
only knows how to turn a Python dict into a prediction, using the loaded
model and the same feature-building logic as the notebooks.
 
Every call is logged (input, output, latency, model version), and bad
input never raises an unhandled exception out of this module -- it comes
back as a structured error result instead, so a caller (like a future API
route) can turn it into a clean 400 response instead of a server crash.
"""
 
import time
 
from src.artifacts import ARTIFACTS
from src.features import build_features, InvalidOrderError
from src.logging_config import get_logger
 
logger = get_logger(__name__)
 
 
def predict_order(order: dict) -> dict:
    """
    Parameters
    ----------
    order : dict
        A raw order, in the shape documented at the top of src/features.py.
 
    Returns
    -------
    On success, a dict with:
        ok            : True
        is_late       : int (0 or 1) -- using the threshold chosen in
                        Notebook 6 (not the default 0.5)
        probability   : float -- the model's raw predicted probability of
                        being late
        threshold     : float
        model_name    : str
        model_version : str
 
    On failure (bad input, or an unexpected internal error), a dict with:
        ok      : False
        error   : str -- a human-readable description of what went wrong
    In both cases this function returns normally; it does not raise.
    """
    start_time = time.perf_counter()
    logger.info("Prediction request received: %s", order)
 
    try:
        features_df = build_features(order)
 
        # predict_proba returns [[P(class=0), P(class=1)]]; we want P(late).
        probability = float(ARTIFACTS.model.predict_proba(features_df)[0, 1])
        is_late = int(probability >= ARTIFACTS.threshold)
 
        result = {
            "ok": True,
            "is_late": is_late,
            "probability": probability,
            "threshold": ARTIFACTS.threshold,
            "model_name": ARTIFACTS.model_name,
            "model_version": ARTIFACTS.model_version,
        }
 
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Prediction succeeded: is_late=%s probability=%.4f latency=%.1fms model=%s v%s",
            result["is_late"], result["probability"], latency_ms,
            ARTIFACTS.model_name, ARTIFACTS.model_version,
        )
        return result
 
    except InvalidOrderError as e:
        # Bad input: the caller's fault, not a bug. Log at WARNING (not
        # ERROR) and return a clean, handleable result.
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.warning("Prediction rejected: %s (latency=%.1fms)", e, latency_ms)
        return {"ok": False, "error": str(e)}
 
    except Exception as e:  # noqa: BLE001 -- last line of defense
        # Anything else is unexpected (a bug, a corrupted artifact, etc.).
        # We still must not let it crash the whole service -- log it with
        # a full traceback so it can be diagnosed, and return a generic,
        # non-leaky error message to the caller.
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "Unexpected error during prediction (latency=%.1fms): %s",
            latency_ms, e, exc_info=True,
        )
        return {"ok": False, "error": "Internal error while making the prediction."}
 
 
if __name__ == "__main__":
    # Manual check: "python -m src.predict" from the project root (task3/)
 
    print("--- Valid order ---")
    valid_order = {
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
    print(predict_order(valid_order))
 
    print("\n--- Invalid order (missing fields) ---")
    invalid_order = {"customer_state": "SP"}
    print(predict_order(invalid_order))
 


























