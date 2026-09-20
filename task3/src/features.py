"""
Turn one raw order into the exact feature row the model expects.
 
This module re-implements, function by function, the transformations built
interactively in Notebook 5 (Feature Engineering). Every fitted object
(imputer, scaler, encoder) is loaded from src.artifacts and only ever
*applied* here -- never re-fit. Given the same raw order, this module must
produce the exact same numbers Notebook 5 would have produced for that row.
 
Expected raw input (one order), as a plain dict:
{
    "order_purchase_timestamp": "2018-05-01T10:00:00",   # ISO string or datetime
    "order_estimated_delivery_date": "2018-05-15",        # ISO string or datetime
    "customer_state": "SP",                                # 2-letter state code
    "seller_ids": ["seller_abc", "seller_abc", "seller_xyz"],  # one entry per item
    "n_items": 3,
    "n_distinct_products": 2,
    "total_price": 189.90,
    "total_freight": 24.50,
    "n_payments": 1,
    "total_payment_value": 214.40,
    "max_installments": 3,
}
 
n_distinct_sellers is derived from seller_ids, not passed separately --
this keeps the caller from having to compute two numbers that must always
agree with each other.
"""
 
from datetime import datetime
 
import numpy as np
import pandas as pd
 
from src.artifacts import ARTIFACTS
from src.logging_config import get_logger
 
logger = get_logger(__name__)
 
# Same skewed-columns list used in Notebook 5, section 7 (log-transform).
# This is a fixed, deterministic transformation -- not something fitted --
# so it is safe to hardcode here exactly as in the notebook.
SKEWED_NUMERIC_COLS = [
    "n_payments", "total_price", "n_distinct_sellers",
    "total_payment_value", "total_freight",
]
 
REQUIRED_FIELDS = [
    "order_purchase_timestamp", "order_estimated_delivery_date",
    "customer_state", "seller_ids", "n_items", "n_distinct_products",
    "total_price", "total_freight", "n_payments",
    "total_payment_value", "max_installments",
]
 
NUMERIC_FIELDS = [
    "n_items", "n_distinct_products", "total_price", "total_freight",
    "n_payments", "total_payment_value", "max_installments",
]
 
 
class InvalidOrderError(ValueError):
    """Raised when a raw order dict is missing fields or has bad values.
 
    This is the boundary between "bad input" and "internal bug": code that
    calls build_features() should catch this specific exception to handle
    a malformed request gracefully (e.g. return an HTTP 400 later in Step 7),
    rather than letting the service crash on an unhandled exception.
    """
 
 
def _validate_order(order: dict) -> None:
    missing = [f for f in REQUIRED_FIELDS if f not in order or order[f] is None]
    if missing:
        logger.warning("Rejected order: missing fields %s", missing)
        raise InvalidOrderError(f"Missing required fields: {missing}")
 
    if not isinstance(order["seller_ids"], list) or len(order["seller_ids"]) == 0:
        logger.warning("Rejected order: seller_ids must be a non-empty list")
        raise InvalidOrderError("seller_ids must be a non-empty list")
 
    for field in NUMERIC_FIELDS:
        value = order[field]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            logger.warning("Rejected order: field '%s' is not numeric (got %r)", field, value)
            raise InvalidOrderError(f"Field '{field}' must be numeric, got {value!r}")
        if value < 0:
            logger.warning("Rejected order: field '%s' is negative (%r)", field, value)
            raise InvalidOrderError(f"Field '{field}' cannot be negative, got {value!r}")
 
    try:
        _parse_datetime(order["order_purchase_timestamp"])
        _parse_datetime(order["order_estimated_delivery_date"])
    except (ValueError, TypeError) as e:
        logger.warning("Rejected order: unparseable date (%s)", e)
        raise InvalidOrderError(f"Could not parse a date field: {e}") from e
 
 
def _parse_datetime(value) -> pd.Timestamp:
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value)
    return pd.to_datetime(value)
 
 
def _mode_or_none(values: list):
    """Same tie-breaking rule used in Notebook 1/4/5: most frequent value,
    first alphabetically on ties (pandas' default .mode() behaviour)."""
    s = pd.Series(values).mode()
    return s.iloc[0] if not s.empty else None
 
 
def _resolve_seller_geography(order: dict) -> dict:
    """
    Reproduces the primary_seller_state / same_state logic from Notebook 5,
    section 3 -- but looks sellers up in the static ARTIFACTS.seller_lookup
    table instead of querying the database live.
    """
    seller_ids = order["seller_ids"]
    lookup = ARTIFACTS.seller_lookup.set_index("seller_id")["seller_state"]
 
    states = [lookup.get(sid) for sid in seller_ids]
    unknown_sellers = [sid for sid, state in zip(seller_ids, states) if state is None]
    if unknown_sellers:
        # Not fatal: this is exactly the situation the OneHotEncoder's
        # handle_unknown="ignore" setting was built for in Notebook 5.
        # We log it because a growing number of unknown sellers over time
        # could mean the static lookup table is going stale.
        logger.warning("Seller ID(s) not found in seller_lookup: %s", unknown_sellers)
 
    states = [s for s in states if s is not None]
    primary_seller_state = _mode_or_none(states) if states else None
    same_state = int(order["customer_state"] == primary_seller_state) if primary_seller_state else 0
    n_distinct_sellers = len(set(seller_ids))
 
    return {
        "primary_seller_state": primary_seller_state,
        "same_state": same_state,
        "n_distinct_sellers": n_distinct_sellers,
    }
 
 
def _build_date_features(order: dict) -> dict:
    """Reproduces Notebook 5, section 4 (date-derived features)."""
    purchase_ts = _parse_datetime(order["order_purchase_timestamp"])
    estimated_ts = _parse_datetime(order["order_estimated_delivery_date"])
 
    estimated_days = (estimated_ts - purchase_ts).days
    if estimated_days < 0:
        # Not necessarily invalid (data quality issues exist in the real
        # world too), but worth flagging since it's an unusual order.
        logger.warning(
            "Order has a negative estimated_days (%d): estimated delivery "
            "date is before the purchase date", estimated_days,
        )
 
    month = purchase_ts.month
    weekday = purchase_ts.dayofweek
 
    return {
        "estimated_days": estimated_days,
        "purchase_month_sin": np.sin(2 * np.pi * month / 12),
        "purchase_month_cos": np.cos(2 * np.pi * month / 12),
        "purchase_weekday_sin": np.sin(2 * np.pi * weekday / 7),
        "purchase_weekday_cos": np.cos(2 * np.pi * weekday / 7),
        "is_peak_season": int(month in (11, 2, 3)),
    }
 
 
def build_features(order: dict) -> pd.DataFrame:
    """
    The main entry point of this module: raw order dict -> one-row
    DataFrame with the exact columns the model expects (same order as
    Notebook 5's train_features.parquet, minus the label column).
 
    Raises InvalidOrderError if the input is malformed -- callers should
    catch this specifically rather than letting it propagate as a crash.
    """
    _validate_order(order)
 
    geo = _resolve_seller_geography(order)
    dates = _build_date_features(order)
 
    # --- Raw feature row, before imputation / transform / scale / encode ---
    row = {
        "n_items": order["n_items"],
        "n_distinct_products": order["n_distinct_products"],
        "n_distinct_sellers": geo["n_distinct_sellers"],
        "total_price": order["total_price"],
        "total_freight": order["total_freight"],
        "n_payments": order["n_payments"],
        "total_payment_value": order["total_payment_value"],
        "max_installments": order["max_installments"],
        "estimated_days": dates["estimated_days"],
        "purchase_month_sin": dates["purchase_month_sin"],
        "purchase_month_cos": dates["purchase_month_cos"],
        "purchase_weekday_sin": dates["purchase_weekday_sin"],
        "purchase_weekday_cos": dates["purchase_weekday_cos"],
        "same_state": geo["same_state"],
        "is_peak_season": dates["is_peak_season"],
        "customer_state": order["customer_state"],
        "primary_seller_state": geo["primary_seller_state"] or "UNKNOWN",
    }
    raw_df = pd.DataFrame([row])
 
    numeric_features = ARTIFACTS.feature_list["numeric_features"]
    cyclical_features = ARTIFACTS.feature_list["cyclical_features"]
    binary_features = ARTIFACTS.feature_list["binary_features"]
    categorical_features = ARTIFACTS.feature_list["categorical_features_original"]
 
    # --- 1. Impute missing numeric values (fitted on train, applied here) ---
    raw_df[numeric_features] = ARTIFACTS.numeric_imputer.transform(raw_df[numeric_features])
 
    # --- 2. Fill missing categorical values with the same fixed rule used in NB5 ---
    for col in categorical_features:
        raw_df[col] = raw_df[col].fillna("UNKNOWN")
 
    # --- 3. Log-transform the same skewed columns as Notebook 5 (deterministic) ---
    for col in SKEWED_NUMERIC_COLS:
        raw_df[col] = np.log1p(raw_df[col])
 
    # --- 4. Scale numeric features (fitted on train, applied here) ---
    raw_df[numeric_features] = ARTIFACTS.numeric_scaler.transform(raw_df[numeric_features])
 
    # --- 5. One-hot encode categoricals (fitted on train, unseen values -> all zeros) ---
    encoded = ARTIFACTS.categorical_encoder.transform(raw_df[categorical_features])
    encoded_cols = ARTIFACTS.categorical_encoder.get_feature_names_out(categorical_features)
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=raw_df.index)
 
    # --- 6. Assemble the final row, in the exact column order the model expects ---
    final_df = pd.concat(
        [raw_df[numeric_features + cyclical_features + binary_features], encoded_df],
        axis=1,
    )
 
    return final_df
 
 
if __name__ == "__main__":
    # Manual check: "python -m src.features" from the project root (task3/)
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
    features_df = build_features(sample_order)
    print("Feature row shape:", features_df.shape)
    print(features_df.T)
 


























