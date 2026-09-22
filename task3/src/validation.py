"""
طبقة تحقق رسمية على سجلات الطلبات الداخلة، مبنية على Great Expectations 1.23.1.

بتغلّف نفس فكرة التحقق اليدوي بـ features.py، بس بشكل رسمي وقابل للتوسعة،
وبترجع نفس نمط {"ok": bool, "error": str} المستخدم بباقي المشروع.

القرار (حسب متطلبات Task 3): REJECT عند فشل أي توقع.
"""

import logging

import pandas as pd
import great_expectations as gx
from great_expectations.core.expectation_suite import ExpectationSuite

logger = logging.getLogger(__name__)

VALID_BRAZIL_STATES = [
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS",
    "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC",
    "SP", "SE", "TO",
]

REQUIRED_COLUMNS = [
    "order_purchase_timestamp",
    "order_estimated_delivery_date",
    "customer_state",
    "seller_ids",
    "n_items",
    "n_distinct_products",
    "total_price",
    "total_freight",
    "n_payments",
    "total_payment_value",
    "max_installments",
]

_context = None
_suite = None
_batch_definition = None


def _get_suite():
    """يبني الـ context والـ Expectation Suite مرة وحدة بس (lazy)، وبيعيد استخدامها."""
    global _context, _suite, _batch_definition
    if _suite is not None:
        return _suite

    logger.info("Building Great Expectations suite for order validation")
    _context = gx.get_context(mode="ephemeral")

    data_source = _context.data_sources.add_pandas("orders_pandas")
    data_asset = data_source.add_dataframe_asset(name="orders")
    _batch_definition = data_asset.add_batch_definition_whole_dataframe("orders_batch")

    _suite = _context.suites.add(ExpectationSuite(name="order_validation_suite"))

    for col in [
        "customer_state", "total_price", "total_freight", "n_items",
        "n_distinct_products", "n_payments", "total_payment_value",
        "max_installments", "order_purchase_timestamp",
        "order_estimated_delivery_date", "seller_ids",
    ]:
        _suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=col))

    _suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="customer_state", value_set=VALID_BRAZIL_STATES
        )
    )
    _suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="total_price", min_value=0, max_value=20000
        )
    )
    _suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="total_freight", min_value=0, max_value=2000
        )
    )
    _suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="total_payment_value", min_value=0, max_value=22000
        )
    )
    _suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="n_items", min_value=1, max_value=50
        )
    )
    _suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="n_distinct_products", min_value=1, max_value=50
        )
    )
    _suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="n_payments", min_value=1, max_value=24
        )
    )
    _suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="max_installments", min_value=1, max_value=24
        )
    )

    return _suite


def validate_order(record: dict) -> dict:
    """
    يتحقق من سجل طلب واحد مقابل الـ Expectation Suite.

    يرجع {"ok": True} لو كل التوقعات نجحت، وإلا
    {"ok": False, "error": "..."}.
    """
    suite = _get_suite()
    df = pd.DataFrame([record]).reindex(columns=REQUIRED_COLUMNS)

    try:
        batch = _batch_definition.get_batch(batch_parameters={"dataframe": df})
        result = batch.validate(suite)
    except Exception as exc:
        logger.exception("Great Expectations validation crashed")
        return {"ok": False, "error": f"Validation error: {exc}"}

    if result.success:
        return {"ok": True}

    failed_columns = [
        r["expectation_config"]["kwargs"].get("column", "?")
        for r in result.results
        if not r["success"]
    ]
    logger.warning("Order validation failed for columns: %s", failed_columns)
    return {
        "ok": False,
        "error": f"Validation failed for field(s): {', '.join(failed_columns)}",
    }
