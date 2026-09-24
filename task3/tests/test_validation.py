from src.validation import validate_order


def make_valid_order():
    return {
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


def test_valid_order_passes_validation():
    order = make_valid_order()

    result = validate_order(order)

    assert result["ok"] is True


def test_invalid_state_is_rejected():
    order = make_valid_order()
    order["customer_state"] = "XX"

    result = validate_order(order)

    assert result["ok"] is False
    assert "customer_state" in result["error"]


def test_negative_price_is_rejected():
    order = make_valid_order()
    order["total_price"] = -100.0

    result = validate_order(order)

    assert result["ok"] is False
    assert "total_price" in result["error"]


def test_missing_required_field_is_rejected():
    order = make_valid_order()
    del order["customer_state"]

    result = validate_order(order)

    assert result["ok"] is False
    assert "customer_state" in result["error"]
