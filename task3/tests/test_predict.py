from src.predict import predict_order


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


def test_valid_order_returns_prediction():
    result = predict_order(make_valid_order())

    assert result["ok"] is True
    assert result["is_late"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0
    assert 0.0 < result["threshold"] < 1.0
    assert result["model_name"] == "late_delivery_classifier"
    assert result["model_version"] == "champion"


def test_missing_fields_return_structured_error():
    result = predict_order({"customer_state": "SP"})

    assert result["ok"] is False
    assert "error" in result
    assert isinstance(result["error"], str)


def test_invalid_data_returns_structured_error():
    order = make_valid_order()
    order["customer_state"] = "XX"
    order["total_price"] = -100.0

    result = predict_order(order)

    assert result["ok"] is False
    assert "error" in result

