from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_home_returns_running_message():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Backorder Prediction API Running"}


def test_predict_returns_prediction_and_probability():
    payload = {
        "national_inv": 117,
        "lead_time": 2.0,
        "in_transit_qty": 0,
        "forecast_3_month": 0,
        "forecast_6_month": 0,
        "forecast_9_month": 0,
        "sales_1_month": 0,
        "sales_3_month": 0,
        "sales_6_month": 15,
        "sales_9_month": 15,
        "min_bank": 1,
        "pieces_past_due": 0,
        "perf_6_month_avg": 0.5,
        "perf_12_month_avg": 0.28,
        "local_bo_qty": 0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    body = response.json()
    assert set(body.keys()) == {"prediction", "probability"}
    assert isinstance(body["prediction"], bool)
    assert isinstance(body["probability"], float)
    assert 0.0 <= body["probability"] <= 1.0
