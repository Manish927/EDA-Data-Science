from fastapi import FastAPI
from api.schema import BackorderRequest
from api.predictor import predict_backorder


app = FastAPI(
    title="Backorder Prediction API"
)


@app.get("/")
def home():

    return {"message": "Backorder Prediction API Running"}


@app.post("/predict")
def predict(request: BackorderRequest):
    prediction, probability = predict_backorder(request.model_dump())
    return {
        "prediction": bool(prediction),
        "probability": float(probability)
    }


# To run the API, use the command: uvicorn api.main:app --reload
# Serve the API at: http://127.0.0.1:8000


# FastAPI automatically generates interactive API documentation at: http://127.0.0.1:8000/docs

# Json Request Example:
# {
# "national_inv": 50,
# "lead_time": 5,
# "in_transit_qty": 10,
# "forecast_3_month": 40,
# "forecast_6_month": 80,
# "forecast_9_month": 120,
# "sales_1_month": 10,
# "sales_3_month": 30,
# "sales_6_month": 50,
# "sales_9_month": 70,
# "min_bank": 20,
# "pieces_past_due": 0,
# "perf_6_month_avg": 0.9,
# "perf_12_month_avg": 0.85,
# "local_bo_qty": 0
# }