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
