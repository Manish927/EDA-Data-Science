import pandas as pd
from pathlib import Path
from src.save_model import load_model


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "xgboost_model.pkl"
model = load_model(str(MODEL_PATH))


def _prepare_features(data):
    df = pd.DataFrame([data])
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_cols]
    return df


def predict_backorder(data):

    df = _prepare_features(data)

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0][1]

    return prediction, probability
